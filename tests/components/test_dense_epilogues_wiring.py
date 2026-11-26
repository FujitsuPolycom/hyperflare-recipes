# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU verification for selective dense MXFP8 epilogue wiring and base-b12x fallbacks."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import SimpleNamespace

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from b12x.gemm.blockscaled._linear import MXFP8LinearWeight
from b12x.preparation.device import DeviceIdentity
from b12x.quantization import mxfp8

from hyperflare_recipes.components import (
    attention_output,
    block_fp8_weights,
    dense_epilogues,
    expert_overlap,
    mlp,
    projection_overlap,
)
from hyperflare_recipes.components.weights import MLPWeights
from hyperflare_recipes.recipes.deepseek41.attention import PreparedAttention
from hyperflare_recipes.recipes.deepseek41.decoder import PreparedLayer
from hyperflare_recipes.recipes.glm53.moe import PreparedGlmMoe


def _fake_mxfp8_linear_weight(out_features: int, in_features: int, *, padded_in_features: int | None = None):
    padded_k = in_features if padded_in_features is None else padded_in_features
    values = torch.zeros((out_features, padded_k), dtype=torch.float8_e4m3fn)
    scale_rows = torch.full((1, out_features, padded_k // 32), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    row_tiles = (out_features + 127) // 128
    k_tiles = (padded_k + 127) // 128
    scale_mma_phys = torch.full((1, row_tiles, k_tiles, 32, 4, 4), 127, dtype=torch.uint8)
    scale_mma = scale_mma_phys.view(torch.float8_e8m0fnu).permute(3, 4, 1, 5, 2, 0)
    rows = mxfp8.MXFP8Rows(
        values=values,
        scale_rows=scale_rows,
        scale_mma=scale_mma,
    )
    return MXFP8LinearWeight(
        weight=rows,
        out_features=out_features,
        in_features=in_features,
        padded_in_features=padded_k,
    )


def _reference_swiglu_bf16(gate_up: torch.Tensor, *, limit: float = float("inf"), round_silu: bool = True):
    half = gate_up.shape[1] // 2
    gate = gate_up[:, :half].float()
    up = gate_up[:, half:].float()
    if math.isfinite(limit):
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    silu = gate * torch.sigmoid(gate)
    if round_silu:
        silu = silu.to(torch.bfloat16).float()
    return (silu * up).to(torch.bfloat16)


def _reference_quantize_mxfp8_cpu(x_bf16: torch.Tensor, padded_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic per-32-element MXFP8 quantization matching UE8M0 block scaling."""
    rows, k = x_bf16.shape
    x = torch.zeros((rows, padded_k), dtype=torch.float32)
    x[:, :k] = x_bf16.float()
    blocks = x.view(rows, padded_k // 32, 32)
    amax = blocks.abs().amax(dim=-1)
    # Power-of-two scale matching E4M3 max normal 448.0.
    scale_val = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))
    exp = torch.ceil(torch.log2(scale_val)).clamp(-127, 127)
    scale_f32 = torch.exp2(exp)
    q = (blocks / scale_f32.unsqueeze(-1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    u8 = (exp.to(torch.int32) + 127).to(torch.uint8)
    return q.view(rows, padded_k), u8


def _reference_dequantize_mxfp8_cpu(values_fp8: torch.Tensor, scale_u8: torch.Tensor) -> torch.Tensor:
    rows, k = values_fp8.shape
    scale_f32 = torch.exp2(scale_u8.to(torch.int32) - 127)
    return (values_fp8.float().view(rows, k // 32, 32) * scale_f32.unsqueeze(-1)).view(rows, k)


def _reference_rotate_tail_split(x_bf16: torch.Tensor, angles_f32: torch.Tensor, *, head_dim: int, rope_dim: int = 64):
    """Exact BF16 tail-64 split-cos/sin rotary reference matching DenseGemmKernel / _rotate_grouped."""
    rows = x_bf16.shape[0]
    heads = x_bf16.shape[1] // head_dim
    out = x_bf16.clone().view(rows, heads, head_dim)
    tail = out[:, :, head_dim - rope_dim :].float()
    half = rope_dim // 2
    cos = angles_f32[:, :half].unsqueeze(1)
    sin = angles_f32[:, half:].unsqueeze(1)
    a = tail[:, :, 0::2]
    b = tail[:, :, 1::2]
    out[:, :, head_dim - rope_dim :: 2] = (a * cos - b * sin).to(torch.bfloat16)
    out[:, :, head_dim - rope_dim + 1 :: 2] = (a * sin + b * cos).to(torch.bfloat16)
    return out.view(rows, -1)


def _mock_lowering(m: int, n: int, k: int, *, swap_ab: bool = False, split_k_slices: int = 1):
    return SimpleNamespace(
        m=m,
        n=n,
        k=k,
        l=1,
        kernel_c_l=1,
        ab_dtype="float8_e4m3fn",
        sf_dtype="float8_e8m0fnu",
        c_dtype="bfloat16",
        alpha_dtype="float32",
        sf_vec_size=32,
        mma_k=32,
        tile_k=128 if k % 128 == 0 else 64,
        mma_tiler_mn=(128, 128),
        cluster_shape_mn=(1, 1),
        policy=SimpleNamespace(split_k_slices=split_k_slices),
        sm_count=170,
        load_path="tma",
        swap_ab=swap_ab,
        sfb_k_reuse=False,
        b_tile_major=False,
        alpha_is_one=True,
        plain_fp8=False,
        block_fp8=False,
        is_mxfp6=False,
        target_occupancy_override=None,
        programmatic_serialization=False,
        evict_first=False,
    )


class _DummySession:
    def __init__(self, *, native_dense_activations: bool = False):
        self.native_dense_activations = native_dense_activations
        self.requests = []
        self.finalizers = []

    def prepare(self, requests):
        self.requests.extend(requests)

    def register_finalizer(self, callback, **_kwargs):
        self.finalizers.append(callback)


@pytest.fixture(autouse=True)
def _clean_rope_runner_cache():
    before = dict(dense_epilogues._DENSE_ROPE_RUNNERS)
    yield
    dense_epilogues._DENSE_ROPE_RUNNERS.clear()
    dense_epilogues._DENSE_ROPE_RUNNERS.update(before)


# ---------------------------------------------------------------------------
# 1. Dynamic feature detection & base-b12x backwards compatibility
# ---------------------------------------------------------------------------


def test_feature_probes_detect_dense_epilogue_capabilities():
    assert dense_epilogues.supports_swiglu_quantize_rows() is True
    assert dense_epilogues.supports_k_alignment_64() is True
    assert dense_epilogues.supports_dense_residual_epilogue() is True
    assert dense_epilogues.supports_dense_rope_epilogue() is True
    assert dense_epilogues.supports_wo_quantize_c_multirow() is True


def test_feature_probes_fall_back_cleanly_on_unmodified_base_b12x(monkeypatch):
    from b12x._lib.dense_gemm import compile as dense_compile
    from b12x._lib.dense_gemm import execution as dense_exec
    from b12x._lib.quant import mxfp8_rows
    from b12x.gemm.blockscaled import _linear as bs_linear
    from b12x.gemm.wo_projection import _preparation as wo_prep

    monkeypatch.setattr(mxfp8, "swiglu_quantize_rows", None)
    monkeypatch.setattr(mxfp8_rows, "_get_compiled_swiglu_mxfp8_rows_quant", None)
    monkeypatch.setattr(bs_linear, "pack_mxfp8_linear_weight", lambda values, scales, allocate=None: None)
    monkeypatch.setattr(bs_linear, "pack_weight", lambda values, scales, recipe="mxfp8", allocate=None: None)
    monkeypatch.setattr(dense_exec, "dense_gemm", lambda a, b, sfa, sfb, c, alpha: None)
    monkeypatch.setattr(dense_compile, "_get_compiled_dense_gemm", lambda n, k, batch: None)
    monkeypatch.setattr(wo_prep, "plan", lambda caps, invocation=None: None)

    assert dense_epilogues.supports_swiglu_quantize_rows() is False
    assert dense_epilogues.supports_k_alignment_64() is False
    assert dense_epilogues.supports_dense_residual_epilogue() is False
    assert dense_epilogues.supports_dense_rope_epilogue() is False
    assert dense_epilogues.supports_wo_quantize_c_multirow() is False
    assert dense_epilogues.should_pack_unpadded_k64(576, torch.device("cpu")) is False

    # Base-b12x fallback must never raise AttributeError or TypeError.
    plan = SimpleNamespace(
        handle=9001,
        _prepared=SimpleNamespace(
            state=SimpleNamespace(
                core=SimpleNamespace(lowering=_mock_lowering(128, 4096, 1280), device=torch.device("cpu"), alpha_one=None)
            )
        ),
    )
    assert dense_epilogues.register_dense_gemm_rope(plan, head_dim=512, rope_dim=64) is False

    down_w = _fake_mxfp8_linear_weight(5120, 576, padded_in_features=640)
    assert (
        dense_epilogues.supports_fused_swiglu_down(
            torch.zeros((128, 576), dtype=torch.bfloat16),
            down_w,
            torch.zeros((128, 5120), dtype=torch.bfloat16),
            activation_mode="auto",
            down_bias=None,
            session=_DummySession(),
        )
        is False
    )


# ---------------------------------------------------------------------------
# 2. Exact Q-RoPE regime gating & standalone/overlap wiring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rows", "head_dim", "expected"),
    [
        # q_b (head_dim=512): ON for M <= 1024, OFF for M > 1024
        (32, 512, True),
        (128, 512, True),
        (512, 512, True),
        (1024, 512, True),
        (1025, 512, False),
        (2048, 512, False),
        (4096, 512, False),
        # index_q (head_dim=128): ON for M <= 128, OFF for M > 128
        (32, 128, True),
        (128, 128, True),
        (129, 128, False),
        (1024, 128, False),
        (2048, 128, False),
        # Unsupported dimensions
        (32, 256, False),
        (0, 512, False),
    ],
)
def test_dense_rope_regime_gating_thresholds(rows, head_dim, expected):
    assert dense_epilogues.should_fuse_dense_rope(rows, head_dim=head_dim, rope_dim=64) is expected


def test_register_dense_gemm_rope_respects_regimes_and_rejects_incompatible_lowerings(monkeypatch):
    compiled = []

    def fake_compile(lowering, device_ordinal, **kwargs):
        compiled.append((lowering.m, lowering.n, kwargs["rope_head_dim"], kwargs["rope_dim"], kwargs["rope_split"]))
        return lambda **call_kw: None

    monkeypatch.setattr(dense_epilogues, "_compile_dense_gemm_from_lowering", fake_compile)

    def make_plan(handle, m, n, *, swap_ab=False, split_k=1):
        return SimpleNamespace(
            handle=handle,
            _prepared=SimpleNamespace(
                state=SimpleNamespace(
                    core=SimpleNamespace(
                        lowering=_mock_lowering(m, n, 1280, swap_ab=swap_ab, split_k_slices=split_k),
                        device=torch.device("cpu"),
                        alpha_one=torch.ones((1,), dtype=torch.float32),
                    )
                )
            ),
        )

    # q_b: ON at 32, 128, 1024; OFF at 2048
    assert dense_epilogues.register_dense_gemm_rope(make_plan(101, 32, 8192), head_dim=512) is True
    assert dense_epilogues.register_dense_gemm_rope(make_plan(102, 128, 8192), head_dim=512) is True
    assert dense_epilogues.register_dense_gemm_rope(make_plan(103, 1024, 8192), head_dim=512) is True
    assert dense_epilogues.register_dense_gemm_rope(make_plan(104, 2048, 8192), head_dim=512) is False

    # index_q: ON at 32, 128; OFF at 1024, 2048
    assert dense_epilogues.register_dense_gemm_rope(make_plan(201, 32, 4096), head_dim=128) is True
    assert dense_epilogues.register_dense_gemm_rope(make_plan(202, 128, 4096), head_dim=128) is True
    assert dense_epilogues.register_dense_gemm_rope(make_plan(203, 1024, 4096), head_dim=128) is False
    assert dense_epilogues.register_dense_gemm_rope(make_plan(204, 2048, 4096), head_dim=128) is False

    # Incompatible lowerings must be rejected cleanly.
    assert dense_epilogues.register_dense_gemm_rope(make_plan(301, 32, 8192, swap_ab=True), head_dim=512) is False
    assert dense_epilogues.register_dense_gemm_rope(make_plan(302, 32, 8192, split_k=2), head_dim=512) is False

    assert [(m, hd) for m, _, hd, _, _ in compiled] == [
        (32, 512),
        (128, 512),
        (1024, 512),
        (32, 128),
        (128, 128),
    ]


def test_encoded_dense_rope_skips_standalone_rotary_and_matches_zero_ulp(monkeypatch):
    """Fused Q-RoPE runner skips standalone _rotate_grouped and matches unfused reference bit-for-bit."""
    rows, k, head_dim, heads, rope_dim = 32, 128, 512, 2, 64
    n = heads * head_dim
    gen = torch.Generator().manual_seed(2026)
    a_bf16 = torch.randn((rows, k), dtype=torch.bfloat16, generator=gen)
    w_bf16 = torch.randn((n, k), dtype=torch.bfloat16, generator=gen) * 0.25
    a_fp8, a_scale = _reference_quantize_mxfp8_cpu(a_bf16, k)
    w_fp8, w_scale = _reference_quantize_mxfp8_cpu(w_bf16, k)
    theta = torch.randn((rows, rope_dim // 2), dtype=torch.float32, generator=gen)
    angles = torch.cat((torch.cos(theta), torch.sin(theta)), dim=-1).contiguous()

    a_deq = _reference_dequantize_mxfp8_cpu(a_fp8, a_scale)
    w_deq = _reference_dequantize_mxfp8_cpu(w_fp8, w_scale)
    unfused_gemm = (a_deq @ w_deq.T).to(torch.bfloat16)
    expected = _reference_rotate_tail_split(unfused_gemm, angles, head_dim=head_dim, rope_dim=rope_dim)

    row_tiles_a = (rows + 127) // 128
    row_tiles_w = (n + 127) // 128
    k_tiles = (k + 127) // 128
    mma_a = torch.full((1, row_tiles_a, k_tiles, 32, 4, 4), 127, dtype=torch.uint8)
    mma_w = torch.full((1, row_tiles_w, k_tiles, 32, 4, 4), 127, dtype=torch.uint8)

    fused_calls = []

    def fake_runner(*, a_tensor_gpu, b_tensor_gpu, c_tensor_gpu, rope_cos_sin_gpu, **_kwargs):
        fused_calls.append((tuple(a_tensor_gpu.shape), tuple(b_tensor_gpu.shape), tuple(rope_cos_sin_gpu.shape)))
        out = _reference_rotate_tail_split(
            (_reference_dequantize_mxfp8_cpu(a_tensor_gpu.squeeze(-1), a_scale)
             @ _reference_dequantize_mxfp8_cpu(b_tensor_gpu.squeeze(-1), w_scale).T).to(torch.bfloat16),
            rope_cos_sin_gpu,
            head_dim=head_dim,
            rope_dim=rope_dim,
        )
        c_tensor_gpu.squeeze(-1).copy_(out)

    plan_handle = 777
    dense_epilogues._DENSE_ROPE_RUNNERS[(plan_handle, head_dim, rope_dim, True, False)] = (
        fake_runner,
        torch.ones((1,), dtype=torch.float32),
        _mock_lowering(rows, n, k),
    )

    standalone_rotary_calls = []
    monkeypatch.setattr(
        projection_overlap,
        "_rotate_grouped",
        SimpleNamespace(__getitem__=lambda self, grid: lambda *args, **kw: standalone_rotary_calls.append(grid)),
    )
    monkeypatch.setattr(
        dense_epilogues,
        "_rotate_grouped",
        SimpleNamespace(__getitem__=lambda self, grid: lambda *args, **kw: standalone_rotary_calls.append(grid)),
    )

    # 1) Direct ConcurrentAttentionAux _run_encoded_with_rope path
    out_overlap = torch.empty((rows, n), dtype=torch.bfloat16)
    projection_overlap._run_encoded_with_rope(
        a_fp8, mma_a, w_fp8, mma_w, out_overlap, torch.empty(0, dtype=torch.uint8),
        plan_handle, angles, head_dim, rope_dim,
    )
    assert len(fused_calls) == 1
    assert standalone_rotary_calls == []
    assert torch.equal(out_overlap.view(torch.int16), expected.view(torch.int16))

    # 2) Standalone PreparedAttention._project_q_b_with_rope + make_fx tracing
    proj = SimpleNamespace(
        output_layout=((rows, n), (n, 1), torch.bfloat16),
        workspace_nbytes=0,
        parts=(w_fp8, mma_w, None),
        plan=SimpleNamespace(handle=plan_handle),
        encoded_layout=((a_fp8.shape, a_fp8.dtype), ((1, rows, k // 32), torch.uint8), (mma_a.shape, mma_a.dtype)),
        quantizer_plan=None,
    )
    attn = object.__new__(PreparedAttention)
    attn.projections = {"q_b": proj}
    attn.query_shape = (rows, heads, head_dim)
    attn.q_b_fused_rope = False
    attn.q_b_dense_rope = True
    attn.angles = SimpleNamespace(apply=lambda *args, **kw: standalone_rotary_calls.append("angles.apply"))

    traced = make_fx(
        lambda q_low, vb, rb, mb, ang: attn._project_q_b_with_rope(q_low, ang, encoded_bases=(vb, rb, mb))
    )(a_bf16, a_fp8, torch.zeros((1, rows, k // 32), dtype=torch.uint8), mma_a, angles)
    op_names = [str(node.target) for node in traced.graph.nodes if node.op == "call_function"]
    assert "flare.mxfp8_encoded_projection_rope.default" in op_names

    out_traced = traced(a_bf16, a_fp8, torch.zeros((1, rows, k // 32), dtype=torch.uint8), mma_a, angles)
    assert standalone_rotary_calls == []
    assert torch.equal(out_traced.view(rows, n).view(torch.int16), expected.view(torch.int16))

    # Negative sensitivity check: corrupting angles must break bit-exact parity.
    bad_angles = angles.clone()
    bad_angles[0, 0] += 0.25
    out_bad = traced(a_bf16, a_fp8, torch.zeros((1, rows, k // 32), dtype=torch.uint8), mma_a, bad_angles)
    assert not torch.equal(out_bad.view(rows, n).view(torch.int16), expected.view(torch.int16))


def test_prepared_attention_finalize_standalone_rope_gates_qb_and_index_q(monkeypatch):
    """PreparedAttention._finalize_standalone_rope gates q_b (<=1024) and index_q (<=128) independently."""
    monkeypatch.setattr(dense_epilogues, "_compile_dense_gemm_from_lowering", lambda *a, **kw: (lambda **c: None))

    def build_attn(q_rows: int, iq_rows: int, *, handle_base: int):
        attn = object.__new__(PreparedAttention)
        attn.row_shard_attention = False
        attn.concurrent_query = None
        attn.angles = SimpleNamespace(split=True)
        attn.index_angles = SimpleNamespace(split=True)
        attn.selectors = [object()]
        attn.q_b_fused_rope = False
        attn.q_b_dense_rope = False
        attn.index_query_fused_rope = False
        attn.index_query_dense_rope = False
        q_plan = SimpleNamespace(
            handle=handle_base,
            _prepared=SimpleNamespace(
                state=SimpleNamespace(
                    core=SimpleNamespace(
                        lowering=_mock_lowering(q_rows, 8192, 1280),
                        device=torch.device("cpu"),
                        alpha_one=torch.ones((1,), dtype=torch.float32),
                    )
                )
            ),
        )
        iq_plan = SimpleNamespace(
            handle=handle_base + 1,
            _prepared=SimpleNamespace(
                state=SimpleNamespace(
                    config=SimpleNamespace(mode="quantized"),
                    query=SimpleNamespace(recipe="mxfp8", num_tokens=iq_rows),
                    device=torch.device("cpu"),
                    dense=SimpleNamespace(
                        lowering=_mock_lowering(iq_rows, 4096, 1280),
                        alpha_one=torch.ones((1,), dtype=torch.float32),
                    ),
                )
            ),
        )
        attn.projections = {"q_b": SimpleNamespace(plan=q_plan, encoded=object(), encoded_layout=())}
        attn.index_query = SimpleNamespace(plan=iq_plan, encoded=None, encoded_layout=None)
        attn._finalize_standalone_rope()
        return attn

    # M=128: both q_b and index_q ON
    a128 = build_attn(128, 128, handle_base=400)
    assert a128.q_b_dense_rope is True
    assert a128.index_query_dense_rope is True

    # M=1024: q_b ON, index_q OFF
    a1024 = build_attn(1024, 1024, handle_base=500)
    assert a1024.q_b_dense_rope is True
    assert a1024.index_query_dense_rope is False

    # M=2048: both q_b and index_q OFF
    a2048 = build_attn(2048, 2048, handle_base=600)
    assert a2048.q_b_dense_rope is False
    assert a2048.index_query_dense_rope is False


# ---------------------------------------------------------------------------
# 3. SwiGLU + quant + K=576 (k_alignment=64) + BF16 residual epilogue wiring
# ---------------------------------------------------------------------------


def test_k576_compatibility_across_tc_gemv_and_dense_gemm_on_sm120_and_sm121():
    """Unpadded K=576 works with TC-GEMV (M<=28) and DenseGemm tile_k=64 (M>=32) on SM120 and SM121."""
    from b12x._lib.dense_gemm.policy import (
        _select_default_dense_gemm_plan,
        _select_mxfp8_tile_k,
        _validate_mxfp8_bk64_plan,
    )
    from b12x.gemm.blockscaled._tuning import BlockscaledQuery, _default_config, _validate_config

    devices = [
        ("sm120", DeviceIdentity(vendor="nvidia", compute_capability=(12, 0), sm_count=170, product_name="RTX 5090")),
        ("sm121", DeviceIdentity(vendor="nvidia", compute_capability=(12, 1), sm_count=48, product_name="GB10")),
    ]
    for _label, identity in devices:
        for m in (1, 8, 16, 28):
            query = BlockscaledQuery(
                recipe="mxfp8",
                num_tokens=m,
                in_features=576,
                padded_in_features=576,
                out_features=5120,
                activation_mode="auto",
            )
            cfg = _default_config(query, identity)
            _validate_config(query, cfg, identity)
            assert cfg.mode == "a16"
            assert cfg.a16_backend == "tc_gemv"

        for m in (32, 128, 512, 1024, 2048, 4096, 8192):
            plan = _select_default_dense_gemm_plan(m, 5120, 576, identity.sm_count, is_mxfp8=True, expected_m=m)
            tile_k = _select_mxfp8_tile_k(m, 5120, 576, m, identity.sm_count)
            assert tile_k == 64
            _validate_mxfp8_bk64_plan(tile_k, plan.mma_tiler_mn, plan.swap_ab, allow_short_m=True)


def test_block_fp8_weight_loader_passes_k_alignment_64_selectively(monkeypatch):
    from b12x.gemm import blockscaled

    recorded = []

    def fake_pack_weight(values, scales, *, recipe="mxfp8", **options):
        recorded.append((tuple(values.shape), tuple(scales.shape), recipe, dict(options)))
        return SimpleNamespace(values=values, scales=scales, options=options)

    monkeypatch.setattr(blockscaled, "pack_weight", fake_pack_weight)

    loader = object.__new__(block_fp8_weights.BlockFP8WeightLoader)
    loader.config = SimpleNamespace(quantization={"weight_block_size": [32, 32]})
    loader.device = torch.device("cpu")
    loader._session = SimpleNamespace(flush=lambda: None)
    loader._require_open = lambda: None
    loader._copy = lambda dst, src: dst.copy_(src)
    w_src = torch.zeros((64, 576), dtype=torch.float8_e4m3fn)
    s_src = torch.full((2, 18), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    loader._sources = {"shared.w2.weight": w_src, "shared.w2.scale": s_src}

    loader.load_linear_group(("shared.w2",), column_range=(0, 576), temporary=True, k_alignment=64)
    loader.load_linear_group(("shared.w2",), column_range=(0, 512), temporary=True)

    assert recorded[0][0] == (64, 576)
    assert recorded[0][3].get("k_alignment") == 64
    assert recorded[1][0] == (64, 512)
    assert "k_alignment" not in recorded[1][3]

    # should_pack_unpadded_k64 gating
    assert dense_epilogues.should_pack_unpadded_k64(576, torch.device("cpu")) is True
    assert dense_epilogues.should_pack_unpadded_k64(2048, torch.device("cpu")) is False
    assert dense_epilogues.should_pack_unpadded_k64(640, torch.device("cpu")) is False
    monkeypatch.setattr(dense_epilogues.sm103_projection, "is_sm103", lambda dev: True)
    assert dense_epilogues.should_pack_unpadded_k64(576, torch.device("cpu")) is False


@pytest.mark.parametrize(
    ("rows", "activation_mode", "native_dense", "expected"),
    [
        (8, "auto", False, False),
        (28, "auto", False, False),
        (29, "auto", False, True),
        (32, "auto", False, True),
        (4096, "auto", False, True),
        (8, "quantized", False, True),
        (8, "auto", True, True),
        (128, "a16", False, False),
    ],
)
def test_supports_fused_swiglu_down_row_and_mode_gating(rows, activation_mode, native_dense, expected):
    down_w = _fake_mxfp8_linear_weight(128, 576, padded_in_features=576)
    actual = dense_epilogues.supports_fused_swiglu_down(
        torch.zeros((rows, 576), dtype=torch.bfloat16),
        down_w,
        torch.zeros((rows, 128), dtype=torch.bfloat16),
        activation_mode=activation_mode,
        down_bias=None,
        session=_DummySession(native_dense_activations=native_dense),
    )
    assert actual is expected


def test_prepared_mlp_fused_swiglu_down_zero_ulp_residual_and_make_fx(monkeypatch):
    """PreparedMLP wires PreparedFusedSwiGLUDown at M>=29, matches 0-ULP with/without residual, and traces in make_fx."""
    rows, hidden, width = 32, 64, 576
    gen = torch.Generator().manual_seed(42)
    source = torch.randn((rows, hidden), dtype=torch.bfloat16, generator=gen) * 0.5
    up_weight = torch.randn((2 * width, hidden), dtype=torch.bfloat16, generator=gen) * 0.25
    down_dense = torch.randn((hidden, width), dtype=torch.bfloat16, generator=gen) * 0.25
    w_fp8, w_scale = _reference_quantize_mxfp8_cpu(down_dense, width)
    down_weight = _fake_mxfp8_linear_weight(hidden, width, padded_in_features=width)
    down_weight.weight.values.copy_(w_fp8)
    residual = torch.randn((rows, hidden), dtype=torch.bfloat16, generator=gen) * 0.5

    # Compute exact unfused reference
    gate_up_ref = (source @ up_weight.T).to(torch.bfloat16)
    swiglu_ref = _reference_swiglu_bf16(gate_up_ref, round_silu=False)
    act_fp8_ref, act_scale_ref = _reference_quantize_mxfp8_cpu(swiglu_ref, width)
    gemm_ref = (
        _reference_dequantize_mxfp8_cpu(act_fp8_ref, act_scale_ref)
        @ _reference_dequantize_mxfp8_cpu(w_fp8, w_scale).T
    ).to(torch.bfloat16)
    gemm_res_ref = (gemm_ref.float() + residual.float()).to(torch.bfloat16)

    residual_kernel_calls = []

    def fake_materialize(query, device):
        def swiglu_quant(gate_up, values, scale_rows, scale_mma):
            mid = _reference_swiglu_bf16(gate_up, limit=query.limit if query.limit_finite else float("inf"),
                                         round_silu=query.round_silu)
            q, s = _reference_quantize_mxfp8_cpu(mid, query.padded_k)
            values.copy_(q)
            scale_rows.copy_(s)

        def dense(*, a_tensor_gpu, b_tensor_gpu, c_tensor_gpu, **_kw):
            # Re-derive activation scale from scale_rows is unneeded; store scale via closure? Better: read a_fp8 directly!
            pass

        # Bind scale_rows cleanly via a custom state matching _FusedSwiGLUDownState.run
        @dataclass(frozen=True)
        class _CpuState:
            query: object
            device: torch.device

            def run(self, gate_up, values, scale_rows, scale_mma, weight_values, weight_mma_base,
                    output, *, residual=None, stream=None):
                swiglu_quant(gate_up, values, scale_rows.view(gate_up.shape[0], -1), scale_mma.view(-1))
                out = (
                    _reference_dequantize_mxfp8_cpu(values, scale_rows.view(gate_up.shape[0], -1))
                    @ _reference_dequantize_mxfp8_cpu(weight_values, w_scale).T
                ).to(torch.bfloat16)
                if residual is not None:
                    residual_kernel_calls.append(tuple(residual.shape))
                    out = (out.float() + residual.float()).to(torch.bfloat16)
                output.copy_(out)

        return _CpuState(query=query, device=device)

    from b12x.gemm import bf16_gemv

    monkeypatch.setattr(dense_epilogues, "_materialize_fused_swiglu_down", fake_materialize)
    monkeypatch.setattr(bf16_gemv, "mm", lambda x, weight, *, out, plan, bias=None: torch.mm(x, weight.T, out=out))

    session = _DummySession()
    gate_up_buf = torch.zeros((rows, 2 * width), dtype=torch.bfloat16)
    inter_buf = torch.zeros((rows, width), dtype=torch.bfloat16)
    out_buf = torch.zeros((rows, hidden), dtype=torch.bfloat16)

    prepared = mlp.PreparedMLP(
        source,
        MLPWeights(up_weight, down_weight),
        gate_up_buf,
        inter_buf,
        out_buf,
        session=session,
        name="shared",
        workspace=None,
        round_silu=False,
        activation_mode="auto",
    )
    assert prepared.fused_swiglu_down is not None
    assert prepared.activation_plan is None
    assert prepared.supports_residual is True
    assert prepared.fused_swiglu_down.k_alignment == 64

    # Install materialized CPU mock state on the native plan so require_prepared succeeds on CPU.
    plan = prepared.fused_swiglu_down.plan
    query = dense_epilogues.FusedSwiGLUDownQuery(
        rows=rows,
        width=width,
        padded_k=width,
        out_features=hidden,
        limit_finite=False,
        limit=0.0,
        round_silu=False,
        clamp_after_silu=False,
        swiglu_alpha=1.0,
        swiglu_beta=0.0,
        residual_capable=True,
    )
    cpu_state = fake_materialize(query, torch.device("cpu"))
    plan._install(
        SimpleNamespace(config=None, state=cpu_state, scratch=(), closed=False, device=SimpleNamespace(ordinal=0))
    )
    prepared.functional(session=session)
    assert dense_epilogues.is_fused_swiglu_down_plan(prepared.fused_swiglu_down.plan.handle) is True

    # Trace both plain and residual variants through make_fx.
    traced_plain = make_fx(lambda x: prepared(x))(source)
    plain_ops = [str(n.target) for n in traced_plain.graph.nodes if n.op == "call_function"]
    assert "flare.swiglu_mxfp8_down.default" in plain_ops
    out_plain = traced_plain(source)
    assert torch.equal(out_plain.view(torch.int16), gemm_ref.view(torch.int16))

    traced_res = make_fx(lambda x, r: prepared(x, residual=r))(source, residual)
    res_ops = [str(n.target) for n in traced_res.graph.nodes if n.op == "call_function"]
    assert "flare.swiglu_mxfp8_down_residual.default" in res_ops
    out_res = traced_res(source, residual)
    assert len(residual_kernel_calls) >= 1
    assert torch.equal(out_res.view(torch.int16), gemm_res_ref.view(torch.int16))

    # Negative sensitivity check: corrupting residual must break bit-exact parity.
    bad_res = residual.clone()
    bad_res[0, 0] += 1.0
    out_bad = traced_res(source, bad_res)
    assert not torch.equal(out_bad.view(torch.int16), gemm_res_ref.view(torch.int16))


def test_decoder_and_glm_moe_wire_shared_residual_without_colliding_with_fused_addend():
    """DeepSeek-V4.1 _run_ffn_gathered and GLM-5.3 PreparedGlmMoe pass residual=routed when supported."""
    rows, hidden = 32, 64
    routed_val = torch.ones((rows, hidden), dtype=torch.bfloat16) * 2.0
    shared_val = torch.ones((rows, hidden), dtype=torch.bfloat16) * 3.0

    # 1) DeepSeek-V4.1 _run_ffn_gathered
    seen_residuals = []

    class _FakeShared:
        def __init__(self, supports_res: bool):
            self.supports_residual = supports_res

        def __call__(self, x, *, encoded_bases=None, residual=None):
            seen_residuals.append(residual)
            return shared_val if residual is None else (shared_val + residual)

    layer = object.__new__(PreparedLayer)
    layer.ffn_quantizer_plan = None
    layer.experts = lambda x, ids, w, **kw: routed_val
    layer.shared = _FakeShared(True)
    meta = torch.zeros((rows, 4), dtype=torch.bfloat16)
    x = torch.zeros((rows, hidden), dtype=torch.bfloat16)
    out_fused, _ = layer._run_ffn_gathered(x, meta)
    assert seen_residuals[-1] is routed_val
    assert torch.equal(out_fused, routed_val + shared_val)

    layer.shared = _FakeShared(False)
    out_unfused, _ = layer._run_ffn_gathered(x, meta)
    assert seen_residuals[-1] is None
    assert torch.equal(out_unfused, routed_val + shared_val)

    # 2) GLM-5.3 PreparedGlmMoe
    glm = object.__new__(PreparedGlmMoe)
    glm.concurrent_shared = None
    glm.router_projection = lambda s: s
    glm.routing = lambda logits: SimpleNamespace(ids=None, weights=None)
    glm.experts = lambda s, ids, w: routed_val
    glm.shared = _FakeShared(True)
    out_glm = glm(x)
    assert seen_residuals[-1] is routed_val
    assert torch.equal(out_glm, routed_val + shared_val)


# ---------------------------------------------------------------------------
# 4. wo_projection multi-row quantize_c (M = 1..2048) & make_fx / fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rows", [1, 8, 32, 128, 1024, 2048])
def test_wo_projection_multirow_quantize_c_make_fx_and_fallback(monkeypatch, rows):
    """PreparedGroupedOutputProjection forwards M=1..2048 to wo_projection.run_inv_rope_out and falls back cleanly."""
    from b12x.gemm import wo_projection

    planned_caps = []
    inv_rope_calls = []

    class _FakeWoPlan:
        def __init__(self, caps, invocation):
            self.caps = caps
            self.invocation = invocation

        def request(self, **_kw):
            return self

        def scratch_specs(self):
            return (SimpleNamespace(shape=(64,), dtype=torch.uint8),)

    @torch.library.custom_op("flare_test::wo_inv_rope_out", mutates_args=("out", "scratch"))
    def _fake_wo_inv_rope_op(o: torch.Tensor, pos: torch.Tensor, cs: torch.Tensor,
                             out: torch.Tensor, scratch: torch.Tensor) -> None:
        out.copy_(o.reshape(o.shape[0], -1)[:, : out.shape[1]])
        scratch.fill_(1)

    @_fake_wo_inv_rope_op.register_fake
    def _fake_wo_inv_rope_op_fake(o, pos, cs, out, scratch):
        pass

    def fake_plan(caps, *, invocation=None):
        planned_caps.append((caps.max_tokens, dict(invocation or {})))
        return _FakeWoPlan(caps, invocation)

    def fake_run_inv_rope_out(o, pos, cs, weights, *, out, scratch, plan):
        inv_rope_calls.append(o.shape[0])
        _fake_wo_inv_rope_op(o, pos, cs, out, scratch)
        return out

    monkeypatch.setattr(wo_projection, "plan", fake_plan)
    monkeypatch.setattr(wo_projection, "run_inv_rope_out", fake_run_inv_rope_out)

    groups, heads_per_group, head_dim, hidden = 2, 2, 512, 128
    fake_packed = SimpleNamespace(
        values=torch.zeros((1,), dtype=torch.float8_e4m3fn),
        scale_rows=None,
        scale_mma=torch.zeros((1,), dtype=torch.uint8),
    )
    weights = SimpleNamespace(
        groups=groups,
        group_width=heads_per_group * head_dim,
        rank=128,
        hidden=hidden,
        wo_a=fake_packed,
        wo_b=fake_packed,
    )
    source = torch.ones((rows, groups * heads_per_group, head_dim), dtype=torch.bfloat16)
    output = torch.zeros((rows, hidden), dtype=torch.bfloat16)
    positions = torch.arange(rows, dtype=torch.int64)
    cos_sin = torch.zeros((rows, 64), dtype=torch.float32)

    finalizers = []
    monkeypatch.setattr(attention_output, "register_finalizer", lambda session, cb, **kw: finalizers.append(cb))
    monkeypatch.setattr(attention_output, "declare", lambda session, reqs: None)

    proj = attention_output.PreparedGroupedOutputProjection(
        source,
        weights,
        output,
        session=None,
        name="wo",
        workspace=None,
        inv_rope=True,
        heads_per_group=heads_per_group,
        nope_dim=448,
        rope_dim=64,
        positions=positions,
        cos_sin_cache=cos_sin,
    ).functional(session=None)
    # Finalize scratch layout (skip rebind which requires ScratchArena).
    proj._finalize_functional()

    assert proj._fused_inv_rope is True
    assert planned_caps[-1][0] == rows
    assert planned_caps[-1][1]["operation"] == "inv_rope"

    traced = make_fx(lambda s, p, c: proj(s, positions=p, cos_sin_cache=c))(source, positions, cos_sin)
    op_names = [str(n.target) for n in traced.graph.nodes if n.op == "call_function"]
    assert "flare_test.wo_inv_rope_out.default" in op_names
    res = traced(source, positions, cos_sin)
    assert res.shape == (rows, hidden)
    assert inv_rope_calls[-1] == rows

    # Verify fallback when wo_projection lacks run_inv_rope_out (base b12x compatibility)
    if rows == 32:
        base_wo = SimpleNamespace(plan=fake_plan, Caps=wo_projection.Caps, run_out=lambda *a, **kw: None)
        monkeypatch.setattr(attention_output, "wo_projection", base_wo)
        proj_fb = attention_output.PreparedGroupedOutputProjection(
            source,
            weights,
            output,
            session=None,
            name="wo_fb",
            workspace=None,
            inv_rope=True,
            heads_per_group=heads_per_group,
            nope_dim=448,
            rope_dim=64,
            positions=positions,
            cos_sin_cache=cos_sin,
        )
        assert proj_fb._fused_inv_rope is False
        assert planned_caps[-1][1] == {}


def test_concurrent_shared_expert_allocates_padded_mma_scales_for_fused_swiglu_down(monkeypatch):
    """ConcurrentSharedExpert fills padded scale_mma rows with UE8M0 1.0 (127) for non-128-aligned M."""
    owner = expert_overlap.ConcurrentSharedExpert.__new__(expert_overlap.ConcurrentSharedExpert)
    owner.outputs = None
    owner.output_layouts = (
        ((32, 1152), (1152, 1), torch.bfloat16),
        ((32, 576), (576, 1), torch.bfloat16),
        ((32, 64), (64, 1), torch.bfloat16),
    )
    owner.workspace_nbytes = 64
    layout = (
        ((32, 576), torch.float8_e4m3fn),
        ((1, 32, 18), torch.uint8),
        ((1, 1, 5, 32, 4, 4), torch.uint8),
    )
    owner.encoded_layouts = list(layout)
    owner.fused_down_layout = (0, layout)
    owner.weights, owner.scales, owner.gains, owner.activation_scales, owner.scale_kinds = [], [], [], [], []
    owner.plans, owner.encoding_offsets, owner.quantizers = [], [0, 0, 3], [0, 0]
    owner.activation, owner.limit, owner.round_silu = "swiglu", float("inf"), False
    owner.stream_handle, owner.launch_order = 1, 0
    captured = []
    monkeypatch.setattr(expert_overlap, "_start_shared_expert", lambda *args: captured.append(args))
    monkeypatch.setattr(expert_overlap, "_join_shared_expert", lambda *args: None)

    pending = owner.start(torch.zeros((32, 64), dtype=torch.bfloat16))
    assert len(pending.encoded) == 3
    assert pending.encoded[0].shape == (32, 576)
    assert pending.encoded[2].shape == (1, 1, 5, 32, 4, 4)
    assert torch.all(pending.encoded[2] == 127)
    out = owner.finish(pending, torch.zeros((32, 64), dtype=torch.bfloat16))
    assert out.shape == (32, 64)


def test_register_dense_gemm_rope_accepts_quantized_mxfp8_alpha_is_one_false(monkeypatch):
    """Quantized blockscaled MXFP8 plans have lowering.alpha_is_one=False; RoPE compiles with unit alpha."""
    lowering = _mock_lowering(128, 4096, 1536)
    lowering.alpha_is_one = False
    plan = SimpleNamespace(
        handle=9876,
        _prepared=SimpleNamespace(
            state=SimpleNamespace(
                config=SimpleNamespace(mode="quantized"),
                query=SimpleNamespace(recipe="mxfp8", num_tokens=128),
                device=torch.device("cpu"),
                dense=SimpleNamespace(lowering=lowering, alpha_one=None),
            )
        ),
    )
    compiled_calls = []

    def fake_compile(low, ordinal, **kwargs):
        compiled_calls.append((low, ordinal, kwargs))
        return object()

    from b12x._lib.dense_gemm import buffers as dense_buffers

    monkeypatch.setattr(dense_epilogues, "supports_dense_rope_epilogue", lambda: True)
    monkeypatch.setattr(dense_epilogues, "_compile_dense_gemm_from_lowering", fake_compile)
    monkeypatch.setattr(dense_buffers, "_cached_alpha_one", lambda dev: torch.ones(1, dtype=torch.float32))

    assert dense_epilogues.register_dense_gemm_rope(plan, head_dim=512, rope_dim=64) is True
    assert len(compiled_calls) == 1
    assert compiled_calls[0][2]["rope_head_dim"] == 512
    assert dense_epilogues.has_dense_gemm_rope(plan.handle, head_dim=512, rope_dim=64) is True


def test_prepared_attention_standalone_rope_flags_have_class_defaults():
    """PreparedAttention instances created via __new__ default all standalone RoPE flags to False."""
    attn = PreparedAttention.__new__(PreparedAttention)
    assert attn.q_b_fused_rope is False
    assert attn.q_b_dense_rope is False
    assert attn.index_query_fused_rope is False
    assert attn.index_query_dense_rope is False
