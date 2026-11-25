# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek V4.1 row-sharded encoder attention weight accounting, routing, and DMA overlap."""

from dataclasses import dataclass
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from b12x.comm import pcie
from hyperflare_recipes.components.moe import RouterScratch
from hyperflare_recipes.components.paged_batch import Metadata
from hyperflare_recipes.components.parallel import TensorParallelShard
from hyperflare_recipes.recipes.deepseek41 import (
    attention as ds_attention,
    config as ds_config,
    decoder,
    recipe as ds_recipe,
    weights as ds_weights,
)
from hyperflare_recipes.recipes.deepseek41.bootstrap import PreparationConfig
from hyperflare_recipes.recipes.deepseek41.pool import ContinuationPool
from ignition.planning import Gpu, Plan, UsageError
from ignition.preparation.rebuild_manifest import (
    declare_weights,
    declared_views,
    weight_payload_declaration,
)
from ignition.preparation.scratch import ScratchArena


_RUN_CUDA_TESTS = os.environ.get("FLARE_RUN_CUDA_TESTS") == "1"
_SNAPSHOT_ENV = os.environ.get("FLARE_DEEPSEEK41_SNAPSHOT", "")

EXPECTED_FULL_ATTN_ENCODER_LAYERS = (
    0, 1, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19,
)
EXPECTED_LAYER_BYTES = 124_780_800
EXPECTED_TOTAL_BYTES = 17 * EXPECTED_LAYER_BYTES  # 2,121,273,600 B


def _qualified_model_config(*, overrides: dict | None = None) -> ds_config.ModelConfig:
    text = {
        **ds_config.TEXT_GEOMETRY,
        **ds_config.QUALIFIED_ENCODER_SCHEDULE,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "compress_rope_theta": 10000.0,
        "rope_scaling": {
            "original_max_position_embeddings": 4096,
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 40.0,
        },
    }
    if overrides:
        text.update(overrides)
    return ds_config.ModelConfig(text, {})


def _sm120_gpus(count: int, capability: tuple[int, int] = (12, 0)) -> tuple[Gpu, ...]:
    return tuple(
        Gpu(i, "NVIDIA RTX PRO 6000 Blackwell", f"GPU-{i}", 96 << 30, 90 << 30, capability)
        for i in range(count)
    )


def test_replicated_attention_layer_schedule_and_exact_byte_accounting(monkeypatch):
    cfg = _qualified_model_config()
    assert cfg.decoder_start == 20

    layers = ds_config.full_attention_encoder_layers(cfg)
    assert layers == EXPECTED_FULL_ATTN_ENCODER_LAYERS
    assert len(layers) == 17
    for mla_wo_layer in (2, 8, 14):
        assert mla_wo_layer not in layers
    for suffix_layer in range(20, 40):
        assert suffix_layer not in layers
    cfg.require_planned_attention_schedule()

    # Exact per-layer breakdown: FP8 values + BOTH MXFP8 scale layouts (scale_rows + scale_mma) + FP32 sinks.
    q_b_vals = 64 * 512 * 1280
    wo_a_vals = 8 * 1024 * 4096
    wo_b_vals = 5120 * (8 * 1024)
    all_vals = q_b_vals + wo_a_vals + wo_b_vals
    both_scales = 2 * (all_vals // 32)
    one_scale_only = all_vals // 32
    fp32_sinks = 64 * 4

    layer_bytes = ds_config.replicated_attention_layer_bytes()
    assert layer_bytes == EXPECTED_LAYER_BYTES
    assert layer_bytes == all_vals + both_scales + fp32_sinks
    # Sensitivity: omitting one scale layout or FP32 sinks must disagree with layer_bytes.
    assert layer_bytes != all_vals + one_scale_only + fp32_sinks
    assert layer_bytes != all_vals + both_scales

    assert ds_config.replicated_attention_bytes(4) == EXPECTED_TOTAL_BYTES
    assert ds_recipe.RECIPE.replicated_attention_bytes(4) == EXPECTED_TOTAL_BYTES
    assert ds_config.replicated_attention_bytes(4, prompt_logprobs=True) == EXPECTED_TOTAL_BYTES
    assert ds_config.replicated_attention_bytes(4, mtp_only=True) == EXPECTED_TOTAL_BYTES

    # Zero charge on single-rank or when no encoder boundary exists.
    assert ds_config.replicated_attention_bytes(1) == 0
    no_encoder_cfg = _qualified_model_config(overrides={
        "compress_ratios": [1] * 40,
        "kv_source_layer_ids": [0],
        "index_source_layer_ids": [0, 20],
        "candidate_source_layer_id": 20,
    })
    assert no_encoder_cfg.decoder_start is None
    assert ds_config.full_attention_encoder_layers(no_encoder_cfg) == ()
    assert ds_config.replicated_attention_bytes(4, config=no_encoder_cfg) == 0

    # Sensitivity to changing layer count or projection dimensions.
    sixteen_layer_cfg = _qualified_model_config(overrides={
        "index_source_layer_ids": [2, 5, 8, 14, 20, 24, 28, 32, 36],
    })
    assert len(ds_config.full_attention_encoder_layers(sixteen_layer_cfg)) == 16
    assert ds_config.replicated_attention_bytes(4, config=sixteen_layer_cfg) == 16 * EXPECTED_LAYER_BYTES
    with pytest.raises(ValueError, match="qualified full-attention encoder layer schedule"):
        sixteen_layer_cfg.require_planned_attention_schedule()

    wider_sink_text = {**ds_config.TEXT_GEOMETRY, "num_attention_heads": 128, "o_groups": 16}
    assert ds_config.replicated_attention_layer_bytes(wider_sink_text) > EXPECTED_LAYER_BYTES

    # PreparationConfig validation and safe plan extraction.
    assert PreparationConfig().row_shard_attention is False
    valid_cfg = PreparationConfig(row_shard_prefill=True, row_shard_attention=True)
    assert valid_cfg.row_shard_attention is True
    for bad in (1, 0, "true", None):
        with pytest.raises(TypeError, match="row_shard_attention must be a boolean"):
            PreparationConfig(row_shard_prefill=True, row_shard_attention=bad)
    with pytest.raises(ValueError, match="row_shard_attention requires row_shard_prefill"):
        PreparationConfig(row_shard_prefill=False, row_shard_attention=True)

    captured = {}

    def fake_prepare_group(snapshot, config, prepare_rank, **kwargs):
        captured["config"] = config
        return "prepared"

    import ignition.preparation.rank_group as rank_group

    monkeypatch.setattr(rank_group, "prepare_group", fake_prepare_group)
    base_plan = Plan(
        recipe=ds_recipe.RECIPE,
        checkpoint=ds_recipe.RECIPE.checkpoints[0],
        gpus=_sm120_gpus(4),
        concurrency=32,
        prefill_chunk=4096,
        speculation=ds_recipe.RECIPE.speculation,
        cache_target=8192,
        cache_states=256,
        pages=1024,
        reserve_bytes=0,
        weights_bytes=1,
        graph_bytes=1,
        image_concurrency=4,
        joint_tuning=False,
        row_shard_prefill=True,
    )
    ds_recipe.prepare("/tmp/snap", base_plan, compile_forwards=True, kernel_cache=None, progress=None)
    assert captured["config"].row_shard_prefill is True
    assert captured["config"].row_shard_attention is False

    if "row_shard_attention" in Plan.__dataclass_fields__:
        rsa_plan = Plan(
            recipe=ds_recipe.RECIPE,
            checkpoint=ds_recipe.RECIPE.checkpoints[0],
            gpus=_sm120_gpus(4),
            concurrency=32,
            prefill_chunk=4096,
            speculation=ds_recipe.RECIPE.speculation,
            cache_target=8192,
            cache_states=256,
            pages=1024,
            reserve_bytes=0,
            weights_bytes=1,
            graph_bytes=1,
            image_concurrency=4,
            joint_tuning=False,
            row_shard_prefill=True,
            row_shard_attention=True,
        )
        ds_recipe.prepare("/tmp/snap", rsa_plan, compile_forwards=True, kernel_cache=None, progress=None)
        assert captured["config"].row_shard_attention is True
        with pytest.raises(UsageError, match="--row-shard-attention requires --row-shard-prefill"):
            Plan(
                recipe=ds_recipe.RECIPE,
                checkpoint=ds_recipe.RECIPE.checkpoints[0],
                gpus=_sm120_gpus(4),
                concurrency=32,
                prefill_chunk=4096,
                speculation=ds_recipe.RECIPE.speculation,
                cache_target=8192,
                cache_states=256,
                pages=1024,
                reserve_bytes=0,
                weights_bytes=1,
                graph_bytes=1,
                image_concurrency=4,
                joint_tuning=False,
                row_shard_prefill=False,
                row_shard_attention=True,
            )
        base_wb = ds_recipe.RECIPE.weights_bytes(
            ds_recipe.RECIPE.checkpoints[0], 4, ds_recipe.RECIPE.speculation, "csf", row_shard_attention=False
        )
        rsa_wb = ds_recipe.RECIPE.weights_bytes(
            ds_recipe.RECIPE.checkpoints[0], 4, ds_recipe.RECIPE.speculation, "csf", row_shard_attention=True
        )
        assert rsa_wb - base_wb == EXPECTED_TOTAL_BYTES


def test_model_weights_replicated_attention_loading_and_misc_component_routing(monkeypatch):
    from b12x.gemm.blockscaled._linear import MXFP8LinearWeight
    from b12x.quantization import mxfp8

    @dataclass(frozen=True)
    class FakePackedWo:
        wo_a: mxfp8.MXFP8Rows
        wo_b: mxfp8.MXFP8Rows
        groups: int
        group_width: int
        rank: int
        hidden: int

    def make_mxfp8_rows(rows: int, cols: int) -> mxfp8.MXFP8Rows:
        # Use scaled-down 1/64 row/col dimensions for fast CPU allocation while preserving exact
        # MXFP8LinearWeight / MXFP8Rows / dataclass container hierarchy for declared_views.
        r, c = max(1, rows // 64), max(32, cols // 64)
        vals = torch.ones((r, c), dtype=torch.float8_e4m3fn)
        srows = torch.ones((1, r, max(1, c // 32)), dtype=torch.uint8)
        smma = torch.ones((1, 1, 1, 1, 1, max(1, (r * c) // 32)), dtype=torch.uint8)
        return mxfp8.MXFP8Rows(vals, srows, smma)

    def make_linear(rows: int, cols: int) -> MXFP8LinearWeight:
        packed = make_mxfp8_rows(rows, cols)
        obj = MXFP8LinearWeight.__new__(MXFP8LinearWeight)
        for k, v in (
            ("weight", packed),
            ("out_features", rows),
            ("in_features", cols),
            ("padded_in_features", cols),
        ):
            object.__setattr__(obj, k, v)
        return obj

    def make_wo(groups: int) -> FakePackedWo:
        return FakePackedWo(
            wo_a=make_mxfp8_rows(groups * 1024, 4096),
            wo_b=make_mxfp8_rows(5120, groups * 1024),
            groups=groups,
            group_width=4096,
            rank=1024,
            hidden=5120,
        )

    cfg = _qualified_model_config()
    replicated_calls = []
    orig_load_replicated = ds_weights.DeepSeekWeightLoader.load_replicated_attention

    class FakeLoader:
        def __init__(self, manifest, device, *, shard, **kwargs):
            self.config = cfg
            self.shard = shard
            self._sources = {}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def load_tensors(self, names):
            return (torch.ones((16, 64), dtype=torch.bfloat16), torch.ones(64, dtype=torch.bfloat16))

        def load_padded_tensor(self, name, *, rows, temporary=False):
            count = rows[1] - rows[0]
            if name.endswith(".attn_sink"):
                return torch.ones((count,), dtype=torch.float32)
            return torch.ones((max(1, count // 1024), 64), dtype=torch.bfloat16)

        def load_linear_group(self, names, *, row_ranges=None, pad=False):
            if row_ranges is None:
                return make_linear(1792, 5120)
            r0, r1 = row_ranges[0]
            return make_linear(r1 - r0, 1280)

        def load_attention_output(self, prefix, groups_range, *, pad=False):
            groups = (groups_range[1] - groups_range[0]) // 1024
            return make_wo(groups)

        def load_layer(self, layer, *, draft=False):
            first, last = self.shard.padded_partition(8)
            attn = {
                "qkv_a": self.load_linear_group((f"layers.{layer}.attn.wq_a",)),
                "q_b": self.load_linear_group((f"layers.{layer}.attn.wq_b",), row_ranges=((first * 4096, last * 4096),)),
                "output": self.load_attention_output(f"layers.{layer}.attn", (first * 1024, last * 1024)),
                "q_norm": torch.ones(1280, dtype=torch.bfloat16),
                "kv_norm": torch.ones(512, dtype=torch.bfloat16),
                "sink": self.load_padded_tensor(f"layers.{layer}.attn.attn_sink", rows=(first * 8, last * 8)),
            }
            shared = ds_weights.MLPWeights(make_linear(1152, 5120), make_linear(5120, 576))
            experts = ds_weights.ExpertWeights(
                router=torch.ones((384, 64), dtype=torch.bfloat16),
                correction=torch.zeros(384, dtype=torch.float32),
                experts=None,
                shared=shared,
            )
            residual = {k: torch.ones(4, dtype=torch.float32) for k in ("hc_attn_base", "hc_ffn_base")}
            return ds_weights.LayerWeights(attn, experts, residual, None)

        def load_replicated_attention(self, layer):
            replicated_calls.append(layer)
            return orig_load_replicated(self, layer)

    monkeypatch.setattr(ds_weights.ModelConfig, "load", classmethod(lambda cls, path: cfg))
    monkeypatch.setattr(ds_weights, "DeepSeekWeightLoader", FakeLoader)
    monkeypatch.setattr(ds_weights, "allocated_tensor_bytes", lambda dev: 0)
    manifest = SimpleNamespace(metadata_path=Path("/tmp/fake"))

    # 1. row_shard_attention=False -> replicated_attention is None.
    w_off = ds_weights.ModelWeights.load(
        manifest, "cpu", shard=TensorParallelShard(1, 4), dspark=False, row_shard_attention=False
    )
    assert w_off.replicated_attention is None
    assert replicated_calls == []

    # 2. world_size == 1 -> replicated_attention is None even if row_shard_attention=True.
    w_tp1 = ds_weights.ModelWeights.load(
        manifest, "cpu", shard=TensorParallelShard(0, 1), dspark=False, row_shard_attention=True
    )
    assert w_tp1.replicated_attention is None
    assert replicated_calls == []

    # 3. row_shard_attention=True at TP4 -> loads exact 17 layers, keeping partitioned decode copies intact.
    w_on = ds_weights.ModelWeights.load(
        manifest, "cpu", shard=TensorParallelShard(1, 4), dspark=False, row_shard_attention=True
    )
    assert tuple(replicated_calls) == EXPECTED_FULL_ATTN_ENCODER_LAYERS
    assert w_on.replicated_attention is not None
    assert tuple(w_on.replicated_attention) == EXPECTED_FULL_ATTN_ENCODER_LAYERS
    for layer_idx in EXPECTED_FULL_ATTN_ENCODER_LAYERS:
        rep = w_on.replicated_attention[layer_idx]
        assert set(rep) == {"q_b_full", "output_full", "sink_full"}
        assert rep["q_b_full"].out_features == 64 * 512
        assert rep["output_full"].groups == 8
        assert rep["sink_full"].shape == (64,) and rep["sink_full"].dtype == torch.float32
        # Partitioned decode weights on rank 1 stay untouched at 2 groups / 16 heads.
        part = w_on.layers[layer_idx].attention
        assert part["q_b"].out_features == 16 * 512
        assert part["output"].groups == 2
        assert part["sink"].shape == (16,)

    # 4. Verify declare_weights + declared_views routes replicated_attention to 'misc' while
    #    keeping layer-000..039 views completely unchanged!
    decl_off = declare_weights(w_off, components=("embedding", "norm", "head", "vision", "draft"))
    decl_on = declare_weights(
        w_on, components=("embedding", "norm", "head", "vision", "draft", "replicated_attention")
    )
    views_off = weight_payload_declaration(decl_off)["views"]
    views_on = weight_payload_declaration(decl_on)["views"]

    layer_views_off = [
        (v["name"], v["group"], v["storage"], v["element_bytes"], v["shape"], v["stride"])
        for v in views_off if v["group"].startswith("layer-")
    ]
    layer_views_on = [
        (v["name"], v["group"], v["storage"], v["element_bytes"], v["shape"], v["stride"])
        for v in views_on if v["group"].startswith("layer-")
    ]
    assert len(layer_views_off) > 0
    assert layer_views_on == layer_views_off
    assert {v[1] for v in layer_views_on} == {f"layer-{i:03d}" for i in range(40)}

    rep_views = [v for v in views_on if "replicated_attention" in v["name"]]
    # Each of the 17 layers contributes 3 (q_b_full) + 6 (wo_a + wo_b) + 1 (sink_full) = 10 tensor views.
    assert len(rep_views) == 17 * 10
    assert all(v["group"] == "misc" and v["storage"] == "stable" for v in rep_views)
    assert not any("replicated_attention" in name for name, *_ in declared_views(decl_off))


def test_prepared_attention_shard_validation_and_rank_invariant_execution(monkeypatch):
    cfg = _qualified_model_config()
    pool = ContinuationPool.allocate(cfg, 4, 4, "cpu")
    rotary = ds_attention.Rotary(cfg, "cpu")

    tokens, world = 1024, 4
    half = tokens // 2
    st = half // world
    meta = Metadata.allocate(tokens, 1, 16, "cpu", max_query_tokens=tokens)
    scr = ds_attention.AttentionScratch.allocate(tokens, world, "cpu")
    src = torch.zeros((tokens, 5120), dtype=torch.bfloat16)

    base_weights = {
        "qkv_a": SimpleNamespace(out_features=1792),
        "q_b": SimpleNamespace(out_features=16 * 512),
        "q_norm": torch.ones(1280, dtype=torch.bfloat16),
        "kv_norm": torch.ones(512, dtype=torch.bfloat16),
        "output": SimpleNamespace(groups=2),
        "sink": torch.zeros(16, dtype=torch.float32),
    }
    valid_rep = {
        "q_b_full": SimpleNamespace(out_features=64 * 512),
        "output_full": SimpleNamespace(groups=8),
        "sink_full": torch.zeros(64, dtype=torch.float32),
    }

    # Validation failure matrix.
    with pytest.raises(ValueError, match="requires both shard_tokens and replicated_weights"):
        ds_attention.PreparedAttention(
            3, base_weights, src, scr, meta, pool, rotary, mode="prefill", world=4,
            session=None, name="bad", workspace=None, pipelined_mode="full",
            shard_tokens=st, replicated_weights=None,
        )
    with pytest.raises(ValueError, match="requires both shard_tokens and replicated_weights"):
        ds_attention.PreparedAttention(
            3, base_weights, src, scr, meta, pool, rotary, mode="prefill", world=4,
            session=None, name="bad", workspace=None, pipelined_mode="full",
            shard_tokens=None, replicated_weights=valid_rep,
        )
    for bad_pipelined in (None, "mla_wo"):
        with pytest.raises(ValueError, match="requires prefill mode with full pipelined attention"):
            ds_attention.PreparedAttention(
                3, base_weights, src, scr, meta, pool, rotary, mode="prefill", world=4,
                session=None, name="bad", workspace=None, pipelined_mode=bad_pipelined,
                shard_tokens=st, replicated_weights=valid_rep,
            )
    with pytest.raises(ValueError, match="requires prefill mode with full pipelined attention"):
        ds_attention.PreparedAttention(
            3, base_weights, src[:8], scr, Metadata.allocate(8, 8, 2, "cpu", max_query_tokens=1),
            pool, rotary, mode="decode", world=4, session=None, name="bad", workspace=None,
            pipelined_mode="full", shard_tokens=2, replicated_weights=valid_rep,
        )
    with pytest.raises(ValueError, match="rejects preencoded qkv_a"):
        ds_attention.PreparedAttention(
            3, base_weights, src, scr, meta, pool, rotary, mode="prefill", world=4,
            session=None, name="bad", workspace=None, pipelined_mode="full",
            qkv_encoded=object(), shard_tokens=st, replicated_weights=valid_rep,
        )
    for bad_st in (0, -128, 192, "128", True):
        with pytest.raises(ValueError, match="shard_tokens must be a positive divisor"):
            ds_attention.PreparedAttention(
                3, base_weights, src, scr, meta, pool, rotary, mode="prefill", world=4,
                session=None, name="bad", workspace=None, pipelined_mode="full",
                shard_tokens=bad_st, replicated_weights=valid_rep,
            )
    for bad_rep, pattern in (
        ({}, "requires q_b_full, output_full, and sink_full"),
        ({**valid_rep, "q_b_full": None}, "requires q_b_full, output_full, and sink_full"),
        ({**valid_rep, "q_b_full": SimpleNamespace(out_features=16 * 512)}, "must span all heads and output groups"),
        ({**valid_rep, "output_full": SimpleNamespace(groups=2)}, "must span all heads and output groups"),
        ({**valid_rep, "sink_full": torch.zeros(16, dtype=torch.float32)}, "must span all heads and output groups"),
        ({**valid_rep, "sink_full": torch.zeros(64, dtype=torch.bfloat16)}, "must span all heads and output groups"),
    ):
        with pytest.raises(ValueError, match=pattern):
            ds_attention.PreparedAttention(
                3, base_weights, src, scr, meta, pool, rotary, mode="prefill", world=4,
                session=None, name="bad", workspace=None, pipelined_mode="full",
                shard_tokens=st, replicated_weights=bad_rep,
            )

    # Stub b12x operators to verify exact constructor declarations and rank-invariant execution.
    declared = {}

    class FakeProj:
        def __init__(self, values, weight, out, **kwargs):
            self.name = kwargs["name"]
            self.weight = weight
            self.plan = object()
            self.quantizer_plan = None
            self.encoded = kwargs.get("encoded")
            self.encoded_layout = None
            declared[self.name] = (tuple(values.shape), tuple(out.shape), weight)

        def functional(self, **kwargs):
            return self

        def __call__(self, x, *, encoded_bases=None):
            if self.name.endswith(".qkv_a"):
                # Propagate row tag in column 0 into both query_low (col 0) and window_values (col 1280).
                out = torch.zeros((x.shape[0], 1792), dtype=torch.bfloat16)
                out[:, 0] = x[:, 0]
                out[:, 1280] = x[:, 0]
                return out
            # q_b_full: [st, 1280] -> [st, 64 * 512]
            out = torch.zeros((x.shape[0], 64, 512), dtype=torch.bfloat16)
            out[:, :, 0] = x[:, :1]
            return out.view(x.shape[0], -1)

    class FakeNorm:
        def __init__(self, values, weight, out, **kwargs):
            self.plan = object()
            declared[kwargs["name"]] = (tuple(values.shape), tuple(out.shape))

        def functional(self, **kwargs):
            return self

        def __call__(self, x):
            return x.clone()

    writer_calls = []

    class FakeWindowWriter:
        def __init__(self, values, cache, slots, **kwargs):
            self.plan = object()
            declared[kwargs["name"]] = (tuple(values.shape), tuple(slots.shape))

        def functional(self, **kwargs):
            return self

        def __call__(self, values, slots, *, cache=None, **kwargs):
            writer_calls.append((values.clone(), slots.clone()))
            # Record written row tags into cache[0, :slots.numel()] for verification.
            for idx, slot in enumerate(slots.tolist()):
                if slot >= 0:
                    cache[0, slot] = int(values[idx, 0].item()) % 251

    mla_calls = []

    class FakeMLA:
        def __init__(self, query, window, win_idx, win_len, sink, attended, **kwargs):
            self.plan = object()
            declared[kwargs["name"]] = (
                tuple(query.shape), tuple(attended.shape), tuple(sink.shape),
                None if "indexed_indices" not in kwargs else tuple(kwargs["indexed_indices"].shape),
            )

        def functional(self, **kwargs):
            return self

        def __call__(self, query, win_idx, win_len, *, indexed_indices=None, indexed_lengths=None,
                     page_table=None, window=None):
            mla_calls.append({
                "win_idx": win_idx.clone(),
                "win_len": win_len.clone(),
                "indexed_indices": None if indexed_indices is None else indexed_indices.clone(),
                "indexed_lengths": None if indexed_lengths is None else indexed_lengths.clone(),
                "page_table": None if page_table is None else page_table.clone(),
                "cache_byte_last_half": int(window[0, win_idx[:, -1].clamp_min(0)].float().sum().item()),
            })
            out = torch.zeros((query.shape[0], 64, 512), dtype=torch.bfloat16)
            # Combine query tag + win_len + optional indexed_indices[:, 0] so mutations propagate to output.
            extra = torch.zeros(query.shape[0], dtype=torch.float32)
            if indexed_indices is not None:
                extra = indexed_indices[:, 0].float()
            out[:, :, 0] = (query[:, 0, 0].float() + win_len.float() + extra).to(torch.bfloat16)[:, None]
            return out

    wo_calls = []

    class FakeWo:
        def __init__(self, attended, weight, output, **kwargs):
            self.plan = object()
            declared[kwargs["name"]] = (
                tuple(attended.shape), tuple(output.shape), weight.groups, kwargs["heads_per_group"],
                tuple(kwargs["positions"].shape), tuple(kwargs["cos_sin_cache"].shape),
            )

        def functional(self, **kwargs):
            return self

        def __call__(self, attended, *, positions=None, cos_sin_cache=None):
            wo_calls.append((positions.clone(), cos_sin_cache.clone()))
            out = torch.zeros((attended.shape[0], 5120), dtype=torch.bfloat16)
            out[:, 0] = attended[:, 0, 0]
            out[:, 1] = cos_sin_cache[:, 0].to(torch.bfloat16)
            return out

    monkeypatch.setattr(ds_attention.sm103_projection, "is_sm103", lambda dev: False)
    monkeypatch.setattr(ds_attention.mla_norm_mxfp8, "is_supported", lambda dev: False)
    monkeypatch.setattr(ds_attention, "QuantizedProjection", FakeProj)
    monkeypatch.setattr(ds_attention, "PreparedRMSNorm", FakeNorm)
    monkeypatch.setattr(ds_attention, "PreparedCompressedCacheWriter", FakeWindowWriter)
    monkeypatch.setattr(ds_attention, "PreparedCompressedMLA", FakeMLA)
    monkeypatch.setattr(ds_attention, "PreparedGroupedOutputProjection", FakeWo)
    monkeypatch.setattr(ds_attention, "register_finalizer", lambda *a, **kw: None)
    monkeypatch.setattr(
        ds_attention.PreparedRotaryAngles,
        "__call__",
        lambda self, pos: torch.zeros((pos.numel(), 64), dtype=torch.float32, device=pos.device),
    )

    wm = ds_attention.AttentionIndices(meta, mode="prefill", tokens=tokens, page_size=128 + tokens,
                                       width=128, ratios=(2,))
    examples = {}
    attn = ds_attention.PreparedAttention(
        3, base_weights, src, scr, meta, pool, rotary, mode="prefill", world=4,
        session=None, name="l3", workspace=None, window_metadata=wm, examples=examples,
        pipelined_mode="full", shard_tokens=st, replicated_weights=valid_rep,
    )
    # Non-row-sharded instance rejects run_shard_* calls.
    plain_attn = ds_attention.PreparedAttention(
        3, base_weights, src, scr, meta, pool, rotary, mode="prefill", world=4,
        session=None, name="l3_plain", workspace=None, window_metadata=wm, examples=examples,
        pipelined_mode="full",
    )
    with pytest.raises(ValueError, match="requires row-sharded attention"):
        plain_attn.run_shard_qkv_q(src[:st], indices=None, shard_rows=torch.arange(st))
    with pytest.raises(ValueError, match="requires row-sharded attention"):
        plain_attn.run_shard_mla_wo(
            torch.zeros((st, 64, 512), dtype=torch.bfloat16), torch.zeros((half, 512), dtype=torch.bfloat16),
            meta, (None, None, None), indices=None, window_cache=None, half_idx=0,
            shard_rows=torch.arange(st), layer_angles=torch.zeros((st, 64)),
        )

    # Disable rotary in-place mutation during deterministic tag check while keeping distinct angle rows.
    attn.fused_qkv_norm_rope = False
    attn.query_norm = FakeNorm(scr.projected[:st, :1280], base_weights["q_norm"], scr.query_low[:st], name="l3.qn")
    attn.window_norm = FakeNorm(scr.projected[:st, 1280:], base_weights["kv_norm"], scr.window_values[:st], name="l3.kvn")
    monkeypatch.setattr(attn.angles, "apply", lambda tensor, angles: None)

    # Check declared shapes: shard_tokens=128 on projections/MLA/Wo, half=512 on window_writer!
    assert declared["l3.qkv_a"][:2] == ((st, 5120), (st, 1792))
    assert declared["l3.q_b"] == ((st, 1280), (st, 64 * 512), valid_rep["q_b_full"])
    assert declared["l3.window_write"] == ((half, 512), (half,))
    assert declared["l3.attention"] == ((st, 64, 512), (st, 64, 512), (64,), (st, 512))
    assert declared["l3.output"] == ((st, 64, 512), (st, 5120), 8, 8, (st,), (st, 64))

    # Populate distinct per-row metadata across all 1024 rows.
    wm.functional()
    meta.positions.copy_(torch.arange(tokens, dtype=torch.int32))
    meta.starts.zero_()
    meta.lengths.fill_(tokens)
    meta.query_start_loc.copy_(torch.tensor([0, tokens], dtype=torch.int32))
    meta.live_counts.copy_(torch.tensor([tokens, 1], dtype=torch.int32))
    meta.state_rows.fill_(1)
    meta.page_table[0, :16] = torch.arange(16, dtype=torch.int32)
    distinct_pt = (
        torch.arange(tokens, dtype=torch.int32)[:, None] * 17
        + torch.arange(meta.token_page_table.shape[1], dtype=torch.int32)[None, :]
    )
    object.__setattr__(meta, "token_page_table", distinct_pt)
    indices = wm(meta)
    # Give each global row a unique angle marker in column 0 so cos_sin_cache slicing is verified.
    indices.angles[(True, 0)][:, 0] = torch.arange(tokens, dtype=torch.float32) * 0.25

    x = torch.zeros((tokens, 5120), dtype=torch.bfloat16)
    x[:, 0] = torch.arange(1, tokens + 1, dtype=torch.float32).to(torch.bfloat16)
    selected = (torch.arange(tokens, dtype=torch.int32)[:, None] * 3 + 7).expand(tokens, 512).contiguous()

    rows_a = [torch.arange(r * st, (r + 1) * st, dtype=torch.int64) for r in range(world)]
    rows_b = [torch.arange(half + r * st, half + (r + 1) * st, dtype=torch.int64) for r in range(world)]

    qa_s, wva_s, anga_s = [], [], []
    qb_s, wvb_s, angb_s = [], [], []
    for r in range(world):
        qa, wva, aa = attn.run_shard_qkv_q(x[r * st : (r + 1) * st], indices=indices, shard_rows=rows_a[r])
        qb, wvb, ab = attn.run_shard_qkv_q(x[half + r * st : half + (r + 1) * st], indices=indices, shard_rows=rows_b[r])
        assert qa.shape == (st, 64, 512) and wva.shape == (st, 512) and aa.shape == (st, 64)
        qa_s.append(qa)
        wva_s.append(wva)
        anga_s.append(aa)
        qb_s.append(qb)
        wvb_s.append(wvb)
        angb_s.append(ab)

    wva_full = torch.cat(wva_s, dim=0)
    wvb_full = torch.cat(wvb_s, dim=0)
    assert wva_full.shape == (half, 512) and wvb_full.shape == (half, 512)

    caches = []
    outs_a, outs_b = [], []
    for r in range(world):
        wc = torch.zeros((1, (128 + tokens) * 528), dtype=torch.uint8)
        oa = attn.run_shard_mla_wo(
            qa_s[r], wva_full, meta, (selected, None, None),
            indices=indices, window_cache=wc, half_idx=0, shard_rows=rows_a[r], layer_angles=anga_s[r],
        )
        ob = attn.run_shard_mla_wo(
            qb_s[r], wvb_full, meta, (selected, None, None),
            indices=indices, window_cache=wc, half_idx=1, shard_rows=rows_b[r], layer_angles=angb_s[r],
        )
        assert oa.shape == (st, 5120) and ob.shape == (st, 5120)
        caches.append(wc)
        outs_a.append(oa)
        outs_b.append(ob)

    # Every rank wrote the exact same full-chunk SWA cache bytes.
    for r in range(1, world):
        assert torch.equal(caches[0], caches[r])

    # Every rank received its disjoint contiguous slice of angles, window_indices, selected, and page_table.
    for r in range(world):
        call_a = mla_calls[2 * r]
        call_b = mla_calls[2 * r + 1]
        s0_a, s0_b = r * st, half + r * st
        assert torch.equal(call_a["win_idx"], indices.window_indices[s0_a : s0_a + st])
        assert torch.equal(call_b["win_idx"], indices.window_indices[s0_b : s0_b + st])
        assert torch.equal(call_a["indexed_indices"], selected[s0_a : s0_a + st])
        assert torch.equal(call_b["indexed_indices"], selected[s0_b : s0_b + st])
        assert torch.equal(call_a["page_table"], meta.token_page_table[s0_a : s0_a + st])
        assert torch.equal(call_b["page_table"], meta.token_page_table[s0_b : s0_b + st])
        pos_a, cs_a = wo_calls[2 * r]
        pos_b, cs_b = wo_calls[2 * r + 1]
        assert pos_a.tolist() == list(range(st)) and pos_b.tolist() == list(range(st))
        assert torch.equal(cs_a, indices.angles[(True, 0)][s0_a : s0_a + st])
        assert torch.equal(cs_b, indices.angles[(True, 0)][s0_b : s0_b + st])

    # Mutation sensitivity: perturbing row (2 * st + 5) in `selected` affects ONLY rank 2 Half A row 5.
    mut_selected = selected.clone()
    mut_selected[2 * st + 5, 0] += 64
    mut_outs_a = [
        attn.run_shard_mla_wo(
            qa_s[r], wva_full, meta, (mut_selected, None, None),
            indices=indices, window_cache=caches[r].clone(), half_idx=0,
            shard_rows=rows_a[r], layer_angles=anga_s[r],
        )
        for r in range(world)
    ]
    for r in range(world):
        if r == 2:
            assert not torch.equal(mut_outs_a[r], outs_a[r])
            assert torch.equal(mut_outs_a[r][:5], outs_a[r][:5])
            assert torch.equal(mut_outs_a[r][6:], outs_a[r][6:])
        else:
            assert torch.equal(mut_outs_a[r], outs_a[r])


class _FakeDmaRuntime:
    def __init__(self, world_size: int = 4, *, kv_gather_fits: bool = True):
        self.world_size = world_size
        self.max_bytes = 64 << 20
        self.shard_capacity = 16 << 20
        self.should_reduce_scatter = lambda tensor: True
        self.should_all_gather = lambda tensor: kv_gather_fits or tensor.shape[1] != 512

    def should_allreduce(self, tensor):
        return True


def test_prepared_decoder_row_shard_attention_layer_routing(monkeypatch):
    monkeypatch.setattr(pcie, "DmaAllReduce", _FakeDmaRuntime)

    seen_layers = []

    class FakeLayer:
        def __init__(self, index, *args, **kwargs):
            self.index = index
            self.has_image_mask = False
            self.pipelined_ffn = kwargs["pipelined_ffn"]
            self.pipelined_attn = kwargs["pipelined_attn"]
            self.row_shard_dma = kwargs["row_shard_dma"]
            self.replicated_attention = kwargs.get("replicated_attention")
            self.boundary = kwargs["boundary"]
            self.engram = None
            self.attention = SimpleNamespace(selectors=[object()], plans=lambda: ())
            self.post = None if kwargs["defer_post"] else object()
            self.cr_mlp_1 = SimpleNamespace(reduce=SimpleNamespace(plan=SimpleNamespace()))
            self.attention_hc = SimpleNamespace(plan=None)
            seen_layers.append(self)

    class FakeBoundary:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(decoder, "PreparedTableRequests", lambda *a, **kw: None)
    monkeypatch.setattr(decoder, "PreparedEmbedding", lambda *a, **kw: SimpleNamespace(functional=lambda **k: None))
    monkeypatch.setattr(decoder, "Rotary", lambda *a, **kw: None)
    monkeypatch.setattr(decoder, "AttentionIndices", lambda *a, **kw: SimpleNamespace(functional=lambda: None))
    monkeypatch.setattr(decoder, "PreparedLayer", FakeLayer)
    monkeypatch.setattr(decoder, "PreparedDecoderBoundary", FakeBoundary)
    monkeypatch.setattr(decoder, "register_finalizer", lambda *a, **kw: None)
    monkeypatch.setattr(decoder.PreparedDecoder, "_prepare_head", lambda *a, **kw: None)

    @dataclass(frozen=True)
    class FakeLayerWeights:
        engram: object
        attention: dict

    cfg = _qualified_model_config()
    kv_sources = set(cfg.text["kv_source_layer_ids"])
    index_sources = set(cfg.text["index_source_layer_ids"])
    layers = tuple(
        FakeLayerWeights(
            engram=object() if i in (1, 14) else None,
            attention=(
                {"compressor": object(), "index_q": object()} if i in kv_sources
                else ({"index_q": object()} if i in index_sources else {})
            ),
        )
        for i in range(40)
    )
    rep_map = {i: {"tag": i} for i in EXPECTED_FULL_ATTN_ENCODER_LAYERS}

    def build(*, row_shard_attention: bool, with_rep: bool = True, kv_fits: bool = True, encoder_only: bool = False):
        seen_layers.clear()
        weights = SimpleNamespace(
            config=SimpleNamespace(
                decoder_start=20,
                sources=cfg.sources,
                text={**cfg.text, "rms_norm_eps": 1e-6},
                draft=SimpleNamespace(target_layers=(37, 38, 39)),
            ),
            draft=None,
            layers=layers,
            replicated_attention=rep_map if with_rep else None,
            shard=TensorParallelShard(0, 4),
            embedding=torch.zeros((16, 5120), dtype=torch.bfloat16),
            norm=torch.ones(5120, dtype=torch.bfloat16),
            head=torch.zeros((16, 5120), dtype=torch.bfloat16),
        )
        tokens = 1024
        scratch = SimpleNamespace(
            embedding=torch.zeros((tokens, 5120), dtype=torch.bfloat16),
            residual=torch.zeros((tokens, 4, 5120), dtype=torch.bfloat16),
            attention=SimpleNamespace(
                output=torch.zeros((tokens, 5120), dtype=torch.bfloat16),
                window_values=torch.zeros((tokens, 512), dtype=torch.bfloat16),
            ),
            attention_output=torch.zeros((tokens, 5120), dtype=torch.bfloat16),
            attention_hc=SimpleNamespace(normalized=torch.zeros((tokens, 5120), dtype=torch.bfloat16)),
            hidden=torch.zeros((1, 5120), dtype=torch.bfloat16),
            local_logits=torch.zeros((1, 16), dtype=torch.float32),
            logits=torch.zeros((1, 16), dtype=torch.float32),
        )
        meta = Metadata.allocate(tokens, 1, 10, "cpu", max_query_tokens=tokens)
        dec = decoder.PreparedDecoder(
            weights, None, scratch, meta, None, mode="prefill",
            runtime=_FakeDmaRuntime(4, kv_gather_fits=kv_fits),
            vocabulary_runtime=None, coordinator=None, session=None, name="dec",
            workspace=ScratchArena("cpu", overlap=True),
            encoder_only=encoder_only, row_shard_prefill=True, row_shard_attention=row_shard_attention,
            examples={},
        )
        return dec, list(seen_layers)

    dec_on, layers_on = build(row_shard_attention=True)
    assert dec_on.row_shard_dma is True and dec_on.row_shard_attention is True
    assert len(layers_on) == 40
    sharded_indices = tuple(l.index for l in layers_on if l.replicated_attention is not None)
    assert sharded_indices == EXPECTED_FULL_ATTN_ENCODER_LAYERS
    for l in layers_on:
        if l.index in EXPECTED_FULL_ATTN_ENCODER_LAYERS:
            assert l.replicated_attention == {"tag": l.index}
        else:
            assert l.replicated_attention is None

    for override in (
        {"row_shard_attention": False},
        {"row_shard_attention": True, "with_rep": False},
        {"row_shard_attention": True, "kv_fits": False},
    ):
        dec_off, layers_off = build(**override)
        assert dec_off.row_shard_dma is True and dec_off.row_shard_attention is False
        assert all(l.replicated_attention is None for l in layers_off)


@pytest.mark.parametrize("pipelined_attn_hc", [False, True])
def test_prepared_layer_row_shard_attention_dma_schedule_and_overlap(pipelined_attn_hc):
    tokens, world, hidden, top_k = 1024, 4, 256, 6
    half = tokens // 2
    st = half // world
    events = []

    class LoggedCollective:
        def __init__(self, tag, fn):
            self.tag = tag
            self.fn = fn
            self.seen_shapes = []

        def start(self, source, *, dependencies=()):
            self.seen_shapes.append(tuple(source.shape))
            events.append(f"{self.tag}.start")
            out = self.fn(source)
            return SimpleNamespace(source=source, output=out, dependencies=dependencies)

        def finish(self, pending, dependency):
            events.append(f"{self.tag}.finish")
            return pending.output

    layer = decoder.PreparedLayer.__new__(decoder.PreparedLayer)
    layer.row_shard_dma = True
    layer.row_shard_attention = True
    layer.pipelined_attn_hc = pipelined_attn_hc
    layer.pipelined_attn = "full"
    layer.defer_mlp_b = False
    layer.half_tokens = half
    layer.shard_tokens = st
    layer.shard_rows_a = torch.arange(0, st, dtype=torch.int64)
    layer.shard_rows_b = torch.arange(half, half + st, dtype=torch.int64)
    layer.shard_offset_a = torch.tensor([0], dtype=torch.int32)
    layer.shard_offset_b = torch.tensor([half], dtype=torch.int32)
    layer.engram = None
    layer.has_image_mask = False
    layer.attn_quantizer_plan = None
    layer.ffn_quantizer_plan = None
    layer.ffn_local_shapes = None
    layer.ffn_meta_scale_cols = 0

    # Confirm attention input gather and attention output reduce-scatter are completely absent.
    layer.attn_gather_0 = layer.attn_gather_1 = None
    layer.cr_attn_gather_0 = layer.cr_attn_gather_1 = None
    layer.att_reduce_0 = layer.att_reduce_1 = None
    layer.cr_att_0 = layer.cr_att_1 = None

    hc_call = [0]

    def fake_attn_hc(*args, **kwargs):
        tag = "A" if hc_call[0] == 0 else "B"
        hc_call[0] += 1
        events.append(f"attn_hc_{tag}")
        return (
            torch.zeros((st, 4, hidden), dtype=torch.bfloat16),
            torch.zeros((st, 4), dtype=torch.float32),
            torch.zeros((st, 4, 4), dtype=torch.float32),
            torch.full((st, hidden), 1.0 if tag == "A" else 2.0, dtype=torch.bfloat16),
        )

    def fake_qkv_q(norm_sh, *, indices, shard_rows):
        tag = "A" if int(shard_rows[0].item()) == 0 else "B"
        events.append(f"qkv_q_{tag}")
        val = 10.0 if tag == "A" else 20.0
        return (
            torch.full((st, 64, 512), val, dtype=torch.bfloat16),
            torch.full((st, 512), val, dtype=torch.bfloat16),
            torch.full((st, 64), val, dtype=torch.float32),
        )

    seen_mla_wo = []

    def fake_mla_wo(q_sh, wv_full, metadata, selection, *, indices, window_cache, half_idx, shard_rows, layer_angles):
        tag = "A" if half_idx == 0 else "B"
        events.append(f"mla_wo_{tag}")
        seen_mla_wo.append((half_idx, tuple(q_sh.shape), tuple(wv_full.shape), shard_rows.clone()))
        return torch.full((st, hidden), 3.0 if half_idx == 0 else 4.0, dtype=torch.bfloat16)

    layer.attention_hc = fake_attn_hc
    layer.attention = SimpleNamespace(
        mode="prefill",
        init_window_cache=lambda metadata, dev: "wc",
        run_shard_qkv_q=fake_qkv_q,
        run_shard_mla_wo=fake_mla_wo,
        commit=lambda lengths, records, metadata: events.append("attn_commit"),
    )
    layer.cr_kv_gather_0 = LoggedCollective("kv_gather_0", lambda sh: sh.repeat(world, 1))
    layer.cr_kv_gather_1 = LoggedCollective("kv_gather_1", lambda sh: sh.repeat(world, 1))

    ffn_hc_call = [0]

    def fake_ffn_hc(att, res, post, comb, *, pre_mix, pre_out):
        tag = "A" if ffn_hc_call[0] == 0 else "B"
        ffn_hc_call[0] += 1
        events.append(f"ffn_hc_{tag}")
        return res, post, comb, att

    layer.ffn_hc = fake_ffn_hc
    layer.router_projection = lambda norm: torch.zeros((norm.shape[0], 384), dtype=torch.float32)
    prep_call = [0]

    def fake_router(logits, **kw):
        tag = "A" if prep_call[0] == 0 else "B"
        prep_call[0] += 1
        events.append(f"ffn_prep_{tag}")
        return RouterScratch(
            torch.zeros((logits.shape[0], top_k), dtype=torch.float32),
            torch.zeros((logits.shape[0], top_k), dtype=torch.int32),
            torch.ones((logits.shape[0], top_k), dtype=torch.float32),
        )

    layer.router = fake_router
    layer.cr_ffn_vals_0 = LoggedCollective("ffn_vals_0", lambda sh: sh.repeat(world, 1))
    layer.cr_ffn_meta_0 = LoggedCollective("ffn_meta_0", lambda sh: sh.repeat(world, 1))
    layer.cr_ffn_vals_1 = LoggedCollective("ffn_vals_1", lambda sh: sh.repeat(world, 1))
    layer.cr_ffn_meta_1 = LoggedCollective("ffn_meta_1", lambda sh: sh.repeat(world, 1))
    layer.experts = lambda vals, ids, weights, **kw: (events.append("ffn_compute") or vals)
    layer.shared = lambda vals, **kw: vals
    layer.cr_mlp_0 = LoggedCollective("mlp_rs_0", lambda full: full[:st].clone())
    layer.cr_mlp_1 = LoggedCollective("mlp_rs_1", lambda full: full[:st].clone())
    layer.post = lambda mlp_sh, res, post, comb: res

    metadata = Metadata.allocate(tokens, 1, 10, "cpu")
    metadata.live_counts[0] = tokens
    metadata.lengths[0] = tokens
    pm_a = torch.zeros((st, 4), dtype=torch.float32)
    pm_b = torch.zeros((st, 4), dtype=torch.float32)
    if pipelined_attn_hc:
        zeros_res = torch.zeros((st, 4, hidden), dtype=torch.bfloat16)
        zeros_post = torch.zeros((st, 4), dtype=torch.float32)
        zeros_comb = torch.zeros((st, 4, 4), dtype=torch.float32)
        mlp_a = torch.zeros((st, hidden), dtype=torch.bfloat16)
        p_mlp_b = SimpleNamespace(output=torch.zeros((st, hidden), dtype=torch.bfloat16))
        incoming = (mlp_a, zeros_res, zeros_post, zeros_comb, p_mlp_b, zeros_res, zeros_post, zeros_comb)
    else:
        incoming = (
            torch.zeros((st, 4, hidden), dtype=torch.bfloat16),
            torch.zeros((st, 4, hidden), dtype=torch.bfloat16),
        )

    layer._run_row_shard_dma(incoming, (pm_a, pm_b), metadata, "indices", ("sel", None, None))

    # 1. Exact narrow KV gather shapes: [128, 512] -> [512, 512] (2 MiB/layer at 8192-token chunk).
    assert layer.cr_kv_gather_0.seen_shapes == [(st, 512)]
    assert layer.cr_kv_gather_1.seen_shapes == [(st, 512)]
    assert seen_mla_wo[0][:3] == (0, (st, 64, 512), (half, 512))
    assert seen_mla_wo[1][:3] == (1, (st, 64, 512), (half, 512))
    assert torch.equal(seen_mla_wo[0][3], layer.shard_rows_a)
    assert torch.equal(seen_mla_wo[1][3], layer.shard_rows_b)

    # 2. Verify exact overlap ordering across the 2-microbatch DMA pipeline!
    idx = {ev: events.index(ev) for ev in (
        "attn_hc_A", "qkv_q_A", "kv_gather_0.start", "attn_hc_B", "qkv_q_B", "kv_gather_0.finish",
        "kv_gather_1.start", "mla_wo_A", "ffn_hc_A", "ffn_prep_A", "kv_gather_1.finish",
        "ffn_vals_0.start", "ffn_meta_0.start", "mla_wo_B", "attn_commit", "ffn_hc_B", "ffn_prep_B",
        "ffn_vals_0.finish", "ffn_meta_0.finish",
    )}
    if pipelined_attn_hc:
        # p_mlp_b from L-1 overlaps Half A attn_hc + qkv_q_A before joining!
        first_mlp_finish = events.index("mlp_rs_1.finish")
        assert idx["attn_hc_A"] < idx["qkv_q_A"] < first_mlp_finish < idx["kv_gather_0.start"]
    assert idx["kv_gather_0.start"] < idx["attn_hc_B"] < idx["qkv_q_B"] < idx["kv_gather_0.finish"]
    assert idx["kv_gather_1.start"] < idx["mla_wo_A"] < idx["ffn_hc_A"] < idx["ffn_prep_A"] < idx["kv_gather_1.finish"]
    # ffn_vals_0 + ffn_meta_0 AGs overlap Half B MLA + Wo + commit + ffn_hc + router/quant!
    assert idx["ffn_vals_0.start"] < idx["ffn_meta_0.start"] < idx["mla_wo_B"] < idx["attn_commit"]
    assert idx["attn_commit"] < idx["ffn_hc_B"] < idx["ffn_prep_B"] < idx["ffn_vals_0.finish"] < idx["ffn_meta_0.finish"]


@pytest.mark.skipif(
    not _RUN_CUDA_TESTS or not torch.cuda.is_available(),
    reason="requires CUDA GPU and FLARE_RUN_CUDA_TESTS=1",
)
def test_row_shard_attention_synthetic_mxfp8_matches_tp4_reference_cuda():
    from b12x.gemm import wo_projection
    from b12x.gemm.blockscaled import pack_weight
    from b12x.preparation import autotune_effort as b12x_autotune_effort
    from ignition.preparation.tuning import RecordedPreparationSession

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    cfg = _qualified_model_config()
    layer_idx = 3
    tokens = 1024
    h = tokens // 2
    st = h // 4
    ratio = cfg.text["compress_ratios"][layer_idx]
    assert ratio == 2 and layer_idx in ds_config.full_attention_encoder_layers(cfg)

    gen = torch.Generator(device=device).manual_seed(20261125)

    def rand_mxfp8_raw(rows: int, cols: int):
        w_bf16 = torch.randn((rows, cols), dtype=torch.bfloat16, device=device, generator=gen) * 0.02
        vals = w_bf16.to(torch.float8_e4m3fn)
        scales_32x32 = torch.full((rows // 32, cols // 32), 127, dtype=torch.uint8, device=device)
        return vals, scales_32x32

    def pack_mxfp8(vals: torch.Tensor, scales_32x32: torch.Tensor):
        expanded = scales_32x32.repeat_interleave(32, dim=0)
        return pack_weight(vals, expanded, recipe="mxfp8")

    qkv_vals, qkv_s = rand_mxfp8_raw(1792, 5120)
    qb_vals, qb_s = rand_mxfp8_raw(32768, 1280)
    woa_vals, woa_s = rand_mxfp8_raw(8192, 4096)
    wob_vals, wob_s = rand_mxfp8_raw(5120, 8192)
    q_norm = torch.ones(1280, dtype=torch.bfloat16, device=device)
    kv_norm = torch.ones(512, dtype=torch.bfloat16, device=device)
    sink_full = torch.linspace(-1.5, 1.5, 64, dtype=torch.float32, device=device)

    qkv_packed = pack_mxfp8(qkv_vals, qkv_s)
    tp_weights = []
    for r in range(4):
        h0, h1 = r * 8192, (r + 1) * 8192
        g0, g1 = r * 2048, (r + 1) * 2048
        tp_weights.append({
            "qkv_a": qkv_packed,
            "q_b": pack_mxfp8(qb_vals[h0:h1], qb_s[h0 // 32 : h1 // 32]),
            "output": wo_projection.pack_weights(
                woa_vals[g0:g1],
                woa_s[g0 // 32 : g1 // 32].view(torch.float8_e8m0fnu),
                wob_vals[:, g0:g1].contiguous(),
                wob_s[:, g0 // 32 : g1 // 32].contiguous().view(torch.float8_e8m0fnu),
                groups=2, group_width=4096, rank=1024, hidden=5120, block_size=(32, 32),
            ),
            "q_norm": q_norm,
            "kv_norm": kv_norm,
            "sink": sink_full[r * 16 : (r + 1) * 16].contiguous(),
        })
    rep_weights = {
        "q_b_full": pack_mxfp8(qb_vals, qb_s),
        "output_full": wo_projection.pack_weights(
            woa_vals,
            woa_s.view(torch.float8_e8m0fnu),
            wob_vals,
            wob_s.view(torch.float8_e8m0fnu),
            groups=8, group_width=4096, rank=1024, hidden=5120, block_size=(32, 32),
        ),
        "sink_full": sink_full,
    }

    pool = ContinuationPool.allocate(cfg, 4, 32, device, dspark=False)
    pool.pages[f"layer{cfg.source(layer_idx).layer}.latent_page"].zero_()
    pool.window(layer_idx).zero_()
    rotary = ds_attention.Rotary(cfg, device)
    meta = Metadata.allocate(tokens, 1, 64, device, max_query_tokens=tokens)
    meta.positions.copy_(torch.arange(tokens, dtype=torch.int32, device=device))
    meta.starts.zero_()
    meta.lengths.fill_(tokens)
    meta.query_start_loc.copy_(torch.tensor([0, tokens], dtype=torch.int32, device=device))
    meta.live_counts.copy_(torch.tensor([tokens, 1], dtype=torch.int32, device=device))
    meta.state_rows.fill_(1)
    meta.requests.zero_()
    meta.page_table.zero_()
    for p in range(16):
        meta.page_table[0, p] = p
    if meta.token_page_table.stride(0) != 0:
        meta.token_page_table.copy_(meta.page_table[0].unsqueeze(0).expand(tokens, -1))

    x = (torch.randn((tokens, 5120), dtype=torch.bfloat16, device=device, generator=gen) * 0.25).contiguous()
    selected = torch.full((tokens, 512), -1, dtype=torch.int32, device=device)
    for t in range(tokens):
        k_avail = min(16, (t + 1) // 2)
        if k_avail > 0:
            selected[t, :k_avail] = torch.arange(k_avail, dtype=torch.int32, device=device)
    selection = (selected, None, None)

    with b12x_autotune_effort("off"):
        session = RecordedPreparationSession(
            device=device, checkpoint="test-rsa-synth", workload_schema="verify",
            numerical_policy="default", autotune_effort="off",
        )
        session.__enter__()
        arena = ScratchArena(device, overlap=False)
        session.register_scratch(arena)
        examples = {}
        wm = ds_attention.AttentionIndices(
            meta, mode="prefill", tokens=tokens, page_size=128 + tokens, width=128, ratios=(ratio,),
        )
        scr = ds_attention.AttentionScratch.allocate(tokens, 4, device)
        ref_attns = [
            ds_attention.PreparedAttention(
                layer_idx, tp_weights[r], x, scr, meta, pool, rotary, mode="prefill", world=4,
                session=session, name=f"ref_r{r}", workspace=arena, mla_compute_mode="bf16",
                window_metadata=wm, examples=examples, pipelined_mode="full",
            )
            for r in range(4)
        ]
        sharded_attn = ds_attention.PreparedAttention(
            layer_idx, tp_weights[0], x, scr, meta, pool, rotary, mode="prefill", world=4,
            session=session, name="sharded", workspace=arena, mla_compute_mode="bf16",
            window_metadata=wm, examples=examples, pipelined_mode="full",
            shard_tokens=st, replicated_weights=rep_weights,
        )
        wm.functional()
        session.tune(show_progress=False)

    try:
        indices = wm(meta)
        ref_out_a = torch.zeros((h, 5120), dtype=torch.float32, device=device)
        ref_out_b = torch.zeros((h, 5120), dtype=torch.float32, device=device)
        ref_wc = None
        for r in range(4):
            wc_r = ref_attns[r].init_window_cache(meta, device)
            oa = ref_attns[r].run_half(x[:h], meta, selection, indices=indices, window_cache=wc_r, half_idx=0)
            ob = ref_attns[r].run_half(x[h:], meta, selection, indices=indices, window_cache=wc_r, half_idx=1)
            ref_out_a += oa.float()
            ref_out_b += ob.float()
            if r == 0:
                ref_wc = wc_r
            else:
                assert torch.equal(ref_wc, wc_r)
        ref_full = torch.cat((ref_out_a.bfloat16(), ref_out_b.bfloat16()), dim=0)

        rows_a = [torch.arange(r * st, (r + 1) * st, dtype=torch.int64, device=device) for r in range(4)]
        rows_b = [torch.arange(h + r * st, h + (r + 1) * st, dtype=torch.int64, device=device) for r in range(4)]
        q_a_l, wv_a_l, ang_a_l = [], [], []
        q_b_l, wv_b_l, ang_b_l = [], [], []
        for r in range(4):
            qa, wva, aa = sharded_attn.run_shard_qkv_q(x[r * st : (r + 1) * st], indices=indices, shard_rows=rows_a[r])
            qb, wvb, ab = sharded_attn.run_shard_qkv_q(x[h + r * st : h + (r + 1) * st], indices=indices, shard_rows=rows_b[r])
            q_a_l.append(qa)
            wv_a_l.append(wva)
            ang_a_l.append(aa)
            q_b_l.append(qb)
            wv_b_l.append(wvb)
            ang_b_l.append(ab)

        wv_a_full = torch.cat(wv_a_l, dim=0)
        wv_b_full = torch.cat(wv_b_l, dim=0)
        sh_out_a, sh_out_b = [], []
        sh_wc = None
        for r in range(4):
            wc_s = sharded_attn.init_window_cache(meta, device)
            oa = sharded_attn.run_shard_mla_wo(
                q_a_l[r], wv_a_full, meta, selection, indices=indices, window_cache=wc_s,
                half_idx=0, shard_rows=rows_a[r], layer_angles=ang_a_l[r],
            )
            ob = sharded_attn.run_shard_mla_wo(
                q_b_l[r], wv_b_full, meta, selection, indices=indices, window_cache=wc_s,
                half_idx=1, shard_rows=rows_b[r], layer_angles=ang_b_l[r],
            )
            sh_out_a.append(oa)
            sh_out_b.append(ob)
            if r == 0:
                sh_wc = wc_s
            else:
                assert torch.equal(sh_wc, wc_s)
        sh_full = torch.cat(sh_out_a + sh_out_b, dim=0)

        assert torch.equal(ref_wc, sh_wc)
        diff = sh_full.float() - ref_full.float()
        rel_rms = diff.square().mean().sqrt().item() / max(ref_full.float().square().mean().sqrt().item(), 1e-12)
        cos = torch.nn.functional.cosine_similarity(sh_full.float().flatten(), ref_full.float().flatten(), dim=0).item()
        assert rel_rms < 0.005, f"rel_rms exceeded BF16 rounding budget: {rel_rms}"
        assert cos > 0.9999, f"cosine similarity below bound: {cos}"

        # Sensitivity check: corrupting gathered KV values breaks both cache equality and output parity.
        bad_wc = sharded_attn.init_window_cache(meta, device)
        bad_wv = wv_a_full.clone()
        bad_wv[st - 4 : st] += 5.0
        bad_oa_r1 = sharded_attn.run_shard_mla_wo(
            q_a_l[1], bad_wv, meta, selection, indices=indices, window_cache=bad_wc,
            half_idx=0, shard_rows=rows_a[1], layer_angles=ang_a_l[1],
        )
        assert not torch.equal(bad_wc, ref_wc[:1])
        bad_rel = (bad_oa_r1.float() - sh_out_a[1].float()).square().mean().sqrt().item() / max(
            sh_out_a[1].float().square().mean().sqrt().item(), 1e-12
        )
        assert bad_rel > 0.05
    finally:
        session.__exit__(None, None, None)


@pytest.mark.skipif(
    not _RUN_CUDA_TESTS or not torch.cuda.is_available() or not _SNAPSHOT_ENV,
    reason="requires CUDA GPU, FLARE_RUN_CUDA_TESTS=1, and FLARE_DEEPSEEK41_SNAPSHOT",
)
@pytest.mark.parametrize(("layer_idx", "tokens"), [(3, 1024), (3, 4096), (0, 1024)])
def test_row_shard_attention_checkpoint_parity_cuda(layer_idx: int, tokens: int):
    from b12x.preparation import autotune_effort as b12x_autotune_effort
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.tuning import RecordedPreparationSession

    snapshot = Path(_SNAPSHOT_ENV)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    manifest = inspect_indexed_checkpoint(snapshot)
    config = ds_config.ModelConfig.load(manifest.metadata_path)

    tp_weights = []
    rep_weights = None
    for r in range(4):
        shard = TensorParallelShard(r, 4)
        with ds_weights.DeepSeekWeightLoader(manifest, device, shard=shard, dspark=False) as loader:
            tp_weights.append(loader.load_layer(layer_idx).attention)
            if r == 0:
                rep_weights = loader.load_replicated_attention(layer_idx)

    h = tokens // 2
    st = h // 4
    pool = ContinuationPool.allocate(config, 4, 32, device, dspark=False)
    rotary = ds_attention.Rotary(config, device)
    meta = Metadata.allocate(tokens, 1, 64, device, max_query_tokens=tokens)
    meta.positions.copy_(torch.arange(tokens, dtype=torch.int32, device=device))
    meta.starts.zero_()
    meta.lengths.fill_(tokens)
    meta.query_start_loc.copy_(torch.tensor([0, tokens], dtype=torch.int32, device=device))
    meta.live_counts.copy_(torch.tensor([tokens, 1], dtype=torch.int32, device=device))
    meta.state_rows.fill_(1)
    meta.requests.zero_()
    meta.page_table.zero_()
    for p in range(16):
        meta.page_table[0, p] = p
    if meta.token_page_table.stride(0) != 0:
        meta.token_page_table.copy_(meta.page_table[0].unsqueeze(0).expand(tokens, -1))

    gen = torch.Generator(device=device).manual_seed(2026)
    ratio = config.text["compress_ratios"][layer_idx]
    if ratio:
        pool.pages[f"layer{config.source(layer_idx).layer}.latent_page"].zero_()
    pool.window(layer_idx).zero_()
    x = (torch.randn((tokens, 5120), dtype=torch.bfloat16, device=device, generator=gen) * 0.25).contiguous()
    selected = torch.full((tokens, 512), -1, dtype=torch.int32, device=device)
    for t in range(tokens):
        k_avail = min(16, (t + 1) // 2)
        if k_avail > 0:
            selected[t, :k_avail] = torch.arange(k_avail, dtype=torch.int32, device=device)
    selection = (selected, None, None)

    with b12x_autotune_effort("off"):
        session = RecordedPreparationSession(
            device=device, checkpoint="verify-rsa", workload_schema="verify",
            numerical_policy="default", autotune_effort="off",
        )
        session.__enter__()
        arena = ScratchArena(device, overlap=False)
        session.register_scratch(arena)
        examples = {}
        wm = ds_attention.AttentionIndices(
            meta, mode="prefill", tokens=tokens, page_size=128 + tokens, width=128,
            ratios=(ratio,) if ratio else (),
        )
        scr = ds_attention.AttentionScratch.allocate(tokens, 4, device)
        ref_attns = [
            ds_attention.PreparedAttention(
                layer_idx, tp_weights[r], x, scr, meta, pool, rotary, mode="prefill", world=4,
                session=session, name=f"ref_r{r}", workspace=arena, mla_compute_mode="bf16",
                window_metadata=wm, examples=examples, pipelined_mode="full",
            )
            for r in range(4)
        ]
        sharded_attn = ds_attention.PreparedAttention(
            layer_idx, tp_weights[0], x, scr, meta, pool, rotary, mode="prefill", world=4,
            session=session, name="sharded", workspace=arena, mla_compute_mode="bf16",
            window_metadata=wm, examples=examples, pipelined_mode="full",
            shard_tokens=st, replicated_weights=rep_weights,
        )
        wm.functional()
        session.tune(show_progress=False)

    try:
        indices = wm(meta)
        ref_out_a = torch.zeros((h, 5120), dtype=torch.float32, device=device)
        ref_out_b = torch.zeros((h, 5120), dtype=torch.float32, device=device)
        ref_wc = None
        for r in range(4):
            wc_r = ref_attns[r].init_window_cache(meta, device)
            oa = ref_attns[r].run_half(x[:h], meta, selection, indices=indices, window_cache=wc_r, half_idx=0)
            ob = ref_attns[r].run_half(x[h:], meta, selection, indices=indices, window_cache=wc_r, half_idx=1)
            ref_out_a += oa.float()
            ref_out_b += ob.float()
            if r == 0:
                ref_wc = wc_r
            else:
                assert torch.equal(ref_wc, wc_r)
        ref_full = torch.cat((ref_out_a.bfloat16(), ref_out_b.bfloat16()), dim=0)

        rows_a = [torch.arange(r * st, (r + 1) * st, dtype=torch.int64, device=device) for r in range(4)]
        rows_b = [torch.arange(h + r * st, h + (r + 1) * st, dtype=torch.int64, device=device) for r in range(4)]
        q_a_l, wv_a_l, ang_a_l = [], [], []
        q_b_l, wv_b_l, ang_b_l = [], [], []
        for r in range(4):
            qa, wva, aa = sharded_attn.run_shard_qkv_q(x[r * st : (r + 1) * st], indices=indices, shard_rows=rows_a[r])
            qb, wvb, ab = sharded_attn.run_shard_qkv_q(x[h + r * st : h + (r + 1) * st], indices=indices, shard_rows=rows_b[r])
            q_a_l.append(qa)
            wv_a_l.append(wva)
            ang_a_l.append(aa)
            q_b_l.append(qb)
            wv_b_l.append(wvb)
            ang_b_l.append(ab)

        wv_a_full = torch.cat(wv_a_l, dim=0)
        wv_b_full = torch.cat(wv_b_l, dim=0)
        sh_out_a, sh_out_b = [], []
        sh_wc = None
        for r in range(4):
            wc_s = sharded_attn.init_window_cache(meta, device)
            oa = sharded_attn.run_shard_mla_wo(
                q_a_l[r], wv_a_full, meta, selection, indices=indices, window_cache=wc_s,
                half_idx=0, shard_rows=rows_a[r], layer_angles=ang_a_l[r],
            )
            ob = sharded_attn.run_shard_mla_wo(
                q_b_l[r], wv_b_full, meta, selection, indices=indices, window_cache=wc_s,
                half_idx=1, shard_rows=rows_b[r], layer_angles=ang_b_l[r],
            )
            sh_out_a.append(oa)
            sh_out_b.append(ob)
            if r == 0:
                sh_wc = wc_s
            else:
                assert torch.equal(sh_wc, wc_s)
        sh_full = torch.cat(sh_out_a + sh_out_b, dim=0)
    finally:
        session.__exit__(None, None, None)

    assert torch.equal(ref_wc, sh_wc)
    diff = sh_full.float() - ref_full.float()
    rel_rms = diff.square().mean().sqrt().item() / max(ref_full.float().square().mean().sqrt().item(), 1e-12)
    cos = torch.nn.functional.cosine_similarity(sh_full.float().flatten(), ref_full.float().flatten(), dim=0).item()
    assert rel_rms < 0.005
    assert cos > 0.9999
