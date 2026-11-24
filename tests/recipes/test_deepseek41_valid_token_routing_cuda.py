# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek V4.1 PreparedLayer excludes inactive suffix rows from MoE routing across modes.

Exercises production `PreparedLayer` router, `SplitLaggedMHC`, `PreparedDecoderSuffix`,
and MXFP4 W4A8 `PreparedRoutedExperts` wiring under compiled CUDA-graph replay with
synthetic weights on one GPU. Attention, non-split hyperconnections, shared MLP, and
collectives use pass-through doubles so the suite isolates routing and routed-expert
execution across `decode`, `verify`, `draft`, `prefill`, `split_mhc`, `pipelined_ffn`,
and `PreparedDecoderSuffix`.
"""

import os
from types import SimpleNamespace

import pytest
from ignition import runtime_patches

runtime_patches.apply()

import torch
import torch._inductor.async_compile  # noqa: F401
from b12x.moe import fused_moe

from hyperflare_recipes.components.mhc import MHCWeights
from hyperflare_recipes.components.moe import PreparedRouter
from hyperflare_recipes.components.paged_batch import Metadata
from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from hyperflare_recipes.recipes.deepseek41 import decoder
from hyperflare_recipes.recipes.deepseek41.decoder_prefill import PreparedDecoderSuffix


pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an explicitly assigned CUDA device",
)

HIDDEN = 5120
INTERMEDIATE = 576


def _make_mxfp4_experts(experts: int, device: torch.device, *, scale_compression: str | None = None):
    generator = torch.Generator(device="cpu").manual_seed(4101 + experts)
    w13 = torch.randint(0, 256, (experts, 2 * INTERMEDIATE, HIDDEN // 2),
                        generator=generator, dtype=torch.uint8).to(device)
    w2 = torch.randint(0, 256, (experts, HIDDEN, INTERMEDIATE // 2),
                       generator=generator, dtype=torch.uint8).to(device)
    s13 = torch.randint(118, 122, (experts, 2 * INTERMEDIATE, HIDDEN // 32),
                        generator=generator, dtype=torch.uint8).to(device)
    s2 = torch.randint(118, 122, (experts, HIDDEN, INTERMEDIATE // 32),
                       generator=generator, dtype=torch.uint8).to(device)
    plan = fused_moe.plan_weights(
        source=fused_moe.PackedSource(format="fp4_e8m0_k32", w13_layout="w31"),
        activation=fused_moe.ActivationSpec(
            mode="a8", nonlinearity="silu", io_dtype=torch.bfloat16, swiglu_limit=10.0),
        geometry=fused_moe.MoEGeometry(
            num_experts=experts, hidden_size=HIDDEN, intermediate_size=INTERMEDIATE),
        constraints=fused_moe.WeightPlanConstraints(scale_compression=scale_compression),
    )
    packed = fused_moe.PackedWeights(
        w13=w13, w2=w2, w13_block_scales=s13, w2_block_scales=s2,
        w13_global_scales=torch.ones(experts, dtype=torch.float32, device=device),
        w2_global_scales=torch.ones(experts, dtype=torch.float32, device=device),
    )
    scale_scratch = (
        (torch.empty((experts, HIDDEN // 32, 2 * INTERMEDIATE), dtype=torch.uint8, device=device),
         torch.empty((experts, INTERMEDIATE // 32, HIDDEN), dtype=torch.uint8, device=device))
        if scale_compression == "csf" else None
    )
    return fused_moe.prepare_weights(plan=plan, weights=packed, scale_scratch=scale_scratch), scale_scratch


@pytest.fixture(scope="module")
def target_experts():
    return _make_mxfp4_experts(32, torch.device("cuda:0"))[0]


@pytest.fixture(scope="module")
def target_csf_experts():
    return _make_mxfp4_experts(32, torch.device("cuda:0"), scale_compression="csf")


@pytest.fixture(scope="module")
def draft_experts():
    return _make_mxfp4_experts(16, torch.device("cuda:0"))[0]


def _assert_within_atomic_repeatability(*, actual_fn, eager_fn, graph_fn):
    actual, eager, reference = actual_fn(), eager_fn(), graph_fn()
    act_fp32, ref_fp32 = actual.float(), reference.float()
    diffs = [
        eager.float() - ref_fp32,
        *(graph_fn().float() - ref_fp32 for _ in range(6)),
        *(actual_fn().float() - act_fp32 for _ in range(6)),
    ]
    floor = max(float(diff.norm()) for diff in diffs)
    peak = max(float(diff.abs().max()) for diff in diffs)
    error = act_fp32 - ref_fp32
    if floor == 0.0:
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    else:
        assert float(error.norm()) <= 2.0 * floor, (float(error.norm()), floor)
        assert float(error.abs().max()) <= 2.0 * peak, (float(error.abs().max()), peak)


class _PassOp:
    plan = None
    fused_addend = False
    up_projection = SimpleNamespace(plan=None)
    activation_plan = None
    down_projection = SimpleNamespace(plan=None)

    def __init__(self, *args, **kwargs):
        pass

    def functional(self, **kwargs):
        return self

    def start(self, x, **kwargs):
        return x

    def finish(self, pending, dep):
        return pending

    def _run(self, x, m, selection, **kwargs):
        return x, None, selection, {}

    def __call__(self, *args, **kwargs):
        x = args[0]
        return (x, x, x, x) if "pre_out" in kwargs else x


def _build_layer(monkeypatch, *, experts, num_experts, top_k, tokens, mode, metadata,
                 session, arena, vision=False, pipelined_ffn=False, split_mhc=False):
    device = metadata.token_ids.device
    scratch = decoder.Scratch.allocate(tokens, 1, 4, device, experts=num_experts, top_k=top_k)

    class ZeroMLP(_PassOp):
        def __call__(self, x):
            return torch.zeros_like(x)

    for attr in ("PreparedMHC", "PreparedAttention", "PreparedAllReduce", "ConcurrentAllReduce", "PreparedMHCPost"):
        monkeypatch.setattr(decoder, attr, _PassOp)
    monkeypatch.setattr(decoder, "PreparedMLP", ZeroMLP)

    gen = torch.Generator(device="cpu").manual_seed(777 + num_experts)
    router_w = (torch.randn((num_experts, HIDDEN), generator=gen) * 0.05).to(device=device, dtype=torch.float32)
    correction = torch.linspace(2.0, -1.0, num_experts, dtype=torch.float32, device=device)
    text = dict(rms_norm_eps=1e-6, hc_eps=1e-6, hc_sinkhorn_iters=20,
                scoring_func="sqrtsoftplus", norm_topk_prob=True, routed_scaling_factor=2.5, swiglu_limit=10.0)
    norm_w = torch.ones(HIDDEN, dtype=torch.bfloat16, device=device)
    if split_mhc:
        mhc_w = MHCWeights(
            (torch.randn((24, 4 * HIDDEN), generator=gen) * 0.02).to(device=device, dtype=torch.float32),
            torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32, device=device),
            (torch.randn(24, generator=gen) * 0.1).to(device=device, dtype=torch.float32),
            norm_w,
        )
        monkeypatch.setattr(decoder, "_mhc_weights", lambda *args: mhc_w)
    else:
        monkeypatch.setattr(decoder, "_mhc_weights", lambda *args: None)

    weights = SimpleNamespace(
        engram=None, attention=None,
        residual={"attn_norm.weight": norm_w, "ffn_norm.weight": norm_w},
        experts=SimpleNamespace(router=router_w, correction=correction,
                                vision_correction=(-correction.clone()) if vision else None,
                                experts=experts, shared=None),
    )
    tag = "pipe" if pipelined_ffn else ("split" if split_mhc else "plain")
    layer = decoder.PreparedLayer(
        3, weights, None, scratch, metadata, SimpleNamespace(config=SimpleNamespace(text=text)), None,
        shard=SimpleNamespace(world_size=4), mode=mode, runtime=object() if split_mhc else None,
        coordinator=None, session=session, name=f"valid_route.{mode}.{tokens}.{tag}",
        workspace=arena, carry=False, defer_post=split_mhc, pipelined_ffn=pipelined_ffn,
    )
    return layer, weights, text


def _populate_metadata(metadata: Metadata, *, mode: str, rows: int, tokens: int, valid: int, vision: bool):
    device = metadata.token_ids.device
    if mode == "prefill":
        metadata.token_ids.zero_()
        if vision and valid > 0:
            metadata.token_ids[:valid] = torch.where(torch.arange(valid, device=device) % 5 == 0, 129280, 42)
        metadata.live_counts.copy_(torch.tensor([valid, int(valid > 0)], dtype=torch.int32, device=device))
        return
    lengths = torch.zeros(rows, dtype=torch.int32, device=device)
    rem, cap = valid, max(1, tokens // rows)
    for row in range(rows):
        lengths[row] = min(rem, cap)
        rem -= int(lengths[row].item())
    metadata.set(
        torch.zeros(rows, dtype=torch.int64, device=device), lengths,
        torch.arange(1, rows + 1, dtype=torch.int64, device=device),
        torch.zeros((rows, 2), dtype=torch.int32, device=device),
        torch.zeros((rows, cap), dtype=torch.int64, device=device),
        torch.tensor([int((lengths > 0).sum().item())], dtype=torch.int32, device=device),
    )


@pytest.mark.parametrize(
    "mode,num_experts,top_k,tokens,valid_cases,vision,use_csf,pipelined_ffn",
    [
        ("decode", 32, 6, 8, (0, 1, 3, 5, 8), False, False, False),
        ("verify", 32, 6, 32, (0, 1, 13, 21, 32), False, False, False),
        ("draft", 16, 3, 56, (0, 7, 21, 56), False, False, False),
        ("prefill", 32, 6, 256, (0, 1, 73, 199, 256), True, False, False),
        ("decode", 32, 6, 8, (0, 1, 5, 8), False, True, False),
        ("verify", 32, 6, 32, (0, 13, 32), False, True, False),
        ("prefill", 32, 6, 1024, (0, 1, 300, 512, 700, 1024), True, False, True),
    ],
    ids=[
        "decode-uncompressed", "verify-uncompressed", "draft-uncompressed",
        "prefill-vision-uncompressed", "decode-csf", "verify-csf", "prefill-pipelined-ffn",
    ],
)
@torch.no_grad()
def test_deepseek41_layer_excludes_inactive_suffix_rows(
    monkeypatch, target_experts, target_csf_experts, draft_experts,
    mode, num_experts, top_k, tokens, valid_cases, vision, use_csf, pipelined_ffn,
):
    torch._dynamo.reset()
    device = torch.device("cuda:0")
    experts = target_csf_experts[0] if use_csf else (draft_experts if mode == "draft" else target_experts)
    rows = 1 if mode == "prefill" else 8
    r_tokens = tokens // 2 if pipelined_ffn else tokens
    metadata = Metadata.allocate(tokens, rows, 8 if pipelined_ffn else 2, device)
    arena = ScratchArena(device, overlap=False)
    graph = ref_graph = None
    with RecordedPreparationSession(
        device=device, checkpoint="ds41-valid-token-routing",
        workload_schema=f"{mode}-{tokens}-{'csf' if use_csf else 'raw'}-{'pipe' if pipelined_ffn else 'plain'}",
        numerical_policy="checkpoint", compile_workers=4,
    ) as session:
        session.register_scratch(arena)
        layer, weights, text = _build_layer(
            monkeypatch, experts=experts, num_experts=num_experts, top_k=top_k, tokens=tokens,
            mode=mode, metadata=metadata, session=session, arena=arena, vision=vision,
            pipelined_ffn=pipelined_ffn,
        )
        base_router = PreparedRouter(
            torch.zeros((r_tokens, num_experts), dtype=torch.float32, device=device),
            decoder.RouterScratch.allocate(r_tokens, device, experts=num_experts, top_k=top_k),
            session=session, name=f"base.{mode}.{r_tokens}",
            score_func=text["scoring_func"], renormalize=text["norm_topk_prob"],
            routed_scaling_factor=text["routed_scaling_factor"], correction_bias=weights.experts.correction,
            image_correction_bias=weights.experts.vision_correction if vision else None,
            image_mask=torch.zeros(r_tokens, dtype=torch.bool, device=device) if vision else None,
        ).functional(session=session)
        session.tune(show_progress=False)
        arena.release_preparation()

        source = torch.zeros((tokens, HIDDEN), dtype=torch.bfloat16, device=device)
        out_ids = torch.empty((tokens, top_k), dtype=torch.int32, device=device)
        out_routed = torch.empty((tokens, HIDDEN), dtype=torch.bfloat16, device=device)
        unmasked_ids = torch.empty((tokens, top_k), dtype=torch.int32, device=device)
        masked_ids = torch.empty((tokens, top_k), dtype=torch.int32, device=device)
        masked_weights = torch.empty((tokens, top_k), dtype=torch.float32, device=device)
        ref_routed = torch.empty((tokens, HIDDEN), dtype=torch.bfloat16, device=device)
        idx = torch.arange(r_tokens, dtype=torch.int32, device=device).unsqueeze(1)
        pre_mix = torch.zeros((tokens, 4), dtype=torch.float32, device=device)
        selection = (torch.full((tokens, 1), -1, dtype=torch.int32, device=device), None, None)

        def forward():
            incoming, _, _, _, obs = layer._run((source,), pre_mix, metadata, None, selection)
            out_ids.copy_(obs["router.ids"])
            out_routed.copy_(incoming[0])

        def reference_forward():
            mask = (metadata.token_ids >= 129280) if vision else None
            valid = metadata.live_counts[:1]
            if pipelined_ffn:
                ra = base_router(layer.router_projection(source[:r_tokens]), image_mask=mask[:r_tokens])
                rb = base_router(layer.router_projection(source[r_tokens:]), image_mask=mask[r_tokens:])
                unmasked_ids[:r_tokens].copy_(ra.ids)
                unmasked_ids[r_tokens:].copy_(rb.ids)
                ia, wa = torch.where(idx < valid.clamp(0, r_tokens), ra.ids, -1), torch.where(idx < valid.clamp(0, r_tokens), ra.weights, 0.0)
                ib, wb = torch.where(idx < (valid - r_tokens).clamp(0, r_tokens), rb.ids, -1), torch.where(idx < (valid - r_tokens).clamp(0, r_tokens), rb.weights, 0.0)
                masked_ids[:r_tokens].copy_(ia)
                masked_ids[r_tokens:].copy_(ib)
                masked_weights[:r_tokens].copy_(wa)
                masked_weights[r_tokens:].copy_(wb)
                ref_routed[:r_tokens].copy_(layer.experts(source[:r_tokens], ia, wa))
                ref_routed[r_tokens:].copy_(layer.experts(source[r_tokens:], ib, wb))
            else:
                routes = base_router(layer.router_projection(source), image_mask=mask)
                unmasked_ids.copy_(routes.ids)
                ids = torch.where(idx < valid, routes.ids, -1)
                wts = torch.where(idx < valid, routes.weights, 0.0)
                masked_ids.copy_(ids)
                masked_weights.copy_(wts)
                ref_routed.copy_(layer.experts(source, ids, wts))

        compiled = compile_forward(forward)
        ref_compiled = compile_forward(reference_forward)
        metadata.live_counts.copy_(torch.tensor([tokens, 1], dtype=torch.int32, device=device))
        compiled()
        ref_compiled()
        torch.cuda.synchronize()
        graph, ref_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        with session.capture(), torch.cuda.graph(ref_graph):
            ref_compiled()
        session.freeze()

        try:
            for valid in valid_cases:
                source.zero_()
                if valid > 0:
                    source[:valid].normal_(std=0.25)
                _populate_metadata(metadata, mode=mode, rows=rows, tokens=tokens, valid=valid, vision=vision)
                out_ids.fill_(999)
                out_routed.fill_(float("nan"))
                unmasked_ids.fill_(-999)
                arena.tensor.fill_(0xA5)
                torch.cuda.synchronize()
                before = torch.cuda.memory_allocated()
                graph.replay()
                ref_graph.replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_allocated() == before

                if valid < tokens:
                    assert bool((unmasked_ids[valid:] >= 0).all())
                    assert bool((out_ids[valid:] == -1).all())
                    assert bool((out_routed[valid:] == 0).all())

                if valid > 0:
                    assert torch.equal(out_ids[:valid], unmasked_ids[:valid])
                    assert torch.isfinite(out_routed[:valid]).all()
                    assert float(out_routed[:valid].abs().max().item()) > 0.0

                    def replay_actual():
                        graph.replay()
                        torch.cuda.synchronize()
                        return out_routed[:valid].clone()

                    def replay_ref():
                        ref_graph.replay()
                        torch.cuda.synchronize()
                        return ref_routed[:valid].clone()

                    def run_eager():
                        if pipelined_ffn:
                            return torch.cat((
                                layer.experts(source[:r_tokens], masked_ids[:r_tokens], masked_weights[:r_tokens]),
                                layer.experts(source[r_tokens:], masked_ids[r_tokens:], masked_weights[r_tokens:]),
                            ), dim=0)[:valid].clone()
                        return layer.experts(source, masked_ids, masked_weights)[:valid].clone()

                    _assert_within_atomic_repeatability(actual_fn=replay_actual, eager_fn=run_eager, graph_fn=replay_ref)
        finally:
            if graph is not None:
                graph.reset()
            if ref_graph is not None:
                ref_graph.reset()


@pytest.mark.parametrize("tokens,valid_cases", [(4, (0, 1, 3, 4)), (8, (0, 1, 3, 5, 8))])
@torch.no_grad()
def test_deepseek41_split_mhc_layer_excludes_inactive_suffix_rows(monkeypatch, target_experts, tokens, valid_cases):
    from b12x.norm.mhc._split_lagged import supports_split_lagged

    device = torch.device("cuda:0")
    if not supports_split_lagged(device, tokens, HIDDEN, torch.bfloat16):
        pytest.skip("SplitLaggedMHC requires SM120+ support")
    torch._dynamo.reset()
    metadata = Metadata.allocate(tokens, tokens, 2, device)
    arena = ScratchArena(device, overlap=True)
    monkeypatch.setattr(decoder.ConcurrentSharedExpert, "__init__", lambda self, shared, **kw: setattr(self, "shared", shared))
    monkeypatch.setattr(decoder.ConcurrentSharedExpert, "functional", lambda self, **kw: self)
    monkeypatch.setattr(decoder.ConcurrentSharedExpert, "start", lambda self, x: x)
    monkeypatch.setattr(decoder.ConcurrentSharedExpert, "finish", lambda self, p, dep: torch.zeros_like(dep))

    graph = None
    with RecordedPreparationSession(
        device=device, checkpoint="ds41-split-mhc-valid-routing",
        workload_schema=f"decode-split-mhc-{tokens}", numerical_policy="checkpoint", compile_workers=4,
    ) as session:
        session.register_scratch(arena)
        layer, _, _ = _build_layer(
            monkeypatch, experts=target_experts, num_experts=32, top_k=6, tokens=tokens,
            mode="decode", metadata=metadata, session=session, arena=arena, split_mhc=True,
        )
        assert layer.split_mhc
        session.tune(show_progress=False)
        arena.release_preparation()
        assert layer.ffn_hc.emit_mxfp8 == layer.experts.accepts_prequant_mxfp8

        residual = torch.zeros((tokens, 4, HIDDEN), dtype=torch.bfloat16, device=device)
        pre_mix = torch.zeros((tokens, 4), dtype=torch.float32, device=device)
        pre_mix[:, 0] = 1.0
        out_ids = torch.empty((tokens, 6), dtype=torch.int32, device=device)
        out_routed = torch.empty((tokens, HIDDEN), dtype=torch.bfloat16, device=device)
        selection = (torch.full((tokens, 1), -1, dtype=torch.int32, device=device), None, None)

        def forward():
            incoming, _, _, _, obs = layer._run((residual,), pre_mix, metadata, None, selection)
            out_ids.copy_(obs["router.ids"])
            out_routed.copy_(incoming[0])

        compiled = compile_forward(forward)
        metadata.live_counts[0] = tokens
        compiled()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()

        try:
            for valid in valid_cases:
                residual.zero_()
                if valid > 0:
                    residual[:valid].normal_(std=0.25)
                _populate_metadata(metadata, mode="decode", rows=tokens, tokens=tokens, valid=valid, vision=False)
                out_ids.fill_(999)
                out_routed.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                if valid < tokens:
                    assert bool((out_ids[valid:] == -1).all())
                    assert bool((out_routed[valid:] == 0).all())
                if valid > 0:
                    assert bool((out_ids[:valid] >= 0).all())
                    assert torch.isfinite(out_routed[:valid]).all()
                    assert float(out_routed[:valid].abs().max().item()) > 0.0
        finally:
            if graph is not None:
                graph.reset()


@torch.no_grad()
def test_deepseek41_decoder_suffix_excludes_inactive_rows(monkeypatch, target_experts):
    torch._dynamo.reset()
    device = torch.device("cuda:0")
    source_meta = Metadata.allocate(256, 1, 4, device)
    packing = PreparedDecoderSuffix(source_meta, window=128)
    arena = ScratchArena(device, overlap=False)
    graph = None
    with RecordedPreparationSession(
        device=device, checkpoint="ds41-suffix-valid-routing",
        workload_schema="suffix-128", numerical_policy="checkpoint", compile_workers=4,
    ) as session:
        session.register_scratch(arena)
        layer, _, _ = _build_layer(
            monkeypatch, experts=target_experts, num_experts=32, top_k=6, tokens=128,
            mode="prefill", metadata=packing.metadata, session=session, arena=arena, vision=True,
        )
        packing.functional()
        session.tune(show_progress=False)
        arena.release_preparation()

        full_hidden = torch.zeros((256, HIDDEN), dtype=torch.bfloat16, device=device)
        out_ids = torch.empty((128, 6), dtype=torch.int32, device=device)
        out_routed = torch.empty((128, HIDDEN), dtype=torch.bfloat16, device=device)
        pre_mix = torch.zeros((128, 4), dtype=torch.float32, device=device)
        selection = (torch.full((128, 1), -1, dtype=torch.int32, device=device), None, None)

        def forward():
            suffix = packing.begin(source_meta)
            incoming, _, _, _, obs = layer._run((suffix.gather(full_hidden),), pre_mix, suffix.metadata, None, selection)
            out_ids.copy_(obs["router.ids"])
            out_routed.copy_(incoming[0])

        compiled = compile_forward(forward)
        source_meta.lengths.fill_(256)
        source_meta.query_start_loc[1] = 256
        source_meta.live_counts.copy_(torch.tensor([256, 1], dtype=torch.int32, device=device))
        source_meta.positions.copy_(torch.arange(256, dtype=torch.int64, device=device))
        compiled()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()

        try:
            for count in (0, 1, 47, 127, 128, 200):
                kept = min(count, 128)
                full_hidden.zero_()
                if count > 0:
                    full_hidden[:count].normal_(std=0.25)
                source_meta.lengths.fill_(count)
                source_meta.query_start_loc[1] = count
                source_meta.live_counts.copy_(torch.tensor([count, int(count > 0)], dtype=torch.int32, device=device))
                source_meta.positions.fill_(-1)
                if count > 0:
                    source_meta.positions[:count] = torch.arange(count, dtype=torch.int64, device=device)
                out_ids.fill_(999)
                out_routed.fill_(float("nan"))
                arena.tensor.fill_(0xA5)
                graph.replay()
                torch.cuda.synchronize()
                if kept < 128:
                    assert bool((out_ids[kept:] == -1).all())
                    assert bool((out_routed[kept:] == 0).all())
                if kept > 0:
                    assert bool((out_ids[:kept] >= 0).all())
                    assert torch.isfinite(out_routed[:kept]).all()
                    assert float(out_routed[:kept].abs().max().item()) > 0.0
        finally:
            if graph is not None:
                graph.reset()
