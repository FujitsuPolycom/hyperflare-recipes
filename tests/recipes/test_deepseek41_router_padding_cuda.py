# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek V4.1 router padding skip excludes inactive tail rows across execution modes.

Exercises production `PreparedLayer` router and MXFP4 W4A8 routed-expert wiring
under compiled CUDA-graph replay with synthetic weights on one GPU. Attention,
hyperconnections, shared MLP, and collectives are replaced with pass-through
doubles so the test isolates router and routed-expert behavior across `decode`,
`verify`, `draft`, `prefill`, and `PreparedDecoderSuffix` metadata.
"""

import os
from types import SimpleNamespace

import pytest
from ignition import runtime_patches

runtime_patches.apply()

import torch
from b12x.moe import fused_moe

from hyperflare_recipes.components.linear import UnquantizedProjection
from hyperflare_recipes.components.moe import PreparedRouter, PreparedRoutedExperts
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
    # Use small valid E8M0 exponents (123..126 -> 2^-4..2^-1) to keep BF16 outputs finite and nonzero.
    s13 = torch.randint(123, 127, (experts, 2 * INTERMEDIATE, HIDDEN // 32),
                        generator=generator, dtype=torch.uint8).to(device)
    s2 = torch.randint(123, 127, (experts, HIDDEN, INTERMEDIATE // 32),
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
    scale_scratch = None
    if scale_compression == "csf":
        scale_scratch = (
            torch.empty((experts, HIDDEN // 32, 2 * INTERMEDIATE), dtype=torch.uint8, device=device),
            torch.empty((experts, INTERMEDIATE // 32, HIDDEN), dtype=torch.uint8, device=device),
        )
    return fused_moe.prepare_weights(plan=plan, weights=packed, scale_scratch=scale_scratch), scale_scratch


@pytest.fixture(scope="module")
def target_experts():
    device = torch.device("cuda:0")
    experts, _ = _make_mxfp4_experts(32, device, scale_compression=None)
    return experts


@pytest.fixture(scope="module")
def target_csf_experts():
    device = torch.device("cuda:0")
    return _make_mxfp4_experts(32, device, scale_compression="csf")


@pytest.fixture(scope="module")
def draft_experts():
    device = torch.device("cuda:0")
    experts, _ = _make_mxfp4_experts(16, device, scale_compression=None)
    return experts


def _build_layer_for_router_test(monkeypatch, *, experts, num_experts, top_k, tokens, mode,
                                 metadata, session, arena, vision=False, pipelined_ffn=False):
    device = metadata.token_ids.device
    scratch = decoder.Scratch.allocate(tokens, 1, 4, device, experts=num_experts, top_k=top_k)

    class PassMHC:
        plan = None

        def __init__(self, incoming, weights, out_scratch, **kwargs):
            self.out = out_scratch

        def functional(self):
            return self

        def __call__(self, *args, pre_mix, pre_out):
            x = args[0]
            return x, x, x, x

    class PassAttention:
        def __init__(self, *args, **kwargs):
            pass

        def _run(self, x, m, selection, **kwargs):
            return x, None, selection, {}

    class PassOp:
        plan = None

        def __init__(self, *args, **kwargs):
            pass

        def functional(self, **kwargs):
            return self

        def __call__(self, x, *args, **kwargs):
            return x

    class PassConcurrentReduce:
        def __init__(self, reduce, **kwargs):
            self.reduce = reduce

        def functional(self, **kwargs):
            return self

        def start(self, x, **kwargs):
            return x

        def finish(self, pending, dep):
            return pending

    class ZeroShared:
        up_projection = SimpleNamespace(plan=None)
        activation_plan = None
        down_projection = SimpleNamespace(plan=None)

        def __init__(self, *args, **kwargs):
            pass

        def functional(self, **kwargs):
            return self

        def __call__(self, x):
            return torch.zeros_like(x)

    monkeypatch.setattr(decoder, "PreparedMHC", PassMHC)
    monkeypatch.setattr(decoder, "PreparedAttention", PassAttention)
    monkeypatch.setattr(decoder, "PreparedAllReduce", PassOp)
    monkeypatch.setattr(decoder, "ConcurrentAllReduce", PassConcurrentReduce)
    monkeypatch.setattr(decoder, "PreparedMLP", ZeroShared)
    monkeypatch.setattr(decoder, "PreparedMHCPost", PassOp)
    monkeypatch.setattr(decoder, "_mhc_weights", lambda *args: None)

    generator = torch.Generator(device="cpu").manual_seed(777 + num_experts)
    router_weight = (torch.randn((num_experts, HIDDEN), generator=generator) * 0.05).to(
        device=device, dtype=torch.float32)
    # Strong positive correction bias ensures unmasked zeroed padding rows select experts 0..top_k-1.
    correction = torch.linspace(2.0, -1.0, num_experts, dtype=torch.float32, device=device)
    vision_correction = (-correction.clone()) if vision else None
    text = dict(
        rms_norm_eps=1e-6, hc_eps=1e-6, hc_sinkhorn_iters=20,
        scoring_func="sqrtsoftplus", norm_topk_prob=True,
        routed_scaling_factor=2.5, swiglu_limit=10.0,
    )
    weights = SimpleNamespace(
        engram=None,
        attention=None,
        experts=SimpleNamespace(
            router=router_weight,
            correction=correction,
            vision_correction=vision_correction,
            experts=experts,
            shared=None,
        ),
    )
    layer = decoder.PreparedLayer(
        3, weights, None, scratch, metadata,
        SimpleNamespace(config=SimpleNamespace(text=text)), None,
        shard=SimpleNamespace(world_size=4), mode=mode, runtime=None, coordinator=None,
        session=session, name=f"pad.{mode}.{tokens}.{'pipe' if pipelined_ffn else 'plain'}",
        workspace=arena, carry=False, defer_post=False, pipelined_ffn=pipelined_ffn,
    )
    return layer, scratch, weights, text


@pytest.mark.parametrize(
    "mode,num_experts,top_k,tokens,valid_cases,vision,use_csf",
    [
        ("decode", 32, 6, 8, (0, 1, 3, 5, 8), False, False),
        ("verify", 32, 6, 32, (0, 1, 13, 21, 32), False, False),
        ("draft", 16, 3, 56, (0, 7, 21, 56), False, False),
        ("prefill", 32, 6, 256, (0, 1, 73, 199, 256), True, False),
        ("decode", 32, 6, 8, (0, 1, 5, 8), False, True),
        ("verify", 32, 6, 32, (0, 13, 32), False, True),
    ],
    ids=[
        "decode-uncompressed",
        "verify-uncompressed",
        "draft-uncompressed",
        "prefill-vision-uncompressed",
        "decode-csf",
        "verify-csf",
    ],
)
@torch.no_grad()
def test_deepseek41_layer_skips_padded_tail_rows(
    monkeypatch, target_experts, target_csf_experts, draft_experts,
    mode, num_experts, top_k, tokens, valid_cases, vision, use_csf,
):
    torch._dynamo.reset()
    device = torch.device("cuda:0")
    if use_csf:
        experts, _csf_owner = target_csf_experts
    elif mode == "draft":
        experts = draft_experts
    else:
        experts = target_experts

    rows = 1 if mode == "prefill" else 8
    metadata = Metadata.allocate(tokens, rows, 2, device)
    arena = ScratchArena(device, overlap=False)
    graph = None
    with RecordedPreparationSession(
        device=device, checkpoint="ds41-router-pad",
        workload_schema=f"{mode}-{tokens}-{'csf' if use_csf else 'raw'}",
        numerical_policy="checkpoint", compile_workers=4,
    ) as session:
        session.register_scratch(arena)
        layer, scratch, weights, text = _build_layer_for_router_test(
            monkeypatch, experts=experts, num_experts=num_experts, top_k=top_k,
            tokens=tokens, mode=mode, metadata=metadata, session=session, arena=arena,
            vision=vision,
        )
        # Baseline router without num_valid_tokens proves unmasked zeroed tail rows route to real experts.
        base_logits = torch.zeros((tokens, num_experts), dtype=torch.float32, device=device)
        base_scratch = decoder.RouterScratch.allocate(tokens, device, experts=num_experts, top_k=top_k)
        base_mask = torch.zeros(tokens, dtype=torch.bool, device=device) if vision else None
        base_router = PreparedRouter(
            base_logits, base_scratch, session=session, name=f"base.{mode}.{tokens}",
            score_func=text["scoring_func"], renormalize=text["norm_topk_prob"],
            routed_scaling_factor=text["routed_scaling_factor"],
            correction_bias=weights.experts.correction,
            image_correction_bias=weights.experts.vision_correction if vision else None,
            image_mask=base_mask,
        ).functional(session=session)
        session.tune(show_progress=False)
        arena.release_preparation()

        source = torch.zeros((tokens, HIDDEN), dtype=torch.bfloat16, device=device)
        out_ids = torch.empty((tokens, top_k), dtype=torch.int32, device=device)
        out_routed = torch.empty((tokens, HIDDEN), dtype=torch.bfloat16, device=device)
        pre_mix = torch.zeros((tokens, 4), dtype=torch.float32, device=device)
        selection = (torch.full((tokens, 1), -1, dtype=torch.int32, device=device), None, None)

        def forward():
            incoming, _, _, _, obs = layer._run((source,), pre_mix, metadata, None, selection)
            out_ids.copy_(obs["router.ids"])
            out_routed.copy_(incoming[0])

        compiled = compile_forward(forward)
        metadata.live_counts[0] = tokens
        compiled()
        base_router(layer.router_projection(source), image_mask=base_mask)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()

        try:
            checked_partial = False
            for valid in valid_cases:
                source.zero_()
                if valid > 0:
                    source[:valid].normal_(std=0.25)
                # Populate realistic mode metadata so live_counts[:1] matches actual packed requests.
                if mode == "prefill":
                    metadata.token_ids.zero_()
                    if vision and valid > 0:
                        metadata.token_ids[:valid] = torch.where(
                            torch.arange(valid, device=device) % 5 == 0, 129280, 42)
                    metadata.live_counts[0] = valid
                    metadata.live_counts[1] = 1 if valid > 0 else 0
                else:
                    lengths = torch.zeros(rows, dtype=torch.int32, device=device)
                    remaining = valid
                    per_row_max = max(1, tokens // rows)
                    for r in range(rows):
                        take = min(remaining, per_row_max)
                        lengths[r] = take
                        remaining -= take
                    assert int(lengths.sum().item()) == valid
                    metadata.set(
                        torch.zeros(rows, dtype=torch.int64, device=device),
                        lengths,
                        torch.arange(1, rows + 1, dtype=torch.int64, device=device),
                        torch.zeros((rows, 2), dtype=torch.int32, device=device),
                        torch.zeros((rows, per_row_max), dtype=torch.int64, device=device),
                        torch.tensor([int((lengths > 0).sum().item())], dtype=torch.int32, device=device),
                    )
                    assert int(metadata.live_counts[0].item()) == valid

                # Poison outputs and arena before replay to ensure tail rows are actively zeroed, not stale zeros.
                out_ids.fill_(999)
                out_routed.fill_(float("nan"))
                arena.tensor.fill_(0xA5)
                torch.cuda.synchronize()
                before = torch.cuda.memory_allocated()
                graph.replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_allocated() == before

                # Negative check on baseline (unmasked router) for a padded case: tail rows must route to valid experts.
                logits_ref = layer.router_projection(source)
                mask_ref = (metadata.token_ids >= 129280) if vision else None
                base_route = base_router(logits_ref, image_mask=mask_ref)
                if valid < tokens:
                    checked_partial = True
                    assert bool((base_route.ids[valid:] >= 0).all()), (
                        "baseline without num_valid_tokens must route zeroed tail rows to valid experts")
                    assert bool((out_ids[valid:] == -1).all()), (
                        f"mode={mode} valid={valid}: expected padded tail ids == -1, got {out_ids[valid:]}")
                    assert bool((out_routed[valid:] == 0).all()), (
                        f"mode={mode} valid={valid}: expected exact-zero routed output on padded tail")

                if valid > 0:
                    assert torch.equal(out_ids[:valid], base_route.ids[:valid]), (
                        f"mode={mode} valid={valid}: valid prefix route IDs must match baseline bit-for-bit")
                    assert torch.isfinite(out_routed[:valid]).all()
                    assert float(out_routed[:valid].abs().max().item()) > 0.0
                    # Compare valid prefix against both (a) explicit tail-masked routes (same active
                    # expert distribution) and (b) unmasked baseline routes. Because dynamic W4A8
                    # sums BF16 expert outputs atomically across top_k experts, check cosine
                    # similarity and relative L2 error grounded in BF16 precision (1 ULP = 2^-7).
                    masked_ids = base_route.ids.clone()
                    masked_weights = base_route.weights.clone()
                    if valid < tokens:
                        masked_ids[valid:] = -1
                        masked_weights[valid:] = 0.0
                    for label, ref_ids, ref_weights in (
                        ("masked", masked_ids, masked_weights),
                        ("unmasked", base_route.ids, base_route.weights),
                    ):
                        ref_routed = layer.experts(source, ref_ids, ref_weights)[:valid].float()
                        actual = out_routed[:valid].float()
                        rel_l2 = float((actual - ref_routed).norm() / ref_routed.norm().clamp_min(1e-6))
                        cos = float(torch.nn.functional.cosine_similarity(
                            actual.reshape(1, -1), ref_routed.reshape(1, -1)).item())
                        assert rel_l2 <= 5e-3 and cos >= 0.99998, (
                            f"mode={mode} valid={valid} ({label}): rel_l2={rel_l2:.6f} cos={cos:.6f}")
            assert checked_partial
        finally:
            if graph is not None:
                graph.reset()


@torch.no_grad()
def test_deepseek41_decoder_suffix_skips_padded_tail_rows(monkeypatch, target_experts):
    """PreparedDecoderSuffix populates live_counts[:1] = min(count, 128) and skips tail rows."""
    torch._dynamo.reset()
    device = torch.device("cuda:0")
    source_meta = Metadata.allocate(256, 1, 4, device)
    packing = PreparedDecoderSuffix(source_meta, window=128)
    suffix_meta = packing.metadata
    arena = ScratchArena(device, overlap=False)
    graph = None
    with RecordedPreparationSession(
        device=device, checkpoint="ds41-suffix-router-pad",
        workload_schema="suffix-128", numerical_policy="checkpoint", compile_workers=4,
    ) as session:
        session.register_scratch(arena)
        layer, _, _, _ = _build_layer_for_router_test(
            monkeypatch, experts=target_experts, num_experts=32, top_k=6,
            tokens=128, mode="prefill", metadata=suffix_meta, session=session, arena=arena,
            vision=True,
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
            gathered = suffix.gather(full_hidden)
            incoming, _, _, _, obs = layer._run((gathered,), pre_mix, suffix.metadata, None, selection)
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
                source_meta.live_counts.copy_(torch.tensor([count, 1 if count else 0], dtype=torch.int32, device=device))
                source_meta.positions.fill_(-1)
                if count > 0:
                    source_meta.positions[:count] = torch.arange(count, dtype=torch.int64, device=device)
                out_ids.fill_(999)
                out_routed.fill_(float("nan"))
                arena.tensor.fill_(0xA5)
                graph.replay()
                torch.cuda.synchronize()
                if kept < 128:
                    assert bool((out_ids[kept:] == -1).all()), f"count={count} kept={kept}"
                    assert bool((out_routed[kept:] == 0).all()), f"count={count} kept={kept}"
                if kept > 0:
                    assert bool((out_ids[:kept] >= 0).all()), f"count={count} kept={kept}"
                    assert torch.isfinite(out_routed[:kept]).all()
                    assert float(out_routed[:kept].abs().max().item()) > 0.0
        finally:
            if graph is not None:
                graph.reset()


@torch.no_grad()
def test_deepseek41_pipelined_ffn_layer_skips_padded_tail_rows_across_microbatches(monkeypatch, target_experts):
    """Two-microbatch pipelined prefill clamps live_counts[:1] independently across halves [0, h) and [h, 2h)."""
    torch._dynamo.reset()
    device = torch.device("cuda:0")
    tokens, h, num_experts, top_k = 1024, 512, 32, 6
    metadata = Metadata.allocate(tokens, 1, 8, device)
    arena = ScratchArena(device, overlap=False)
    graph = None
    with RecordedPreparationSession(
        device=device, checkpoint="ds41-pipe-router-pad",
        workload_schema="prefill-pipe-1024", numerical_policy="checkpoint", compile_workers=4,
    ) as session:
        session.register_scratch(arena)
        layer, _, weights, text = _build_layer_for_router_test(
            monkeypatch, experts=target_experts, num_experts=num_experts, top_k=top_k,
            tokens=tokens, mode="prefill", metadata=metadata, session=session, arena=arena,
            vision=True, pipelined_ffn=True,
        )
        base_logits = torch.zeros((h, num_experts), dtype=torch.float32, device=device)
        base_scratch = decoder.RouterScratch.allocate(h, device, experts=num_experts, top_k=top_k)
        base_mask = torch.zeros(h, dtype=torch.bool, device=device)
        base_router = PreparedRouter(
            base_logits, base_scratch, session=session, name="base.prefill.pipe.512",
            score_func=text["scoring_func"], renormalize=text["norm_topk_prob"],
            routed_scaling_factor=text["routed_scaling_factor"],
            correction_bias=weights.experts.correction,
            image_correction_bias=weights.experts.vision_correction,
            image_mask=base_mask,
        ).functional(session=session)
        session.tune(show_progress=False)
        arena.release_preparation()

        source = torch.zeros((tokens, HIDDEN), dtype=torch.bfloat16, device=device)
        out_ids = torch.empty((tokens, top_k), dtype=torch.int32, device=device)
        out_routed = torch.empty((tokens, HIDDEN), dtype=torch.bfloat16, device=device)
        pre_mix = torch.zeros((tokens, 4), dtype=torch.float32, device=device)
        selection = (torch.full((tokens, 1), -1, dtype=torch.int32, device=device), None, None)

        def forward():
            incoming, _, _, _, obs = layer._run((source,), pre_mix, metadata, None, selection)
            out_ids.copy_(obs["router.ids"])
            out_routed.copy_(incoming[0])

        compiled = compile_forward(forward)
        metadata.live_counts.copy_(torch.tensor([tokens, 1], dtype=torch.int32, device=device))
        compiled()
        base_router(layer.router_projection(source[:h]), image_mask=base_mask)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()

        try:
            for valid in (0, 1, 300, 512, 700, 1024):
                source.zero_()
                metadata.token_ids.zero_()
                if valid > 0:
                    source[:valid].normal_(std=0.25)
                    metadata.token_ids[:valid] = torch.where(
                        torch.arange(valid, device=device) % 7 == 0, 129280, 19)
                metadata.live_counts.copy_(torch.tensor([valid, 1 if valid > 0 else 0], dtype=torch.int32, device=device))

                out_ids.fill_(999)
                out_routed.fill_(float("nan"))
                arena.tensor.fill_(0xA5)
                torch.cuda.synchronize()
                before = torch.cuda.memory_allocated()
                graph.replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_allocated() == before

                mask_full = metadata.token_ids >= 129280
                logits_a = layer.router_projection(source[:h])
                logits_b = layer.router_projection(source[h:])
                base_a = base_router(logits_a, image_mask=mask_full[:h])
                base_b = base_router(logits_b, image_mask=mask_full[h:])
                base_ids = torch.cat((base_a.ids, base_b.ids), dim=0)

                if valid < tokens:
                    assert bool((base_ids[valid:] >= 0).all())
                    assert bool((out_ids[valid:] == -1).all()), f"valid={valid}: tail ids not -1"
                    assert bool((out_routed[valid:] == 0).all()), f"valid={valid}: tail routed not zero"

                if valid > 0:
                    assert torch.equal(out_ids[:valid], base_ids[:valid]), f"valid={valid}: active ids mismatch"
                    assert torch.isfinite(out_routed[:valid]).all()
                    assert float(out_routed[:valid].abs().max().item()) > 0.0
                    ref_a = layer.experts(source[:h], base_a.ids, base_a.weights)
                    ref_b = layer.experts(source[h:], base_b.ids, base_b.weights)
                    ref_full = torch.cat((ref_a, ref_b), dim=0)[:valid].float()
                    actual = out_routed[:valid].float()
                    rel_l2 = float((actual - ref_full).norm() / ref_full.norm().clamp_min(1e-6))
                    cos = float(torch.nn.functional.cosine_similarity(
                        actual.reshape(1, -1), ref_full.reshape(1, -1)).item())
                    assert rel_l2 <= 5e-3 and cos >= 0.99998, (
                        f"valid={valid}: rel_l2={rel_l2:.6f} cos={cos:.6f}")
        finally:
            if graph is not None:
                graph.reset()
