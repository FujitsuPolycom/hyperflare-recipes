# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Decoder composition selects expert overlap and valid-token routing bounds across modes."""

from types import SimpleNamespace

import pytest
import torch

from hyperflare_recipes.recipes.deepseek41 import decoder


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 8, 33, 48, 64, 96, 128, 129, 1024])
@pytest.mark.parametrize("mode", ["decode", "verify", "draft", "prefill"])
@pytest.mark.parametrize("overlap", [False, True])
def test_layer_overlap_uses_physical_capacity_and_enabled_workspace(monkeypatch, rows, mode, overlap):
    scratch = decoder.Scratch.allocate(rows, 1, 4, "meta")
    calls = []

    class Component:
        plan = object()
        up_projection = SimpleNamespace(plan=object())
        activation_plan = object()
        down_projection = SimpleNamespace(plan=object())

        def functional(self, **kwargs):
            return self

    def component(*args, **kwargs):
        return Component()

    def concurrent(shared, **kwargs):
        calls.append((shared, kwargs))
        return Component()

    router_calls = []

    def router_component(logits, router_scratch, **kwargs):
        router_calls.append((tuple(logits.shape), kwargs))
        return Component()

    for name in ("PreparedMHC", "PreparedAttention", "PreparedAllReduce", "ConcurrentAllReduce",
                 "UnquantizedProjection", "PreparedRoutedExperts", "PreparedMLP", "PreparedMHCPost"):
        monkeypatch.setattr(decoder, name, component)
    monkeypatch.setattr(decoder, "_mhc_weights", lambda *args: None)
    monkeypatch.setattr(decoder, "register_finalizer", lambda *args, **kwargs: None)
    monkeypatch.setattr(decoder, "ConcurrentSharedExpert", concurrent)
    monkeypatch.setattr(decoder, "PreparedRouter", router_component)
    text = dict(rms_norm_eps=1e-6, hc_eps=1e-6, hc_sinkhorn_iters=20,
                scoring_func="sigmoid", norm_topk_prob=True, routed_scaling_factor=1., swiglu_limit=10.)
    weights = SimpleNamespace(engram=None, attention=None, experts=SimpleNamespace(
        vision_correction=None, router=None, correction=None, experts=None, shared=None))
    workspace = SimpleNamespace(overlap=overlap)
    session = object()
    live_counts = torch.zeros(2, dtype=torch.int32, device="meta")
    pipelined = mode == "prefill" and rows >= 1024 and rows % 2 == 0 and overlap
    layer = decoder.PreparedLayer(3, weights, None, scratch,
        SimpleNamespace(token_ids=torch.empty(rows, device="meta"), live_counts=live_counts),
        SimpleNamespace(config=SimpleNamespace(text=text)), None,
        shard=SimpleNamespace(world_size=4), mode=mode, runtime=None, coordinator=None,
        session=session, name="layer", workspace=workspace, carry=False, defer_post=True,
        pipelined_ffn=pipelined)
    assert len(router_calls) == 1
    logits_shape, r_kwargs = router_calls[0]
    assert logits_shape[0] == (rows // 2 if pipelined else rows)
    assert r_kwargs["num_valid_tokens"].shape == (1,)
    assert r_kwargs["num_valid_tokens"].dtype == torch.int32
    expected = overlap and mode in ("decode", "verify", "draft") and rows <= 128
    assert (layer.concurrent_shared is not None) == expected
    if expected:
        shared, options = calls.pop()
        assert shared is layer.shared
        assert options == dict(parallel_plans=(layer.router_projection.plan, layer.router.plan,
            layer.experts.plan), session=session, workspace=workspace, defer_launch=rows > 4)
    assert not calls


def test_layer_consumes_standalone_router_results_with_runtime_image_mask():
    layer = object.__new__(decoder.PreparedLayer)
    layer.engram = layer.concurrent_shared = None
    layer.has_image_mask = True
    residual = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    metadata = SimpleNamespace(token_ids=torch.tensor([7, 129280, 9]),
                               live_counts=torch.tensor([2, 1], dtype=torch.int32))
    selection = (torch.zeros(3, 1, dtype=torch.int32), None, None)
    layer.attention_hc = lambda source, **kwargs: (source, source, source, source)
    layer.attention = SimpleNamespace(_run=lambda source, m, selected, **kwargs:
        (source * 2, source, selected, {}))
    layer.attention_reduce = lambda source: source
    layer.ffn_hc = lambda source, *args, **kwargs: (source, source, source, source)
    layer.router_projection = lambda source: source + 1
    observed = []
    def route(logits, *, image_mask, num_valid_tokens):
        observed.append((logits.clone(), image_mask.clone(), num_valid_tokens.clone()))
        return SimpleNamespace(ids=torch.tensor([[0], [1], [0]]), weights=logits[:, :1],
                               selected_logits=logits[:, :1])
    layer.router = route
    def experts(source, ids, weights, **kwargs):
        assert torch.equal(ids, torch.tensor([[0], [1], [0]]))
        assert torch.equal(weights, source[:, :1] + 1)
        return source * weights
    layer.experts = experts
    layer.shared = lambda source: source + 3
    layer.mlp_reduce = lambda source: source
    layer.post = lambda source, *args: source
    incoming, _, records, actual_selection = layer((residual,), torch.zeros(3, 4),
                                                  metadata, None, selection)
    normalized = residual * 2
    assert torch.equal(incoming[0], normalized * (normalized[:, :1] + 1) + normalized + 3)
    assert torch.equal(observed[0][0], normalized + 1)
    assert torch.equal(observed[0][1], torch.tensor([False, True, False]))
    assert torch.equal(observed[0][2], torch.tensor([2], dtype=torch.int32))
    assert records is residual
    assert actual_selection is selection


def test_pipelined_ffn_layer_clamps_valid_tokens_per_microbatch():
    layer = object.__new__(decoder.PreparedLayer)
    layer.engram = layer.concurrent_shared = layer.boundary = layer.post = None
    layer.pipelined_ffn = True
    layer.pipelined_attn = None
    layer.pipelined_attn_hc = False
    layer.defer_mlp_b = False
    layer.half_tokens = 4
    layer.has_image_mask = True

    layer.attention_hc = lambda source, **kwargs: (source, source, source, source)
    layer.attention = SimpleNamespace(_run=lambda source, m, selected, **kwargs:
        (source, None, selected, {}))
    layer.att_reduce_0 = lambda source, **kwargs: source
    layer.cr_att_1 = SimpleNamespace(start=lambda x, **kw: x, finish=lambda p, d: p)
    layer.cr_mlp_0 = SimpleNamespace(start=lambda x, **kw: x, finish=lambda p, d: p)
    layer.cr_mlp_1 = SimpleNamespace(start=lambda x, **kw: x, finish=lambda p, d: p)
    layer.ffn_hc = lambda source, *args, **kwargs: (source, source, source, source)
    layer.router_projection = lambda source: source
    observed = []

    def route(logits, *, image_mask, num_valid_tokens):
        observed.append((image_mask.tolist(), num_valid_tokens.tolist(), num_valid_tokens.dtype))
        rows = logits.shape[0]
        return SimpleNamespace(
            ids=torch.zeros((rows, 1), dtype=torch.int32),
            weights=torch.ones((rows, 1), dtype=torch.float32),
            selected_logits=logits[:, :1],
        )

    layer.router = route
    layer.experts = lambda source, ids, weights, **kwargs: source
    layer.shared = lambda source: torch.zeros_like(source)

    residual = torch.ones((8, 2), dtype=torch.float32)
    tokens = torch.tensor([1, 129280, 2, 3, 129281, 4, 5, 6])
    selection = (torch.zeros((8, 1), dtype=torch.int32), None, None)

    for live, expected_a, expected_b in ((0, 0, 0), (3, 3, 0), (4, 4, 0), (6, 4, 2), (8, 4, 4)):
        observed.clear()
        meta = SimpleNamespace(token_ids=tokens, live_counts=torch.tensor([live, int(live > 0)], dtype=torch.int32))
        layer((residual,), torch.zeros((8, 4)), meta, None, selection)
        assert observed == [
            ([False, True, False, False], [expected_a], torch.int32),
            ([True, False, False, False], [expected_b], torch.int32),
        ]


def test_split_mhc_layer_passes_num_valid_tokens_before_prequant_experts():
    layer = object.__new__(decoder.PreparedLayer)
    layer.engram = layer.concurrent_shared = layer.boundary = layer.post = None
    layer.pipelined_ffn = False
    layer.pipelined_attn = None
    layer.pipelined_attn_hc = False
    layer.split_mhc = True
    layer.has_image_mask = False

    pending = SimpleNamespace(pre_out=torch.zeros((4, 4)))
    layer.attention_hc = SimpleNamespace(
        post_y_norm=lambda *args, **kwargs: (args[0], args[0], None, None),
        start_sinkhorn=lambda *args: pending,
        finish_sinkhorn=lambda p, dep: (dep, dep, p.pre_out),
    )
    layer.attention = SimpleNamespace(_run=lambda source, m, selected, **kwargs:
        (source, None, selected, {}))
    layer.attention_reduce = lambda source, **kwargs: source
    prequant_v = torch.ones((4, 4), dtype=torch.uint8)
    prequant_s = torch.ones((4, 1), dtype=torch.uint8)
    layer.ffn_hc = SimpleNamespace(
        post_y_norm=lambda *args, **kwargs: (args[0], args[0], prequant_v, prequant_s),
        start_sinkhorn=lambda *args: pending,
        finish_sinkhorn=lambda p, dep: (dep, dep, p.pre_out),
    )
    layer.router_projection = lambda source: source
    seen = []

    def route(logits, *, image_mask, num_valid_tokens):
        seen.append(("route", image_mask, num_valid_tokens.tolist()))
        return SimpleNamespace(ids=torch.zeros((4, 1), dtype=torch.int32),
                               weights=torch.ones((4, 1), dtype=torch.float32),
                               selected_logits=logits[:, :1])

    def experts(source, ids, weights, **kwargs):
        seen.append(("experts", kwargs["prequant_a_values"] is prequant_v, kwargs["prequant_a_scales"] is prequant_s))
        return source

    layer.router = route
    layer.experts = experts
    layer.shared = lambda source: torch.zeros_like(source)
    layer.mlp_reduce = lambda source, **kwargs: source

    meta = SimpleNamespace(token_ids=torch.zeros(4, dtype=torch.int64),
                           live_counts=torch.tensor([3, 1], dtype=torch.int32))
    layer((torch.ones((4, 4)),), torch.zeros((4, 4)), meta, None, (torch.zeros((4, 1), dtype=torch.int32), None, None))
    assert seen == [("route", None, [3]), ("experts", True, True)]
