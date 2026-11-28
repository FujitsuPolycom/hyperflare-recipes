# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU verification for DeepSeek V4.1 MXFP4 indexer request_ids wiring and auxiliary q_b overlap."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from b12x.gemm.blockscaled._linear import MXFP8LinearWeight
from b12x.quantization import mxfp8
from ignition.preparation.scratch import ScratchArena

from hyperflare_recipes.components import mxfp4_indexer, projection_overlap
from hyperflare_recipes.fixtures import mxfp4_indexer as mxfp4_fixture
from hyperflare_recipes.recipes.deepseek41 import attention as ds_attention


def _fake_mxfp8_linear_weight(out_features: int, in_features: int) -> MXFP8LinearWeight:
    values = torch.zeros((out_features, in_features), dtype=torch.float8_e4m3fn)
    scale_rows = torch.full((1, out_features, in_features // 32), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    row_tiles = (out_features + 127) // 128
    k_tiles = (in_features + 127) // 128
    scale_mma_phys = torch.full((1, row_tiles, k_tiles, 32, 4, 4), 127, dtype=torch.uint8)
    scale_mma = scale_mma_phys.view(torch.float8_e8m0fnu).permute(3, 4, 1, 5, 2, 0)
    rows = mxfp8.MXFP8Rows(values=values, scale_rows=scale_rows, scale_mma=scale_mma)
    return MXFP8LinearWeight(
        weight=rows,
        out_features=out_features,
        in_features=in_features,
        padded_in_features=in_features,
    )


def test_mxfp4_select_binds_request_ids_for_multiquery_and_expands_fallback(monkeypatch):
    rows, num_requests, blocks = 8, 2, 16
    query = torch.zeros((rows, 32, 64), dtype=torch.uint8)
    scales = torch.full((rows, 32, 4), 127, dtype=torch.uint8)
    weights = torch.ones((rows, 32), dtype=torch.float32)
    cache = torch.zeros((blocks, 64 * 68), dtype=torch.uint8)
    pages = torch.arange(num_requests * blocks, dtype=torch.int32).view(num_requests, blocks)
    lengths = torch.full((rows,), 640, dtype=torch.int32)
    active = torch.tensor([640], dtype=torch.int32)
    output = torch.empty((rows, 512), dtype=torch.int32)
    scratch = torch.empty(16, dtype=torch.uint8)
    request_ids = torch.tensor([1, 1, 0, 1, 0, 0, 1, 0], dtype=torch.int32)

    fake_plan = object()
    monkeypatch.setattr(mxfp4_indexer, "plan_from_handle", lambda handle: fake_plan)

    # Case 1: multiquery state + dsa.bind accepts request_ids.
    calls: list[tuple[str, dict]] = []
    mq_state = SimpleNamespace(multiquery_dense=object(), multiquery_hierarchical=None)
    monkeypatch.setattr(mxfp4_indexer, "require_prepared", lambda plan, name, dev: mq_state)

    def bind_with_req(
        plan, *, scratch, q_mxfp4, q_scales, query_weights, index_k_cache, page_table,
        cache_lengths, active_width, output_indices, candidate_indices=None,
        candidate_lengths=None, candidate_output=None, candidate_output_lengths=None,
        request_ids=None,
    ):
        calls.append(("dsa.bind", dict(page_table=page_table, request_ids=request_ids)))
        return "binding-dsa"

    monkeypatch.setattr(mxfp4_indexer.dsa, "bind", bind_with_req)
    monkeypatch.setattr(mxfp4_indexer.dsa, "run", lambda binding: calls.append(("dsa.run", dict(binding=binding))))

    mxfp4_indexer._select(
        query, scales, weights, cache, pages, lengths, active, output, scratch,
        None, None, None, None, request_ids, 1,
    )
    assert [c[0] for c in calls] == ["dsa.bind", "dsa.run"]
    assert calls[0][1]["page_table"] is pages
    assert calls[0][1]["request_ids"] is request_ids

    # Case 2: multiquery state + dsa.bind lacks request_ids -> binds directly via state.bind/state.run.
    calls.clear()

    def bind_without_req(
        plan, *, scratch, q_mxfp4, q_scales, query_weights, index_k_cache, page_table,
        cache_lengths, active_width, output_indices, candidate_indices=None,
        candidate_lengths=None, candidate_output=None, candidate_output_lengths=None,
    ):
        calls.append(("dsa.bind", dict(page_table=page_table)))
        return "binding-dsa"

    def state_bind(**kwargs):
        calls.append(("state.bind", kwargs))
        return "binding-state"

    mq_state_direct = SimpleNamespace(
        multiquery_dense=None,
        multiquery_hierarchical=object(),
        bind=state_bind,
        run=lambda b: calls.append(("state.run", dict(binding=b))),
    )
    monkeypatch.setattr(mxfp4_indexer, "require_prepared", lambda plan, name, dev: mq_state_direct)
    monkeypatch.setattr(mxfp4_indexer.dsa, "bind", bind_without_req)

    mxfp4_indexer._select(
        query, scales, weights, cache, pages, lengths, active, output, scratch,
        None, None, None, None, request_ids, 1,
    )
    assert [c[0] for c in calls] == ["state.bind", "state.run"]
    assert calls[0][1]["page_table"] is pages
    assert calls[0][1]["request_ids"] is request_ids

    # Case 3: non-multiquery state -> expands [B, blocks] to [rows, blocks] without request_ids.
    calls.clear()
    plain_state = SimpleNamespace(multiquery_dense=None, multiquery_hierarchical=None)
    monkeypatch.setattr(mxfp4_indexer, "require_prepared", lambda plan, name, dev: plain_state)

    mxfp4_indexer._select(
        query, scales, weights, cache, pages, lengths, active, output, scratch,
        None, None, None, None, request_ids, 1,
    )
    assert [c[0] for c in calls] == ["dsa.bind", "dsa.run"]
    expanded = calls[0][1]["page_table"]
    assert expanded.shape == (rows, blocks)
    assert expanded.is_contiguous()
    torch.testing.assert_close(expanded, pages[request_ids.long()], rtol=0, atol=0)
    # Sensitivity: permuting request_ids changes expanded row assignment.
    alt_req_ids = torch.zeros_like(request_ids)
    assert not torch.equal(expanded, pages[alt_req_ids.long()])


def test_mxfp4_indexer_validates_request_ids_and_fixture_bind_arguments(monkeypatch):
    rows, num_requests, blocks = 8, 2, 16
    query = torch.zeros((rows, 32, 128), dtype=torch.bfloat16)
    weights = torch.ones((rows, 32), dtype=torch.float32)
    cache = torch.zeros((blocks, 64 * 68), dtype=torch.uint8)
    pages = torch.zeros((num_requests, blocks), dtype=torch.int32)
    lengths = torch.full((rows,), 640, dtype=torch.int32)
    active = torch.tensor([640], dtype=torch.int32)
    output = torch.empty((rows, 512), dtype=torch.int32)

    monkeypatch.setattr(mxfp4_indexer, "supports_mxfp4_request_ids", lambda: False)
    with pytest.raises(ValueError, match="request_ids support"):
        mxfp4_indexer.PreparedMXFP4Indexer(
            query, weights, cache, pages, lengths, active, output,
            page_size=64, session=object(), name="idx", workspace=object(),
            request_ids=torch.zeros(rows, dtype=torch.int32),
        )

    monkeypatch.setattr(mxfp4_indexer, "supports_mxfp4_request_ids", lambda: True)
    for bad_req_ids in (
        torch.zeros(rows, dtype=torch.int64),
        torch.zeros((rows, 1), dtype=torch.int32),
        torch.zeros(rows - 1, dtype=torch.int32),
        torch.arange(rows * 2, dtype=torch.int32)[::2],
    ):
        with pytest.raises(ValueError, match="contiguous int32"):
            mxfp4_indexer.PreparedMXFP4Indexer(
                query, weights, cache, pages, lengths, active, output,
                page_size=64, session=object(), name="idx", workspace=object(),
                request_ids=bad_req_ids,
            )

    # Verify fixtures/mxfp4_indexer.Inputs initializes valid request_ids and adapts non-multiquery candidates.
    fake_state = SimpleNamespace(
        caps=SimpleNamespace(max_page_table_width=blocks, page_size=64),
        quantize_query=lambda q, q_mxfp4, q_scales: None,
        write_index_keys=lambda k, index_k_cache, slot_mapping: None,
    )
    original = dict(
        q_mxfp4=torch.empty((rows, 32, 64), dtype=torch.uint8),
        q_scales=torch.empty((rows, 32, 4), dtype=torch.uint8),
        query_weights=torch.empty((rows, 32), dtype=torch.float32),
        index_k_cache=cache,
        page_table=torch.empty((num_requests, blocks), dtype=torch.int32),
        cache_lengths=torch.empty((rows,), dtype=torch.int32),
        active_width=torch.empty((1,), dtype=torch.int32),
        output_indices=torch.empty((rows, 512), dtype=torch.int32),
        candidate_indices=None,
        candidate_lengths=None,
        candidate_output=None,
        candidate_output_lengths=None,
        request_ids=torch.full((rows,), -999, dtype=torch.int32),
    )
    inputs = mxfp4_fixture.Inputs(fake_state, original)
    assert inputs.arguments["request_ids"].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert inputs.expanded_page_table is not None
    assert inputs.expanded_page_table.shape == (rows, blocks)

    mq_args = inputs.bind_arguments(SimpleNamespace(multiquery_dense=object(), multiquery_hierarchical=None))
    assert "request_ids" in mq_args
    assert mq_args["page_table"].shape == (num_requests, blocks)

    plain_args = inputs.bind_arguments(SimpleNamespace(multiquery_dense=None, multiquery_hierarchical=None))
    assert "request_ids" not in plain_args
    assert plain_args["page_table"] is inputs.expanded_page_table
    assert plain_args["page_table"].shape == (rows, blocks)


def test_attention_request_page_table_gating_and_sentinel_clamping(monkeypatch):
    assert ds_attention.INDEXER_QUERY_ROWS == 2048
    monkeypatch.setattr(ds_attention, "supports_mxfp4_request_ids", lambda: True)

    base = dict(
        mode="verify",
        row_shard_attention=False,
        layer=2,
        candidate_source=20,
        idx_tokens=8,
        num_requests=1,
        width_cap=1024,
    )
    assert ds_attention._should_use_request_page_table(**base) is True
    assert ds_attention._should_use_request_page_table(**{**base, "mode": "decode", "num_requests": 2}) is True
    assert ds_attention._should_use_request_page_table(**{**base, "mode": "draft", "layer": 20}) is True
    assert ds_attention._should_use_request_page_table(**{**base, "mode": "prefill"}) is False
    assert ds_attention._should_use_request_page_table(**{**base, "row_shard_attention": True}) is False
    assert ds_attention._should_use_request_page_table(**{**base, "layer": 26}) is False
    assert ds_attention._should_use_request_page_table(**{**base, "idx_tokens": 1, "num_requests": 1}) is False
    assert ds_attention._should_use_request_page_table(**{**base, "num_requests": 0}) is False
    assert ds_attention._should_use_request_page_table(**{**base, "idx_tokens": 8, "num_requests": 8}) is False
    assert ds_attention._should_use_request_page_table(**{**base, "width_cap": 512}) is False
    assert ds_attention._should_use_request_page_table(**{**base, "layer": 20, "width_cap": 32769 * 8}) is False

    monkeypatch.setattr(ds_attention, "supports_mxfp4_request_ids", lambda: False)
    assert ds_attention._should_use_request_page_table(**base) is False

    # Drive PreparedAttention._run with negative and out-of-range request_ids sentinels.
    tokens, num_requests, blocks = 4, 2, 16
    selector_calls: list[dict] = []
    events: list[str] = []

    class _Selector:
        cache = torch.zeros(1)

        def __call__(self, iq, iw, pages, lengths, active, **kwargs):
            events.append("selector")
            selector_calls.append(dict(pages=pages, request_ids=kwargs.get("request_ids"), quantized=kwargs.get("quantized")))
            return torch.zeros((tokens, 512), dtype=torch.int32)

    class _Concurrent:
        has_indexer = True

        def start(self, query_low, source, layer_angles, *, index_angles=None, encoded_bases=None):
            events.append("start")
            return ("pending",)

        def finish_indexer(self, pending, dependency, *, global_dependency=None):
            events.append("finish_indexer")
            assert pending == ("pending",)
            assert global_dependency is selector.cache
            iq_flat = torch.zeros((tokens, 32 * 128), dtype=torch.bfloat16)
            iw = torch.ones((tokens, 32), dtype=torch.float32)
            return iq_flat, iw, ((torch.zeros(1), torch.zeros(1)),)

        def finish(self, pending, dependency, *, global_dependency=None):
            events.append("finish")
            assert pending == ("pending",)
            assert dependency.shape == (tokens, 512)
            return torch.zeros((tokens, 4, 4), dtype=torch.bfloat16)

    selector = _Selector()
    attn = object.__new__(ds_attention.PreparedAttention)
    attn.owns_window_metadata = False
    attn.mode = "verify"
    attn.dfly_indexed = False
    attn.use_compressed_rope = True
    attn.window_page_size = 128
    attn.window = torch.zeros((2, 128 * 528), dtype=torch.uint8)
    qkv_fn = lambda src, encoded_bases=None: torch.zeros((tokens, 1792), dtype=torch.bfloat16)  # noqa: E731
    qkv_fn.encoded_layout = None
    attn.projections = {"qkv_a": qkv_fn}
    attn.fused_qkv_norm_rope = False
    attn.norm_quant = None
    attn.norm_quant_fused_kv_rope = False
    attn.query_norm = lambda x: x
    attn.window_norm = lambda x: x
    attn.angles = SimpleNamespace(apply=lambda wv, ang: None)
    attn.concurrent_query = _Concurrent()
    attn.window_writer = lambda wv, slots, history_rows=None, cache=None: events.append("window_writer")
    attn.ratio = 2
    attn.layer = 2
    attn.candidate_source = 20
    attn.global_kv = lambda src, sel0, m, geom, angles=None: events.append("global_kv") or "records"
    attn.selectors = [selector]
    attn.use_request_page_table = True
    attn.query_shape = (tokens, 4, 4)
    attn.token_offsets = torch.arange(tokens, dtype=torch.int32)
    attn.attention = lambda q, wi, wl, **kw: events.append("attention") or q
    attn.output_projection = lambda att, **kw: events.append("output_projection") or att

    page_table = torch.arange(num_requests * blocks, dtype=torch.int32).view(num_requests, blocks)
    token_page_table = torch.arange(tokens * blocks, dtype=torch.int32).view(tokens, blocks)
    metadata = SimpleNamespace(
        starts=torch.zeros(num_requests, dtype=torch.int64),
        state_rows=torch.arange(num_requests, dtype=torch.int64),
        page_table=page_table,
        token_page_table=token_page_table,
        requests=torch.tensor([0, 1, -1, 99], dtype=torch.int64),
    )
    geometry = SimpleNamespace(
        lengths=torch.full((tokens,), 640, dtype=torch.int32),
        active_width=torch.tensor([640], dtype=torch.int32),
        index_lengths=torch.full((tokens,), 512, dtype=torch.int32),
    )
    indices = SimpleNamespace(
        position=torch.arange(tokens, dtype=torch.int64),
        angles={(True, 0): torch.zeros((tokens, 64)), (True, 2): torch.zeros((tokens, 64))},
        window_slots=torch.arange(tokens, dtype=torch.int64),
        window_indices=torch.zeros((tokens, 4), dtype=torch.int32),
        window_lengths=torch.ones(tokens, dtype=torch.int32),
        compressed=lambda ratio: geometry,
    )

    attn._run(torch.zeros((tokens, 16), dtype=torch.bfloat16), metadata, (None, None, None), indices=indices)
    assert events == [
        "start",
        "window_writer",
        "global_kv",
        "finish_indexer",
        "selector",
        "finish",
        "attention",
        "output_projection",
    ]
    assert len(selector_calls) == 1
    assert selector_calls[0]["pages"] is page_table
    req_ids = selector_calls[0]["request_ids"]
    assert req_ids.dtype == torch.int32 and req_ids.is_contiguous()
    assert req_ids.tolist() == [0, 1, 0, 1]
    assert selector_calls[0]["quantized"] is not None

    # Disabling use_request_page_table falls back to token_page_table and request_ids=None.
    selector_calls.clear()
    events.clear()
    attn.use_request_page_table = False
    attn._run(torch.zeros((tokens, 16), dtype=torch.bfloat16), metadata, (None, None, None), indices=indices)
    torch.testing.assert_close(selector_calls[0]["pages"], token_page_table, rtol=0, atol=0)
    assert selector_calls[0]["request_ids"] is None


@pytest.mark.parametrize("preencoded", [False, True])
def test_concurrent_attention_aux_runs_indexer_before_qb_and_splits_joins(monkeypatch, preencoded):
    events: list[str] = []

    class _FakeEvent:
        def record(self, stream):
            events.append(f"event.record:{stream.name}")

    class _FakeStream:
        def __init__(self, name: str):
            self.name = name

        def wait_stream(self, other):
            events.append(f"{self.name}.wait_stream:{other.name}")

        def wait_event(self, event):
            events.append(f"{self.name}.wait_event")

    main_stream = _FakeStream("main")
    aux_stream = _FakeStream("aux")

    @contextmanager
    def _stream_ctx(s):
        events.append(f"enter:{s.name}")
        yield
        events.append(f"exit:{s.name}")

    monkeypatch.setattr(torch.cuda, "ExternalStream", lambda handle, device=None: aux_stream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: main_stream)
    monkeypatch.setattr(torch.cuda, "stream", _stream_ctx)
    monkeypatch.setattr(
        projection_overlap,
        "_run_mxfp8_with_rope",
        lambda *args, **kwargs: events.append(f"mxfp8_rope:plan={args[8]}"),
    )
    monkeypatch.setattr(
        projection_overlap,
        "_run_encoded_with_rope",
        lambda *args, **kwargs: events.append(f"encoded_rope:plan={args[6]}"),
    )

    from hyperflare_recipes.components import sm103_projection
    from b12x.attention import dsa_indexer as dsa
    from b12x.preparation import types as b12x_types

    monkeypatch.setattr(
        sm103_projection,
        "bf16_out",
        lambda src, w, out, plan: (out.fill_(2.0), events.append(f"bf16_out:plan={plan}")),
    )
    monkeypatch.setattr(b12x_types, "plan_from_handle", lambda handle: f"sel-{handle}")

    iw_seen_at_quantize: list[float] = []

    def _fake_quantize(plan, q_slice, *, q_mxfp4, q_scales):
        events.append(f"quantize:{plan}")
        iw_seen_at_quantize.append(float(iw_out[0, 0].item()))

    monkeypatch.setattr(dsa, "quantize_q_mxfp4", _fake_quantize)

    class _Owner:
        def __init__(self, event):
            self.index_ready_event = event

    owner = _Owner(_FakeEvent())
    handle = 999_001
    projection_overlap._AUX_INDEXED_OWNERS[handle] = owner
    try:
        rows = 4
        query_low = torch.zeros((rows, 128), dtype=torch.bfloat16)
        source = torch.zeros((rows, 128), dtype=torch.bfloat16)
        layer_angles = torch.zeros((rows, 64), dtype=torch.float32)
        index_angles = torch.zeros((rows, 64), dtype=torch.float32)
        w = torch.zeros((128, 128), dtype=torch.uint8)
        s = torch.zeros((128, 4), dtype=torch.uint8)
        q_out = torch.zeros((rows, 512), dtype=torch.bfloat16)
        iq_out = torch.zeros((rows, 32 * 128), dtype=torch.bfloat16)
        iw_out = torch.zeros((rows, 32), dtype=torch.float32)
        packed_list = [torch.zeros((rows, 32, 64), dtype=torch.uint8)]
        scales_list = [torch.zeros((rows, 32, 4), dtype=torch.uint8)]
        workspace = torch.zeros(64, dtype=torch.uint8)
        q_vb = torch.zeros((rows, 128), dtype=torch.float8_e4m3fn) if preencoded else None
        q_mb = torch.zeros((rows, 4), dtype=torch.uint8) if preencoded else None
        iw_scale = float(128 ** -0.5 * 32 ** -0.5)

        projection_overlap._start_attn_aux_indexed(
            query_low, source, layer_angles, index_angles,
            w, s, None, None, 0, 101, q_vb, q_mb,
            w, s, None, None, 0, 202,
            w, 303,
            q_out, iq_out, iw_out, packed_list, scales_list,
            [404], [0], [rows], workspace,
            512, 128, 64, iw_scale, 777, handle,
        )
        expected_q = "encoded_rope:plan=101" if preencoded else "mxfp8_rope:plan=101"
        assert events == [
            "aux.wait_stream:main",
            "enter:aux",
            "mxfp8_rope:plan=202",
            "bf16_out:plan=303",
            "quantize:sel-404",
            "event.record:aux",
            expected_q,
            "exit:aux",
        ]
        assert iw_seen_at_quantize == [pytest.approx(2.0 * iw_scale)]

        events.clear()
        dep = torch.zeros(1)
        projection_overlap._join_attn_aux_indexer(
            query_low, source, index_angles, dep, dep,
            iq_out, iw_out, packed_list, scales_list, 777, handle,
        )
        assert events == ["main.wait_event"]

        events.clear()
        projection_overlap._join_attn_aux_indexed(
            query_low, layer_angles, dep, q_vb, None, q_mb, q_out, workspace, 777,
        )
        assert events == ["main.wait_stream:aux"]

        # Without an event (CPU/non-CUDA fallback), finish_indexer joins the auxiliary stream.
        events.clear()
        owner.index_ready_event = None
        projection_overlap._join_attn_aux_indexer(
            query_low, source, index_angles, dep, dep,
            iq_out, iw_out, packed_list, scales_list, 777, handle,
        )
        assert events == ["main.wait_stream:aux"]
    finally:
        projection_overlap._AUX_INDEXED_OWNERS.pop(handle, None)


def test_concurrent_attention_aux_registers_selector_qb_parallel_scratch(monkeypatch):
    registered_pairs: list[tuple] = []

    class _FakeArena(ScratchArena):
        def __init__(self):
            self._stream = SimpleNamespace(cuda_stream=888)

        @property
        def auxiliary_stream(self):
            return self._stream

        def register_parallel(self, *plans):
            registered_pairs.append(plans)

    monkeypatch.setattr(projection_overlap, "register_finalizer", lambda *args, **kwargs: None)
    arena = _FakeArena()
    q_plan, iq_plan, iw_plan, main_plan, sel_plan = ("q", "iq", "iw", "main", "sel")
    q_b = SimpleNamespace(
        bias=None,
        encoded=None,
        quantizer_plan=None,
        weight=_fake_mxfp8_linear_weight(128, 128),
        activation_scale=None,
        plan=q_plan,
        output=torch.zeros((4, 128), dtype=torch.bfloat16),
    )
    index_query = SimpleNamespace(
        bias=None,
        encoded=None,
        weight=_fake_mxfp8_linear_weight(128, 128),
        activation_scale=None,
        plan=iq_plan,
        output=torch.zeros((4, 128), dtype=torch.bfloat16),
    )
    index_weight = SimpleNamespace(
        bias=None,
        weight=torch.zeros((32, 128), dtype=torch.bfloat16),
        plan=iw_plan,
        output=torch.zeros((4, 32), dtype=torch.float32),
    )
    selector = SimpleNamespace(plan=sel_plan)

    aux = projection_overlap.ConcurrentAttentionAux(
        q_b,
        parallel_plans=(main_plan,),
        session=object(),
        workspace=arena,
        index_query=index_query,
        index_weight=index_weight,
        selectors=(selector,),
        selector_rows=2048,
    )
    assert (sel_plan, q_plan) in registered_pairs
    assert sel_plan in aux.parallel_plans
    assert aux.handle in projection_overlap._AUX_INDEXED_OWNERS
