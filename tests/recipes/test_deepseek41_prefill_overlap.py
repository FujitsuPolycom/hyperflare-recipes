# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU parity and schedule tests for DeepSeek V4.1 prefill collective overlap, fused packing, KV RoPE, and PDL."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from b12x.comm import pcie
from hyperflare_recipes.components import linear as comp_linear, mla_norm_quant
from hyperflare_recipes.components.moe import RouteRecord, RouterScratch
from hyperflare_recipes.components.paged_batch import Metadata
from hyperflare_recipes.components.parallel import TensorParallelShard
from hyperflare_recipes.recipes.deepseek41 import attention as ds_attention, decoder, engram as ds_engram
from ignition.preparation.scratch import ScratchArena


class _FakeDmaRuntime:
    def __init__(self, world_size: int = 4, *, max_bytes: int = 256 << 20, shard_capacity: int = 64 << 20,
                 allow_all_to_all: bool = True):
        self.world_size = world_size
        self.max_bytes = max_bytes
        self.shard_capacity = shard_capacity
        self._allow_all_to_all = allow_all_to_all

    def should_allreduce(self, tensor):
        return True

    def should_reduce_scatter(self, tensor):
        return True

    def should_all_gather(self, tensor):
        return True

    def should_all_to_all(self, tensor):
        return (
            self._allow_all_to_all
            and tensor.ndim >= 2
            and tensor.shape[0] == self.world_size
            and tensor.is_contiguous()
            and tensor.numel() * tensor.element_size() <= self.max_bytes
            and (tensor.numel() // self.world_size) * tensor.element_size() <= self.shard_capacity
        )


@pytest.mark.parametrize(
    ("tokens", "min_chunks"),
    [(512, 1), (512, 2), (8192, 1), (8192, 2), (8192, 4), (16384, 2), (16384, 4)],
)
def test_owner_row_engram_all_to_all_exact_parity_across_chunks_and_world(monkeypatch, tokens, min_chunks):
    world = 4
    half = tokens // 2
    shard_rows = half // world
    local_cols = 6144 // world
    pool = SimpleNamespace(config=SimpleNamespace(text={"rms_norm_eps": 1e-6}))

    class FakeAllToAll:
        @staticmethod
        def fits(runtime, source, output):
            return (
                isinstance(runtime, _FakeDmaRuntime)
                and runtime.should_all_to_all(source)
                and tuple(source.shape) == tuple(output.shape)
                and source.is_contiguous()
                and output.is_contiguous()
            )

        def __init__(self, source, output, *, runtime, session, coordinator, name):
            self.source_shape = tuple(source.shape)
            self.output_shape = tuple(output.shape)
            self.plan = SimpleNamespace(tag=name)
            self.single = False

        def functional(self, *, session):
            return self

    class FakeAllGather:
        def __init__(self, source, output, *, runtime, session, coordinator, name):
            self.source_shape = tuple(source.shape)
            self.output_shape = tuple(output.shape)
            self.plan = SimpleNamespace(tag=name)
            self.single = False

        def functional(self, *, session):
            return self

    class IdentityProj:
        def __init__(self, *args, **kwargs):
            self.plan = SimpleNamespace()

        def functional(self, *, session):
            return self

        def __call__(self, x):
            return x

    monkeypatch.setattr(pcie, "DmaAllReduce", _FakeDmaRuntime)
    monkeypatch.setattr(ds_engram, "PreparedEngramLookup", lambda *a, **k: SimpleNamespace(functional=lambda **kw: None))
    monkeypatch.setattr(ds_engram, "PreparedAllToAll", FakeAllToAll)
    monkeypatch.setattr(ds_engram, "PreparedAllGather", FakeAllGather)
    monkeypatch.setattr(ds_engram, "QuantizedProjection", IdentityProj)
    monkeypatch.setattr(ds_engram.hyperconnection, "plan", lambda *a, **k: SimpleNamespace(request=lambda **kw: None))
    monkeypatch.setattr(ds_engram, "declare", lambda *a, **k: None)

    weights = {"projection_full": "pf", "projection": "ps", "weight": torch.ones(5120, dtype=torch.bfloat16)}
    rt_a2a = _FakeDmaRuntime(world, allow_all_to_all=True)
    rt_ag = _FakeDmaRuntime(world, allow_all_to_all=False)

    # Distinct per-(rank, token, col) BF16 payloads across all 4 ranks.
    rank_locals = []
    for r in range(world):
        row_ids = torch.arange(tokens, dtype=torch.int32)[:, None]
        col_ids = torch.arange(local_cols, dtype=torch.int32)[None, :]
        bits = ((r * 16381 + row_ids * 131 + col_ids * 7) & 0x7FFF).to(torch.uint16)
        rank_locals.append(bits.view(torch.bfloat16).contiguous())

    def build_engram(rank, rt):
        tables = SimpleNamespace(
            columns=torch.arange(local_cols),
            shard=TensorParallelShard(rank, world),
            inputs={1: object()},
            token_map=(0, 1),
            geometry=object(),
            readiness_runtime=None,
        )
        meta = Metadata.allocate(tokens, 1, 10, "cpu")
        scr = ds_engram.EngramScratch.allocate(tokens, "cpu", world=world)
        src = torch.zeros((tokens, 4, 5120), dtype=torch.bfloat16)
        out = torch.zeros((tokens, 4, 5120), dtype=torch.bfloat16)
        return ds_engram.PreparedEngram(
            1, weights, tables, src, out, meta, pool,
            scratch=scr, session=None, name=f"eng_r{rank}", workspace=None,
            runtime=rt, row_shard=(rank, half, shard_rows), min_gather_chunks=min_chunks,
        )

    a2a_engrams = [build_engram(r, rt_a2a) for r in range(world)]
    ag_engrams = [build_engram(r, rt_ag) for r in range(world)]

    assert all(e.use_all_to_all is True for e in a2a_engrams)
    assert all(e.use_all_to_all is False for e in ag_engrams)
    assert len(a2a_engrams[0].gather_bounds) == min_chunks

    # Simulate exact 4-rank AllGather + index_select baseline vs 4-rank AllToAll.
    ag_bytes_per_rank = 0
    ag_chunks_per_rank = [[] for _ in range(world)]
    for b, e in ag_engrams[0].gather_bounds:
        stacked = torch.stack([rank_locals[r][b:e] for r in range(world)], dim=0)
        ag_bytes_per_rank += (world - 1) * rank_locals[0][b:e].nbytes
        for r in range(world):
            ag_chunks_per_rank[r].append(stacked)

    a2a_bytes_per_rank = 0
    a2a_chunks_per_rank = [[] for _ in range(world)]
    for b, e in a2a_engrams[0].gather_bounds:
        sent_by_rank = [a2a_engrams[r].slice_exchange_chunk(rank_locals[r], b, e) for r in range(world)]
        assert all(t.shape == (world, e - b, local_cols) and t.is_contiguous() for t in sent_by_rank)
        a2a_bytes_per_rank += (world - 1) * sent_by_rank[0][0].nbytes
        for dst in range(world):
            received = torch.stack([sent_by_rank[src][dst] for src in range(world)], dim=0)
            a2a_chunks_per_rank[dst].append(received)

    # Exact 4x wire traffic reduction at TP4.
    assert a2a_bytes_per_rank * world == ag_bytes_per_rank

    for r in range(world):
        ref = ag_engrams[r].project_gathered(ag_chunks_per_rank[r])
        got = a2a_engrams[r].project_gathered(a2a_chunks_per_rank[r])
        assert got.shape == (2 * shard_rows, 6144)
        assert torch.equal(got.view(torch.uint16), ref.view(torch.uint16))

    # Sensitivity check: flipping 1 bit in rank 2's row for owner rank 1 half B row 3 alters ONLY that exact slot.
    if tokens == 512 and min_chunks == 2:
        b, e = a2a_engrams[0].gather_bounds[1]
        sent = [a2a_engrams[r].slice_exchange_chunk(rank_locals[r], b, e) for r in range(world)]
        mutated = sent[2].clone()
        mutated.view(torch.uint16)[1, 3, 17] ^= 1
        sent[2] = mutated
        recv_r1 = [a2a_chunks_per_rank[1][0], torch.stack([sent[src][1] for src in range(world)], dim=0)]
        got_mut = a2a_engrams[1].project_gathered(recv_r1)
        ref_r1 = ag_engrams[1].project_gathered(ag_chunks_per_rank[1])
        diff = (got_mut.view(torch.uint16) != ref_r1.view(torch.uint16)).nonzero(as_tuple=False).tolist()
        assert diff == [[shard_rows + 3, 2 * local_cols + 17]]

    # Backwards-compatibility: absent PreparedAllToAll falls back cleanly to PreparedAllGather.
    monkeypatch.setattr(ds_engram, "PreparedAllToAll", None)
    fb = build_engram(0, rt_a2a)
    assert fb.use_all_to_all is False
    assert all(isinstance(p, FakeAllGather) for p in fb.gather_parts)


def test_engram_mix_half_exact_parity_with_full_mix(monkeypatch):
    tokens, world = 1024, 4
    half = tokens // 2
    st = half // world
    rank = 2

    def fake_run_engram_mix(state, projected_kv, norm_weights, *, eps, plan, out, token_mask):
        # Deterministic row-independent transformation matching per-row masking semantics.
        scale = projected_kv[:, :state.shape[1]].float() * norm_weights.float()[None, :state.shape[1]]
        updated = (state.float() + scale * 0.125).to(torch.bfloat16)
        out.copy_(torch.where(token_mask[:, None], updated, state))

    monkeypatch.setattr(ds_engram.hyperconnection, "run_engram_mix", fake_run_engram_mix)

    eng = ds_engram.PreparedEngram.__new__(ds_engram.PreparedEngram)
    s0 = rank * st
    eng.row_shard_indices = torch.cat((
        torch.arange(s0, s0 + st, dtype=torch.int64),
        torch.arange(half + s0, half + s0 + st, dtype=torch.int64),
    ))
    eng.norm_weight = torch.linspace(0.5, 1.5, 4 * 64, dtype=torch.float32)
    eng.eps = 1e-6
    eng.mix_plan = SimpleNamespace()

    meta = Metadata.allocate(tokens, 1, 10, "cpu")
    meta.positions.copy_(torch.tensor([i if i < 700 else -1 for i in range(tokens)], dtype=torch.int64))
    meta.token_ids.copy_(torch.tensor([130000 if (i % 19 == 0) else (i % 1000) for i in range(tokens)], dtype=torch.int64))

    gen = torch.Generator().manual_seed(7)
    out_a = torch.randn((st, 4, 64), dtype=torch.bfloat16, generator=gen)
    out_b = torch.randn((st, 4, 64), dtype=torch.bfloat16, generator=gen)
    proj = torch.randn((2 * st, 5 * 64), dtype=torch.bfloat16, generator=gen)

    full_mixed = eng.mix(torch.cat((out_a, out_b), dim=0), proj, meta)
    half_a = eng.mix_half(out_a, proj[:st], meta, 0)
    half_b = eng.mix_half(out_b, proj[st:], meta, 1)

    assert torch.equal(half_a.view(torch.uint16), full_mixed[:st].view(torch.uint16))
    assert torch.equal(half_b.view(torch.uint16), full_mixed[st:].view(torch.uint16))


@pytest.mark.parametrize("st", [128, 1024])
def test_fused_ffn_packed_gather_pack_unpack_and_swizzle_exact_parity(st):
    world, hidden, top_k = 4, 5120, 6
    half = st * world
    scale_cols_u8 = hidden // 32
    vals_bf16_cols = hidden // 2
    scale_bf16_cols = scale_cols_u8 // 2
    route_bf16_cols = 4 * top_k
    expected_packed_cols = vals_bf16_cols + scale_bf16_cols + route_bf16_cols
    assert expected_packed_cols == 2664
    assert (expected_packed_cols * 2) % 16 == 0

    gen = torch.Generator().manual_seed(123 + st)
    shard_vals, shard_srows, shard_ids, shard_weights = [], [], [], []
    packed_shards = []
    split_vals_shards, split_meta_shards = [], []
    for r in range(world):
        v = ((torch.randint(1, 240, (st, hidden), generator=gen, dtype=torch.int32)) & 0xFF).to(torch.uint8).view(torch.float8_e4m3fn)
        sr = torch.randint(1, 254, (1, st, scale_cols_u8), generator=gen, dtype=torch.int32).to(torch.uint8)
        ids = torch.randint(0, 384, (st, top_k), generator=gen, dtype=torch.int32)
        w = torch.randn((st, top_k), generator=gen, dtype=torch.float32)
        shard_vals.append(v)
        shard_srows.append(sr)
        shard_ids.append(ids)
        shard_weights.append(w)

        packed = decoder._pack_ffn_shard(v, sr, ids, w)
        assert packed.shape == (st, expected_packed_cols) and packed.dtype == torch.bfloat16 and packed.is_contiguous()
        packed_shards.append(packed)

        rr = RouteRecord.empty(st, top_k, "cpu")
        rr.write(ids, w)
        split_vals_shards.append(v.view(torch.bfloat16))
        split_meta_shards.append(torch.cat((sr.view(st, -1).view(torch.bfloat16), rr.packed), dim=1))

    # Simulate rank-major GatherShards across all 4 ranks.
    packed_full = torch.cat(packed_shards, dim=0)
    vals_full_ref = torch.cat(split_vals_shards, dim=0).view(torch.float8_e4m3fn)
    meta_full_ref = torch.cat(split_meta_shards, dim=0)
    srows_ref_2d = meta_full_ref[:, :scale_bf16_cols].contiguous().view(torch.uint8)
    srows_ref = srows_ref_2d.view(1, half, scale_cols_u8)
    smma_ref = decoder._mxfp8_scale_mma(srows_ref_2d, hidden)
    route_words_ref = meta_full_ref[:, scale_bf16_cols:].contiguous().view(torch.int32)
    ids_ref = torch.empty((half, top_k), dtype=torch.int32)
    weights_ref = torch.empty((half, top_k), dtype=torch.float32)
    RouteRecord(route_words_ref).read(ids_ref, weights_ref)

    got_vals, got_srows, got_smma, got_ids, got_weights = decoder._unpack_ffn_gathered(
        packed_full, hidden=hidden, top_k=top_k
    )
    assert torch.equal(got_vals.view(torch.uint8), vals_full_ref.view(torch.uint8))
    assert torch.equal(got_srows, srows_ref)
    assert torch.equal(got_smma, smma_ref)
    assert torch.equal(got_ids, ids_ref)
    assert torch.equal(got_weights.view(torch.int32), weights_ref.view(torch.int32))

    # Sensitivity checks: flipping 1 bit in a scale byte and 1 bit in a route weight.
    mut = packed_full.clone()
    mut_u8 = mut[:, vals_bf16_cols:vals_bf16_cols + scale_bf16_cols].contiguous().view(torch.uint8)
    mut_u8[165, 5] ^= 0x01
    mut[:, vals_bf16_cols:vals_bf16_cols + scale_bf16_cols] = mut_u8.view(torch.bfloat16)
    _, _, mut_smma, _, _ = decoder._unpack_ffn_gathered(mut, hidden=hidden, top_k=top_k)
    diff_mask = mut_smma != got_smma
    assert int(diff_mask.sum().item()) == 1
    assert bool(diff_mask[0, 1, 1, 5, 1, 1].item())

    mut2 = packed_full.clone()
    mut2_i32 = mut2[:, vals_bf16_cols + scale_bf16_cols:].contiguous().view(torch.int32)
    mut2_i32[17, top_k + 3] ^= 1
    mut2[:, vals_bf16_cols + scale_bf16_cols:] = mut2_i32.view(torch.bfloat16)
    _, _, _, _, mut_w = decoder._unpack_ffn_gathered(mut2, hidden=hidden, top_k=top_k)
    w_diff = (mut_w.view(torch.int32) != got_weights.view(torch.int32)).nonzero(as_tuple=False).tolist()
    assert w_diff == [[17, 3]]


@pytest.mark.parametrize("kv_cols", [512, 1024, 2560])
@pytest.mark.parametrize("st", [128, 1024])
def test_mla_wo_p2p_kv_gather_transpose_parity(kv_cols, st):
    world = 4
    half = st * world
    factor = kv_cols // 512

    layer = decoder.PreparedLayer.__new__(decoder.PreparedLayer)
    layer.world_size = world
    layer.half_tokens = half

    shards = []
    for r in range(world):
        rows = torch.arange(st, dtype=torch.int32)[:, None]
        cols = torch.arange(kv_cols, dtype=torch.int32)[None, :]
        bits = ((r * 8191 + rows * 97 + cols * 3) & 0x7FFF).to(torch.uint16)
        shards.append(bits.view(torch.bfloat16).contiguous())

    # Simulate DcpAllToAll.all_gather_heads on [st * factor, 512] -> [st * factor, 512 * world].
    reshaped = [s.view(st * factor, 512) for s in shards]
    col_interleaved = torch.cat(reshaped, dim=1).contiguous()
    assert col_interleaved.shape == (st * factor, 512 * world)

    seen_shapes = []

    class FakeGather:
        single = False

        def __call__(self, send, *, dependencies=()):
            seen_shapes.append(tuple(send.shape))
            return col_interleaved

    got = layer._gather_kv_shard(FakeGather(), shards[0])
    expected = torch.cat(shards, dim=0)
    assert seen_shapes == [(st * factor, 512)]
    assert got.shape == (half, kv_cols) and got.is_contiguous()
    assert torch.equal(got.view(torch.uint16), expected.view(torch.uint16))

    # Sensitivity: flip 1 bit in rank 3, shard row 9, col (kv_cols - 13).
    target_col = kv_cols - 13
    k_blk, in_blk = divmod(target_col, 512)
    bad = col_interleaved.clone()
    bad.view(torch.uint16)[9 * factor + k_blk, 3 * 512 + in_blk] ^= 1
    bad_got = layer._gather_kv_shard(
        type("G", (), {"single": False, "__call__": lambda self, x, **kw: bad})(), shards[0]
    )
    diff = (bad_got.view(torch.uint16) != expected.view(torch.uint16)).nonzero(as_tuple=False).tolist()
    assert diff == [[3 * st + 9, target_col]]


@pytest.mark.parametrize("pipelined_attn", ["full", "mla_wo"])
@pytest.mark.parametrize("ffn_packed", [False, True])
def test_defer_mlp_b_across_engram_transitions_and_mla_wo_p2p_schedule(pipelined_attn, ffn_packed):
    tokens, world, hidden, top_k = 1024, 4, 256, 6
    half = tokens // 2
    st = half // world
    kv_cols = 512 if pipelined_attn == "full" else 2560
    events = []
    copy_input_seen = []

    class LoggedCollective:
        def __init__(self, tag, fn):
            self.tag = tag
            self.fn = fn
            self.seen_shapes = []

        def start(self, source, *, dependencies=(), copy_input=True):
            self.seen_shapes.append(tuple(source.shape))
            copy_input_seen.append((self.tag, copy_input))
            events.append(f"{self.tag}.start")
            return SimpleNamespace(source=source, output=self.fn(source), dependencies=dependencies)

        def finish(self, pending, dependency):
            events.append(f"{self.tag}.finish")
            return pending.output

    class LoggedSyncGather:
        single = False

        def __init__(self, tag):
            self.tag = tag
            self.seen_shapes = []

        def __call__(self, source, *, dependencies=()):
            self.seen_shapes.append(tuple(source.shape))
            events.append(self.tag)
            return source.repeat(1, world)

    def make_base_layer(*, is_prev: bool):
        layer = decoder.PreparedLayer.__new__(decoder.PreparedLayer)
        layer.row_shard_dma = True
        layer.row_shard_attention = True
        layer.pipelined_attn_hc = not is_prev
        layer.pipelined_attn = "full" if is_prev else pipelined_attn
        layer.p2p_kv_gather = not is_prev
        layer.world_size = world
        layer.defer_mlp_b = is_prev
        layer.half_tokens = half
        layer.shard_tokens = st
        layer.shard_rows_a = torch.arange(0, st, dtype=torch.int64)
        layer.shard_rows_b = torch.arange(half, half + st, dtype=torch.int64)
        layer.shard_offset_a = torch.tensor([0], dtype=torch.int32)
        layer.shard_offset_b = torch.tensor([half], dtype=torch.int32)
        layer.has_image_mask = False
        layer.attn_quantizer_plan = None
        layer.ffn_quantizer_plan = None
        layer.ffn_local_shapes = None
        layer.ffn_meta_scale_cols = 0
        layer.ffn_packed_gather = ffn_packed
        layer.ffn_vals_bf16_cols = hidden // 2
        layer.ffn_top_k = top_k
        layer.attn_gather_0 = layer.attn_gather_1 = None
        layer.cr_attn_gather_0 = layer.cr_attn_gather_1 = None
        layer.att_reduce_0 = layer.att_reduce_1 = None
        layer.cr_att_0 = layer.cr_att_1 = None
        layer.engram = None
        layer.incoming_post = None

        prefix = "L0" if is_prev else "L1"
        hc_idx = [0]

        def fake_attn_hc(*args, **kwargs):
            tag = "A" if hc_idx[0] == 0 else "B"
            hc_idx[0] += 1
            events.append(f"{prefix}.attn_hc_{tag}")
            return (
                torch.zeros((st, 4, hidden), dtype=torch.bfloat16),
                torch.zeros((st, 4), dtype=torch.float32),
                torch.zeros((st, 4, 4), dtype=torch.float32),
                torch.ones((st, hidden), dtype=torch.bfloat16),
            )

        layer.attention_hc = fake_attn_hc
        if layer.pipelined_attn == "full":
            layer.attention = SimpleNamespace(
                mode="prefill",
                init_window_cache=lambda m, d: "wc",
                run_shard_qkv_q=lambda norm, **kw: (
                    events.append(f"{prefix}.qkv_q") or (
                        torch.ones((st, 64, 512), dtype=torch.bfloat16),
                        torch.ones((st, 512), dtype=torch.bfloat16),
                        torch.ones((st, 64), dtype=torch.float32),
                    )
                ),
                run_shard_mla_wo=lambda q, wv, m, sel, **kw: (
                    events.append(f"{prefix}.mla_wo_{'A' if kw['half_idx'] == 0 else 'B'}")
                    or torch.ones((st, hidden), dtype=torch.bfloat16)
                ),
                commit=lambda lengths, r, m: events.append(f"{prefix}.commit"),
            )
        else:
            layer.attention = SimpleNamespace(
                mode="prefill",
                global_kv=SimpleNamespace(compressor=SimpleNamespace(init_pending=lambda m, d: "pending")),
                init_window_cache=lambda m, d: "wc",
                run_shard_mla_wo_local=lambda norm, **kw: (
                    events.append(f"{prefix}.mla_wo_local") or (
                        torch.ones((st, 64, 512), dtype=torch.bfloat16),
                        torch.ones((st, kv_cols), dtype=torch.bfloat16),
                        torch.ones((st, 32, 128), dtype=torch.bfloat16),
                        torch.ones((st, 32), dtype=torch.bfloat16),
                        torch.ones((st, 64), dtype=torch.float32),
                    )
                ),
                run_shard_mla_wo_source=lambda q, kv, iq, iw, m, sel, **kw: (
                    assert_shape(kv, (half, kv_cols))
                    or events.append(f"{prefix}.mla_wo_source_{'A' if kw['half_idx'] == 0 else 'B'}")
                    or (torch.ones((st, hidden), dtype=torch.bfloat16), torch.zeros((st, 512), dtype=torch.int32))
                ),
                commit=lambda lengths, r, m: events.append(f"{prefix}.commit"),
            )

        if layer.p2p_kv_gather:
            layer.kv_gather_0 = LoggedSyncGather(f"{prefix}.p2p_kv_0")
            layer.kv_gather_1 = LoggedSyncGather(f"{prefix}.p2p_kv_1")
        else:
            layer.cr_kv_gather_0 = LoggedCollective(f"{prefix}.dma_kv_0", lambda sh: sh.repeat(world, 1))
            layer.cr_kv_gather_1 = LoggedCollective(f"{prefix}.dma_kv_1", lambda sh: sh.repeat(world, 1))

        layer.ffn_hc = lambda att, r, p, c, *, pre_mix, pre_out: (events.append(f"{prefix}.ffn_hc") or (r, p, c, att))
        if ffn_packed:
            packed_cols = hidden // 2 + hidden // 64 + 4 * top_k
            layer._prepare_ffn_shard = lambda norm, mask, valid: (
                events.append(f"{prefix}.ffn_prep") or torch.ones((st, packed_cols), dtype=torch.bfloat16)
            )
            layer._run_ffn_gathered = lambda packed, meta=None: (
                events.append(f"{prefix}.ffn_compute") or (
                    torch.ones((half, hidden), dtype=torch.bfloat16),
                    torch.zeros((half, top_k), dtype=torch.int32),
                )
            )
            layer.cr_ffn_packed_0 = LoggedCollective(f"{prefix}.ffn_packed_0", lambda sh: sh.repeat(world, 1))
            layer.cr_ffn_packed_1 = LoggedCollective(f"{prefix}.ffn_packed_1", lambda sh: sh.repeat(world, 1))
        else:
            layer.router_projection = lambda norm: torch.zeros((norm.shape[0], 384), dtype=torch.float32)
            layer.router = lambda logits, **kw: (
                events.append(f"{prefix}.ffn_prep") or RouterScratch(
                    torch.zeros((logits.shape[0], top_k), dtype=torch.float32),
                    torch.zeros((logits.shape[0], top_k), dtype=torch.int32),
                    torch.ones((logits.shape[0], top_k), dtype=torch.float32),
                )
            )
            layer.cr_ffn_vals_0 = LoggedCollective(f"{prefix}.ffn_vals_0", lambda sh: sh.repeat(world, 1))
            layer.cr_ffn_meta_0 = LoggedCollective(f"{prefix}.ffn_meta_0", lambda sh: sh.repeat(world, 1))
            layer.cr_ffn_vals_1 = LoggedCollective(f"{prefix}.ffn_vals_1", lambda sh: sh.repeat(world, 1))
            layer.cr_ffn_meta_1 = LoggedCollective(f"{prefix}.ffn_meta_1", lambda sh: sh.repeat(world, 1))
            layer.experts = lambda vals, ids, weights, **kw: (events.append(f"{prefix}.ffn_compute") or vals)
            layer.shared = lambda vals, **kw: vals

        layer.cr_mlp_0 = LoggedCollective(f"{prefix}.mlp_rs_0", lambda full: full[:st].clone())
        layer.cr_mlp_1 = LoggedCollective(f"{prefix}.mlp_rs_1", lambda full: full[:st].clone())
        layer.post = (lambda mlp_sh, res, post, comb: (events.append(f"{prefix}.post") or res)) if is_prev else None
        return layer

    def assert_shape(tensor, expected):
        assert tuple(tensor.shape) == expected
        return False

    l0 = make_base_layer(is_prev=True)
    l1 = make_base_layer(is_prev=False)
    l1.incoming_post = l0.post
    l1.cr_mlp_1 = LoggedCollective("L1.mlp_rs_1", lambda full: full[:st].clone())

    # L1's Engram: overlapped during L0's FFN windows, projected on L0, mixed per-half on L1!
    engram_state = {"partial": None, "chunks": []}
    partial_buf = torch.ones((tokens, 1536), dtype=torch.bfloat16)
    seen_eng_ptrs = []

    class LoggedEngramCollective(LoggedCollective):
        def start(self, source, *, dependencies=(), copy_input=True):
            seen_eng_ptrs.append(source.data_ptr())
            return super().start(source, dependencies=dependencies, copy_input=copy_input)

    fake_engram = SimpleNamespace(
        lookup_local=lambda src, m: (events.append("engram.lookup") or partial_buf),
        slice_exchange_chunk=lambda p, b, e: p[b:e],
        project_gathered=lambda chunks: (events.append("engram.project") or torch.ones((2 * st, 25600), dtype=torch.bfloat16)),
        mix_half=lambda src, proj, m, h_idx: (events.append(f"engram.mix_half_{'A' if h_idx == 0 else 'B'}") or src),
    )
    l1.engram = fake_engram
    cr_eng_0 = LoggedEngramCollective("engram_0", lambda sh: sh.unsqueeze(0).expand(world, -1, -1).contiguous())
    cr_eng_1 = LoggedEngramCollective("engram_1", lambda sh: sh.unsqueeze(0).expand(world, -1, -1).contiguous())
    ffn_overlaps = (
        (fake_engram, cr_eng_0, (0, half), engram_state, True),
        (None, cr_eng_1, (half, tokens), engram_state, True),
    )

    metadata = Metadata.allocate(tokens, 1, 10, "cpu")
    metadata.live_counts[0] = tokens
    metadata.lengths[0] = tokens
    pm = (torch.zeros((st, 4), dtype=torch.float32), torch.zeros((st, 4), dtype=torch.float32))
    init_in = (torch.zeros((st, 4, hidden), dtype=torch.bfloat16), torch.zeros((st, 4, hidden), dtype=torch.bfloat16))

    # Execute L0 -> L1 transition!
    l0_out, l0_mix, _, sel = l0._run_row_shard_dma(
        init_in, pm, metadata, "indices", ((None, None), None, None),
        ffn_engram_overlaps=ffn_overlaps, engram_project=(fake_engram, engram_state),
    )
    assert len(l0_out) == 5  # (out_a, p_mlp_b, res_b, post_b, comb_b)
    assert "L0.mlp_rs_1.finish" not in events
    assert "projected" in engram_state
    # Verify zero-copy slicing of Engram partial buffer (no .clone()!).
    assert seen_eng_ptrs[0] == partial_buf[0:half].data_ptr()
    assert seen_eng_ptrs[1] == partial_buf[half:tokens].data_ptr()

    # Now execute L1 consuming L0's in-flight p_mlp_b!
    l1.cr_mlp_1 = l0.cr_mlp_1
    l1._run_row_shard_dma(
        l0_out, l0_mix, metadata, "indices", sel,
        engram_projected=engram_state["projected"],
    )

    # Verify zero DMA stream overlap conflict and full cross-layer p_mlp_b hiding!
    first_mla_a = "L1.mla_wo_A" if pipelined_attn == "full" else "L1.mla_wo_source_A"
    first_ffn_start = "L1.ffn_packed_0.start" if ffn_packed else "L1.ffn_vals_0.start"
    idx = {name: events.index(name) for name in (
        "engram_0.finish", "engram_1.finish", "L0.mlp_rs_1.start", "L0.post", "engram.project",
        "engram.mix_half_A", "L1.attn_hc_A", "L1.p2p_kv_0", first_mla_a,
        "L0.mlp_rs_1.finish", first_ffn_start, "engram.mix_half_B", "L1.attn_hc_B",
    )}
    # All L0 Engram gathers finish strictly BEFORE L0.mlp_rs_1.start!
    assert idx["engram_0.finish"] < idx["engram_1.finish"] < idx["L0.mlp_rs_1.start"]
    # L0.mlp_rs_1 stays in flight across L0.post, engram.project, AND all of L1 Half A (mix_half_A + attn + p2p_kv_0)!
    assert idx["L0.mlp_rs_1.start"] < idx["L0.post"] < idx["engram.project"]
    assert idx["engram.project"] < idx["engram.mix_half_A"] < idx["L1.attn_hc_A"] < idx["L1.p2p_kv_0"] < idx[first_mla_a]
    assert idx[first_mla_a] < idx["L0.mlp_rs_1.finish"] < idx[first_ffn_start]
    # L1 Half B runs incoming L0.post + engram.mix_half_B AFTER L0.mlp_rs_1.finish!
    second_l0_post = events.index("L0.post", idx["L0.mlp_rs_1.finish"])
    assert idx["L0.mlp_rs_1.finish"] < first_ffn_start_idx(events, first_ffn_start) < second_l0_post < idx["engram.mix_half_B"] < idx["L1.attn_hc_B"]
    # Verify p2p_kv_gather shape on L1: (st, 512) for full, (5*st, 512) for mla_wo!
    assert l1.kv_gather_0.seen_shapes == [(st * (kv_cols // 512), 512)]
    assert l1.kv_gather_1.seen_shapes == [(st * (kv_cols // 512), 512)]
    # Verify every concurrent collective received copy_input=False.
    assert all(ci is False for _, ci in copy_input_seen)


def first_ffn_start_idx(events, tag):
    return events.index(tag)


def test_zero_copy_concurrent_collectives_backwards_compatibility():
    init_calls = []
    start_calls = []

    class LegacyConcurrent:
        def __init__(self, collective, *, workspace, session, parallel_plans):
            init_calls.append(("legacy", None))

        def start(self, source, *, dependencies=()):
            start_calls.append(("legacy", None))
            return SimpleNamespace(source=source, output=source, dependencies=dependencies)

    class ModernConcurrent:
        def __init__(self, collective, *, workspace, session, parallel_plans, copy_input=True):
            init_calls.append(("modern", copy_input))
            self.copy_input = copy_input

        def start(self, source, *, dependencies=(), copy_input=True):
            start_calls.append(("modern", copy_input))
            return SimpleNamespace(source=source, output=source, dependencies=dependencies)

    legacy = decoder._make_concurrent(LegacyConcurrent, "c", workspace="w", session="s", parallel_plans=("p",), copy_input=False)
    modern = decoder._make_concurrent(ModernConcurrent, "c", workspace="w", session="s", parallel_plans=("p",), copy_input=False)
    x = torch.zeros((4, 8), dtype=torch.bfloat16)
    decoder._start_concurrent(legacy, x, copy_input=False)
    decoder._start_concurrent(modern, x, copy_input=False)

    assert init_calls == [("legacy", None), ("modern", False)]
    assert start_calls == [("legacy", None), ("modern", False)]
    assert modern.copy_input is False


def test_mla_norm_quant_kv_rope_and_prefill_pdl_feature_detection(monkeypatch):
    # 1. Verify feature detection + cos/sin plumbing when b12x Query supports kv_rope_dim.
    @dataclass(frozen=True, kw_only=True)
    class ModernQuery:
        rows: int
        query_weight_dtype: str
        kv_weight_dtype: str
        query_features: int = 1280
        kv_features: int = 512
        epsilon: float = 1e-20
        min_amax: float = 1e-4
        affine_prefetch_ready: bool = False
        release_dependents: bool = False
        kv_rope_dim: int = 0

    seen_queries = []
    seen_runs = []

    def fake_plan(query):
        seen_queries.append(query)
        return SimpleNamespace(request=lambda **kw: kw)

    def fake_run(projected, qw, kw, query, kv, v, sr, sm, *, plan, cos=None, sin=None):
        seen_runs.append((tuple(projected.shape), cos, sin))

    monkeypatch.setattr(mla_norm_quant.mla_norm_mxfp8, "Query", ModernQuery)
    monkeypatch.setattr(mla_norm_quant.mla_norm_mxfp8, "plan", fake_plan)
    monkeypatch.setattr(mla_norm_quant.mla_norm_mxfp8, "run", fake_run)
    monkeypatch.setattr(mla_norm_quant, "declare", lambda *a, **k: None)
    monkeypatch.setattr(mla_norm_quant, "register_finalizer", lambda *a, **k: None)
    monkeypatch.setattr(mla_norm_quant, "input_factory", lambda *a, **k: (lambda: None))
    monkeypatch.setattr(mla_norm_quant, "record_numeric_tensor", lambda *a, **k: None)
    monkeypatch.setattr(mla_norm_quant.mxfp8, "dense_rows_storage_shapes", lambda r, c: ((r, c), (1, r, c // 32), (1, 1, 1, 32, 4, 4)))
    monkeypatch.setattr(mla_norm_quant.mxfp8, "dense_rows_from_bases", lambda *b: SimpleNamespace(bases=b))

    assert mla_norm_quant.PreparedMLANormQuant.supports_kv_rope() is True
    rows = 1024
    proj = torch.zeros((rows, 1792), dtype=torch.bfloat16)
    q = torch.zeros((rows, 1280), dtype=torch.bfloat16)
    kv = torch.zeros((rows, 512), dtype=torch.bfloat16)
    qw = torch.ones(1280, dtype=torch.bfloat16)
    kw = torch.ones(512, dtype=torch.bfloat16)
    angles = torch.randn((rows, 64), dtype=torch.float32)

    nq = mla_norm_quant.PreparedMLANormQuant(
        proj, (qw, kw), q, kv, eps=1e-20, session=None, name="nq",
        release_dependents=True, kv_rope_dim=64, angles=angles,
    ).functional(session=None)
    nq._release_examples()

    assert seen_queries[-1].kv_rope_dim == 64
    assert seen_queries[-1].release_dependents is True
    assert seen_queries[-1].rows == 1024

    # Verify calling with split cos/sin views forwards zero-copy 16-byte aligned views!
    cos_v, sin_v = angles[:, :32], angles[:, 32:]
    assert cos_v.stride() == (64, 1) and sin_v.stride() == (64, 1)
    assert cos_v.data_ptr() % 16 == 0 and sin_v.data_ptr() % 16 == 0
    nq(proj, cos=cos_v, sin=sin_v)
    assert seen_runs[-1][1] is cos_v and seen_runs[-1][2] is sin_v

    # Verify PreparedAttention skips standalone angles.apply(window_values) when norm_quant_fused_kv_rope is True.
    apply_targets = []
    attn = ds_attention.PreparedAttention.__new__(ds_attention.PreparedAttention)
    attn.ratio = 0
    attn.fused_qkv_norm_rope = False
    attn.norm_quant_fused_kv_rope = True
    attn.norm_quant = nq
    attn.query_shape = (rows, 64, 512)
    attn.angles = SimpleNamespace(apply=lambda t, a: apply_targets.append(tuple(t.shape)))
    class _FakeQkv:
        encoded_layout = None

        def __call__(self, s, encoded_bases=None):
            return proj

    attn.projections = {
        "qkv_a": _FakeQkv(),
        "q_b": lambda ql, encoded_bases=None: torch.zeros((rows, 64 * 512), dtype=torch.bfloat16),
    }
    fake_indices = SimpleNamespace(angles={(False, 0): angles})
    attn._run_shard_qkv_core(torch.zeros((rows, 5120), dtype=torch.bfloat16), indices=fake_indices, shard_rows=torch.arange(rows))
    # Only query ([1024, 64, 512]) should receive angles.apply; window_values ([1024, 512]) was fused in norm_quant!
    assert apply_targets == [(rows, 64, 512)]

    # 2. Verify backwards-compatibility when base b12x Query lacks kv_rope_dim.
    @dataclass(frozen=True, kw_only=True)
    class LegacyQuery:
        rows: int
        query_weight_dtype: str
        kv_weight_dtype: str
        query_features: int = 1280
        kv_features: int = 512
        epsilon: float = 1e-20
        min_amax: float = 1e-4
        affine_prefetch_ready: bool = False
        release_dependents: bool = False

    def legacy_run(projected, qw, kw, query, kv, v, sr, sm, *, plan):
        seen_runs.append(("legacy", tuple(projected.shape)))

    monkeypatch.setattr(mla_norm_quant.mla_norm_mxfp8, "Query", LegacyQuery)
    monkeypatch.setattr(mla_norm_quant.mla_norm_mxfp8, "run", legacy_run)
    assert mla_norm_quant.PreparedMLANormQuant.supports_kv_rope() is False
    with pytest.raises(ValueError, match="does not support fused KV RoPE"):
        mla_norm_quant.PreparedMLANormQuant(
            proj, (qw, kw), q, kv, eps=1e-20, session=None, name="nq_bad", kv_rope_dim=64,
        )
    nq_legacy = mla_norm_quant.PreparedMLANormQuant(
        proj, (qw, kw), q, kv, eps=1e-20, session=None, name="nq_leg", release_dependents=True,
    )
    nq_legacy(proj)
    assert seen_runs[-1] == ("legacy", (rows, 1792))

    # 3. Verify QuantizedProjection unlocks release_dependents=True for prefill rows > 8 when rhs_prefetch_ready=True.
    from b12x.gemm.blockscaled._linear import MXFP8LinearWeight
    from b12x.quantization import mxfp8

    captured_mxfp8_queries = []

    def fake_mxfp8_qfc(src, v, sr, sm, *, min_amax=0.0, release_dependents=False):
        captured_mxfp8_queries.append((src.shape[0], release_dependents))
        return SimpleNamespace(rows=src.shape[0], release_dependents=release_dependents)

    @dataclass(frozen=True)
    class _FakeGemmQuery:
        workspace_form: str = "none"
        workspace_nbytes: int | None = 0

    monkeypatch.setattr(comp_linear.sm103_projection, "is_sm103", lambda dev: False)
    monkeypatch.setattr(mxfp8, "query_from_call", fake_mxfp8_qfc)
    monkeypatch.setattr(mxfp8, "plan", lambda q: SimpleNamespace(request=lambda **kw: kw))
    monkeypatch.setattr(mxfp8, "dense_rows_storage_shapes", lambda r, c: ((r, c), (1, r, c // 32), (1, 1, 1, 32, 4, 4)))
    monkeypatch.setattr(mxfp8, "dense_rows_from_bases", lambda v, sr, sm: SimpleNamespace(values=v, scale_rows=sr, scale_mma=sm))
    monkeypatch.setattr(mxfp8, "dense_scale_storage", lambda s: s)
    monkeypatch.setattr(comp_linear.blockscaled, "query_from_call", lambda *a, **k: _FakeGemmQuery())
    monkeypatch.setattr(comp_linear.blockscaled, "plan", lambda q: SimpleNamespace(request=lambda **kw: kw, memory_requirements=lambda: SimpleNamespace(scratch_nbytes=0)))
    monkeypatch.setattr(comp_linear, "declare", lambda *a, **k: None)
    monkeypatch.setattr(comp_linear, "register_finalizer", lambda *a, **k: None)
    monkeypatch.setattr(comp_linear, "record_numeric_tensor", lambda *a, **k: None)

    fake_weight = MXFP8LinearWeight.__new__(MXFP8LinearWeight)
    object.__setattr__(fake_weight, "out_features", 128)
    object.__setattr__(fake_weight, "in_features", 128)
    object.__setattr__(fake_weight, "padded_in_features", 128)
    object.__setattr__(fake_weight, "weight", SimpleNamespace(
        values=torch.zeros((128, 128), dtype=torch.float8_e4m3fn),
        scale_rows=torch.zeros((1, 128, 4), dtype=torch.uint8),
        scale_mma=torch.zeros((1, 1, 1, 32, 4, 4), dtype=torch.uint8),
    ))

    for r_ready in (True, False):
        comp_linear.QuantizedProjection(
            torch.zeros((1024, 128), dtype=torch.bfloat16),
            fake_weight,
            torch.zeros((1024, 128), dtype=torch.bfloat16),
            session=None,
            name="pdl_test",
            workspace=ScratchArena("cpu", overlap=False),
            activation_mode="auto",
            mxfp8_min_amax=1e-4,
            rhs_prefetch_ready=r_ready,
        )
    assert captured_mxfp8_queries == [(1024, True), (1024, False)]


def test_unpack_ffn_gathered_rejects_non_128_aligned_rows():
    packed = torch.zeros((64, 5120 // 2 + 5120 // 64 + 8 * 4), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="divisible by 128"):
        decoder._unpack_ffn_gathered(packed, hidden=5120, top_k=8)


def test_run_ffn_gathered_packed_path_preserves_shared_residual_epilogue():
    rows, hidden, top_k = 128, 512, 4
    vals = torch.zeros((rows, hidden), dtype=torch.float8_e4m3fn)
    srows = torch.full((1, rows, hidden // 32), 127, dtype=torch.uint8)
    ids = torch.zeros((rows, top_k), dtype=torch.int32)
    weights = torch.ones((rows, top_k), dtype=torch.float32)
    packed = decoder._pack_ffn_shard(vals, srows, ids, weights)

    routed_val = torch.ones((rows, hidden), dtype=torch.bfloat16) * 2.0
    shared_val = torch.ones((rows, hidden), dtype=torch.bfloat16) * 3.0
    seen_residuals = []

    class _FakeShared:
        def __init__(self, supports_res: bool):
            self.supports_residual = supports_res

        def __call__(self, x, *, encoded_bases=None, residual=None):
            seen_residuals.append(residual)
            return shared_val if residual is None else (shared_val + residual)

    layer = object.__new__(decoder.PreparedLayer)
    layer.ffn_packed_gather = True
    layer.ffn_vals_bf16_cols = hidden // 2
    layer.ffn_top_k = top_k
    layer.experts = lambda *a, **kw: routed_val
    layer.shared = _FakeShared(True)
    out_fused, _ = layer._run_ffn_gathered(packed)
    assert seen_residuals[-1] is routed_val
    assert torch.equal(out_fused, routed_val + shared_val)

    layer.shared = _FakeShared(False)
    out_plain, _ = layer._run_ffn_gathered(packed)
    assert seen_residuals[-1] is None
    assert torch.equal(out_plain, routed_val + shared_val)
