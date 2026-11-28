# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""0-ULP GPU parity tests for compact packed target verification, logprobs, and overlapped PreparedContext."""

import os
from types import SimpleNamespace

import pytest
import torch

from hyperflare_recipes.components.dspark import PreparedContext
from hyperflare_recipes.components.paged_batch import Metadata
from hyperflare_recipes.components.parallel import TensorParallelShard
from hyperflare_recipes.components.speculation import PreparedSpeculativeCommit, SpeculativeRngState
from hyperflare_recipes.components.vocabulary_logprobs import PreparedVocabularyParallelLogprobs
from hyperflare_recipes.components.vocabulary_sampling import (
    PreparedVocabularyBlockVerification,
    PreparedVocabularySoftmax,
)
from hyperflare_recipes.recipes.deepseek41.config import ModelConfig, QUALIFIED_ENCODER_SCHEDULE, TEXT_GEOMETRY
from hyperflare_recipes.recipes.deepseek41.pool import ContinuationPool
from ignition.components.completion import CompletionOutputs, VECTOR_TOKENS_FIELD
from ignition.components.sampling import DeviceRequestState, RUNNING
from ignition.preparation.aot import aot_forward
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an explicitly assigned CUDA device",
)


def _deepseek41_config():
    text = {
        **TEXT_GEOMETRY,
        **QUALIFIED_ENCODER_SCHEDULE,
        "hidden_size": 5120,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "compress_rope_theta": 10000.0,
        "rope_scaling": {
            "original_max_position_embeddings": 4096,
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 40.0,
        },
        "num_nextn_predict_layers": 3,
        "dspark_block_size": 5,
        "dspark_noise_token_id": 128799,
        "dspark_target_layer_ids": [37, 38, 39],
        "dspark_markov_rank": 256,
        "dspark_n_routed_experts": 128,
        "dspark_num_experts_per_tok": 3,
        "compress_ratios": [*QUALIFIED_ENCODER_SCHEDULE["compress_ratios"], 0, 0, 0],
    }
    quant = {
        "quant_method": "fp8",
        "weight_block_size": [32, 32],
        "scale_fmt": "ue8m0",
        "expert_dtype": "fp4",
        "activation_scheme": "dynamic",
    }
    return ModelConfig(text, quant)


@pytest.mark.parametrize(
    "batch,width,tokens,lengths,query_lengths,temperatures,request_logprobs",
    [
        (2, 8, 8, [5, 3], [5, 3], [1.0, 0.7], False),
        (4, 8, 19, [7, 6, 4, 2], [7, 6, 4, 2], [1.0, 0.0, 0.8, 1.2], True),
        # Inactive row with nonzero query_length but zero packed length (e.g. unadmitted / padding lane).
        (4, 8, 12, [6, 4, 0, 2], [6, 4, 3, 2], [1.0, 1.0, 1.0, 0.0], True),
        (
            16,
            8,
            47,
            [6, 5, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1],
            [6, 5, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1],
            [1.0] * 16,
            True,
        ),
    ],
)
@torch.no_grad()
def test_compact_target_softmax_verify_and_logprobs_0ulp_parity(
    batch, width, tokens, lengths, query_lengths, temperatures, request_logprobs
):
    """Compact [P, V] target softmax, BlockVerify, and logprobs match expanded [B, W, V] bit-for-bit."""
    device = torch.device("cuda:0")
    vocab = 4096
    capacity = 32 if batch > 16 else 16
    gen = torch.Generator(device="cpu").manual_seed(1234 + batch * 100 + tokens)

    def make_state():
        req = DeviceRequestState.allocate(device, vocab_size=vocab, capacity=capacity)
        req.status.fill_(0)
        for r in range(batch):
            req.status[r] = RUNNING
            req.generation[r] = 7 + r
            req.seed[r] = 1000 + r
            req.max_new_tokens[r] = 64
            req.parameters[r, 0] = temperatures[r]
            req.parameters[r, 1] = 1.0  # top_p
            req.parameters[r, 2] = 0.0  # min_p
            req.parameters[r, 3] = 1.0  # repeat_penalty
            req.parameters[r, 4] = 0.0  # presence
            req.parameters[r, 5] = 0.0  # frequency
            req.top_k[r] = 0
        rng = SpeculativeRngState.allocate(device, capacity=capacity)
        rng.generation[:batch].copy_(req.generation[:batch])
        rng.target_offset[:batch].fill_(40)
        comp = CompletionOutputs.allocate((2, capacity, VECTOR_TOKENS_FIELD + width), device=device)
        if request_logprobs:
            comp.requested[:batch].fill_(1)
        return req, rng, comp

    slots = torch.arange(batch, dtype=torch.int32, device=device)
    generations = torch.tensor([7 + r for r in range(batch)], dtype=torch.int64, device=device)
    live_rows = torch.tensor([batch], dtype=torch.int32, device=device)
    q_limits = torch.tensor(query_lengths, dtype=torch.int32, device=device)
    s_limits = torch.tensor(lengths, dtype=torch.int32, device=device)
    starts = torch.full((batch,), 10, dtype=torch.int64, device=device)
    states = torch.arange(batch, dtype=torch.int64, device=device)
    pages = torch.zeros((batch, 2), dtype=torch.int32, device=device)
    candidates = torch.randint(0, vocab, (batch, width), generator=gen, dtype=torch.int64).to(device)
    drafts = candidates[:, 1:].contiguous()

    metadata = Metadata.pack(
        starts, s_limits, states, pages, candidates, live_rows,
        token_capacity=tokens, sample_count=batch * width, max_query_tokens=width,
    )

    compact_logits = (torch.randn((tokens, vocab), generator=gen) * 2.0).to(device=device, dtype=torch.float32)
    expanded_logits = compact_logits[metadata.sample_indices].view(batch, width, vocab).contiguous()

    # Build valid proposal probabilities q and tile masses.
    draft_logits = (torch.randn((batch, width - 1, vocab), generator=gen) * 2.0).to(device=device, dtype=torch.float32)
    draft_probs = torch.softmax(draft_logits, dim=-1).contiguous()
    blocks = (vocab + 255) // 256
    draft_masses = draft_probs.view(batch, width - 1, blocks, 256).sum(dim=-1).contiguous()
    draft_valid = torch.ones((batch, width - 1), dtype=torch.bool, device=device)
    for r, lim in enumerate(lengths):
        if lim == 0:
            draft_valid[r].fill_(False)

    frame = torch.zeros((1,), dtype=torch.int32, device=device)
    seq = torch.ones((1,), dtype=torch.int64, device=device)

    # Reference 3D expanded path
    req_ref, rng_ref, comp_ref = make_state()
    ref_probs = torch.empty((batch, width, vocab), dtype=torch.float32, device=device)
    ref_valid = torch.empty((batch, width), dtype=torch.bool, device=device)
    ref_masses = torch.empty((batch, width, blocks), dtype=torch.float32, device=device)
    t_ref = PreparedVocabularySoftmax(
        expanded_logits, ref_probs, ref_valid, req_ref, slots, generations, live_rows, drafts,
        start=0, runtime=None, session=None, coordinator=None, name="ref.target",
        tile_masses=ref_masses, token_limits=q_limits,
    )
    t_ref.prepare()
    t_ref.functional(session=None)
    v_ref = PreparedVocabularyBlockVerification(
        draft_probs, drafts, ref_probs, req_ref, slots, generations, live_rows,
        start=0, runtime=None, session=None, coordinator=None, name="ref.verify",
        rng=rng_ref, draft_valid=draft_valid, target_valid=ref_valid, token_limits=q_limits,
        draft_tile_masses=draft_masses, target_tile_masses=ref_masses,
    )
    v_ref.prepare()
    lp_ref = PreparedVocabularyParallelLogprobs(
        expanded_logits, comp_ref, slots, generations, live_rows, frame,
        vocabulary=vocab, offset=0, runtime=None, session=None, coordinator=None, name="ref.lp",
    )
    c_ref = PreparedSpeculativeCommit(
        req_ref, v_ref.output, candidates, slots, generations, live_rows, comp_ref, frame, seq,
        rng=rng_ref, token_limits=q_limits, logprobs=lp_ref,
    )
    c_ref.prepare()
    v_ref.functional(session=None)
    c_ref.functional(session=None)

    ref_tm = torch.empty((batch, width, blocks), dtype=torch.float32, device=device)
    ref_p, ref_v, _ = t_ref.run(expanded_logits, drafts, tile_masses=ref_tm, token_limits=q_limits)
    acc_ref = v_ref.run(
        draft_probs, drafts, ref_p, draft_valid, ref_v,
        token_limits=q_limits, draft_tile_masses=draft_masses, target_tile_masses=ref_tm,
    )
    # Mask invalid rows matching parallel_draft.py (proposal_valid = s_limits > 0)
    acc_ref.valid.logical_and_(s_limits > 0)
    com_ref = c_ref(expanded_logits, acceptance=acc_ref, inputs=candidates, token_limits=q_limits)

    # Compact 2D path
    req_cmp, rng_cmp, comp_cmp = make_state()
    cmp_probs = torch.empty((tokens, vocab), dtype=torch.float32, device=device)
    cmp_valid = torch.empty((batch, width), dtype=torch.bool, device=device)
    cmp_masses = torch.empty((batch, width, blocks), dtype=torch.float32, device=device)
    t_cmp = PreparedVocabularySoftmax(
        compact_logits, cmp_probs, cmp_valid, req_cmp, slots, generations, live_rows, drafts,
        start=0, runtime=None, session=None, coordinator=None, name="cmp.target",
        tile_masses=cmp_masses, token_limits=q_limits,
        sample_indices=metadata.sample_indices, sample_limits=s_limits,
    )
    t_cmp.prepare()
    t_cmp.functional(session=None)
    v_cmp = PreparedVocabularyBlockVerification(
        draft_probs, drafts, cmp_probs, req_cmp, slots, generations, live_rows,
        start=0, runtime=None, session=None, coordinator=None, name="cmp.verify",
        rng=rng_cmp, draft_valid=draft_valid, target_valid=cmp_valid, token_limits=q_limits,
        draft_tile_masses=draft_masses, target_tile_masses=cmp_masses,
        sample_indices=metadata.sample_indices, sample_limits=s_limits,
    )
    v_cmp.prepare()
    lp_cmp = PreparedVocabularyParallelLogprobs(
        compact_logits, comp_cmp, slots, generations, live_rows, frame,
        vocabulary=vocab, offset=0, runtime=None, session=None, coordinator=None, name="cmp.lp",
        sample_indices=metadata.sample_indices, batch=batch, width=width,
    )
    c_cmp = PreparedSpeculativeCommit(
        req_cmp, v_cmp.output, candidates, slots, generations, live_rows, comp_cmp, frame, seq,
        rng=rng_cmp, token_limits=q_limits, logprobs=lp_cmp,
    )
    c_cmp.prepare()
    v_cmp.functional(session=None)
    c_cmp.functional(session=None)

    cmp_tm = torch.empty((batch, width, blocks), dtype=torch.float32, device=device)
    cmp_p, cmp_v, _ = t_cmp.run(
        compact_logits, drafts, tile_masses=cmp_tm, token_limits=q_limits,
        sample_indices=metadata.sample_indices, sample_limits=s_limits,
    )
    acc_cmp = v_cmp.run(
        draft_probs, drafts, cmp_p, draft_valid, cmp_v,
        token_limits=q_limits, draft_tile_masses=draft_masses, target_tile_masses=cmp_tm,
        sample_indices=metadata.sample_indices, sample_limits=s_limits,
    )
    acc_cmp.valid.logical_and_(s_limits > 0)
    com_cmp = c_cmp(
        compact_logits, acceptance=acc_cmp, inputs=candidates,
        token_limits=q_limits, sample_indices=metadata.sample_indices,
    )
    torch.cuda.synchronize()

    # Strict 0-ULP bit-exact assertions on all live positions.
    idx_2d = metadata.sample_indices.view(batch, width)
    for r, lim in enumerate(lengths):
        for p in range(lim):
            phys = int(idx_2d[r, p].item())
            assert torch.equal(cmp_p[phys].view(torch.int32), ref_p[r, p].view(torch.int32))
            assert torch.equal(cmp_tm[r, p].view(torch.int32), ref_tm[r, p].view(torch.int32))
            assert bool(cmp_v[r, p].item()) == bool(ref_v[r, p].item())

    for r, lim in enumerate(lengths):
        if lim > 0:
            assert torch.equal(acc_cmp.token_ids[r], acc_ref.token_ids[r])
            assert int(acc_cmp.proposed_count[r]) == int(acc_ref.proposed_count[r])
            assert int(acc_cmp.accepted_drafts[r]) == int(acc_ref.accepted_drafts[r])
            assert int(acc_cmp.offered_drafts[r]) == int(acc_ref.offered_drafts[r])
            assert bool(acc_cmp.valid[r]) == bool(acc_ref.valid[r])
            assert torch.equal(acc_cmp.survival[r].view(torch.int32), acc_ref.survival[r].view(torch.int32))
        assert int(com_cmp.emitted_count[r]) == int(com_ref.emitted_count[r])
        assert torch.equal(com_cmp.emitted_ids[r], com_ref.emitted_ids[r])

    assert torch.equal(comp_cmp.records, comp_ref.records)
    if request_logprobs:
        assert torch.equal(comp_cmp.logprobs, comp_ref.logprobs)


class _SyncContextFwd:
    def __init__(self, ctx, aux, counts, meta):
        self.ctx, self.aux, self.counts, self.meta = ctx, aux, counts, meta

    def __call__(self):
        self.ctx(self.aux, self.counts, self.meta)


class _OverlapContextFwd:
    def __init__(self, ctx, aux, counts, meta):
        self.ctx, self.aux, self.counts, self.meta = ctx, aux, counts, meta

    def __call__(self):
        # Produce a forward-local temporary auxiliary tensor (matching decoder output)
        # and allocate same-sized clobber buffers on the main stream before join to
        # verify TorchInductor never aliases auxiliary or positions while aux_stream reads them.
        aux_temp = self.aux + 0.0
        pending = self.ctx.start_prepare(aux_temp, self.meta)
        clobber_aux = torch.full_like(aux_temp, float("nan"))
        clobber_pos = torch.full_like(self.meta.positions, -999999)
        dummy = (
            self.counts.float().sum()
            + clobber_aux[:1, :1].isfinite().float().sum()
            + (clobber_pos[:1] * 0).float().sum()
        )
        self.ctx.commit_prepared(pending, self.counts + (dummy * 0.0).to(torch.int32), self.meta)


@pytest.mark.parametrize("tokens", [8, 19, 29, 32, 47])
@torch.no_grad()
def test_overlapped_dspark_context_0ulp_parity_and_cudagraph(tokens):
    """Overlapped PreparedContext.start_prepare/commit_prepared matches synchronous __call__ bit-for-bit."""
    from b12x.gemm._shared.wo_mxfp8 import quantize_mxfp8_rows_torch
    from b12x.gemm.blockscaled import pack_weight

    torch._dynamo.reset()
    device = torch.device("cuda:0")
    config = _deepseek41_config()
    hidden = 5120
    aux_cols = len(config.draft.target_layers) * hidden  # 15360
    draft_layers = config.draft.layers  # 3

    gen = torch.Generator(device="cpu").manual_seed(4242 + tokens)
    main_bf16 = (torch.randn((hidden, aux_cols), generator=gen) * 0.05).to(device=device, dtype=torch.bfloat16)
    kv_bf16 = (torch.randn((512 * draft_layers, hidden), generator=gen) * 0.05).to(device=device, dtype=torch.bfloat16)

    def pack_mxfp8(w_bf16):
        q = quantize_mxfp8_rows_torch(w_bf16)
        return pack_weight(
            q.values.contiguous(),
            q.scale_rows.view(torch.uint8).reshape(w_bf16.shape[0], -1).contiguous(),
            recipe="mxfp8",
        )

    main_w = pack_mxfp8(main_bf16)
    kv_w = pack_mxfp8(kv_bf16)
    main_norm = torch.ones((hidden,), dtype=torch.bfloat16, device=device)
    kv_norms = [torch.ones((512,), dtype=torch.bfloat16, device=device) for _ in range(draft_layers)]
    weights = SimpleNamespace(
        config=config,
        shard=TensorParallelShard(0, 1),
        draft=SimpleNamespace(
            main_projection=main_w,
            context_kv=kv_w,
            main_norm=main_norm,
            layers=tuple(SimpleNamespace(attention={"kv_norm": n}) for n in kv_norms),
        ),
    )

    batch = 4
    width = 8
    pool_sync = ContinuationPool.allocate(config, batch + 2, 4, device, dspark=True)
    pool_ovlp = ContinuationPool.allocate(config, batch + 2, 4, device, dspark=True)
    meta = Metadata.allocate(tokens, batch, 2, device, samples=batch * width, max_query_tokens=width)
    aux = (torch.randn((tokens, aux_cols), generator=gen) * 0.1).to(device=device, dtype=torch.bfloat16)
    counts = torch.tensor([min(tokens // batch, 5), 2, 0, 3], dtype=torch.int32, device=device)
    starts = torch.tensor([125, 4, 200, 127], dtype=torch.int64, device=device)
    lengths = torch.tensor([min(tokens - 6, 8), 3, 1, 2], dtype=torch.int32, device=device)
    states = torch.arange(1, batch + 1, dtype=torch.int64, device=device)
    pages = torch.zeros((batch, 2), dtype=torch.int32, device=device)
    meta.set(
        starts, lengths, states, pages, torch.zeros((batch, width), dtype=torch.int64, device=device),
        torch.tensor([batch], dtype=torch.int32, device=device),
    )

    arena = ScratchArena(device, overlap=True)
    with RecordedPreparationSession(
        device=device, checkpoint=f"test-ctx-ovlp-{tokens}", workload_schema="test", numerical_policy="default",
        compile_workers=4, autotune_effort="default",
    ) as session:
        session.register_scratch(arena)
        ctx_sync = PreparedContext(weights, aux, meta, pool_sync, session=session, name="sync", workspace=arena)
        ctx_ovlp = PreparedContext(weights, aux, meta, pool_ovlp, session=session, name="ovlp", workspace=arena)
        session.tune(show_progress=False)
        assert ctx_ovlp.can_overlap_with(None, None)

        f_sync = _SyncContextFwd(ctx_sync, aux, counts, meta)
        f_ovlp = _OverlapContextFwd(ctx_ovlp, aux, counts, meta)
        c_sync = aot_forward(f_sync)
        c_ovlp = aot_forward(f_ovlp)
        c_sync(f_sync)
        c_ovlp(f_ovlp)
        torch.cuda.synchronize()

        g_sync = torch.cuda.CUDAGraph()
        g_ovlp = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(g_sync):
            c_sync(f_sync)
        with session.capture(), torch.cuda.graph(g_ovlp):
            c_ovlp(f_ovlp)
        try:
            for trial in range(3):
                aux.normal_(std=0.1)
                counts.copy_(torch.tensor([min(int(lengths[0]), trial + 1), 1, 0, min(int(lengths[3]), 2)],
                                          dtype=torch.int32, device=device))
                pool_sync.clear_for_preparation()
                pool_ovlp.clear_for_preparation()
                before = torch.cuda.memory_stats()["allocation.all.allocated"]
                g_sync.replay()
                g_ovlp.replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
                for layer_idx in range(draft_layers):
                    assert torch.equal(
                        pool_ovlp.window(layer_idx, draft=True),
                        pool_sync.window(layer_idx, draft=True),
                    )
        finally:
            g_sync.reset()
            g_ovlp.reset()


class _StepGateFwd:
    def __init__(self, op_eager, verify_op, logits, pending, seeds, offsets, limits,
                 target_p, target_v, target_tm, out_q, out_v, out_t, out_tm, out_conf,
                 out_acc_valid, out_acc_offered):
        self.op_eager, self.verify_op = op_eager, verify_op
        self.logits, self.pending, self.seeds, self.offsets, self.limits = logits, pending, seeds, offsets, limits
        self.target_p, self.target_v, self.target_tm = target_p, target_v, target_tm
        self.out_q, self.out_v, self.out_t, self.out_tm, self.out_conf = out_q, out_v, out_t, out_tm, out_conf
        self.out_acc_valid, self.out_acc_offered = out_acc_valid, out_acc_offered

    def __call__(self):
        self.op_eager.run(
            self.logits, self.out_t, pending=self.pending, seeds=self.seeds, offsets=self.offsets,
            probabilities=self.out_q, valid=self.out_v, tokens=self.out_t,
            tile_masses=self.out_tm, token_limits=self.limits, confidence=self.out_conf,
        )
        acc = self.verify_op.run(
            self.out_q, self.out_t, self.target_p, self.out_v, self.target_v,
            token_limits=self.limits, draft_tile_masses=self.out_tm, target_tile_masses=self.target_tm,
        )
        self.out_acc_valid.copy_(acc.valid)
        self.out_acc_offered.copy_(acc.offered_drafts)


@pytest.mark.parametrize("batch,width,fp8", [(1, 7, True), (4, 7, True), (16, 7, False)])
@torch.no_grad()
def test_proposal_step_gates_skip_inactive_steps_0ulp_and_cudagraph(batch, width, fp8):
    """Per-step proposal gating skips steps >= max(limits - 1) with 0-ULP parity under CUDAGraph."""
    from hyperflare_recipes.components.markov_sampling import fp8_markov_head
    from hyperflare_recipes.components.speculation import DRAFT_RNG_DOMAIN

    torch._dynamo.reset()
    device = torch.device("cuda:0")
    vocab = 1024
    rank = 64
    blocks = (vocab + 255) // 256
    gen = torch.Generator(device="cpu").manual_seed(777 + batch * 17 + int(fp8))

    state = DeviceRequestState.allocate(device, vocab_size=vocab, capacity=max(16, batch))
    state.status.fill_(RUNNING)
    state.generation.fill_(11)
    state.seed[:batch].copy_(torch.arange(batch, dtype=torch.int64) + 9001)
    state.parameters[:batch].copy_(torch.tensor([0.8, 1.0, 0.0, 1.0, 0.0, 0.0]))
    state.top_k.fill_(-1)
    slots = torch.arange(batch, dtype=torch.int32, device=device)
    epochs = torch.full((batch,), 11, dtype=torch.int64, device=device)
    live = torch.tensor([batch], dtype=torch.int32, device=device)

    emb = (torch.randn((vocab, rank), generator=gen) * 0.1).to(device=device, dtype=torch.bfloat16)
    head_bf16 = (torch.randn((vocab, rank), generator=gen) * 0.1).to(device=device, dtype=torch.bfloat16)
    head = fp8_markov_head(head_bf16) if fp8 else head_bf16
    logits = (torch.randn((batch, width, vocab), generator=gen) * 0.5).to(device=device, dtype=torch.bfloat16)
    pending = torch.randint(0, vocab, (batch,), generator=gen, dtype=torch.int64).to(device)
    seeds = state.seed[:batch].view(torch.int64) ^ DRAFT_RNG_DOMAIN
    offsets = torch.full((batch,), 123, dtype=torch.int64, device=device)
    limits = torch.full((batch,), 3, dtype=torch.int32, device=device)

    # Eager check: verify step_gates tensor values and skipped-step storage preservation.
    e_probs = torch.full((batch, width, vocab), -99.0, dtype=torch.float32, device=device)
    e_valid = torch.ones((batch, width), dtype=torch.bool, device=device)
    e_tokens = torch.full((batch, width), -7, dtype=torch.int64, device=device)
    e_masses = torch.full((batch, width, blocks), -55.0, dtype=torch.float32, device=device)
    e_conf = torch.full((batch, width), -33.0, dtype=torch.float32, device=device)
    op_eager = PreparedVocabularySoftmax(
        logits, e_probs, e_valid, state, slots, epochs, live, e_tokens,
        start=0, runtime=None, session=None, coordinator=None, name="stepgate.eager",
        embedding=emb, head=head, pending=pending, seeds=seeds, offsets=offsets,
        tokens=e_tokens, tile_masses=e_masses, token_limits=limits, confidence=e_conf,
    )
    assert op_eager.step_gated
    op_eager.prepare()
    # Reference without token_limits (computes all steps unconditionally).
    ref_probs = torch.empty((batch, width, vocab), dtype=torch.float32, device=device)
    ref_valid = torch.empty((batch, width), dtype=torch.bool, device=device)
    ref_tokens = torch.zeros((batch, width), dtype=torch.int64, device=device)
    ref_masses = torch.empty((batch, width, blocks), dtype=torch.float32, device=device)
    ref_conf = torch.empty((batch, width), dtype=torch.float32, device=device)
    op_ref = PreparedVocabularySoftmax(
        logits, ref_probs, ref_valid, state, slots, epochs, live, ref_tokens,
        start=0, runtime=None, session=None, coordinator=None, name="stepgate.ref",
        embedding=emb, head=head, pending=pending, seeds=seeds, offsets=offsets,
        tokens=ref_tokens, tile_masses=ref_masses, token_limits=None, confidence=ref_conf,
    )
    assert not op_ref.step_gated
    op_ref.prepare()

    op_eager()
    op_ref()
    torch.cuda.synchronize()
    # max(limits - 1) == 2 -> steps 0, 1 open; steps 2..6 closed.
    gates = op_eager.scratch["step_gates"][:, 0].tolist()
    assert gates == [1, 1] + [0] * (width - 2)
    assert e_valid[:, :2].all() and not e_valid[:, 2:].any()
    assert (e_tokens[:, 2:] == 0).all()
    assert (e_conf[:, 2:] == 0.0).all()
    assert torch.equal(e_tokens[:, :2], ref_tokens[:, :2])
    assert torch.equal(e_probs[:, :2].view(torch.int32), ref_probs[:, :2].view(torch.int32))
    assert torch.equal(e_masses[:, :2].view(torch.int32), ref_masses[:, :2].view(torch.int32))
    assert torch.equal(e_conf[:, :2].view(torch.int32), ref_conf[:, :2].view(torch.int32))
    # Inactive steps >= max_drafts did NOT touch tile_masses (sentinel preserved).
    assert (e_masses[:, 2:] == -55.0).all()

    # Functional + AOT + CUDAGraph replay across varying dynamic token_limits + BlockVerification.
    rng_g = SpeculativeRngState.allocate(device, capacity=max(16, batch))
    rng_g.generation.fill_(11)
    target_p = torch.full((batch, width + 1, vocab), 1.0 / vocab, dtype=torch.float32, device=device)
    target_tm = torch.full((batch, width + 1, blocks), 256.0 / vocab, dtype=torch.float32, device=device)
    target_v = torch.ones((batch, width + 1), dtype=torch.bool, device=device)

    verify_op = PreparedVocabularyBlockVerification(
        e_probs, e_tokens, target_p, state, slots, epochs, live,
        start=0, runtime=None, session=None, coordinator=None, name="stepgate.verify",
        rng=rng_g, draft_valid=e_valid, target_valid=target_v, token_limits=limits,
        draft_tile_masses=e_masses, target_tile_masses=target_tm,
    )
    verify_op.prepare()
    op_eager.functional(session=None)
    verify_op.functional(session=None)

    out_q = torch.empty((batch, width, vocab), dtype=torch.float32, device=device)
    out_v = torch.empty((batch, width), dtype=torch.bool, device=device)
    out_t = torch.empty((batch, width), dtype=torch.int64, device=device)
    out_tm = torch.empty((batch, width, blocks), dtype=torch.float32, device=device)
    out_conf = torch.empty((batch, width), dtype=torch.float32, device=device)
    out_acc_valid = torch.empty((batch,), dtype=torch.bool, device=device)
    out_acc_offered = torch.empty((batch,), dtype=torch.int32, device=device)

    fwd = _StepGateFwd(
        op_eager, verify_op, logits, pending, seeds, offsets, limits,
        target_p, target_v, target_tm, out_q, out_v, out_t, out_tm, out_conf,
        out_acc_valid, out_acc_offered,
    )
    compiled = aot_forward(fwd)
    compiled(fwd)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled(fwd)
    try:
        # Test dynamic limit transitions: 1 (0 drafts), mixed [4, 2, ...], full 8 (7 drafts).
        test_schedules = [
            [1] * batch,
            [min(width + 1, 2 + (r % 4)) for r in range(batch)],
            [width + 1] * batch,
        ]
        for sched in test_schedules:
            limits.copy_(torch.tensor(sched, dtype=torch.int32, device=device))
            out_tm.fill_(float("nan"))
            out_v.fill_(True)
            out_t.fill_(-1)
            out_conf.fill_(float("nan"))
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
            assert out_acc_valid.all()
            for r, lim in enumerate(sched):
                nd = lim - 1
                assert int(out_acc_offered[r].item()) == nd
                if nd > 0:
                    assert out_v[r, :nd].all()
                    assert torch.equal(out_t[r, :nd], ref_tokens[r, :nd])
                    assert torch.equal(out_q[r, :nd].view(torch.int32), ref_probs[r, :nd].view(torch.int32))
                    assert torch.equal(out_conf[r, :nd].view(torch.int32), ref_conf[r, :nd].view(torch.int32))
                if nd < width:
                    assert not out_v[r, nd:].any()
                    assert (out_t[r, nd:] == 0).all()
                    assert (out_conf[r, nd:] == 0.0).all()
    finally:
        graph.reset()
