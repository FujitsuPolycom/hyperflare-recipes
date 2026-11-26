# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU and FX verification of lossless PromptOutputSAM copy-draft wiring in DeepSeek-V4.1."""

from types import SimpleNamespace

import pytest
import torch

from ignition.components.completion import CompletionOutputs, VECTOR_TOKENS_FIELD
from ignition.components.sampling import DeviceRequestState, RUNNING
from ignition.preparation import runtime_core_pb2 as pb2
from ignition.preparation.execution import GraphSignature

from hyperflare_recipes.components.dspark import DSparkForwards, SpeculativePreparation
from hyperflare_recipes.components.parallel_draft import (
    DraftProposalState,
    PreparedParallelDraftRound,
    _apply_copy_proposals,
    _resolve_copy_proposals,
)
from hyperflare_recipes.components.speculation import (
    DRAFT_RNG_DOMAIN,
    PreparedSpeculativeCommit,
    SpeculativeAcceptance,
    SpeculativeRngState,
)
from hyperflare_recipes.recipes.deepseek41.bootstrap import (
    PreparationConfig,
    _prepare_speculation_catalog,
)
from hyperflare_recipes.recipes.deepseek41.contract import prepare as prepare_ds41_contract


class _DummyDraftScratch:
    @staticmethod
    def allocate(tokens, samples, world_size, device, *, experts, top_k):
        return SimpleNamespace(tokens=tokens, samples=samples, device=device)


class _RecordingProposal:
    """Deterministic per-position proposal that records Markov predecessors and candidate prefixes."""

    has_source_logits = True

    def __init__(self, batch, vocab, step_index, distributions):
        self.batch = batch
        self.vocab = vocab
        self.step_index = step_index
        self.distributions = distributions
        self.layout = {
            "probabilities": ((batch, 1, vocab), torch.float32),
            "valid": ((batch, 1), torch.bool),
        }
        self.seen_candidates = None

    def run(self, scratch, logits, *, source_logits, shortlist_ids, candidates, seed, offset, sampled, source_bias=None):
        self.seen_candidates = candidates.clone()
        probs = self.distributions[self.step_index].clone()
        # Deterministic sampled token per row: argmax of Markov-biased distribution.
        if source_bias is not None:
            scores = torch.log(probs.clamp_min(1e-12)) + source_bias
            probs = torch.softmax(scores, dim=-1)
        chosen = probs.argmax(dim=-1)
        sampled.copy_(chosen)
        scratch["valid"].fill_(True)
        scratch["probabilities"][:, 0].copy_(probs)
        return scratch["probabilities"]


def _make_round_and_owner(
    *,
    batch=2,
    width=8,
    capacity=4,
    vocab=16,
    with_copy=True,
    copy_width=7,
    markov_fn=None,
    distributions=None,
    query_lengths=None,
    row_slot_ids=None,
    consumed=None,
    generations=None,
    grammar_enabled=None,
    active_lengths=None,
):
    if distributions is None:
        gen = torch.Generator().manual_seed(12345)
        raw = torch.rand((width - 1, batch, vocab), generator=gen, dtype=torch.float32) + 0.05
        distributions = raw / raw.sum(dim=-1, keepdim=True)
    if query_lengths is None:
        query_lengths = torch.full((batch,), width, dtype=torch.int32)
    if row_slot_ids is None:
        row_slot_ids = torch.arange(batch, dtype=torch.int32)
    if consumed is None:
        consumed = torch.tensor([10 + i * 5 for i in range(capacity)], dtype=torch.int64)
    if generations is None:
        generations = torch.tensor([100 + i for i in range(capacity)], dtype=torch.int64)
    if grammar_enabled is None:
        grammar_enabled = torch.zeros(capacity, dtype=torch.int32)
    if active_lengths is None:
        active_lengths = torch.ones(batch, dtype=torch.int32)

    requests = SimpleNamespace(
        capacity=capacity,
        vocab_size=vocab,
        histogram=torch.zeros(capacity, vocab, dtype=torch.int32),
        consumed_tokens=consumed.clone(),
        pending_token=torch.tensor([1 + i for i in range(capacity)], dtype=torch.int64),
        generation=generations.clone(),
        seed=torch.tensor([101 + i for i in range(capacity)], dtype=torch.int64),
        grammar_enabled=grammar_enabled.clone(),
    )
    inputs = SimpleNamespace(
        row_slot_ids=row_slot_ids.clone(),
        row_generations=generations[row_slot_ids.long().clamp(0, capacity - 1)].clone(),
        query_lengths=query_lengths.clone(),
        request_state_rows=torch.arange(capacity, dtype=torch.int64)[:, None],
        page_tables=torch.zeros((capacity, 2), dtype=torch.int32),
        live_rows=torch.tensor([batch], dtype=torch.int32),
    )
    owner_fields = dict(
        requests=requests,
        inputs=inputs,
        block_size=width - 1,
        logit_offset=0,
        weights=SimpleNamespace(
            shard=SimpleNamespace(rank=0, world_size=1),
            config=SimpleNamespace(draft=SimpleNamespace(noise_token=0)),
        ),
        generation=generations.clone(),
        offset=torch.zeros(capacity, dtype=torch.int64),
    )
    if with_copy:
        owner_fields.update(
            copy_ids=torch.full((capacity, copy_width), -1, dtype=torch.int64),
            copy_count=torch.zeros(capacity, dtype=torch.int32),
            copy_expected=torch.full((capacity,), -1, dtype=torch.int64),
            copy_generation=torch.zeros(capacity, dtype=torch.int64),
        )
    owner = SimpleNamespace(**owner_fields)

    proposals = tuple(_RecordingProposal(batch, vocab, i, distributions) for i in range(width - 1))
    op = object.__new__(PreparedParallelDraftRound)
    op.owner, op.batch, op.width = owner, batch, width
    op.vocabulary_parallel = False
    op.generate_proposals = True
    op.operations = None, proposals, None, None
    op.validator = SimpleNamespace(
        validate=lambda: (torch.zeros(1, 8), torch.ones(batch, dtype=torch.bool)),
        begin=lambda totals, present: SimpleNamespace(lengths=active_lengths.clone()),
    )
    op.validation = lambda values: values
    op.draft = lambda metadata: (torch.zeros(batch * (width - 1), vocab), None)
    op.shortlisted_selector = op.selector = None
    op.markov = markov_fn
    op.full_proposal = op.proposal_block = op.recurrent = None
    return op, owner, proposals


def test_speculative_preparation_buffers_and_helios_contract_abi(monkeypatch):
    """SpeculativePreparation allocates Helios copy buffers and deepseek41 contract binds them."""
    from hyperflare_recipes.components import packed_speculation, parallel_draft

    capacity = 4
    requests = DeviceRequestState.allocate(torch.device("cpu"), vocab_size=129280, capacity=capacity)
    pool = SimpleNamespace(
        backing=torch.zeros((8, 64), dtype=torch.uint8),
        pages={},
        layout=SimpleNamespace(
            row_bytes=64,
            region=lambda name: SimpleNamespace(offset=0, nbytes=8),
        ),
        config=SimpleNamespace(text={"compress_ratios": {}}),
    )
    weights = SimpleNamespace(
        shard=SimpleNamespace(world_size=4, rank=0),
        config=SimpleNamespace(draft=SimpleNamespace(experts=64, top_k=6, noise_token=0)),
    )
    forwards = DSparkForwards(
        decoder=object,
        scratch=_DummyDraftScratch,
        context=object,
        draft=object,
        markov=object,
    )

    for draft_tokens in (5, 7):
        spec = SpeculativePreparation(
            weights,
            tables=None,
            pool=pool,
            requests=requests,
            inputs=None,
            scratch=None,
            draft_tokens=draft_tokens,
            completion=None,
            frame=None,
            sequence=None,
            session=None,
            workspace=None,
            runtime=None,
            vocabulary_runtime=None,
            markov_runtime=None,
            coordinator=None,
            forwards=forwards,
        )
        bufs = spec.buffers()
        assert set(bufs) == {
            "copy_token_ids",
            "copy_token_count",
            "copy_expected_consumed",
            "copy_generation",
        }
        assert bufs["copy_token_ids"].shape == (capacity, 7)
        assert bufs["copy_token_ids"].dtype == torch.int64
        assert torch.equal(bufs["copy_token_ids"], torch.full((capacity, 7), -1, dtype=torch.int64))
        assert bufs["copy_token_count"].shape == (capacity,)
        assert bufs["copy_token_count"].dtype == torch.int32
        assert torch.equal(bufs["copy_token_count"], torch.zeros(capacity, dtype=torch.int32))
        assert bufs["copy_expected_consumed"].shape == (capacity,)
        assert bufs["copy_expected_consumed"].dtype == torch.int64
        assert torch.equal(bufs["copy_expected_consumed"], torch.full((capacity,), -1, dtype=torch.int64))
        assert bufs["copy_generation"].shape == (capacity,)
        assert bufs["copy_generation"].dtype == torch.int64
        assert torch.equal(bufs["copy_generation"], torch.zeros(capacity, dtype=torch.int64))
        # Persistent copy buffers must survive release_examples() for graph replay.
        spec.release_examples()
        assert spec.copy_ids is bufs["copy_token_ids"]
        assert spec.copy_count is bufs["copy_token_count"]

    # Wire through _prepare_speculation_catalog and deepseek41.contract.prepare.
    class _DraftRound:
        def __init__(self, owner, batch, width, name, **options):
            self.owner, self.batch, self.width, self.name, self.options = owner, batch, width, name, options

        def prepare(self):
            pass

    class _Packed:
        def __init__(self, source, target, prefix_limits, confidence_output, **options):
            self.source, self.target, self.prefix_limits = source, target, prefix_limits

    monkeypatch.setattr(parallel_draft, "PreparedParallelDraftRound", _DraftRound)
    monkeypatch.setattr(packed_speculation, "PreparedPackedSpeculativeRound", _Packed)

    config = PreparationConfig(max_active=4, speculative_widths=(1, 8), speculative_token_capacity=16)
    spec7 = SpeculativePreparation(
        weights, None, pool, requests, None, None, draft_tokens=config.draft_tokens,
        completion=None, frame=None, sequence=None, session=None, workspace=None,
        runtime=None, vocabulary_runtime=None, markov_runtime=None, coordinator=None,
        forwards=forwards,
    )
    signatures, buffers = {}, {}

    def add(name, forward, kind, rows, tokens, fields, *, enabled=None, packed_tokens=0, draft_tokens=None):
        signatures[name] = GraphSignature(
            kind, rows, tokens, tuple(fields), speculation_enabled=enabled,
            packed_token_capacity=packed_tokens, draft_tokens=draft_tokens,
            confidence_width=draft_tokens if "speculative_prefix_limits" in fields else 0,
        )

    def host_allocate(name, shape, dtype):
        tensor = torch.zeros(shape, dtype=dtype)
        return SimpleNamespace(device_view=tensor, host_view=tensor)

    _prepare_speculation_catalog(
        config,
        spec7,
        add=add,
        buffers=buffers,
        forward_inputs=("row_slot_ids", "row_generations", "live_rows", "query_lengths", "request_state_rows", "page_tables"),
        host_allocate=host_allocate,
        device="cpu",
    )
    for name in ("copy_token_ids", "copy_token_count", "copy_expected_consumed", "copy_generation"):
        assert name in buffers
        assert buffers[name] is spec7.buffers()[name]

    contract = prepare_ds41_contract(signatures, buffers, pool, page_tokens=128, draft_lag=0, rows=None)
    packed_graphs = [g for g in contract["graphs"] if signatures[g["graph"]].token_capacity > 1]
    width_one_graphs = [g for g in contract["graphs"] if signatures[g["graph"]].token_capacity == 1]
    assert len(packed_graphs) > 0 and len(width_one_graphs) > 0

    expected_copy_bindings = {
        "copy_token_ids": (pb2.COPY_TOKEN_IDS, pb2.REQUEST_ROW),
        "copy_token_count": (pb2.COPY_TOKEN_COUNT, pb2.REQUEST_ROW),
        "copy_expected_consumed": (pb2.COPY_CONSUMED_TOKENS, pb2.REQUEST_ROW),
        "copy_generation": (pb2.COPY_GENERATION, pb2.REQUEST_ROW),
    }
    for graph in packed_graphs:
        by_buffer = {entry["buffer"]: (entry["source"], entry["indexing"]) for entry in graph["inputs"]}
        for buf_name, expected in expected_copy_bindings.items():
            assert by_buffer[buf_name] == expected
    for graph in width_one_graphs:
        bound_names = {entry["buffer"] for entry in graph["inputs"]}
        assert bound_names.isdisjoint(expected_copy_bindings)


def test_bit_exact_fallback_when_copy_inactive_stale_or_mismatched():
    """Copy count 0, stale generation, wrong consumed cursor, grammar, and bad IDs match baseline bit-for-bit."""
    base_op, _, _ = _make_round_and_owner(with_copy=False)
    base_probs = torch.empty(2, 7, 16, dtype=torch.float32)
    base = base_op.propose(probabilities=base_probs)

    scenarios = [
        ("zero_count", lambda o: None),
        (
            "stale_generation",
            lambda o: (
                o.copy_ids.__setitem__((slice(None), slice(0, 4)), torch.tensor([9, 8, 7, 6])),
                o.copy_count.fill_(4),
                o.copy_expected.copy_(o.requests.consumed_tokens),
                o.copy_generation.copy_(o.requests.generation + 1),
            ),
        ),
        (
            "mismatched_consumed",
            lambda o: (
                o.copy_ids.__setitem__((slice(None), slice(0, 4)), torch.tensor([9, 8, 7, 6])),
                o.copy_count.fill_(4),
                o.copy_expected.copy_(o.requests.consumed_tokens + 1),
                o.copy_generation.copy_(o.requests.generation),
            ),
        ),
        (
            "grammar_constrained",
            lambda o: (
                o.requests.grammar_enabled.fill_(1),
                o.copy_ids.__setitem__((slice(None), slice(0, 3)), torch.tensor([5, 6, 7])),
                o.copy_count.fill_(3),
                o.copy_expected.copy_(o.requests.consumed_tokens),
                o.copy_generation.copy_(o.requests.generation),
            ),
        ),
        (
            "negative_count",
            lambda o: (
                o.copy_ids.fill_(3),
                o.copy_count.fill_(-2),
                o.copy_expected.copy_(o.requests.consumed_tokens),
                o.copy_generation.copy_(o.requests.generation),
            ),
        ),
        (
            "invalid_first_token",
            lambda o: (
                o.copy_ids.__setitem__((slice(None), 0), 16),  # out of vocab [0, 16)
                o.copy_ids.__setitem__((slice(None), 1), 4),
                o.copy_count.fill_(2),
                o.copy_expected.copy_(o.requests.consumed_tokens),
                o.copy_generation.copy_(o.requests.generation),
            ),
        ),
    ]
    for label, configure in scenarios:
        op, owner, _ = _make_round_and_owner(with_copy=True)
        configure(owner)
        probs = torch.empty(2, 7, 16, dtype=torch.float32)
        out = op.propose(probabilities=probs)
        assert torch.equal(out.drafts, base.drafts), label
        assert torch.equal(out.candidates, base.candidates), label
        assert torch.equal(out.valid, base.valid), label
        assert torch.equal(probs.view(torch.int32), base_probs.view(torch.int32)), label
        assert out.copy_offered_mask.tolist() == [0, 0], label


@pytest.mark.parametrize("k", [1, 3, 7])
def test_hybrid_copy_prefix_and_markov_tail_unfused_and_fused(k):
    """Copy prefix 0..k-1 gets q=1.0 point-mass and seeds Markov tail k..D-1 seamlessly."""
    vocab = 16
    seen_markov_predecessors = []

    def markov_bias(tokens):
        seen_markov_predecessors.append(tokens.clone())
        # Shift preferred token deterministically from predecessor so tail sensitivity is observable.
        bias = torch.zeros((tokens.numel(), vocab), dtype=torch.float32)
        target = (tokens + 3) % vocab
        bias.scatter_(1, target[:, None], 5.0)
        return bias

    op, owner, proposals = _make_round_and_owner(
        batch=2, width=8, capacity=4, vocab=vocab, with_copy=True, markov_fn=markov_bias,
    )
    copy_seq_row0 = [11, 2, 14, 7, 15, 0, 9]
    owner.copy_ids[0, :k] = torch.tensor(copy_seq_row0[:k], dtype=torch.int64)
    owner.copy_count[0] = k
    owner.copy_expected[0] = owner.requests.consumed_tokens[0]
    owner.copy_generation[0] = owner.requests.generation[0]
    # Row 1 has no copy proposals and must remain pure DSpark/Markov.
    owner.copy_count[1] = 0

    probs = torch.empty(2, 7, vocab, dtype=torch.float32)
    proposal = op.propose(probabilities=probs)

    # Row 0: positions 0..k-1 are exact copy tokens with q=1.0 point mass.
    assert proposal.drafts[0, :k].tolist() == copy_seq_row0[:k]
    assert proposal.candidates[0, 1:1 + k].tolist() == copy_seq_row0[:k]
    assert proposal.valid[0, :k].all()
    for pos in range(k):
        expected_one_hot = torch.zeros(vocab, dtype=torch.float32)
        expected_one_hot[copy_seq_row0[pos]] = 1.0
        assert torch.equal(probs[0, pos], expected_one_hot)
    assert proposal.copy_offered_mask.tolist() == [(1 << k) - 1, 0]

    # When k < 7, position k's Markov predecessor MUST be copy_seq_row0[k - 1]
    # and proposals[k] must observe the copy prefix in candidates[:, :k].
    if k < 7:
        assert seen_markov_predecessors[k][0].item() == copy_seq_row0[k - 1]
        assert proposals[k].seen_candidates[0, :k].tolist() == copy_seq_row0[:k]
        # Tail probabilities remain non-degenerate Markov softmax distributions summing to 1.
        tail_probs = probs[0, k:]
        assert torch.allclose(tail_probs.sum(dim=-1), torch.ones(7 - k), atol=1e-6)
        assert (tail_probs > 0).all()

    # Row 1: compare against a no-copy reference to confirm zero cross-row bleed.
    ref_op, _, _ = _make_round_and_owner(
        batch=2, width=8, capacity=4, vocab=vocab, with_copy=False, markov_fn=markov_bias,
    )
    ref_probs = torch.empty(2, 7, vocab, dtype=torch.float32)
    ref = ref_op.propose(probabilities=ref_probs)
    assert torch.equal(proposal.drafts[1], ref.drafts[1])
    assert torch.equal(probs[1].view(torch.int32), ref_probs[1].view(torch.int32))

    # Also test fused full_proposal path (PreparedMarkovProposals + PreparedDraftTokenBroadcast contract).
    fused_op, fused_owner, _ = _make_round_and_owner(batch=2, width=8, capacity=4, vocab=vocab, with_copy=True)
    fused_op.operations = None, (), None, None
    fused_drafts_ref = torch.tensor([[3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7]], dtype=torch.int64)
    gen = torch.Generator().manual_seed(99)
    fused_probs_ref = torch.rand((2, 7, vocab), generator=gen) + 0.1
    fused_probs_ref = fused_probs_ref / fused_probs_ref.sum(dim=-1, keepdim=True)

    class _FakeFullProposal:
        def run(self, base, pending, drafts, probabilities, valid, seed, base_offset):
            drafts.copy_(fused_drafts_ref)
            probabilities.copy_(fused_probs_ref)
            valid.fill_(True)

    class _FakeTokenBroadcast:
        def finish(self, drafts, valid, active, limits):
            positions = torch.arange(drafts.shape[1], device=drafts.device)
            return drafts, valid & active[:, None] & (positions[None, :] < limits[:, None] - 1), drafts[:, -1]

    fused_op.full_proposal = _FakeFullProposal()
    fused_op.proposal_block = _FakeTokenBroadcast()
    fused_owner.copy_ids[0, :k] = torch.tensor(copy_seq_row0[:k], dtype=torch.int64)
    fused_owner.copy_count[0] = k
    fused_owner.copy_expected[0] = fused_owner.requests.consumed_tokens[0]
    fused_owner.copy_generation[0] = fused_owner.requests.generation[0]

    fused_probs = torch.empty(2, 7, vocab, dtype=torch.float32)
    fused_out = fused_op.propose(probabilities=fused_probs)
    assert fused_out.drafts[0, :k].tolist() == copy_seq_row0[:k]
    assert fused_out.copy_offered_mask.tolist() == [(1 << k) - 1, 0]
    for pos in range(k):
        assert fused_probs[0, pos, copy_seq_row0[pos]].item() == 1.0
        assert fused_probs[0, pos].sum().item() == 1.0
    if k < 7:
        assert torch.equal(fused_out.drafts[0, k:], fused_drafts_ref[0, k:])
        assert torch.equal(fused_probs[0, k:].view(torch.int32), fused_probs_ref[0, k:].view(torch.int32))
    assert torch.equal(fused_out.drafts[1], fused_drafts_ref[1])
    assert torch.equal(fused_probs[1].view(torch.int32), fused_probs_ref[1].view(torch.int32))


def test_copy_prefix_truncation_on_corrupted_interior_token_and_query_limit():
    """Interior invalid token halts copy prefix contiguously; query_lengths bounds active draft width."""
    op, owner, _ = _make_round_and_owner(
        batch=2, width=8, capacity=4, vocab=16, with_copy=True,
        query_lengths=torch.tensor([8, 4], dtype=torch.int32),
    )
    # Row 0: copy_count=4 but index 2 is -1 -> contiguous prefix is k=2 (positions 0, 1).
    owner.copy_ids[0, :4] = torch.tensor([12, 13, -1, 15], dtype=torch.int64)
    owner.copy_count[0] = 4
    owner.copy_expected[0] = owner.requests.consumed_tokens[0]
    owner.copy_generation[0] = owner.requests.generation[0]
    # Row 1: copy_count=7 but query_lengths=4 -> active draft width is 3 -> k=3.
    owner.copy_ids[1, :7] = torch.tensor([4, 5, 6, 7, 8, 9, 10], dtype=torch.int64)
    owner.copy_count[1] = 7
    owner.copy_expected[1] = owner.requests.consumed_tokens[1]
    owner.copy_generation[1] = owner.requests.generation[1]

    probs = torch.empty(2, 7, 16, dtype=torch.float32)
    out = op.propose(probabilities=probs)
    assert out.copy_offered_mask.tolist() == [0b11, 0b111]
    assert out.drafts[0, :2].tolist() == [12, 13]
    assert out.drafts[0, 2].item() != -1
    assert out.drafts[1, :3].tolist() == [4, 5, 6]
    assert out.valid[1, :3].all() and (~out.valid[1, 3:]).all()


def test_vocabulary_parallel_sharded_point_mass_and_tile_masses():
    """Across 4 TP vocabulary shards, point-mass copy proposals and tile masses sum to 1.0 globally."""
    world, local_vocab, tile_block = 4, 512, 256
    vocab = world * local_vocab
    blocks = local_vocab // tile_block
    copy_tokens = [10, 520, 1100, 1900]  # One copy token in each of the 4 TP shards.
    k = len(copy_tokens)

    gen = torch.Generator().manual_seed(77)
    global_tail = torch.rand((1, 7, vocab), generator=gen, dtype=torch.float32) + 0.01
    global_tail = global_tail / global_tail.sum(dim=-1, keepdim=True)

    rank_probs = []
    rank_masses = []
    rank_drafts = []
    rank_masks = []
    for rank in range(world):
        op, owner, _ = _make_round_and_owner(batch=1, width=8, capacity=2, vocab=vocab, with_copy=True)
        owner.weights.shard = SimpleNamespace(rank=rank, world_size=world)
        owner.copy_ids[0, :k] = torch.tensor(copy_tokens, dtype=torch.int64)
        owner.copy_count[0] = k
        owner.copy_expected[0] = owner.requests.consumed_tokens[0]
        owner.copy_generation[0] = owner.requests.generation[0]

        op.vocabulary_parallel = True
        op.draft_mass_shape = (1, 7, blocks)
        op.operations = None, (), None, None
        seen_prev_at_k = []

        class _FakeVocabParallelSoftmax:
            def __init__(self, r):
                self.r = r

            def run_step(self, step, base, drafts, *, pending, seeds, offsets, probabilities, valid, tokens, tile_masses, token_limits):
                if step == k:
                    seen_prev_at_k.append(drafts[0, step - 1].item())
                shard = global_tail[:, step, self.r * local_vocab:(self.r + 1) * local_vocab]
                probabilities[:, step].copy_(shard)
                tile_masses[:, step].copy_(shard.view(1, blocks, tile_block).sum(dim=-1))
                valid[:, step].fill_(True)
                tokens[:, step].fill_(100 + step)

        op.full_proposal = _FakeVocabParallelSoftmax(rank)
        probs = torch.empty(1, 7, local_vocab, dtype=torch.float32)
        out = op.propose(probabilities=probs)
        assert seen_prev_at_k == [copy_tokens[k - 1]]
        rank_probs.append(probs)
        rank_masses.append(out.tile_masses)
        rank_drafts.append(out.drafts)
        rank_masks.append(out.copy_offered_mask)

    for rank in range(world):
        assert rank_drafts[rank][0, :k].tolist() == copy_tokens
        assert rank_drafts[rank][0, k:].tolist() == [100 + s for s in range(k, 7)]
        assert rank_masks[rank].tolist() == [(1 << k) - 1]

    reconstructed_probs = torch.cat(rank_probs, dim=-1)
    reconstructed_masses = torch.cat(rank_masses, dim=-1)
    for pos, tok in enumerate(copy_tokens):
        assert reconstructed_probs[0, pos, tok].item() == 1.0
        assert reconstructed_probs[0, pos].sum().item() == 1.0
        assert reconstructed_masses[0, pos].sum().item() == 1.0
        owner_rank = tok // local_vocab
        for r in range(world):
            expected_rank_sum = 1.0 if r == owner_rank else 0.0
            assert rank_probs[r][0, pos].sum().item() == expected_rank_sum
            assert rank_masses[r][0, pos].sum().item() == expected_rank_sum
    assert torch.allclose(reconstructed_probs[0, k:], global_tail[0, k:], atol=1e-7)
    assert torch.allclose(reconstructed_masses[0, k:].sum(dim=-1), torch.ones(7 - k), atol=1e-6)


def test_packed_truncation_clips_copy_offered_mask_in_completion_record():
    """When causal prefix allocation truncates query_lengths, verify clips copy_offered_mask to offered_drafts."""
    b, w, vocab, cap = 2, 6, 16, 4
    device = torch.device("cpu")
    requests = DeviceRequestState.allocate(device, vocab_size=vocab, capacity=cap)
    requests.status[:2] = RUNNING
    requests.generation[:2] = torch.tensor([11, 22], dtype=torch.int64)
    requests.consumed_tokens[:2] = torch.tensor([5, 8], dtype=torch.int64)
    requests.pending_token[:2] = torch.tensor([2, 3], dtype=torch.int64)
    requests.max_new_tokens[:2] = 64

    completion = CompletionOutputs.allocate((2, cap, VECTOR_TOKENS_FIELD + w), device=device)
    frame = torch.zeros(1, dtype=torch.int32, device=device)
    sequence = torch.tensor([7], dtype=torch.int64, device=device)
    rng = SpeculativeRngState.allocate(device, capacity=cap)

    # Row 0 had 5 copy drafts offered during propose (mask=0b11111=31), then allocator truncated query_lengths to 3 (offered=2).
    # Row 1 had 4 copy drafts offered during propose (mask=0b1111=15), then allocator truncated query_lengths to 1 (offered=0).
    inputs = SimpleNamespace(
        row_slot_ids=torch.tensor([0, 1], dtype=torch.int32),
        row_generations=torch.tensor([11, 22], dtype=torch.int64),
        query_lengths=torch.tensor([3, 1], dtype=torch.int32),
        request_state_rows=torch.arange(cap, dtype=torch.int64)[:, None],
        page_tables=torch.zeros((cap, 2), dtype=torch.int32),
        live_rows=torch.tensor([2], dtype=torch.int32),
    )
    candidates = torch.tensor([[2, 4, 5, 6, 7, 8], [3, 9, 10, 11, 12, 13]], dtype=torch.int64)
    drafts = candidates[:, 1:].clone()
    proposal = DraftProposalState(
        before=torch.tensor([5, 8], dtype=torch.int64),
        active=torch.tensor([True, True]),
        candidates=candidates,
        drafts=drafts,
        valid=torch.ones((b, w - 1), dtype=torch.bool),
        base_offset=torch.zeros(b, dtype=torch.int64),
        metadata=None,
        copy_offered_mask=torch.tensor([0b11111, 0b1111], dtype=torch.int64),
    )

    acceptance = SpeculativeAcceptance.allocate(b, w, device)
    # Row 0 accepts 2 copy drafts + bonus token (emitted=3); row 1 emits 1 token (offered=0, accepted=0).
    acceptance.token_ids[0, :3] = torch.tensor([4, 5, 14], dtype=torch.int64)
    acceptance.token_ids[1, :1] = torch.tensor([15], dtype=torch.int64)
    acceptance.proposed_count.copy_(torch.tensor([3, 1], dtype=torch.int32))
    acceptance.accepted_drafts.copy_(torch.tensor([2, 0], dtype=torch.int32))
    acceptance.offered_drafts.copy_(torch.tensor([2, 0], dtype=torch.int32))
    acceptance.valid.fill_(True)
    acceptance.rng_increment.fill_(8)
    acceptance.survival[0, :2] = torch.tensor([0.9, 0.8])

    class _NoopLogprobs:
        def prepare(self):
            pass

        def functional(self, **kwargs):
            return self

        def __call__(self):
            pass

        def run(self, logits, **kwargs):
            pass

    commit = PreparedSpeculativeCommit(
        requests,
        acceptance,
        candidates,
        inputs.row_slot_ids,
        inputs.row_generations,
        inputs.live_rows,
        completion,
        frame,
        sequence,
        logprobs=_NoopLogprobs(),
        rng=rng,
        draft_context_lag=0,
        token_limits=inputs.query_lengths,
    )
    commit.prepare()
    commit.functional()

    pool = SimpleNamespace(last_hidden=torch.full((cap, 4), -1.0))
    commit_calls, context_calls = [], []

    class _Decoder:
        def __call__(self, metadata):
            hidden = torch.arange(b * w * 4, dtype=torch.float32).view(b * w, 4)
            logits = torch.zeros((b * w, vocab), dtype=torch.float32)
            return hidden, logits, torch.ones((b * w, 4)), {"rec": True}

        def commit(self, counts, records, metadata):
            commit_calls.append(counts.tolist())

    class _Target:
        layout = {"probabilities": ((b, w, vocab), torch.float32), "valid": ((b, w), torch.bool)}

        def run(self, scratch, logits, *, candidates):
            scratch["probabilities"].fill_(1.0 / vocab)
            scratch["valid"].fill_(True)

    class _Verify:
        layout = {}

        def run(self, scratch, probabilities, target, valid, *, drafts, draft_valid, disjoint):
            return acceptance

    op = object.__new__(PreparedParallelDraftRound)
    op.owner = SimpleNamespace(
        inputs=inputs,
        requests=requests,
        weights=SimpleNamespace(config=SimpleNamespace(text={"hidden_size": 4})),
        pool=pool,
    )
    op.batch, op.width, op.target_token_capacity = b, w, b * w
    op.vocabulary_parallel = False
    op.operations = _Target(), (), _Verify(), commit
    op.decoder = _Decoder()
    op.context = lambda aux, counts, metadata: context_calls.append(counts.tolist())

    _, _, committed = op.verify(probabilities=torch.full((b, w - 1, vocab), 1.0 / vocab), proposal=proposal)
    assert committed.emitted_count.tolist() == [3, 1]
    assert commit_calls == [[3, 1]]
    assert context_calls == [[3, 1]]
    records = completion.records[0]
    # Field 9 = offered_drafts, field 10 = accepted_drafts, field 11 = copy_offered_mask.
    assert records[0, 9].item() == 2 and records[0, 10].item() == 2
    assert records[0, 11].item() == 0b11
    assert 0 <= records[0, 11].item() < (1 << records[0, 9].item())
    assert records[1, 9].item() == 0 and records[1, 10].item() == 0
    assert records[1, 11].item() == 0
    assert 0 <= records[1, 11].item() < (1 << records[1, 9].item())


def test_lossless_verification_exact_target_distribution_and_negative_sensitivity():
    """Point-mass q=1.0 copy proposals preserve exact target distribution (greedy and stochastic) and survival."""
    vocab = 8

    def exact_marginal_distribution(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Compute exact emitted marginal Pr[emit v] for single-step speculative rejection sampling."""
        # For each draft token x drawn with probability q(x):
        #   accept x with prob min(1, p(x) / q(x)) -> contributes min(p(x), q(x)) to emit(x).
        # Total reject probability R = 1 - sum_x min(p(x), q(x)).
        # On reject, sample v from residual r(v) = max(p(v) - q(v), 0) / R.
        accepted = torch.minimum(p, q)
        residual = torch.clamp(p - q, min=0.0)
        z = residual.sum()
        if z.item() == 0.0:
            return accepted
        reject_prob = 1.0 - accepted.sum()
        return accepted + reject_prob * (residual / z)

    # 1. Greedy limit (one-hot target p = delta_{x*}): bit-identical whether copy matches or mismatches.
    target_greedy = torch.zeros(vocab, dtype=torch.float32)
    target_greedy[5] = 1.0
    for copy_tok in (5, 2):
        q_copy = torch.zeros(vocab, dtype=torch.float32)
        q_copy[copy_tok] = 1.0
        emitted = exact_marginal_distribution(target_greedy, q_copy)
        assert torch.equal(emitted, target_greedy)
        survival = torch.minimum(target_greedy[copy_tok] / q_copy[copy_tok], torch.tensor(1.0)).item()
        assert survival == (1.0 if copy_tok == 5 else 0.0)

    # 2. Stochastic target distribution across hybrid copy-prefix (q=1.0) + Markov-tail steps.
    gen = torch.Generator().manual_seed(2026)
    p_steps = torch.softmax(torch.randn((7, vocab), generator=gen), dim=-1)
    q_markov = torch.softmax(torch.randn((7, vocab), generator=gen), dim=-1)
    copy_tokens = [3, 0, 7]
    for step in range(7):
        if step < len(copy_tokens):
            q = torch.zeros(vocab, dtype=torch.float32)
            q[copy_tokens[step]] = 1.0
            # Acceptance probability for copy token c under q=delta_c is exactly p(c).
            alpha = min(1.0, (p_steps[step, copy_tokens[step]] / q[copy_tokens[step]]).item())
            assert alpha == pytest.approx(p_steps[step, copy_tokens[step]].item(), abs=0.0)
            # Residual distribution puts exact 0.0 mass on rejected copy token c.
            residual = torch.clamp(p_steps[step] - q, min=0.0)
            assert residual[copy_tokens[step]].item() == 0.0
        else:
            q = q_markov[step]
        emitted = exact_marginal_distribution(p_steps[step], q)
        assert torch.allclose(emitted, p_steps[step], atol=1e-7, rtol=0.0)

    # 3. Monte Carlo check (50,000 draws) on a stochastic target with copy proposal q = delta_c.
    p = p_steps[0]
    c = copy_tokens[0]
    q = torch.zeros(vocab, dtype=torch.float32)
    q[c] = 1.0
    n_draws = 50_000
    u = torch.rand(n_draws, generator=gen)
    accept = u < p[c]
    residual_probs = torch.clamp(p - q, min=0.0)
    residual_probs = residual_probs / residual_probs.sum()
    bonus = torch.multinomial(residual_probs, n_draws, replacement=True, generator=gen)
    emitted_tokens = torch.where(accept, torch.full_like(bonus, c), bonus)
    empirical = torch.bincount(emitted_tokens, minlength=vocab).float() / n_draws
    assert torch.allclose(empirical, p, atol=0.015)

    # 4. Negative sensitivity: corrupted copy_expected or copy_generation MUST reject copy proposal.
    owner = SimpleNamespace(
        requests=SimpleNamespace(
            vocab_size=vocab,
            generation=torch.tensor([42], dtype=torch.int64),
            grammar_enabled=torch.tensor([0], dtype=torch.int32),
        ),
        copy_ids=torch.tensor([[3, 1, 2, -1, -1, -1, -1]], dtype=torch.int64),
        copy_count=torch.tensor([3], dtype=torch.int32),
        copy_expected=torch.tensor([100], dtype=torch.int64),
        copy_generation=torch.tensor([42], dtype=torch.int64),
    )
    active = torch.tensor([True])
    safe = torch.tensor([0], dtype=torch.int64)
    before = torch.tensor([100], dtype=torch.int64)
    limits = torch.tensor([8], dtype=torch.int32)

    valid_mask, _ = _resolve_copy_proposals(owner, active, safe, before, limits, 7)
    assert valid_mask[0, :3].all() and (~valid_mask[0, 3:]).all()

    # Corrupt expected consumed cursor -> must reject completely.
    owner.copy_expected[0] = 101
    bad_expected, _ = _resolve_copy_proposals(owner, active, safe, before, limits, 7)
    assert not bad_expected.any()
    owner.copy_expected[0] = 100

    # Corrupt generation -> must reject completely.
    owner.copy_generation[0] = 43
    bad_gen, _ = _resolve_copy_proposals(owner, active, safe, before, limits, 7)
    assert not bad_gen.any()


def test_fx_traceable_copy_overlay_has_no_data_dependent_control_flow():
    """Copy resolution and point-mass overlay trace cleanly via make_fx without host syncs."""
    from torch.fx.experimental.proxy_tensor import make_fx

    b, d, v = 2, 7, 16

    def overlay(
        active,
        before,
        limits,
        req_generation,
        grammar_enabled,
        copy_ids,
        copy_count,
        copy_expected,
        copy_generation,
        drafts,
        probabilities,
        valid,
    ):
        owner = SimpleNamespace(
            requests=SimpleNamespace(
                vocab_size=v,
                generation=req_generation,
                grammar_enabled=grammar_enabled,
            ),
            copy_ids=copy_ids,
            copy_count=copy_count,
            copy_expected=copy_expected,
            copy_generation=copy_generation,
        )
        safe = torch.arange(b, dtype=torch.int64)
        copy_active, copy_tokens = _resolve_copy_proposals(owner, active, safe, before, limits, d)
        mask = _apply_copy_proposals(drafts, probabilities, valid, copy_active, copy_tokens)
        return drafts, probabilities, valid, mask

    active = torch.tensor([True, True])
    before = torch.tensor([20, 30], dtype=torch.int64)
    limits = torch.tensor([8, 8], dtype=torch.int32)
    req_gen = torch.tensor([5, 6], dtype=torch.int64)
    grammar = torch.zeros(2, dtype=torch.int32)
    copy_ids = torch.tensor([[7, 8, 9, -1, -1, -1, -1], [-1] * 7], dtype=torch.int64)
    copy_count = torch.tensor([3, 0], dtype=torch.int32)
    copy_expected = torch.tensor([20, -1], dtype=torch.int64)
    copy_generation = torch.tensor([5, 0], dtype=torch.int64)
    drafts = torch.full((b, d), 1, dtype=torch.int64)
    probs = torch.full((b, d, v), 1.0 / v, dtype=torch.float32)
    valid = torch.ones((b, d), dtype=torch.bool)

    traced = make_fx(overlay)(
        active, before, limits, req_gen, grammar,
        copy_ids, copy_count, copy_expected, copy_generation,
        drafts.clone(), probs.clone(), valid.clone(),
    )
    # Re-execute traced FX graph with completely different runtime copy inputs (row 0 off, row 1 k=5).
    copy_ids_2 = torch.tensor([[-1] * 7, [2, 4, 6, 8, 10, -1, -1]], dtype=torch.int64)
    copy_count_2 = torch.tensor([0, 5], dtype=torch.int32)
    copy_expected_2 = torch.tensor([-1, 30], dtype=torch.int64)
    copy_generation_2 = torch.tensor([0, 6], dtype=torch.int64)

    out_drafts, out_probs, out_valid, out_mask = traced(
        active, before, limits, req_gen, grammar,
        copy_ids_2, copy_count_2, copy_expected_2, copy_generation_2,
        drafts.clone(), probs.clone(), valid.clone(),
    )
    assert out_drafts[0].tolist() == [1] * 7
    assert out_drafts[1, :5].tolist() == [2, 4, 6, 8, 10]
    assert out_drafts[1, 5:].tolist() == [1, 1]
    assert out_probs[0, 0].sum().item() == pytest.approx(1.0)
    assert out_probs[1, 0, 2].item() == 1.0 and out_probs[1, 0].sum().item() == 1.0
    assert out_probs[1, 4, 10].item() == 1.0 and out_probs[1, 4].sum().item() == 1.0
    assert out_probs[1, 5].sum().item() == pytest.approx(1.0)
    assert out_valid.all()
    assert out_mask.tolist() == [0, 0b11111]
