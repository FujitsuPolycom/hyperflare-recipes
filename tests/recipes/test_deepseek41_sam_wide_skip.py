# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU and FX verification of lossless W=16/24 DSpark-skip wide copy-verify rounds in DeepSeek-V4.1."""

from types import SimpleNamespace

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from ignition.components.completion import CompletionOutputs, VECTOR_TOKENS_FIELD
from ignition.components.sampling import DeviceRequestState, RUNNING
from ignition.preparation import runtime_core_pb2 as pb2
from ignition.preparation.execution import GraphSignature

from hyperflare_recipes.components.dspark import DSparkForwards, SpeculativePreparation
from hyperflare_recipes.components.parallel_draft import (
    PreparedCopyOnlySpeculativeRound,
    PreparedParallelDraftRound,
)
from hyperflare_recipes.components.speculation import (
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


class _DummyHostView:
    def __init__(self, shape, dtype):
        self.device_view = torch.zeros(shape, dtype=dtype)
        self.host_view = self.device_view


class _RecordingTarget:
    """Target distribution operator computing softmax over logits."""

    def __init__(self, batch, width, vocab):
        self.layout = {
            "probabilities": ((batch, width, vocab), torch.float32),
            "valid": ((batch, width), torch.bool),
        }
        self.last_token_limits = None

    def run(self, scratch, logits, *, candidates=None, token_limits=None):
        self.last_token_limits = None if token_limits is None else token_limits.clone()
        probs = torch.softmax(logits.float(), dim=-1)
        scratch["probabilities"].copy_(probs)
        if token_limits is None:
            scratch["valid"].fill_(True)
        else:
            cols = torch.arange(logits.shape[1], device=logits.device)
            scratch["valid"].copy_(cols[None, :] < token_limits[:, None])
        return scratch["probabilities"]


class _ExactGreedyOrDeterministicVerify:
    """Lossless q=1.0 chain verifier supporting exact greedy argmax or fixed rejection index."""

    def __init__(self, batch, width, vocab, *, force_reject_at=None, bonus_token=0):
        self.batch, self.width, self.vocab = batch, width, vocab
        self.force_reject_at = force_reject_at
        self.bonus_token = bonus_token
        self.layout = {}
        self.seen_draft_probs = None
        self.seen_draft_valid = None
        self.seen_drafts = None
        self.seen_token_limits = None

    def run(
        self,
        scratch,
        draft_probs,
        target_probs,
        target_valid,
        *,
        disjoint=True,
        drafts=None,
        draft_valid=None,
        token_limits=None,
    ):
        b, w, _ = target_probs.shape
        device = target_probs.device
        self.seen_draft_probs = draft_probs.clone()
        self.seen_draft_valid = draft_valid.clone()
        self.seen_drafts = drafts.clone()
        self.seen_token_limits = None if token_limits is None else token_limits.clone()

        out = SpeculativeAcceptance.allocate(b, w, device)
        limits = (
            torch.full((b,), w, dtype=torch.int32, device=device)
            if token_limits is None
            else token_limits.to(torch.int32)
        )
        offered = (limits - 1).clamp(0, w - 1)
        cols = torch.arange(w - 1, device=device)
        used = cols[None, :] < offered[:, None]

        # Exact q = 1.0 greedy verification: accept draft i iff draft_i == argmax(p_i) and q_i(draft_i) == 1.0.
        greedy_ids = target_probs.argmax(dim=-1)
        gathered_q = draft_probs.gather(-1, drafts.clamp(0, self.vocab - 1).unsqueeze(-1)).squeeze(-1)
        gathered_p = target_probs[:, :-1].gather(-1, drafts.clamp(0, self.vocab - 1).unsqueeze(-1)).squeeze(-1)
        if self.force_reject_at is None:
            step_accept = used & draft_valid & (gathered_q == 1.0) & (drafts == greedy_ids[:, :-1])
        else:
            reject_pos = torch.as_tensor(self.force_reject_at, dtype=torch.int64, device=device)
            if reject_pos.ndim == 0:
                reject_pos = reject_pos.expand(b)
            step_accept = used & draft_valid & (gathered_q == 1.0) & (cols[None, :] < reject_pos[:, None])
        prefix_accept = step_accept.to(torch.int32).cumprod(dim=1)
        accepted = prefix_accept.sum(dim=1).to(torch.int32)

        # Build emitted token_ids: accepted drafts followed by bonus/correction token at index `accepted`.
        if self.force_reject_at is None:
            bonus = greedy_ids.gather(1, accepted.long().clamp(0, w - 1)[:, None])[:, 0]
        else:
            bonus = torch.full((b,), self.bonus_token, dtype=torch.int64, device=device)
        pos = torch.arange(w, device=device)
        padded_drafts = torch.cat([drafts, torch.full((b, 1), -1, dtype=torch.int64, device=device)], dim=1)
        emitted = torch.where(
            pos[None, :] < accepted[:, None],
            padded_drafts,
            torch.where(pos[None, :] == accepted[:, None], bonus[:, None], -1),
        )

        # Compute exact survival prefix product: for q = 1.0, alpha_i = min(1, p_i(d_i) / 1.0) = p_i(d_i).
        ratio = torch.where(gathered_q > 0, (gathered_p / gathered_q).clamp(max=1.0), 0.0)
        prefixes = [ratio[:, 0]]
        for d in range(1, w - 1):
            prefixes.append(prefixes[-1] * ratio[:, d])
        survival = torch.stack(prefixes, dim=-1)

        out.token_ids.copy_(emitted)
        out.offered_drafts.copy_(offered)
        out.accepted_drafts.copy_(accepted)
        out.proposed_count.copy_(accepted + 1)
        out.valid.copy_(limits >= 1)
        out.rng_increment.copy_(torch.where(limits >= 1, 4 * ((w + 4) // 4), 0))
        out.survival.copy_(torch.where(used, survival, -1.0))
        return out


class _RecordingDecoder:
    def __init__(self, batch, width, vocab, hidden_size, logits):
        self.batch, self.width, self.vocab, self.hidden_size = batch, width, vocab, hidden_size
        self._logits = logits
        self.calls = 0
        self.commit_counts = None
        self.last_metadata = None

    def __call__(self, metadata):
        self.calls += 1
        self.last_metadata = metadata
        hidden = torch.zeros((self.batch * self.width, self.hidden_size), dtype=torch.bfloat16)
        auxiliary = torch.ones((self.batch * self.width, self.hidden_size), dtype=torch.bfloat16)
        return hidden, self._logits.view(self.batch * self.width, self.vocab), auxiliary, ("records",)

    def commit(self, counts, records, metadata):
        self.commit_counts = counts.clone()


class _RecordingContext:
    def __init__(self):
        self.calls = 0
        self.last_counts = None
        self.last_auxiliary_shape = None

    def __call__(self, auxiliary, counts, metadata):
        self.calls += 1
        self.last_counts = counts.clone()
        self.last_auxiliary_shape = tuple(auxiliary.shape)


class _DummyLogprobs:
    def prepare(self):
        return None

    def functional(self, **_kwargs):
        return self

    def run(self, *_args, **_kwargs):
        return None

    def __call__(self, *_args, **_kwargs):
        return None


def _make_copy_only_round(
    *,
    batch=1,
    width=24,
    capacity=4,
    vocab=32,
    hidden_size=8,
    logits=None,
    query_lengths=None,
    row_slot_ids=None,
    consumed=None,
    generations=None,
    grammar_enabled=None,
    active_lengths=None,
    force_reject_at=None,
    bonus_token=0,
    vocabulary_parallel=False,
    rank=0,
    world_size=1,
):
    if logits is None:
        logits = torch.zeros((batch, width, vocab), dtype=torch.float32)
    if query_lengths is None:
        query_lengths = torch.full((batch,), width, dtype=torch.int32)
    if row_slot_ids is None:
        row_slot_ids = torch.arange(batch, dtype=torch.int32)
    if consumed is None:
        consumed = torch.tensor([16 + i * 4 for i in range(capacity)], dtype=torch.int64)
    if generations is None:
        generations = torch.tensor([100 + i for i in range(capacity)], dtype=torch.int64)
    if grammar_enabled is None:
        grammar_enabled = torch.zeros(capacity, dtype=torch.int32)
    if active_lengths is None:
        active_lengths = torch.ones(batch, dtype=torch.int32)

    requests = DeviceRequestState.allocate(torch.device("cpu"), vocab_size=vocab, capacity=capacity)
    requests.status.fill_(RUNNING)
    requests.generation.copy_(generations)
    requests.consumed_tokens.copy_(consumed)
    requests.generated_tokens.fill_(1)
    requests.max_new_tokens.fill_(128)
    requests.pending_token.copy_(torch.tensor([1 + i for i in range(capacity)], dtype=torch.int64))
    requests.grammar_enabled.copy_(grammar_enabled)

    inputs = SimpleNamespace(
        row_slot_ids=row_slot_ids.clone(),
        row_generations=generations[row_slot_ids.long().clamp(0, capacity - 1)].clone(),
        query_lengths=query_lengths.clone(),
        request_state_rows=torch.arange(1, capacity + 1, dtype=torch.int64)[:, None],
        page_tables=torch.zeros((capacity, 4), dtype=torch.int32),
        live_rows=torch.tensor([batch], dtype=torch.int32),
    )
    completion = CompletionOutputs.allocate((2, capacity, VECTOR_TOKENS_FIELD + max(24, width)), device=torch.device("cpu"))
    rng = SpeculativeRngState.allocate(torch.device("cpu"), capacity=capacity)
    pool = SimpleNamespace(last_hidden=torch.zeros((capacity + 1, hidden_size), dtype=torch.bfloat16))
    owner = SimpleNamespace(
        requests=requests,
        inputs=inputs,
        pool=pool,
        completion=completion,
        frame=torch.zeros(1, dtype=torch.int32),
        sequence=torch.tensor([7], dtype=torch.int64),
        rng=rng,
        weights=SimpleNamespace(
            shard=SimpleNamespace(rank=rank, world_size=world_size),
            config=SimpleNamespace(text={"hidden_size": hidden_size}, draft=SimpleNamespace(noise_token=0)),
        ),
        copy_ids=torch.full((capacity, max(7, width - 1)), -1, dtype=torch.int64),
        copy_count=torch.zeros(capacity, dtype=torch.int32),
        copy_expected=torch.full((capacity,), -1, dtype=torch.int64),
        copy_generation=torch.zeros(capacity, dtype=torch.int64),
    )

    local_vocab = vocab // world_size if vocabulary_parallel else vocab
    target_round = object.__new__(PreparedParallelDraftRound)
    target_round.owner = owner
    target_round.batch = batch
    target_round.width = width
    target_round.name = f"copy-verification{batch}x{width}"
    target_round.target_token_capacity = batch * width
    target_round.vocabulary_parallel = vocabulary_parallel
    target_round.generate_proposals = False
    target_round.proposal_only = False
    target_round.draft = None
    target_round.markov = None
    target_round.full_proposal = None
    target_round.probability_shape = (batch, width - 1, local_vocab)
    target_round.draft_mass_shape = (batch, width - 1, (local_vocab + 255) // 256)
    target_round.target_mass_shape = (batch, width, (local_vocab + 255) // 256)
    target_round.validator = SimpleNamespace(
        validate=lambda: (active_lengths, True),
        begin=lambda val, present=True: SimpleNamespace(lengths=val),
    )
    target_round.validation = lambda x: x

    decoder = _RecordingDecoder(batch, width, local_vocab, hidden_size, logits)
    context = _RecordingContext()
    target_round.decoder = decoder
    target_round.context = context

    if vocabulary_parallel:
        vp_state = {}

        class _VPTarget:
            def run(self, local_logits, drafts, *, tile_masses=None, token_limits=None):
                vp_state["target_limits"] = token_limits.clone()
                probs = torch.full_like(local_logits, 1.0 / vocab)
                if tile_masses is not None:
                    tile_masses.fill_(float(local_vocab) / float(vocab))
                cols = torch.arange(width, device=local_logits.device)
                valid = cols[None, :] < token_limits[:, None]
                return probs, valid, None

        class _VPVerify:
            def run(
                self,
                draft_probs,
                drafts,
                target_probs,
                draft_valid,
                target_valid,
                *,
                token_limits=None,
                draft_tile_masses=None,
                target_tile_masses=None,
            ):
                vp_state["draft_probs"] = draft_probs.clone()
                vp_state["draft_tile_masses"] = draft_tile_masses.clone()
                vp_state["draft_valid"] = draft_valid.clone()
                vp_state["drafts"] = drafts.clone()
                vp_state["token_limits"] = token_limits.clone()
                out = SpeculativeAcceptance.allocate(batch, width, local_logits_device := draft_probs.device)
                offered = (token_limits - 1).clamp(0, width - 1)
                out.offered_drafts.copy_(offered)
                out.accepted_drafts.copy_(offered)
                out.proposed_count.copy_(offered + 1)
                out.valid.copy_(token_limits >= 1)
                out.rng_increment.copy_(torch.where(token_limits >= 1, 4 * ((width + 4) // 4), 0))
                padded = torch.cat([drafts, torch.zeros((batch, 1), dtype=torch.int64, device=local_logits_device)], dim=1)
                pos = torch.arange(width, device=local_logits_device)
                out.token_ids.copy_(torch.where(pos[None, :] <= offered[:, None], padded, -1))
                out.survival.fill_(-1.0)
                return out

        vp_target = _VPTarget()
        vp_verify = _VPVerify()
        dummy_acceptance = SpeculativeAcceptance.allocate(batch, width, torch.device("cpu"))
        commit = PreparedSpeculativeCommit(
            requests,
            dummy_acceptance,
            torch.zeros((batch, width), dtype=torch.int64),
            inputs.row_slot_ids[:batch],
            inputs.row_generations[:batch],
            inputs.live_rows,
            completion,
            owner.frame,
            owner.sequence,
            logprobs=_DummyLogprobs(),
            rng=rng,
            draft_context_lag=0,
            token_limits=inputs.query_lengths[:batch],
        )
        commit.functional()
        target_round.operations = (vp_target, (), vp_verify, commit)
        copy_round = PreparedCopyOnlySpeculativeRound(target_round, name=f"copy-speculative{batch}x{width}")
        return copy_round, owner, decoder, context, vp_state

    target_dist = _RecordingTarget(batch, width, vocab)
    verifier = _ExactGreedyOrDeterministicVerify(
        batch, width, vocab, force_reject_at=force_reject_at, bonus_token=bonus_token
    )
    dummy_acceptance = SpeculativeAcceptance.allocate(batch, width, torch.device("cpu"))
    commit = PreparedSpeculativeCommit(
        requests,
        dummy_acceptance,
        torch.zeros((batch, width), dtype=torch.int64),
        inputs.row_slot_ids[:batch],
        inputs.row_generations[:batch],
        inputs.live_rows,
        completion,
        owner.frame,
        owner.sequence,
        logits=torch.zeros((batch, width, vocab), dtype=torch.float32),
        rng=rng,
        draft_context_lag=0,
        token_limits=inputs.query_lengths[:batch],
    )
    commit.logprobs = SimpleNamespace(run=lambda *a, **k: None, __call__=lambda *a, **k: None, prepare=lambda: None, functional=lambda **_: None)
    commit.functional()
    target_round.operations = (target_dist, (), verifier, commit)
    copy_round = PreparedCopyOnlySpeculativeRound(target_round, name=f"copy-speculative{batch}x{width}")
    return copy_round, owner, decoder, context, verifier


def test_copy_verify_shapes_config_and_catalog_registration(monkeypatch):
    from dataclasses import replace
    from ignition.parameters import SpeculationParameters

    base = PreparationConfig()

    def _make_cfg(*, max_active=32, dspark=True, draft_tokens=7, token_capacity=128,
                  graph_budget=64, copy_verify_shapes=()):
        spec = (
            SpeculationParameters(
                draft_tokens=draft_tokens, max_draft_tokens=7, token_capacity=token_capacity,
                packed_verification=True, draft_prefix_selection="gpu", graph_budget=graph_budget,
            )
            if dspark else None
        )
        return replace(
            base,
            parameters=replace(base.parameters, max_concurrency=max_active, speculation=spec),
            copy_verify_shapes=copy_verify_shapes,
        )

    # Default config stays 100% identical to baseline when copy_verify_shapes=().
    default_cfg = _make_cfg()
    assert default_cfg.copy_verify_shapes == ()
    assert default_cfg.max_copy_drafts == 0
    assert default_cfg.all_speculative_widths == default_cfg.speculative_widths
    assert default_cfg.completion_token_capacity == 8
    assert default_cfg.decode_tokens == 128

    # Opt-in wide copy-verify shapes resize copy_ids and completion capacity with 0 B extra decode_tokens.
    wide_shapes = ((1, 16), (1, 24), (2, 16), (4, 16))
    wide_cfg = _make_cfg(max_active=4, token_capacity=64, graph_budget=32, copy_verify_shapes=wide_shapes)
    assert wide_cfg.max_copy_drafts == 23
    assert wide_cfg.all_speculative_widths == (*wide_cfg.speculative_widths, 16, 24)
    assert wide_cfg.completion_token_capacity == 24
    assert wide_cfg.decode_tokens == 64

    # Invalid configurations are rejected explicitly.
    with pytest.raises(ValueError, match="exceed base MTP width"):
        _make_cfg(max_active=4, copy_verify_shapes=((1, 8),))
    with pytest.raises(ValueError, match="within 64"):
        _make_cfg(max_active=4, copy_verify_shapes=((1, 65),))
    with pytest.raises(ValueError, match="speculative_token_capacity"):
        _make_cfg(max_active=4, token_capacity=32, graph_budget=16, copy_verify_shapes=((4, 16),))
    with pytest.raises(ValueError, match="enabled DSpark"):
        _make_cfg(max_active=4, dspark=False, copy_verify_shapes=((1, 16),))
    with pytest.raises(ValueError, match="distinct"):
        _make_cfg(max_active=4, copy_verify_shapes=((1, 16), (1, 16)))

    # Verify SpeculativePreparation buffer sizing and _prepare_speculation_catalog wiring.
    device = torch.device("cpu")
    pool = SimpleNamespace(
        backing=torch.zeros((8, 16), dtype=torch.uint8, device=device),
        pages={},
        layout=SimpleNamespace(
            row_bytes=16,
            region=lambda name: SimpleNamespace(offset=0, nbytes=8),
        ),
        config=SimpleNamespace(text={"compress_ratios": []}),
    )
    requests = SimpleNamespace(capacity=4, generation=torch.zeros(4, dtype=torch.int64, device=device))
    weights = SimpleNamespace(
        config=SimpleNamespace(draft=SimpleNamespace(experts=8, top_k=2)),
        shard=SimpleNamespace(world_size=1),
    )
    spec = SpeculativePreparation(
        weights, None, pool, requests, None, None,
        draft_tokens=wide_cfg.draft_tokens,
        copy_draft_tokens=wide_cfg.max_copy_drafts,
        completion=None, frame=None, sequence=None,
        session=None, workspace=None,
        runtime=None, vocabulary_runtime=None, markov_runtime=None,
        coordinator=None,
        forwards=DSparkForwards(None, _DummyDraftScratch, None, None, None),
    )
    assert spec.copy_ids.shape == (4, 23)

    constructed = []

    class _FakeParallelDraftRound:
        def __init__(self, owner, batch, width, name, *, generate_proposals=True, proposal_only=False,
                     target_token_capacity=None):
            self.owner, self.batch, self.width, self.name = owner, batch, width, name
            self.generate_proposals = generate_proposals
            self.proposal_only = proposal_only
            self.target_token_capacity = target_token_capacity
            self.draft = None if (not generate_proposals or width == 1) else object()
            self.markov = None
            constructed.append((name, batch, width, generate_proposals, proposal_only, target_token_capacity))

        def prepare(self):
            pass

    class _FakePackedRound:
        def __init__(self, proposal, target, prefix_limits, confidence, *, name):
            self.name = name

    import hyperflare_recipes.components.parallel_draft as pd_mod
    import hyperflare_recipes.components.packed_speculation as ps_mod

    monkeypatch.setattr(pd_mod, "PreparedParallelDraftRound", _FakeParallelDraftRound)
    monkeypatch.setattr(ps_mod, "PreparedPackedSpeculativeRound", _FakePackedRound)

    signatures = {}
    forwards = {}
    buffers = {}

    def add(name, forward, kind, rows, tokens, fields, *, enabled=None, packed_tokens=0, draft_tokens=None,
            draft_source=1):
        forwards[name] = forward
        source_kwargs = {"draft_source": draft_source} if draft_source != 1 else {}
        signatures[name] = GraphSignature(
            kind, rows, tokens, tuple(fields), speculation_enabled=enabled,
            packed_token_capacity=packed_tokens, draft_tokens=draft_tokens,
            confidence_width=draft_tokens if "speculative_prefix_limits" in fields else 0,
            **source_kwargs,
        )

    forward_inputs = ("request_state_rows", "row_slot_ids", "row_generations", "live_rows",
                      "page_tables", "sampling_grammar_mask", "sample_enabled", "query_lengths")
    _prepare_speculation_catalog(
        wide_cfg, spec, add=add, buffers=buffers, forward_inputs=forward_inputs,
        host_allocate=lambda _name, shape, dtype: _DummyHostView(shape, dtype),
        device=device,
    )

    for batch, width in wide_shapes:
        name = f"copy-speculative{batch}x{width}"
        assert isinstance(forwards[name], PreparedCopyOnlySpeculativeRound)
        sig = signatures[name]
        assert sig.kind == "speculative"
        assert sig.batch_capacity == batch
        assert sig.token_capacity == width
        assert sig.packed_token_capacity == batch * width
        assert sig.draft_tokens == width - 1
        assert sig.confidence_width == 0
        assert getattr(sig, "draft_source", 1) == 2
        assert "query_lengths" in sig.inputs
        assert "speculative_prefix_limits" not in sig.inputs
        for copy_buf in ("copy_token_ids", "copy_token_count", "copy_expected_consumed", "copy_generation"):
            assert copy_buf in sig.inputs

    # Target-only rounds constructed for wide copy graphs must never allocate draft proposals.
    copy_targets = [entry for entry in constructed if entry[0].startswith("copy-verification")]
    assert len(copy_targets) == len(wide_shapes)
    assert all(gen is False and prop_only is False for _, _, _, gen, prop_only, _ in copy_targets)

    # Contract binds all four copy buffers with REQUEST_ROW indexing and keeps state_rows_per_request == 8.
    buffers["request_pending_token"] = torch.zeros(4, dtype=torch.int64)
    buffers["state_pool"] = pool.backing
    contract = prepare_ds41_contract(
        signatures, buffers, pool, page_tokens=128, draft_lag=0, rows=8
    )
    by_name = {g["graph"]: g for g in contract["graphs"]}
    for batch, width in wide_shapes:
        graph_contract = by_name[f"copy-speculative{batch}x{width}"]
        assert graph_contract["state_rows_per_request"] == 8
        sources = {item["buffer"]: (item["source"], item["indexing"]) for item in graph_contract["inputs"]}
        assert sources["copy_token_ids"] == (pb2.COPY_TOKEN_IDS, pb2.REQUEST_ROW)
        assert sources["copy_token_count"] == (pb2.COPY_TOKEN_COUNT, pb2.REQUEST_ROW)
        assert sources["copy_expected_consumed"] == (pb2.COPY_CONSUMED_TOKENS, pb2.REQUEST_ROW)
        assert sources["copy_generation"] == (pb2.COPY_GENERATION, pb2.REQUEST_ROW)
        assert sources["query_lengths"] == (pb2.QUERY_LENGTH, pb2.SUBMISSION_ROW)


@pytest.mark.parametrize("width", [16, 24])
def test_copy_only_round_skips_dspark_and_accepts_full_wide_span(width):
    vocab = 32
    drafts = width - 1
    copy_tokens = [((i * 3 + 5) % vocab) for i in range(drafts)]
    bonus_token = 19

    # Target logits greedily prefer copy_tokens[0..drafts-1] and bonus_token at the bonus column.
    logits = torch.full((1, width, vocab), -10.0, dtype=torch.float32)
    for pos, tok in enumerate(copy_tokens):
        logits[0, pos, tok] = 10.0
    logits[0, drafts, bonus_token] = 10.0

    copy_round, owner, decoder, context, verifier = _make_copy_only_round(
        batch=1, width=width, vocab=vocab, logits=logits
    )
    assert copy_round.target.draft is None
    assert copy_round.target.markov is None
    assert copy_round.target.full_proposal is None

    owner.copy_ids[0, :drafts] = torch.tensor(copy_tokens, dtype=torch.int64)
    owner.copy_count[0] = drafts
    owner.copy_expected[0] = owner.requests.consumed_tokens[0]
    owner.copy_generation[0] = owner.requests.generation[0]

    metadata, acceptance, committed = copy_round()

    assert decoder.calls == 1
    assert metadata.max_query_tokens == width
    assert verifier.seen_drafts[0].tolist() == copy_tokens
    assert verifier.seen_draft_valid[0].all()
    # Every draft position has an exact q = 1.0 point-mass distribution.
    expected_q = torch.zeros((1, drafts, vocab), dtype=torch.float32)
    for pos, tok in enumerate(copy_tokens):
        expected_q[0, pos, tok] = 1.0
    assert torch.equal(verifier.seen_draft_probs, expected_q)
    assert acceptance.offered_drafts.tolist() == [drafts]
    assert acceptance.accepted_drafts.tolist() == [drafts]
    assert committed.emitted_count.tolist() == [width]
    assert committed.emitted_ids[0].tolist() == [*copy_tokens, bonus_token]
    assert decoder.commit_counts.tolist() == [width]
    assert context.calls == 1
    assert context.last_counts.tolist() == [width]

    # Inspect completion record fields including full 15-bit / 23-bit copy_offered_mask.
    rec = owner.completion.records[0, 0]
    expected_mask = (1 << drafts) - 1
    assert int(rec[8].item()) == width            # emitted_count
    assert int(rec[9].item()) == drafts           # offered_drafts
    assert int(rec[10].item()) == drafts          # accepted_drafts
    assert int(rec[11].item()) == expected_mask   # copy_offered_mask (15-bit or 23-bit)
    assert int(rec[15].item()) == width           # verification_width
    assert rec[VECTOR_TOKENS_FIELD:VECTOR_TOKENS_FIELD + width].tolist() == [*copy_tokens, bonus_token]


@pytest.mark.parametrize("reject_at", [0, 1, 7, 14, 15, 22, 23])
def test_copy_only_round_partial_rejection_across_w24_positions(reject_at):
    width = 24
    drafts = 23
    vocab = 32
    copy_tokens = [((i + 2) % vocab) for i in range(drafts)]
    bonus_token = 29

    logits = torch.zeros((1, width, vocab), dtype=torch.float32)
    for pos, tok in enumerate(copy_tokens):
        logits[0, pos, tok] = 4.0
    logits[0, drafts, bonus_token] = 4.0

    copy_round, owner, decoder, context, _ = _make_copy_only_round(
        batch=1, width=width, vocab=vocab, logits=logits,
        force_reject_at=reject_at, bonus_token=bonus_token,
    )
    before_consumed = int(owner.requests.consumed_tokens[0].item())
    before_generated = int(owner.requests.generated_tokens[0].item())
    owner.copy_ids[0, :drafts] = torch.tensor(copy_tokens, dtype=torch.int64)
    owner.copy_count[0] = drafts
    owner.copy_expected[0] = before_consumed
    owner.copy_generation[0] = owner.requests.generation[0]

    _, acceptance, committed = copy_round()

    expected_emitted = reject_at + 1
    assert acceptance.offered_drafts.tolist() == [drafts]
    assert acceptance.accepted_drafts.tolist() == [reject_at]
    assert committed.emitted_count.tolist() == [expected_emitted]
    assert committed.accepted_count.tolist() == [expected_emitted]
    assert committed.emitted_ids[0, :reject_at].tolist() == copy_tokens[:reject_at]
    assert int(committed.emitted_ids[0, reject_at].item()) == bonus_token
    assert decoder.commit_counts.tolist() == [expected_emitted]
    assert context.calls == 1
    assert context.last_counts.tolist() == [expected_emitted]
    assert int(owner.requests.consumed_tokens[0].item()) == before_consumed + expected_emitted
    assert int(owner.requests.generated_tokens[0].item()) == before_generated + expected_emitted

    rec = owner.completion.records[0, 0]
    assert int(rec[8].item()) == expected_emitted
    assert int(rec[9].item()) == drafts
    assert int(rec[10].item()) == reject_at
    assert int(rec[11].item()) == (1 << drafts) - 1  # 0x7fffff (23 bits)


def test_copy_only_round_freshness_grammar_and_contiguity_validation():
    width = 16
    drafts = 15
    vocab = 32
    logits = torch.zeros((4, width, vocab), dtype=torch.float32)
    logits[:, :, 3] = 5.0
    copy_round, owner, decoder, context, verifier = _make_copy_only_round(
        batch=4,
        width=width,
        capacity=4,
        vocab=vocab,
        logits=logits,
        grammar_enabled=torch.tensor([0, 0, 1, 0], dtype=torch.int32),
        active_lengths=torch.tensor([1, 1, 1, 1], dtype=torch.int32),
    )
    original_query_lengths = owner.inputs.query_lengths.clone()

    # Slot 0: valid full 15 tokens.
    owner.copy_ids[0, :drafts] = 3
    owner.copy_count[0] = drafts
    owner.copy_expected[0] = owner.requests.consumed_tokens[0]
    owner.copy_generation[0] = owner.requests.generation[0]

    # Slot 1: interior out-of-vocab token at index 6 -> contiguous prefix truncates to k=6.
    owner.copy_ids[1, :drafts] = 3
    owner.copy_ids[1, 6] = vocab + 5
    owner.copy_count[1] = drafts
    owner.copy_expected[1] = owner.requests.consumed_tokens[1]
    owner.copy_generation[1] = owner.requests.generation[1]

    # Slot 2: grammar enabled -> copy drafts rejected, falls back cleanly to limit=1 (0 drafts).
    owner.copy_ids[2, :drafts] = 3
    owner.copy_count[2] = drafts
    owner.copy_expected[2] = owner.requests.consumed_tokens[2]
    owner.copy_generation[2] = owner.requests.generation[2]

    # Slot 3: stale generation -> copy drafts rejected, falls back cleanly to limit=1 (0 drafts).
    owner.copy_ids[3, :drafts] = 3
    owner.copy_count[3] = drafts
    owner.copy_expected[3] = owner.requests.consumed_tokens[3]
    owner.copy_generation[3] = owner.requests.generation[3] - 1

    _, acceptance, committed = copy_round()

    # Host-uploaded query_lengths buffer must not be mutated in place.
    assert torch.equal(owner.inputs.query_lengths, original_query_lengths)
    assert verifier.seen_token_limits.tolist() == [16, 7, 1, 1]
    assert acceptance.offered_drafts.tolist() == [15, 6, 0, 0]
    assert committed.emitted_count.tolist() == [16, 7, 1, 1]
    assert decoder.commit_counts.tolist() == [16, 7, 1, 1]
    assert context.last_counts.tolist() == [16, 7, 1, 1]

    masks = [int(owner.completion.records[0, slot, 11].item()) for slot in range(4)]
    assert masks == [(1 << 15) - 1, (1 << 6) - 1, 0, 0]


def test_copy_only_point_mass_exact_zero_ulp_stochastic_marginal():
    # Exact analytical verification of q = 1.0 point-mass rejection sampling across W=24 positions.
    torch.manual_seed(2026)
    vocab = 32
    drafts = 23
    target_probs = torch.softmax(torch.randn((drafts, vocab), dtype=torch.float64), dim=-1)
    copy_tokens = torch.randint(0, vocab, (drafts,))

    for pos in range(drafts):
        p = target_probs[pos]
        c = int(copy_tokens[pos].item())
        q = torch.zeros_like(p)
        q[c] = 1.0

        # Exact speculative acceptance rate: min(1, p(c) / q(c)) = p(c).
        accept_prob = min(1.0, float(p[c].item()) / float(q[c].item()))
        assert accept_prob == pytest.approx(float(p[c].item()), abs=0.0)

        # Residual distribution on rejection: (p - q)_+ / sum((p - q)_+).
        residual_unnorm = torch.clamp(p - q, min=0.0)
        residual_mass = residual_unnorm.sum()
        residual = residual_unnorm / residual_mass

        # Marginal emitted distribution: accept_prob * delta_c + (1 - accept_prob) * residual == p.
        marginal = accept_prob * q + (1.0 - accept_prob) * residual
        assert torch.allclose(marginal, p, atol=1e-15, rtol=1e-14)

        # Negative sensitivity check: perturbing the residual breaks exactness immediately.
        wrong_residual = torch.roll(residual, 1)
        wrong_marginal = accept_prob * q + (1.0 - accept_prob) * wrong_residual
        assert not torch.allclose(wrong_marginal, p, atol=1e-4)


def test_copy_only_round_vocabulary_parallel_sharded_tile_masses():
    batch, width, vocab, world_size = 2, 16, 1024, 4
    local_vocab = vocab // world_size
    drafts = width - 1
    # Place copy tokens across all 4 vocabulary shards (0..255, 256..511, 512..767, 768..1023).
    row0_tokens = [((i * 271 + 17) % vocab) for i in range(drafts)]
    row1_tokens = [((i * 257 + 5) % vocab) for i in range(9)]

    total_probs = torch.zeros((batch, drafts, vocab), dtype=torch.float32)
    total_tile_mass = torch.zeros((batch, drafts), dtype=torch.float32)
    for rank in range(world_size):
        logits = torch.zeros((batch, width, local_vocab), dtype=torch.float32)
        copy_round, owner, _, _, vp_state = _make_copy_only_round(
            batch=batch, width=width, capacity=4, vocab=vocab, logits=logits,
            vocabulary_parallel=True, rank=rank, world_size=world_size,
        )
        assert isinstance(copy_round.local_vocab_start, torch.Tensor)
        assert copy_round.local_vocab_start.shape == (1,)
        assert copy_round.local_vocab_start.dtype == torch.int64
        assert int(copy_round.local_vocab_start.item()) == rank * local_vocab
        owner.copy_ids[0, :drafts] = torch.tensor(row0_tokens, dtype=torch.int64)
        owner.copy_count[0] = drafts
        owner.copy_expected[0] = owner.requests.consumed_tokens[0]
        owner.copy_generation[0] = owner.requests.generation[0]

        owner.copy_ids[1, :9] = torch.tensor(row1_tokens, dtype=torch.int64)
        owner.copy_count[1] = 9
        owner.copy_expected[1] = owner.requests.consumed_tokens[1]
        owner.copy_generation[1] = owner.requests.generation[1]

        copy_round()
        assert vp_state["token_limits"].tolist() == [16, 10]
        start = rank * local_vocab
        total_probs[:, :, start:start + local_vocab] = vp_state["draft_probs"]
        total_tile_mass += vp_state["draft_tile_masses"].sum(dim=-1)

    # Across all 4 ranks, every active copy position has exact unit point-mass probability and tile mass.
    for pos, tok in enumerate(row0_tokens):
        assert total_probs[0, pos, tok].item() == 1.0
        assert total_probs[0, pos].sum().item() == 1.0
        assert total_tile_mass[0, pos].item() == 1.0
    for pos, tok in enumerate(row1_tokens):
        assert total_probs[1, pos, tok].item() == 1.0
        assert total_probs[1, pos].sum().item() == 1.0
        assert total_tile_mass[1, pos].item() == 1.0
    assert total_tile_mass[1, 9:].sum().item() == 0.0


def test_copy_only_round_passes_persistent_local_vocab_start_tensor(monkeypatch):
    """PreparedCopyOnlySpeculativeRound passes persistent int64 local_vocab_start tensor (no rank-0 Python guard)."""
    from hyperflare_recipes.components import parallel_draft

    seen_vocab_starts = []
    real_apply = parallel_draft._apply_copy_proposals

    def spy_apply(*args, **kwargs):
        seen_vocab_starts.append(kwargs.get("vocab_start"))
        return real_apply(*args, **kwargs)

    monkeypatch.setattr(parallel_draft, "_apply_copy_proposals", spy_apply)

    for rank in (0, 2):
        copy_round, owner, _, _, _ = _make_copy_only_round(
            batch=1, width=16, capacity=4, vocab=1024,
            logits=torch.zeros((1, 16, 256), dtype=torch.float32),
            vocabulary_parallel=True, rank=rank, world_size=4,
        )
        owner.copy_ids[0, :15] = rank * 256 + 7
        owner.copy_count[0] = 15
        owner.copy_expected[0] = owner.requests.consumed_tokens[0]
        owner.copy_generation[0] = owner.requests.generation[0]
        copy_round()
        assert seen_vocab_starts[-1] is copy_round.local_vocab_start
        assert isinstance(seen_vocab_starts[-1], torch.Tensor)
        assert int(seen_vocab_starts[-1].item()) == rank * 256


@pytest.mark.parametrize("width", [16, 24])
def test_copy_only_round_make_fx_traces_cleanly(width):
    vocab = 32
    drafts = width - 1
    copy_tokens = [((i * 5 + 1) % vocab) for i in range(drafts)]
    bonus_token = 11

    logits = torch.full((1, width, vocab), -8.0, dtype=torch.float32)
    for pos, tok in enumerate(copy_tokens):
        logits[0, pos, tok] = 8.0
    logits[0, drafts, bonus_token] = 8.0

    # Build two identical rounds to compare eager vs make_fx traced execution bit-for-bit.
    eager_round, eager_owner, _, _, _ = _make_copy_only_round(batch=1, width=width, vocab=vocab, logits=logits)
    fx_round, fx_owner, _, _, _ = _make_copy_only_round(batch=1, width=width, vocab=vocab, logits=logits)
    for owner in (eager_owner, fx_owner):
        owner.copy_ids[0, :drafts] = torch.tensor(copy_tokens, dtype=torch.int64)
        owner.copy_count[0] = drafts
        owner.copy_expected[0] = owner.requests.consumed_tokens[0]
        owner.copy_generation[0] = owner.requests.generation[0]

    def _run_fx(copy_ids, copy_count):
        fx_owner.copy_ids.copy_(copy_ids)
        fx_owner.copy_count.copy_(copy_count)
        _, acc, comm = fx_round()
        return acc.offered_drafts, acc.accepted_drafts, comm.emitted_count, comm.emitted_ids

    # Trace with dummy inputs, then run the traced graph on a fresh owner state.
    traced = make_fx(_run_fx)(fx_owner.copy_ids.clone(), fx_owner.copy_count.clone())
    assert len(list(traced.graph.nodes)) > 0

    # Reset fx_owner request state to match eager_owner before comparing eager vs traced results.
    fx_owner.requests.consumed_tokens.copy_(eager_owner.requests.consumed_tokens)
    fx_owner.requests.generated_tokens.copy_(eager_owner.requests.generated_tokens)
    fx_owner.requests.pending_token.copy_(eager_owner.requests.pending_token)
    fx_owner.requests.status.copy_(eager_owner.requests.status)
    fx_owner.requests.histogram.zero_()
    fx_owner.rng.target_offset.zero_()

    _, eager_acc, eager_comm = eager_round()
    fx_offered, fx_accepted, fx_emitted_count, fx_emitted_ids = traced(
        eager_owner.copy_ids.clone(), eager_owner.copy_count.clone()
    )
    assert torch.equal(fx_offered, eager_acc.offered_drafts)
    assert torch.equal(fx_accepted, eager_acc.accepted_drafts)
    assert torch.equal(fx_emitted_count, eager_comm.emitted_count)
    assert torch.equal(fx_emitted_ids, eager_comm.emitted_ids)
    assert torch.equal(fx_owner.completion.records, eager_owner.completion.records)
