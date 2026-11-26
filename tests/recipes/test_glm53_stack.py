# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared scratch and persistent state ownership for complete GLM text stacks."""

import os
from dataclasses import fields
from pathlib import Path

import pytest
import torch

from hyperflare_recipes.components.parallel import TensorParallelShard
from hyperflare_recipes.recipes.glm53.model import create_model_tree
from hyperflare_recipes.recipes.glm53.stack import DecoderScratch
from hyperflare_recipes.recipes.glm53.state import DecoderMetadata, DecoderState
from hyperflare_recipes.recipes.glm53.mla import MLAState, PreparedGlmMLA
from hyperflare_recipes.recipes.glm53.indexer import PreparedGlmIndexer


def scratch_tensors(scratch, prefix=""):
    for field in fields(scratch):
        value = getattr(scratch, field.name)
        name = f"{prefix}{field.name}"
        if isinstance(value, torch.Tensor):
            yield name, value
        elif value is not None:
            yield from scratch_tensors(value, f"{name}.")


def test_sparse_attention_rebind_uses_replacement_cache_at_execution():
    from types import SimpleNamespace

    state = MLAState.allocate(1, 2, "cpu")
    previous = torch.empty_like(state.cache)
    indexer = object.__new__(PreparedGlmIndexer)
    indexer.state, indexer.cache = state.index, previous
    indexer.selectors, indexer.logical, indexer.expand_options = (), False, {}

    class Attention:
        def rebind_cache(self, cache):
            self.cache = cache

    prepared = object.__new__(PreparedGlmMLA)
    prepared.state, prepared.cache = state, previous
    prepared.writer, prepared.indexer = SimpleNamespace(cache=previous), indexer
    prepared.attention, prepared.context = Attention(), None
    prepared.rebind_pages()
    assert prepared.cache is state.cache
    assert prepared.writer.cache is state.cache
    assert prepared.attention.cache is state.cache
    assert prepared.indexer.cache is state.index.cache
    assert prepared.indexer.expand_options["num_cache_blocks"] == state.index.cache.shape[0]


@pytest.mark.parametrize("world,rank", [(2, 0), (4, 3)])
def test_serial_graph_variants_share_storage_with_exact_shapes(world, rank):
    shard = TensorParallelShard(rank, world)
    base = DecoderScratch.allocate(64, 8, "decode", "cpu", shard=shard, sample_rows=4)
    storage = dict(scratch_tensors(base))
    for mode, tokens, rows in (("prefill", 64, 1), ("prefill", 16, 1), ("prefill", 1, 1),
                              ("decode", 4, 4), ("decode", 2, 2), ("decode", 1, 1)):
        view = base.rows(tokens, mode, sample_rows=rows)
        expected = dict(scratch_tensors(DecoderScratch.allocate(tokens, 8, mode, "meta",
            shard=shard, sample_rows=rows)))
        actual = dict(scratch_tensors(view))
        assert actual.keys() == expected.keys()
        for name, tensor in actual.items():
            assert tensor.shape == expected[name].shape, name
            assert tensor.stride() == expected[name].stride(), name
            assert tensor.dtype == expected[name].dtype, name
            backing = base.kda.projected_input[:, -128:] if name == "kda.forget_input" and tokens == 1 else storage[name]
            assert tensor.data_ptr() == backing.data_ptr(), name
        assert view.mla.index.page_table.shape == (tokens if mode == "decode" else 1, 8)
        view.residual.fill_(tokens)
        assert torch.all(base.residual[:tokens] == tokens)
        assert view.local_logits.shape == (rows, 154880 // world)
        assert view.logits.shape == (rows, 154880)


def test_scratch_views_reject_unavailable_token_sample_and_request_capacity():
    base = DecoderScratch.allocate(64, 8, "prefill", "meta", shard=TensorParallelShard(0, 2), sample_rows=4)
    for tokens, mode, rows in ((0, "prefill", 1), (65, "prefill", 1), (16, "prefill", 5),
                               (1, "prefill", 2), (4, "decode", 4), (4, "invalid", 1)):
        with pytest.raises(ValueError):
            base.rows(tokens, mode, sample_rows=rows)


def test_cp_prefill_query_value_alias_survives_declaration_views():
    base = DecoderScratch.allocate(65, 2, "prefill", "cpu", shard=TensorParallelShard(0, 2),
        sample_rows=1, context_ranks=2)
    for rows in (65, 16, 65):
        view = base.rows(rows, "prefill", sample_rows=1).mla
        view.query.fill_(float("nan"))
        view.expanded.fill_(rows)
        assert torch.all(view.query == rows)
    for mode, ranks in (("prefill", 1), ("decode", 2)):
        separate = DecoderScratch.allocate(2, 2, mode, "cpu", shard=TensorParallelShard(0, 2),
            sample_rows=1, context_ranks=ranks).mla
        separate.query.fill_(7)
        separate.expanded.zero_()
        assert torch.all(separate.query == 7)


@pytest.mark.parametrize("world,rank", [(2, 0), (4, 3)])
def test_canonical_layers_share_intermediates_and_own_distinct_state(world, rank):
    snapshot = Path(os.environ.get("FLARE_GLM53_SNAPSHOT",
        "/data/checkpoint-releases/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD"))
    if not snapshot.is_dir():
        pytest.skip("GLM checkpoint is not available")
    tree = create_model_tree(snapshot)
    layers = tree.model.language_model.layers
    initialized = torch.cuda.is_initialized()
    scratch = DecoderScratch.allocate(65, 1, "prefill", "cpu",
        shard=TensorParallelShard(rank, world), sample_rows=1)
    state = DecoderState.allocate(tree.config.text_config.layer_types, world, 3, 1, "cpu")
    metadata = DecoderMetadata.allocate(65, 1, 1, "cpu")
    buffers = list(state.named_tensors())
    assert len(buffers) == 34 * 2 + 11 * 3
    assert len({name for name, _ in buffers}) == len(buffers)
    assert len({value.untyped_storage().data_ptr() for _, value in buffers}) == len(buffers)
    for index, module in enumerate(layers):
        layer = scratch.layer(module.block_type, sparse=hasattr(module.mlp, "experts"))
        assert layer.attention_hc is scratch.attention_hc and layer.mlp_hc is scratch.mlp_hc
        assert layer.attention_output is scratch.attention_output and layer.mlp_output is scratch.mlp_output
        assert layer.attention is (scratch.mla if module.block_type == "indexed_attention" else scratch.kda)
        assert layer.mlp is (scratch.dense if index < 3 else scratch.moe)
    assert scratch.local_logits.shape == (1, 154880 // world)
    assert scratch.logits.shape == (1, 154880)
    assert not metadata.recurrent.num_tokens.count_nonzero()
    assert not metadata.recurrent.num_seqs.count_nonzero()
    assert (metadata.indexed.positions == -1).all()
    assert (metadata.indexed.token_slots == -1).all()
    assert (metadata.indexed.state_slots == -1).all()
    for _, buffer in buffers:
        buffer.fill_(1)
    state.clear_for_preparation()
    assert all(not value.count_nonzero() for _, value in buffers)
    assert torch.cuda.is_initialized() == initialized


@pytest.mark.parametrize("fits", [False, True])
def test_prepared_mtp_selects_fused_all_reduce_rmsnorm_when_runtime_fits(monkeypatch, fits):
    from types import SimpleNamespace
    import hyperflare_recipes.recipes.glm53.mtp as mtp_module

    calls = []

    class FakeFused:
        @staticmethod
        def fits(runtime, source):
            return fits

        def __init__(self, source, residual, weight, output, **kwargs):
            self.name = kwargs["name"]
            self.weight = weight
            calls.append(("fused_init", self.name, weight))

        def functional(self, *, session):
            return self

        def __call__(self, source, residual):
            calls.append(("fused_call", self.name))
            residual.add_(source)
            return residual.clone()

    class FakePlain:
        def __init__(self, source, output, **kwargs):
            self.name = kwargs["name"]
            calls.append(("plain_init", self.name))

        def functional(self, *, session):
            return self

        def __call__(self, source):
            calls.append(("plain_call", self.name))
            return source

    def fake_rms_add(x, residual, weight, eps, x_out=None):
        calls.append(("rms_add", weight))
        residual.add_(x)
        x_out.copy_(residual)

    monkeypatch.setattr(mtp_module, "PreparedAllReduceRMSNorm", FakeFused)
    monkeypatch.setattr(mtp_module, "PreparedAllReduce", FakePlain)
    monkeypatch.setattr(mtp_module, "rms_norm_add_residual", fake_rms_add)
    monkeypatch.setattr(mtp_module, "rms_norm", lambda x, w, out, eps, **kw: out.copy_(x))

    # Exercise PreparedMTP.__init__ selection logic with minimal lightweight stubs.
    class FakeEmbedding:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *, token_ids, num_rows):
            return torch.ones((token_ids.numel(), 4096), dtype=torch.bfloat16)

    class FakeProjection:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, joined):
            return joined[:, :4096].clone()

    class FakeAttention:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, x, indexed):
            return x * 2

    class FakeMoe:
        def __init__(self, module, normalized, weights, scratch, **kwargs):
            self.output = scratch.output if hasattr(scratch, "output") else torch.zeros_like(normalized)

        def __call__(self, x):
            return x * 3

    class FakeHead:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, x):
            return x[:, :8]

    import transformers.models.glm5_next.modeling_glm5_next as glm_mod

    monkeypatch.setattr(glm_mod, "Glm5NextTextAttention", lambda cfg, idx: SimpleNamespace(eval=lambda: None))
    monkeypatch.setattr(glm_mod, "Glm5NextTextMoE", lambda cfg: SimpleNamespace(eval=lambda: None))
    monkeypatch.setattr(mtp_module, "PreparedEmbedding", FakeEmbedding)
    monkeypatch.setattr(mtp_module, "UnquantizedProjection", FakeProjection)
    monkeypatch.setattr(mtp_module, "PreparedGlmMLA", FakeAttention)
    monkeypatch.setattr(mtp_module, "PreparedGlmMoe", FakeMoe)
    monkeypatch.setattr(mtp_module, "PreparedVocabularyProjection", FakeHead)

    scratch = mtp_module.MTPScratch.allocate(2, 2, 1, "decode", "cpu")
    weights = SimpleNamespace(
        tree=SimpleNamespace(
            config=SimpleNamespace(text_config=SimpleNamespace(indexer_types=[], rms_norm_eps=1e-5)),
            model=SimpleNamespace(language_model=SimpleNamespace(embed_tokens=SimpleNamespace(weight=None))),
        ),
        shard=SimpleNamespace(world_size=2),
        head=None,
        mtp=SimpleNamespace(
            tensors={
                "eh_proj": None,
                "enorm": "enorm",
                "hnorm": "hnorm",
                "input_layernorm": "in_norm",
                "post_attention_layernorm": "post_norm",
                "shared_head.norm": "head_norm",
            },
            attention=None,
            mlp=None,
            draft_head=None,
        ),
    )
    meta = SimpleNamespace(
        token_ids=torch.zeros(2, dtype=torch.int64),
        num_tokens=torch.tensor([2], dtype=torch.int32),
        hidden=torch.ones((2, 4096), dtype=torch.bfloat16),
        indexed=SimpleNamespace(
            block_table=torch.zeros((2, 1), dtype=torch.int32),
            positions=torch.tensor([1, 2], dtype=torch.int64),
        ),
    )
    op = mtp_module.PreparedMTP(
        weights, None, meta, scratch=scratch, mode="decode", workspace=None,
        runtime=object(), vocabulary_runtime=None, coordinator=None, session=None, name="mtp",
    )
    assert op.fused_attention_reduce is fits
    assert op.fused_mlp_reduce is fits

    hidden, logits = op(meta)
    # Initial residual = 1, attention = 2 -> residual = 3, moe = 9 -> residual = 12.
    assert hidden.shape == (2, 4096) and logits.shape == (2, 8)
    assert torch.all(hidden == 12)
    if fits:
        assert [c[0] for c in calls] == ["fused_init", "fused_init", "fused_call", "fused_call"]
        assert calls[0][2] == "post_norm" and calls[1][2] == "head_norm"
    else:
        assert [c[0] for c in calls] == ["plain_init", "plain_init", "plain_call", "rms_add", "plain_call", "rms_add"]
        assert calls[3][1] == "post_norm" and calls[5][1] == "head_norm"
