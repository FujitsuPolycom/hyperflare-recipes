# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Captured DSpark proposals, packed target verification, and accepted-state commits."""

from dataclasses import dataclass
import itertools
import weakref

import torch
import triton
import triton.language as tl

from hyperflare_recipes.components.speculation import SpeculativeRngState

from hyperflare_recipes.components.collectives import PreparedAllGather
from ignition.preparation.persistent import persistent_copy
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.tuning import register_finalizer
from .compressed_mla import PreparedCompressedCacheWriter
from .embedding import PreparedEmbedding
from .linear import QuantizedProjection, UnquantizedProjection
from .markov_sampling import PreparedRowScaledFP8MarkovProjection, RowScaledFP8MarkovWeight
from .rms_norm import PreparedRMSNorm
from .rotary import CompressedRotary as Rotary


@dataclass(frozen=True)
class DSparkForwards:
    decoder: type
    scratch: type
    context: type
    draft: type
    markov: type


class SpeculativePreparation:
    def __init__(
        self,
        weights,
        tables,
        pool,
        requests,
        inputs,
        scratch,
        *,
        draft_tokens,
        completion,
        frame,
        sequence,
        session,
        workspace,
        runtime,
        vocabulary_runtime,
        markov_runtime,
        coordinator,
        forwards,
        page_tokens=128,
        result_runtime=None,
        block_proposal_broadcast=False,
        vocabulary_parallel_sampling=False,
        sampling_runtime=None,
        logprob_runtime=None,
        copy_draft_tokens=None,
        block_size=None,
        logit_offset=0,
    ):
        self.forwards = forwards
        self.page_tokens = page_tokens
        self.weights, self.tables, self.pool = weights, tables, pool
        self.requests, self.inputs, self.scratch = requests, inputs, scratch
        self.completion, self.frame, self.sequence = completion, frame, sequence
        self.session, self.workspace = session, workspace
        self.collectives = {
            "runtime": runtime, "vocabulary_runtime": vocabulary_runtime, "coordinator": coordinator
        }
        self.vocabulary_parallel_sampling = vocabulary_parallel_sampling
        self.sampling_runtime, self.logprob_runtime = sampling_runtime, logprob_runtime
        self.markov_runtime = markov_runtime
        # Only sampling uses the result channel; model forwards receive the collectives above.
        self.result_runtime = result_runtime
        self.block_proposal_broadcast = block_proposal_broadcast
        device = pool.backing.device
        self.rng = SpeculativeRngState.allocate(device, capacity=requests.capacity)
        self.generation = torch.zeros_like(requests.generation)
        self.offset = torch.zeros_like(requests.generation)
        copy_width = max(7, draft_tokens, copy_draft_tokens or 0)
        self.copy_ids = torch.full((requests.capacity, copy_width), -1, dtype=torch.int64, device=device)
        self.copy_count = torch.zeros(requests.capacity, dtype=torch.int32, device=device)
        self.copy_expected = torch.full((requests.capacity,), -1, dtype=torch.int64, device=device)
        self.copy_generation = torch.zeros(requests.capacity, dtype=torch.int64, device=device)
        draft = weights.config.draft
        self.logit_offset = logit_offset
        self.block_size = max(1, draft_tokens) if block_size is None else block_size
        tokens = requests.capacity * self.block_size
        self.draft_scratch = forwards.scratch.allocate(
            tokens,
            tokens,
            weights.shard.world_size,
            device,
            experts=draft.experts,
            top_k=draft.top_k,
        )

    def buffers(self):
        return {
            "copy_token_ids": self.copy_ids,
            "copy_token_count": self.copy_count,
            "copy_expected_consumed": self.copy_expected,
            "copy_generation": self.copy_generation,
        }

    def release_examples(self):
        self.scratch = self.draft_scratch = None


class PreparedFeedbackCapture:
    """Capture target stream means as forward-local accepted-context features."""

    def __init__(self, metadata, columns, *, compact_prefill=False):
        rows = metadata.positions.numel()
        self.compact = compact_prefill and columns > 0 and metadata.starts.numel() == 1 and rows > 128
        self.shape = (128 if self.compact else rows, columns)

    def begin(self, metadata):
        indices = live = None
        if self.compact:
            offsets = torch.arange(self.shape[0], device=metadata.positions.device)
            count = metadata.lengths[0].long()
            kept = count.clamp(max=self.shape[0])
            indices = (metadata.query_start_loc[0] + count - kept + offsets).clamp(0, metadata.positions.numel() - 1)
            live = (offsets < kept) & (metadata.positions[indices] >= 0)
        return torch.empty(self.shape, dtype=torch.bfloat16, device=metadata.positions.device), indices, live

    def __call__(self, state, residual, column):
        output, indices, live = state
        values = residual.float().mean(1) if indices is None else residual[indices].float().mean(1)
        if live is not None:
            values = torch.where(live[:, None], values, 0)
        output[:, column:column + residual.shape[-1]].copy_(values)
        return state


@triton.jit
def _context_kv_norm_rope_kernel(
    KV, Weights, Positions, Frequencies, Out,
    kv_row_stride, out_stage_stride, out_row_stride,
    EPS: tl.constexpr,
):
    """Fused multi-layer context KV RMSNorm (512) + RoPE (last 64), BF16-exact seam."""
    row = tl.program_id(0).to(tl.int64)
    stage = tl.program_id(1).to(tl.int64)
    col = tl.arange(0, 512)
    src = KV + row * kv_row_stride + stage * 512
    dst = Out + stage * out_stage_stride + row * out_row_stride
    x = tl.load(src + col).to(tl.float32)
    w = tl.load(Weights + stage * 512 + col).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / 512.0 + EPS)
    tl.store(dst + col, (x * inv * w).to(tl.bfloat16), mask=col < 448)
    pair = tl.arange(0, 32)
    xa = tl.load(src + 448 + 2 * pair).to(tl.float32)
    xb = tl.load(src + 448 + 2 * pair + 1).to(tl.float32)
    wa = tl.load(Weights + stage * 512 + 448 + 2 * pair).to(tl.float32)
    wb = tl.load(Weights + stage * 512 + 448 + 2 * pair + 1).to(tl.float32)
    a = (xa * inv * wa).to(tl.bfloat16).to(tl.float32)
    b = (xb * inv * wb).to(tl.bfloat16).to(tl.float32)
    pos = tl.maximum(tl.load(Positions + row), 0).to(tl.float32)
    angle = pos * tl.load(Frequencies + pair)
    cos, sin = tl.cos(angle), tl.sin(angle)
    tl.store(dst + 448 + 2 * pair, (a * cos - b * sin).to(tl.bfloat16))
    tl.store(dst + 448 + 2 * pair + 1, (a * sin + b * cos).to(tl.bfloat16))


_CONTEXT_OWNERS = weakref.WeakValueDictionary()
_CONTEXT_HANDLES = itertools.count()


@torch.library.custom_op(
    "flare::start_dspark_context",
    mutates_args=("buffers", "workspace", "normalized_all"),
)
def _start_dspark_context(
    auxiliary: torch.Tensor,
    positions: torch.Tensor,
    buffers: list[torch.Tensor],
    workspace: torch.Tensor,
    normalized_all: torch.Tensor,
    handle: int,
    stream: int,
) -> None:
    owner = _CONTEXT_OWNERS[handle]
    aux = torch.cuda.ExternalStream(stream, device=auxiliary.device)
    aux.wait_stream(torch.cuda.current_stream(auxiliary.device))
    with torch.cuda.stream(aux):
        owner._run_prepare_inplace(auxiliary, positions, buffers, workspace, normalized_all)


@_start_dspark_context.register_fake
def _fake_start_dspark_context(auxiliary, positions, buffers, workspace, normalized_all, handle, stream):
    pass


@torch.library.custom_op(
    "flare::join_dspark_context",
    mutates_args=("buffers", "workspace", "normalized_all"),
)
def _join_dspark_context(
    dependency: torch.Tensor,
    auxiliary: torch.Tensor,
    positions: torch.Tensor,
    buffers: list[torch.Tensor],
    workspace: torch.Tensor,
    normalized_all: torch.Tensor,
    stream: int,
) -> None:
    torch.cuda.current_stream(normalized_all.device).wait_stream(
        torch.cuda.ExternalStream(stream, device=normalized_all.device)
    )


@_join_dspark_context.register_fake
def _fake_join_dspark_context(
    dependency, auxiliary, positions, buffers, workspace, normalized_all, stream
):
    pass


class PreparedContext:
    """Project accepted target feedback into persistent draft context rings."""

    def __init__(self, weights, auxiliary, metadata, pool, *, session, name, workspace,
                 cache_format="deepseek_v41", compact_prefill=False, runtime=None, coordinator=None):
        from hyperflare_recipes.components import collectives

        tokens, device = auxiliary.shape[0], auxiliary.device
        self.compact = compact_prefill and metadata.starts.numel() == 1 and metadata.positions.numel() > 128
        if self.compact and tokens != 128:
            raise ValueError("compact prefill feedback must contain the retained 128-row suffix")
        hidden = weights.config.text["hidden_size"]
        main_weight = weights.draft.main_projection
        local_hidden = main_weight.shape[0] if isinstance(main_weight, torch.Tensor) else main_weight.out_features
        world_size = getattr(getattr(weights, "shard", None), "world_size", 1)
        projected = torch.empty((tokens, hidden), dtype=torch.bfloat16, device=device)
        main = torch.empty_like(projected)
        kv = torch.empty((tokens, 512 * len(weights.draft.layers)), dtype=torch.bfloat16, device=device)
        normalized = torch.empty((tokens, 512), dtype=torch.bfloat16, device=device)
        slots = torch.full((tokens,), -1, dtype=torch.int64, device=device)
        self._gather_runtime = runtime
        if local_hidden == hidden:
            self.gather = None
            self._main_proj_op = QuantizedProjection(auxiliary, main_weight, projected,
                session=session, name=name + ".main", workspace=workspace, activation_mode="auto").functional(session=session)
            self.project = self._main_proj_op
            overlap_buffers = [projected, main, kv]
        else:
            if world_size < 2 or hidden % world_size != 0 or local_hidden != hidden // world_size:
                raise ValueError(
                    f"draft main_projection output width {local_hidden} must equal hidden_size {hidden} "
                    f"or hidden_size // world_size ({hidden} // {world_size})"
                )
            if runtime is None:
                raise ValueError("output-sharded draft main_projection requires a collective all-gather runtime")
            local_projected = torch.empty((tokens, local_hidden), dtype=torch.bfloat16, device=device)
            self._main_proj_op = QuantizedProjection(auxiliary, main_weight, local_projected,
                session=session, name=name + ".main", workspace=workspace, activation_mode="auto").functional(session=session)
            self.gather = _prepare_context_gather(local_projected, projected, runtime=runtime,
                coordinator=coordinator, session=session, name=name + ".main_gather")
            self.project = _ShardedMainProjection(self._main_proj_op, self.gather)
            overlap_buffers = [local_projected, projected, main, kv]
        self.norm = PreparedRMSNorm(projected, weights.draft.main_norm, main,
            eps=weights.config.text["rms_norm_eps"], session=session, name=name + ".main_norm").functional()
        self.context_projection = QuantizedProjection(main, weights.draft.context_kv, kv,
            session=session, name=name + ".kv", workspace=workspace, activation_mode="auto").functional(session=session)
        self.rotary = Rotary(weights.config, device)
        self.eps = float(weights.config.text["rms_norm_eps"])
        self.fused_stage_norm_rope = device.type == "cuda" and len(weights.draft.layers) > 0
        self.kv_norm_weights = None
        self.stages = []
        if self.fused_stage_norm_rope:
            self.kv_norm_weights = persistent_copy(torch.stack([
                layer.attention["kv_norm"] for layer in weights.draft.layers
            ]))
            record_numeric_tensor(self.kv_norm_weights)
        for index, layer in enumerate(weights.draft.layers):
            norm = (None if self.fused_stage_norm_rope else PreparedRMSNorm(
                kv[:, index * 512:(index + 1) * 512], layer.attention["kv_norm"],
                normalized, eps=self.eps, session=session,
                name=f"{name}.{index}.norm").functional())
            writer = PreparedCompressedCacheWriter(normalized, pool.window(index, draft=True), slots,
                page_size=128, kind="swa", session=session, name=f"{name}.{index}.write", cache_format=cache_format)
            self.stages.append((norm, writer.functional(session=session)))
        self.handle = next(_CONTEXT_HANDLES)
        _CONTEXT_OWNERS[self.handle] = self
        self.stream_handle = (
            workspace.auxiliary_stream.cuda_stream
            if isinstance(workspace, ScratchArena) and getattr(workspace, "overlap", False)
            else None
        )
        self._overlap_buffer_layouts = tuple((tuple(t.shape), t.dtype) for t in overlap_buffers)
        self._overlap_workspace_nbytes = 1
        self._overlap_supported = (
            not self.compact
            and self.fused_stage_norm_rope
            and self.stream_handle is not None
            and not isinstance(main_weight, torch.Tensor)
            and not isinstance(weights.draft.context_kv, torch.Tensor)
            and not getattr(self._main_proj_op, "sm103", True)
            and not getattr(self.context_projection, "sm103", True)
            and getattr(self._main_proj_op, "encoded", None) is None
            and getattr(self.context_projection, "encoded", None) is None
            and getattr(self._main_proj_op, "bias", None) is None
            and getattr(self.context_projection, "bias", None) is None
            and not getattr(self.norm, "add_residual", True)
            and not getattr(self.norm, "stage_source", True)
            and (self.gather is None or (hasattr(self.gather, "parts") and not collectives._roce(runtime)))
        )
        if self._overlap_supported:
            register_finalizer(session, self._finalize_overlap)

    def _finalize_overlap(self):
        main_bytes = getattr(self._main_proj_op, "workspace_nbytes", None)
        if main_bytes is None:
            main_bytes = self._main_proj_op.plan.memory_requirements().scratch_nbytes
        kv_bytes = getattr(self.context_projection, "workspace_nbytes", None)
        if kv_bytes is None:
            kv_bytes = self.context_projection.plan.memory_requirements().scratch_nbytes
        self._overlap_workspace_nbytes = max(256, ((max(int(main_bytes), int(kv_bytes)) + 255) // 256) * 256)

    def can_overlap_with(self, *other_runtimes) -> bool:
        return self._overlap_supported and (
            self._gather_runtime is None or all(rt is not self._gather_runtime for rt in other_runtimes)
        )

    def _run_prepare_inplace(self, auxiliary, positions, buffers, workspace_buf, normalized_all):
        from b12x.gemm import blockscaled

        if self.gather is None:
            projected, main, kv = buffers
            blockscaled.mm(
                auxiliary, self._main_proj_op.weight, out=projected,
                plan=self._main_proj_op.plan, workspace=workspace_buf,
            )
        else:
            local_projected, projected, main, kv = buffers
            blockscaled.mm(
                auxiliary, self._main_proj_op.weight, out=local_projected,
                plan=self._main_proj_op.plan, workspace=workspace_buf,
            )
            for (begin, end), part in zip(self.gather.bounds, self.gather.parts, strict=True):
                part.run_eager(local_projected[begin:end], projected[begin:end])
        torch.ops.b12x.hyperconnection_grouped_rmsnorm(
            projected, self.norm.weight, main, self.norm.eps, self.norm.plan.handle, self.norm.zero_centered,
        )
        blockscaled.mm(
            main, self.context_projection.weight, out=kv,
            plan=self.context_projection.plan, workspace=workspace_buf,
        )
        _context_kv_norm_rope_kernel[(kv.shape[0], len(self.stages))](
            kv, self.kv_norm_weights, positions, self.rotary.frequencies[False], normalized_all,
            kv.stride(0), normalized_all.stride(0), normalized_all.stride(1), self.eps, num_warps=4,
        )

    def start_prepare(self, auxiliary, metadata):
        """Launch context projection, gather, RMSNorm, and RoPE on the auxiliary stream."""
        device = auxiliary.device
        buffers = [torch.empty(shape, dtype=dtype, device=device) for shape, dtype in self._overlap_buffer_layouts]
        workspace_buf = torch.empty((self._overlap_workspace_nbytes,), dtype=torch.uint8, device=device)
        normalized_all = torch.empty(
            (len(self.stages), auxiliary.shape[0], 512), dtype=torch.bfloat16, device=device
        )
        _start_dspark_context(
            auxiliary, metadata.positions, buffers, workspace_buf, normalized_all,
            self.handle, self.stream_handle,
        )
        return auxiliary, metadata.positions, buffers, workspace_buf, normalized_all

    def commit_prepared(self, state, counts, metadata):
        """Join the auxiliary context preparation stream and commit accepted KV rows."""
        auxiliary, positions, buffers, workspace_buf, normalized_all = state
        _join_dspark_context(
            counts, auxiliary, positions, buffers, workspace_buf, normalized_all, self.stream_handle
        )
        m = metadata
        positions = m.positions
        offset = positions - m.starts[m.requests]
        count = counts[m.requests]
        live = (positions >= 0) & (offset < count) & (offset >= (count - 128).clamp_min(0))
        slots = torch.where(live, m.state_rows[m.requests] * 128 + positions % 128, -1)
        for index, (_, writer) in enumerate(self.stages):
            writer(normalized_all[index], slots)

    def functional(self):
        return self

    def __call__(self, auxiliary, counts, metadata):
        m = metadata
        if self.compact:
            offsets = torch.arange(auxiliary.shape[0], device=auxiliary.device)
            count = counts[0].long()
            kept = count.clamp(max=128)
            indices = (m.query_start_loc[0] + count - kept + offsets).clamp(0, m.positions.numel() - 1)
            positions = m.positions[indices]
            live = (offsets < kept) & (positions >= 0)
            slots = torch.where(live, m.state_rows[0] * 128 + positions % 128, -1)
            positions = torch.where(live, positions, -1)
        else:
            positions = m.positions
            offset = positions - m.starts[m.requests]
            count = counts[m.requests]
            live = (positions >= 0) & (offset < count) & (offset >= (count - 128).clamp_min(0))
            slots = torch.where(live, m.state_rows[m.requests] * 128 + positions % 128, -1)
        kv = self.context_projection(self.norm(self.project(auxiliary)))
        if self.fused_stage_norm_rope:
            normalized_all = torch.empty((len(self.stages), kv.shape[0], 512), dtype=torch.bfloat16, device=kv.device)
            torch.library.wrap_triton(_context_kv_norm_rope_kernel)[(kv.shape[0], len(self.stages))](
                kv, self.kv_norm_weights, positions, self.rotary.frequencies[False], normalized_all,
                kv.stride(0), normalized_all.stride(0), normalized_all.stride(1), self.eps, num_warps=4)
            for index, (_, writer) in enumerate(self.stages):
                writer(normalized_all[index], slots)
            return
        for index, (norm, writer) in enumerate(self.stages):
            normalized = norm(kv[:, index * 512:(index + 1) * 512])
            self.rotary.apply(normalized, positions.clamp_min(0), compressed=False)
            writer(normalized, slots)


class PreparedMarkovEmbedding:
    """Retain replicated predecessor embeddings for fused transition proposals."""

    def __init__(self, weights, tokens, *, session, name):
        embedding = torch.zeros((tokens.numel(), weights.config.draft.markov_rank),
                                     dtype=torch.bfloat16, device=tokens.device)
        self.embed = PreparedEmbedding(weights.draft.markov_embedding, tokens, embedding,
                                       session=session, name=name + ".embed").functional(session=session, dynamic_inputs=True)


    def __call__(self, tokens):
        return self.embed(token_ids=tokens)


class PreparedMarkov:
    """Project FP32 transition logits from replicated or vocabulary-sharded weights.

    A replicated head runs independently on each rank without a collective.
    A sharded head uses the supplied runtime to gather its FP32 bits unchanged.
    NVFP4 A16 projections produce BF16 logits that are widened before addition;
    row-scaled E4M3 projections write FP32 logits directly.
    """

    def __init__(self, weights, tokens, *, runtime, coordinator, session, name):
        rows, device = tokens.numel(), tokens.device
        embedding = torch.zeros(
            (rows, weights.config.draft.markov_rank), dtype=torch.bfloat16, device=device
        )
        head = weights.draft.markov_head
        columns = head.shape[0] if isinstance(head, torch.Tensor) else head.out_features
        expected_columns = 129280 if runtime is None else 129280 // weights.shard.world_size
        if columns != expected_columns:
            raise ValueError("Markov head vocabulary partition must match its gather runtime")
        local = torch.zeros((rows, columns), dtype=torch.float32, device=device)
        projected = local if isinstance(head, (torch.Tensor, RowScaledFP8MarkovWeight)) else torch.empty(
            (rows, columns), dtype=torch.bfloat16, device=device)
        logits = local if runtime is None else torch.zeros((rows, 129280), dtype=torch.float32, device=device)
        self.embed = PreparedEmbedding(
            weights.draft.markov_embedding, tokens, embedding, session=session, name=name + ".embed"
        ).functional(session=session, dynamic_inputs=True)
        if isinstance(head, RowScaledFP8MarkovWeight):
            self.projection = PreparedRowScaledFP8MarkovProjection(embedding, head, projected,
                session=session, name=name + ".head").functional(session=session)
        else:
            projection = UnquantizedProjection if isinstance(head, torch.Tensor) else QuantizedProjection
            options = {} if isinstance(head, torch.Tensor) else {"activation_mode": "a16"}
            self.projection = projection(embedding, head, projected,
                                         session=session, name=name + ".head", **options).functional(session=session)
        self.gather = None if runtime is None else PreparedAllGather(
            local.view(torch.bfloat16),
            logits.view(torch.bfloat16),
            runtime=runtime,
            coordinator=coordinator,
            session=session,
            name=name + ".gather",
        ).functional(session=session)

    def __call__(self, tokens):
        logits = self.projection(self.embed(token_ids=tokens)).float()
        if self.gather is not None:
            logits = self.gather(logits.view(torch.bfloat16)).view(torch.float32)
        return logits


class _ShardedMainProjection:
    """Run an output-sharded ``mtp.0.main_proj`` projection and gather its BF16 columns."""

    def __init__(self, local, gather):
        self.local = local
        self.gather = gather

    def __call__(self, auxiliary):
        return self.gather(self.local(auxiliary))


class _ChunkedRoceAllGather:
    """Row-chunked column all-gather for RoCE payloads exceeding ``runtime.max_gather_bytes``."""

    def __init__(self, source, output, *, runtime, session, coordinator, name):
        from hyperflare_recipes.components import collectives

        rows = source.shape[0]
        row_bytes = source.shape[1] * source.element_size()
        step = runtime.max_gather_bytes // row_bytes
        if step < 1:
            raise ValueError("one draft context row exceeds the RoCE all-gather capacity")
        self.bounds = tuple((start, min(start + step, rows)) for start in range(0, rows, step))
        self.parts = tuple(
            collectives.PreparedAllGather(
                source[begin:end],
                output[begin:end],
                runtime=runtime,
                session=session,
                coordinator=coordinator,
                name=name if len(self.bounds) == 1 else f"{name}.{index}",
            ).functional(session=session)
            for index, (begin, end) in enumerate(self.bounds)
        )
        self.output_layout = (tuple(output.shape), output.stride(), output.dtype)

    def __call__(self, source):
        shape, stride, dtype = self.output_layout
        output = torch.empty_strided(shape, stride, dtype=dtype, device=source.device)
        for (begin, end), part in zip(self.bounds, self.parts, strict=True):
            part(source[begin:end], output[begin:end])
        return output


def _prepare_context_gather(source, output, *, runtime, coordinator, session, name):
    """Prepare a functional BF16 column all-gather sized for any context row count."""
    from hyperflare_recipes.components import collectives

    if collectives._roce(runtime) and source.numel() * source.element_size() > runtime.max_gather_bytes:
        return _ChunkedRoceAllGather(
            source, output, runtime=runtime, session=session, coordinator=coordinator, name=name
        )
    return collectives.PreparedChunkedAllGather(
        source, output, runtime=runtime, session=session, coordinator=coordinator, name=name
    ).functional(session=session)
