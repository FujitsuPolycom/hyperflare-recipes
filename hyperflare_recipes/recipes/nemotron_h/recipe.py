# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Nemotron-H Bolt and Super VL preparation capacities and serving defaults."""

from pathlib import Path
from types import MappingProxyType

from ignition.declaration import Checkpoint, RankCosts, RankUsage, Recipe, SamplingDefaults, Speculation


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False,
            core_storage=None, ranks=None):
    from .bootstrap import PreparationConfig, prepare as prepare_model, prepare_rank

    if plan.world_size not in (1, 2):
        raise ValueError("Nemotron-H preparation requires one or two GPUs")
    if plan.world_size == 1 and ranks is not None:
        raise ValueError("single-rank Nemotron-H preparation does not use a multi-host rendezvous")
    drafting = {}
    if plan.speculation is not None:
        from huggingface_hub import snapshot_download
        from .draft_config import draft_source

        repo, rev = draft_source(plan.checkpoint.repository, Path(snapshot))
        draft = snapshot_download(repo, revision=rev, cache_dir=plan.hub_cache,
                                  allow_patterns=("config.json", "model.safetensors"))
        drafting = dict(draft_checkpoint=draft, speculative_widths=plan.speculation.widths,
            speculative_fixed_width=plan.speculation.fixed_width, speculative_token_capacity=plan.speculative_token_capacity,
            speculative_graph_budget=plan.speculation.graph_budget or 64)
    config = PreparationConfig(mtp_only=plan.speculation is not None and plan.speculation.mtp_only,
        prompt_logprobs=plan.prompt_logprobs,
        compile_workers=plan.compile_workers, autotune_effort=plan.autotune_effort, tuning_replica=plan.tuning_replica,
        native_dense_activations=plan.native_dense_activations, benchmark_selection=plan.benchmark_selection,
        lock_gpu_clocks=plan.lock_gpu_clocks and plan.autotune_effort != "off",
        devices=tuple(range(plan.world_size)) if ranks is None else (0,) * plan.world_size,
        hosts=None if ranks is None else ranks.hosts,
        roce_ring=plan.roce_ring, roce_chain=plan.roce_chain,
        max_active=plan.concurrency, image_concurrency=plan.image_concurrency, context=plan.recipe.context,
        stream_overlap=plan.stream_overlap,
        prefill_chunk_size=plan.prefill_chunk, snapshot_budget_bytes=plan.snapshots * plan.costs.state_row_bytes,
        headroom_bytes=plan.headroom_bytes + plan.reserve_bytes, compile_decoder=compile_forwards,
        retain_graphs=retain_graphs, kernel_cache=kernel_cache, hub_cache=plan.hub_cache, **drafting)
    if plan.world_size == 1:
        return prepare_model(snapshot, config, progress=progress, core_storage=core_storage)
    from ignition.preparation.rank_group import prepare_group

    return prepare_group(snapshot, config, prepare_rank, progress=progress, core_storage=core_storage,
                         ranks=ranks, checkpoint=plan.checkpoint)


def usage(prepared):
    report = prepared.memory_report
    return tuple(RankUsage(rank["pages"], rank["free_device_bytes"]) for rank in report.get("ranks", (report,)))


RECIPE = Recipe(
    name="nemotron_h", prepare=prepare, usage=usage,
    served_model_name="NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB",
    # Text sampling defaults verified against both model cards:
    # https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB
    # https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4.
    sampling_defaults=SamplingDefaults(temperature=1., top_p=.95, top_k=-1, min_p=0.,
        repetition_penalty=1., presence_penalty=0., frequency_penalty=0.),
    checkpoints=(
        Checkpoint(alias="nemotron-bolt-24gb", repository="nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB",
            revision="c6cc2e8ffa353403213d6b9b18aaf8eb684d705a", architecture="NemotronH_Omni_Reasoning_V3",
            stored_bytes=24_079_678_824, device_bytes=25_063_117_344, draft_bytes=983_438_520,
            ranks={1: RankCosts(107_365_120, 524_288, 4 << 30)},
            plain_ranks={1: RankCosts(86_876_416, 524_288, 4 << 30)}),
        # Super VL NVFP4 skips the 1.65 GiB MTP tensors and video embedder, loading backbone,
        # C-RADIO vision (with vision_final_layernorm), and LiLiCorr draft weights. Plain dicts
        # keep checkpoint metadata picklable across multi-host preparation ranks.
        Checkpoint(alias="nemotron-3.5-super-vl-nvfp4", repository="nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4",
            revision="ae1368329b1fc76b71534c5d04b87b35d9cc6e15", architecture="NemotronH_Omni_Reasoning_V3",
            served_model_name="NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4",
            stored_bytes=74_986_817_096, device_bytes=74_531_164_588, draft_bytes=1_317_933_948,
            expansion_bytes=6 << 30, decode_weight_bytes=9_765_491_976,
            ranks={
                1: RankCosts(188_629_760, 524_288, 6 << 30),
                2: RankCosts(94_323_456, 262_144, 6 << 30),
            },
            plain_ranks={
                1: RankCosts(86_876_416, 524_288, 6 << 30),
                2: RankCosts(43_442_432, 262_144, 6 << 30),
            }),
    ),
    context=262144, page_tokens=128, concurrency=32, max_concurrency=64, prefill_chunk=4096,
    speculation=Speculation(draft_tokens=7, token_capacity=128, packed_verification=True, graph_budget=64), max_draft_tokens=7, verification_rows=0,
    max_images=4, image_bytes=32 << 20,
    stream_overlap=True,
    benchmark_world_sizes=(1,),
    ranks=MappingProxyType({
        1: RankCosts(107_365_120, 524_288, 4 << 30),
        2: RankCosts(94_323_456, 262_144, 6 << 30),
    }),
    plain_ranks=MappingProxyType({
        1: RankCosts(86_876_416, 524_288, 4 << 30),
        2: RankCosts(43_442_432, 262_144, 6 << 30),
    }),
)
