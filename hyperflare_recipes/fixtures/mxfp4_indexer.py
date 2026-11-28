# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Populated short/full-context MXFP4 indexer workloads for kernel selection."""

from functools import partial

import torch
from b12x.preparation import PreparedCall

from ignition.preparation.scratch import planned_scratch
from ignition.preparation.tuning import benchmark_resource


class Inputs:
    def __init__(self, state, original):
        caps = state.caps
        pages, page_size = caps.max_page_table_width, caps.page_size
        self.arguments = {name: None if value is None else torch.empty_like(value)
                          for name, value in original.items() if name != "index_k_cache"}
        a = self.arguments
        device = a["q_mxfp4"].device
        generator = torch.Generator(device=device).manual_seed(410512)
        self.query = torch.randn((*a["q_mxfp4"].shape[:-1], 128), dtype=torch.bfloat16,
                                 device=device, generator=generator) * .2
        state.quantize_query(self.query, q_mxfp4=a["q_mxfp4"], q_scales=a["q_scales"])
        a["query_weights"].normal_(std=1 / 64, generator=generator)
        a["index_k_cache"] = torch.empty((pages, page_size * 68), dtype=torch.uint8, device=device)
        keys = torch.randn((pages * page_size, 128), dtype=torch.bfloat16, device=device, generator=generator) * .2
        state.write_index_keys(keys, index_k_cache=a["index_k_cache"],
                               slot_mapping=torch.arange(pages * page_size, device=device))
        a["page_table"].copy_(torch.arange(pages, dtype=torch.int32, device=device)[None])
        self.expanded_page_table = None
        if a.get("request_ids") is not None:
            rows = a["request_ids"].shape[0]
            num_requests = a["page_table"].shape[0]
            a["request_ids"].copy_(
                (torch.arange(rows, dtype=torch.int32, device=device) * num_requests) // max(rows, 1)
            )
            if num_requests not in (1, rows):
                self.expanded_page_table = a["page_table"][a["request_ids"].long()].contiguous()
        a["active_width"].fill_(pages * page_size)
        if a["candidate_indices"] is not None:
            width = a["candidate_indices"].shape[1]
            candidates = torch.arange(width, dtype=torch.int32, device=device)
            a["candidate_indices"].copy_(torch.where(candidates < pages * page_size, candidates, -1)[None])
            a["candidate_lengths"].fill_(min(width, pages * page_size))
        self.producers = tuple(partial(a["cache_lengths"].fill_, length)
                               for length in sorted({min(512, pages * page_size), pages * page_size}))
        self.producers[-1]()

    def bind_arguments(self, state):
        if self.arguments.get("request_ids") is None:
            return self.arguments
        uses_multiquery = (
            getattr(state, "multiquery_dense", None) is not None
            or getattr(state, "multiquery_hierarchical", None) is not None
        )
        if uses_multiquery:
            return self.arguments
        adapted = {key: value for key, value in self.arguments.items() if key != "request_ids"}
        if self.expanded_page_table is not None:
            adapted["page_table"] = self.expanded_page_table
        return adapted

    @property
    def nbytes(self):
        extra = 0 if self.expanded_page_table is None else self.expanded_page_table.nbytes
        return self.query.nbytes + extra + sum(t.nbytes for t in self.arguments.values() if t is not None)


def acquire_inputs(state, arguments, session):
    key = tuple((name, None if value is None else
        (value.shape if name != "index_k_cache" else value.shape[1:], value.dtype, value.device))
        for name, value in arguments.items())
    return benchmark_resource(session, ("mxfp4_indexer", key), lambda: Inputs(state, arguments))


def benchmark_call(state, *, arguments, session, workspace):
    inputs = acquire_inputs(state, arguments, session)
    scratch = planned_scratch(state.layout.scratch_specs(), workspace)
    binding = state.bind(scratch=scratch, **inputs.bind_arguments(state))
    return PreparedCall(run=lambda: state.run(binding), produce=inputs.producers[0],
        benchmark_producers=inputs.producers, owners=(inputs, scratch, binding))
