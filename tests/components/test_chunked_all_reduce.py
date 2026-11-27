# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for PreparedChunkedAllReduce range dispatch and ConcurrentChunkedAllReduce."""

from types import SimpleNamespace

import pytest
import torch

from hyperflare_recipes.components import collectives


def test_prepared_chunked_all_reduce_ranges_and_dependencies():
    calls = []

    class FakePart:
        def __init__(self, idx):
            self.idx = idx
            self.output_layout = ((4, 8), (8, 1), torch.bfloat16)

        def __call__(self, source, output=None, *, dependencies=()):
            calls.append((self.idx, tuple(source.shape), len(dependencies)))
            output.copy_(source + self.idx + 1)
            return output

    chunked = object.__new__(collectives.PreparedChunkedAllReduce)
    chunked.bounds = [(0, 4), (4, 8), (8, 12)]
    chunked.parts = [FakePart(0), FakePart(1), FakePart(2)]
    chunked.output = None
    chunked.output_layout = ((12, 8), (8, 1), torch.bfloat16)

    dep = torch.zeros(1)
    inp = torch.zeros((8, 8), dtype=torch.bfloat16)
    out = chunked(inp, parts=(1, 3), dependencies=(dep,))
    assert out.shape == (8, 8)
    assert torch.equal(out[:4], torch.full((4, 8), 2, dtype=torch.bfloat16))
    assert torch.equal(out[4:], torch.full((4, 8), 3, dtype=torch.bfloat16))
    # First chunk carries caller dep; second chunk chains prior chunk output as dependency.
    assert calls == [(1, (4, 8), 1), (2, (4, 8), 2)]

    with pytest.raises(ValueError, match="out of bounds"):
        chunked(inp, parts=(2, 4))
    with pytest.raises(ValueError, match="expected 8"):
        chunked(torch.zeros((4, 8), dtype=torch.bfloat16), parts=(1, 3))


def test_concurrent_chunked_all_reduce_chains_starts_and_joins(monkeypatch):
    started = []
    joined = []

    class FakeReduce:
        def __init__(self, handle, rows):
            self.handle = handle
            self.output = None
            self.output_layout = ((rows, 8), (8, 1), torch.bfloat16)
            self.concurrent_input = None

        def functional(self, *, session):
            return self

    class FakeConcurrent:
        def __init__(self, part, *, workspace, session, parallel_plans, send_buffer=None):
            self.reduce = part
            self.stream_handle = workspace.auxiliary_stream.cuda_stream
            self.send_buffer = send_buffer
            self.validated = False

        def validate_scratch(self):
            self.validated = True

        def functional(self, *, session):
            self.reduce.functional(session=session)
            return self

    monkeypatch.setattr(collectives, "ConcurrentAllReduce", FakeConcurrent)
    monkeypatch.setattr(
        collectives,
        "_start_all_reduce",
        lambda send, out, deps, handle, stream: started.append((handle, tuple(send.shape), len(deps), stream)),
    )
    monkeypatch.setattr(
        collectives,
        "_join_all_reduce",
        lambda send, out, deps, dep, stream: joined.append((tuple(out.shape), len(deps), stream)),
    )

    workspace = SimpleNamespace(auxiliary_stream=SimpleNamespace(cuda_stream=77))
    parts = [FakeReduce(10, 4), FakeReduce(11, 4)]
    cc = collectives.ConcurrentChunkedAllReduce(
        parts, [(4, 8), (8, 12)], workspace=workspace, session=None, parallel_plans=()
    ).functional(session=None)
    cc.validate_scratch()
    assert all(p.validated for p in cc.parts)

    src = torch.randn((8, 8), dtype=torch.bfloat16)
    dep = torch.zeros(1)
    pending = cc.start(src, dependencies=(dep,))
    assert started == [(10, (4, 8), 1, 77), (11, (4, 8), 2, 77)]
    res = cc.finish(pending, torch.zeros(1))
    assert res.shape == (8, 8)
    assert joined == [((8, 8), 2, 77)]

    with pytest.raises(ValueError, match="contiguous bounds"):
        collectives.ConcurrentChunkedAllReduce(
            parts, [(0, 4), (5, 8)], workspace=workspace, session=None, parallel_plans=()
        )
