# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Read-only ZeRO-3 parameter partition ownership queries.

:meth:`deepspeed.zero.Init.get_partition_ownership` (and the
:func:`compute_zero_param_partitions` helper it wraps) report which rank owns
which half-open ``[start, end)`` interval of a flattened parameter, without
triggering any parameter all-gather. The intervals are non-overlapping and
cover exactly the parameter's real elements; an empty parameter yields empty
intervals on every rank.

Run the pure (single-process) tests::

    pytest tests/unit/runtime/zero/test_zero_partition_ownership.py -k "not distributed"

Run the 2-process CPU example (works on any platform, including Windows where
the pytest ``DistributedTest`` harness cannot use ``forkserver``)::

    python tests/unit/runtime/zero/test_zero_partition_ownership.py
"""

import os
import sys

# Allow running this file directly (e.g. ``python test_zero_partition_ownership.py``)
# where the ``unit`` package under ``tests/`` is not yet on ``sys.path``.
_TESTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import pytest
import torch

import deepspeed
import deepspeed.comm as dist
from deepspeed.comm.comm import init_distributed
from deepspeed.runtime.zero.partition_parameters import (ZeroParamPartition, ZeroParamStatus,
                                                         compute_zero_param_partitions)

from unit.common import DistributedTest

WORLD_SIZE = 2


# ---------------------------------------------------------------------------
# Pure, single-process tests (no distributed initialization required)
# ---------------------------------------------------------------------------


def _check_coverage_and_nonoverlap(numel, world_size):
    parts = compute_zero_param_partitions(numel, world_size)
    assert len(parts) == world_size
    assert [p.rank for p in parts] == list(range(world_size))
    # Every interval is well-formed ...
    for p in parts:
        assert 0 <= p.start <= p.end <= numel
    # ... and intervals do not overlap.
    for left, right in zip(parts, parts[1:]):
        assert left.end <= right.start
    # Coverage: the union of intervals is exactly [0, numel).
    assert parts[0].start == 0
    assert parts[-1].end == numel
    assert sum(p.numel for p in parts) == numel
    return parts


def test_divisible_partition():
    parts = _check_coverage_and_nonoverlap(8, 2)
    assert parts == [ZeroParamPartition(0, 0, 4), ZeroParamPartition(1, 4, 8)]


def test_non_divisible_partition():
    # numel=5, world=2: aligned to 6; rank 1 owns [3, 5) (2 real elements),
    # the 6th slot is padding owned by no one.
    parts = _check_coverage_and_nonoverlap(5, 2)
    assert parts == [ZeroParamPartition(0, 0, 3), ZeroParamPartition(1, 3, 5)]


def test_empty_parameter():
    # numel=0: every rank reports a deterministic empty interval.
    parts = _check_coverage_and_nonoverlap(0, 2)
    assert len(parts) == 2
    assert all(p.is_empty for p in parts)
    assert all(p.numel == 0 for p in parts)
    assert parts == [ZeroParamPartition(0, 0, 0), ZeroParamPartition(1, 0, 0)]


def test_numel_less_than_world_size():
    # numel=2, world=4: ranks 2 and 3 own nothing.
    parts = _check_coverage_and_nonoverlap(2, 4)
    assert parts[0] == ZeroParamPartition(0, 0, 1)
    assert parts[1] == ZeroParamPartition(1, 1, 2)
    assert parts[2].is_empty and parts[3].is_empty
    assert parts[2] == ZeroParamPartition(2, 2, 2)
    assert parts[3] == ZeroParamPartition(3, 2, 2)


def test_single_rank_owns_full_parameter():
    parts = _check_coverage_and_nonoverlap(5, 1)
    assert parts == [ZeroParamPartition(0, 0, 5)]


def test_three_ranks_seven_elements():
    # numel=7, world=3: aligned to 9; rank 2 owns [6, 7).
    parts = _check_coverage_and_nonoverlap(7, 3)
    assert parts == [ZeroParamPartition(0, 0, 3), ZeroParamPartition(1, 3, 6), ZeroParamPartition(2, 6, 7)]


def test_invalid_world_size_raises():
    with pytest.raises(ValueError):
        compute_zero_param_partitions(5, 0)


def test_negative_numel_raises():
    with pytest.raises(ValueError):
        compute_zero_param_partitions(-1, 2)


# ---------------------------------------------------------------------------
# Distributed 2-process verification
# ---------------------------------------------------------------------------


class _ToyModel(torch.nn.Module):
    """Parameters of lengths 5 (non-divisible), 0 (empty), 8 (divisible), 2 (< world)."""

    def __init__(self):
        super().__init__()
        self.p5 = torch.nn.Parameter(torch.arange(5, dtype=torch.float32))
        self.p0 = torch.nn.Parameter(torch.zeros(0, dtype=torch.float32))
        self.p8 = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))
        self.p2 = torch.nn.Parameter(torch.arange(2, dtype=torch.float32))

    def forward(self, x):
        return x


def verify_partition_ownership():
    """Run the 2-process ZeRO-3 partition ownership verification.

    Must be called inside a ``deepspeed.init_distributed`` process group with
    ``WORLD_SIZE=2``.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == WORLD_SIZE, f"expected world_size={WORLD_SIZE}, got {world_size}"

    ctx = deepspeed.zero.Init(remote_device="cpu")
    with ctx:
        model = _ToyModel()
    dist.barrier()

    expected = {
        "p5": {
            0: ZeroParamPartition(0, 0, 3),
            1: ZeroParamPartition(1, 3, 5),
        },
        "p0": {
            0: ZeroParamPartition(0, 0, 0),
            1: ZeroParamPartition(1, 0, 0),
        },
        "p8": {
            0: ZeroParamPartition(0, 0, 4),
            1: ZeroParamPartition(1, 4, 8),
        },
        "p2": {
            0: ZeroParamPartition(0, 0, 1),
            1: ZeroParamPartition(1, 1, 2),
        },
    }

    for name, param in model.named_parameters():
        table = ctx.get_partition_ownership(param)
        local = ctx.get_local_partition(param)

        # Read-only guarantee: the query must not aggregate parameters. Sabotage
        # every gather entry point; if the query triggers one, this raises.
        def _boom(*args, **kwargs):
            raise RuntimeError(f"read-only query triggered parameter all-gather for {name}")

        param.all_gather = _boom
        orig_all_gather = dist.all_gather
        orig_all_gather_into_tensor = dist.all_gather_into_tensor
        dist.all_gather = _boom
        dist.all_gather_into_tensor = _boom
        try:
            table_again = ctx.get_partition_ownership(param)
            local_again = ctx.get_local_partition(param)
        finally:
            dist.all_gather = orig_all_gather
            dist.all_gather_into_tensor = orig_all_gather_into_tensor

        assert table == table_again, f"{name}: ownership table changed across read-only calls"
        assert local == local_again, f"{name}: local partition changed across read-only calls"

        # The parameter stays partitioned / not available; the query does not
        # materialize the full tensor.
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE, (
            f"{name}: query changed ds_status to {param.ds_status}")
        assert param.ds_tensor is not None, f"{name}: expected a partition buffer"

        # Coverage and non-overlap across ranks.
        assert [p.rank for p in table] == list(range(world_size))
        for p in table:
            assert 0 <= p.start <= p.end <= param.ds_numel, f"{name}: malformed interval {p}"
        for left, right in zip(table, table[1:]):
            assert left.end <= right.start, f"{name}: overlapping intervals {left} {right}"
        assert table[0].start == 0, f"{name}: coverage does not start at 0"
        assert table[-1].end == param.ds_numel, f"{name}: coverage does not end at numel"
        assert sum(p.numel for p in table) == param.ds_numel, f"{name}: coverage is not exact"

        # The local rank's interval matches the expected ownership table.
        assert local == expected[name][rank], (
            f"{name}: rank {rank} expected {expected[name][rank]}, got {local}")
        assert table[rank] == expected[name][rank]

        # The local partition buffer is at least as large as the owned valid
        # interval (the buffer may hold padding for the last rank).
        assert local.numel <= param.ds_tensor.numel(), (
            f"{name}: owned {local.numel} > buffer {param.ds_tensor.numel()}")

    dist.barrier()
    if rank == 0:
        by_name = dict(model.named_parameters())
        rows = "\n".join(
            f"    {n:>3} (numel={by_name[n].ds_numel:>2}): "
            + ", ".join(f"r{p.rank}:[{p.start},{p.end})" for p in ctx.get_partition_ownership(by_name[n]))
            for n in expected)
        print(f"[rank 0] ZeRO-3 partition ownership (world_size={world_size}):\n" + rows, flush=True)
        print("[rank 0] OK: coverage exact, intervals non-overlapping, no aggregation triggered", flush=True)


class TestZeroPartitionOwnershipDistributed(DistributedTest):
    """Distributed verification for CI (uses the pytest ``DistributedTest`` harness)."""

    world_size = WORLD_SIZE
    requires_cuda_env = False
    backend = "gloo"
    non_daemonic_procs = True  # spawn instead of forkserver (forkserver is not available on all platforms)

    def test_partition_ownership(self):
        verify_partition_ownership()


# ---------------------------------------------------------------------------
# Minimal example: 2-process CPU launch using torch.multiprocessing.spawn
# (works on any platform, including Windows where forkserver is unavailable)
# ---------------------------------------------------------------------------


def _spawn_worker(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = os.environ.get("DS_TEST_MASTER_PORT", "29531")
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["LOCAL_SIZE"] = str(world_size)
    init_distributed(dist_backend="gloo", rank=rank, world_size=world_size)
    try:
        verify_partition_ownership()
    finally:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.spawn(_spawn_worker, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)
