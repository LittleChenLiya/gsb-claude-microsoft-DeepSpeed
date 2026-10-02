# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Read-only ZeRO partition ownership queries.

``compute_partition_ownership`` / ``get_param_partition_ownership`` report which
rank of the partition group owns which half-open element interval of a
ZeRO-partitioned parameter. The query is pure metadata arithmetic: it must never
trigger an all-gather of the real parameter, and it must give deterministic
results for empty parameters and for sizes that do not divide evenly.

This file is both a pytest module (see ``TestZeroPartitionOwnership``) and a
self-contained minimal example. Run it directly to execute the two-process CPU
verification without pytest:

    python tests/unit/runtime/zero/test_zero_partition_ownership.py
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch


def _alias_symlinked_packages():
    """Allow importing deepspeed from a source checkout whose directory
    symlinks (deepspeed/accelerator -> ../accelerator, deepspeed/ops/op_builder
    -> ../../op_builder) were not materialized, e.g. Windows checkouts with
    core.symlinks=false. No-op on checkouts where the symlinks exist."""
    repo_root = Path(__file__).resolve().parents[4]
    for path in (repo_root, repo_root / "tests"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    try:
        import deepspeed.accelerator  # noqa: F401
        return
    except ImportError:
        pass
    import importlib
    for alias, target in (("deepspeed.accelerator", "accelerator"), ("deepspeed.ops.op_builder", "op_builder")):
        if alias not in sys.modules:
            try:
                sys.modules[alias] = importlib.import_module(target)
            except ImportError:
                pass


_alias_symlinked_packages()

import deepspeed
import deepspeed.comm as dist
from deepspeed.runtime.zero import partition_parameters
from deepspeed.runtime.zero.partition_parameters import (PartitionOwnership, ZeroParamStatus,
                                                         compute_partition_ownership, get_param_partition_ownership)

from unit.common import DistributedTest

CONFIG_DICT = {
    "train_micro_batch_size_per_gpu": 1,
    "zero_optimization": {
        "stage": 3
    },
}

NUMEL = 5  # deliberately not divisible by the 2 ranks


class TinyModel(torch.nn.Module):
    """One parameter of 5 elements (uneven partition) and one empty parameter."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(NUMEL, dtype=torch.float32))
        self.empty = torch.nn.Parameter(torch.empty(0))

    def forward(self, x):
        return x


def _assert_coverage(ownership, numel, world_size):
    """The intervals must tile [0, numel) exactly: contiguous and non-overlapping."""
    assert len(ownership) == world_size
    assert [o.rank for o in ownership] == list(range(world_size))
    prev_end = 0
    for o in ownership:
        assert o.start == prev_end, f"gap or overlap before rank {o.rank}: {ownership}"
        assert 0 <= o.start <= o.end <= numel
        prev_end = o.end
    assert prev_end == numel, f"intervals do not cover [0, {numel}): {ownership}"


class TestComputePartitionOwnership:
    """Pure-arithmetic ownership layout; no distributed environment needed."""

    def test_even_division(self):
        assert compute_partition_ownership(4, 2) == [PartitionOwnership(0, 0, 2), PartitionOwnership(1, 2, 4)]

    def test_uneven_division(self):
        # ceil(5 / 2) == 3 elements per rank, clamped to the real 5 elements
        assert compute_partition_ownership(5, 2) == [PartitionOwnership(0, 0, 3), PartitionOwnership(1, 3, 5)]

    def test_empty_param(self):
        assert compute_partition_ownership(0, 2) == [PartitionOwnership(0, 0, 0), PartitionOwnership(1, 0, 0)]

    def test_single_rank(self):
        assert compute_partition_ownership(5, 1) == [PartitionOwnership(0, 0, 5)]

    def test_more_ranks_than_elements(self):
        ownership = compute_partition_ownership(2, 4)
        assert ownership == [
            PartitionOwnership(0, 0, 1),
            PartitionOwnership(1, 1, 2),
            PartitionOwnership(2, 2, 2),
            PartitionOwnership(3, 2, 2),
        ]

    def test_invalid_arguments(self):
        with pytest.raises(ValueError):
            compute_partition_ownership(5, 0)
        with pytest.raises(ValueError):
            compute_partition_ownership(-1, 2)

    @pytest.mark.parametrize("numel", [0, 1, 5, 6, 7, 100])
    @pytest.mark.parametrize("world_size", [1, 2, 3, 8])
    def test_coverage_and_non_overlap(self, numel, world_size):
        _assert_coverage(compute_partition_ownership(numel, world_size), numel, world_size)

    def test_plain_param_reports_single_owner(self):
        param = torch.nn.Parameter(torch.randn(4))
        assert get_param_partition_ownership(param) == [PartitionOwnership(0, 0, 4)]


def _verify_zero_init_ownership():
    """Two-rank verification of the ownership query against real ZeRO-3 partitions.

    Assumes deepspeed.comm is already initialized. Checks
    that the query matches the actual ds_tensor shard on each rank, that the
    intervals cover the parameter without overlapping, and that asking the
    question does not gather the parameter.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2

    with deepspeed.zero.Init(config_dict_or_path=CONFIG_DICT):
        model = TinyModel()

    weight, empty = model.weight, model.empty
    assert weight.ds_status == ZeroParamStatus.NOT_AVAILABLE
    assert empty.ds_status == ZeroParamStatus.NOT_AVAILABLE

    # The ownership query must be read-only: turn any all-gather into a failure.
    def _forbidden_all_gather(self, *args, **kwargs):
        raise AssertionError("partition ownership query triggered a parameter all-gather")

    original_all_gather = partition_parameters.Init._all_gather
    partition_parameters.Init._all_gather = _forbidden_all_gather
    try:
        ownership_w = get_param_partition_ownership(weight)
        ownership_e = get_param_partition_ownership(empty)
    finally:
        partition_parameters.Init._all_gather = original_all_gather

    # Querying must not have changed the partitioned state.
    assert weight.ds_status == ZeroParamStatus.NOT_AVAILABLE
    assert empty.ds_status == ZeroParamStatus.NOT_AVAILABLE
    assert weight.data.numel() == 0

    # numel == 5 over 2 ranks: [0, 3) on rank 0, [3, 5) on rank 1.
    assert ownership_w == [PartitionOwnership(0, 0, 3), PartitionOwnership(1, 3, 5)]
    _assert_coverage(ownership_w, NUMEL, world_size)

    # Empty parameter: deterministic all-empty intervals.
    assert ownership_e == [PartitionOwnership(0, 0, 0), PartitionOwnership(1, 0, 0)]
    _assert_coverage(ownership_e, 0, world_size)
    assert empty.ds_tensor.ds_numel == 0

    # The reported interval must match the shard this rank actually holds.
    mine = ownership_w[rank]
    assert weight.ds_tensor.ds_numel == 3  # ceil(5 / 2), padding included
    local = weight.ds_tensor[:mine.end - mine.start].flatten().cpu()
    expected = torch.arange(NUMEL, dtype=torch.float32)[mine.start:mine.end]
    assert torch.equal(local, expected), f"rank {rank}: shard {local} != elements [{mine.start}, {mine.end})"

    # Reassemble the parameter from the per-rank shards to prove the reported
    # intervals really tile the original data (test-side communication only).
    gathered = [None] * world_size
    dist.all_gather_object(gathered, (rank, local))
    shards = {r: t for r, t in gathered}
    reassembled = torch.cat([shards[r] for r in range(world_size)])
    assert torch.equal(reassembled, torch.arange(NUMEL, dtype=torch.float32))

    if rank == 0:
        print(f"ownership(numel=5, world_size=2): {ownership_w}")
        print(f"ownership(numel=0, world_size=2): {ownership_e}")
        print("coverage/non-overlap/shard-content checks passed")


class TestZeroPartitionOwnership(DistributedTest):
    world_size = 2

    def test_ownership_matches_actual_partitions(self):
        _verify_zero_init_ownership()


def _standalone_worker(rank, world_size, init_file):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29519'
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    deepspeed.init_distributed(dist_backend="gloo",
                               init_method=f"file://{init_file}",
                               rank=rank,
                               world_size=world_size)
    try:
        _verify_zero_init_ownership()
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _main():
    # 1) In-process checks of the pure ownership arithmetic.
    for numel, world_size in [(0, 2), (1, 2), (5, 2), (6, 3), (7, 4), (100, 8)]:
        _assert_coverage(compute_partition_ownership(numel, world_size), numel, world_size)
    print("pure ownership arithmetic: coverage/non-overlap OK for numel in {0,1,5,6,7,100}")

    # 2) Two CPU processes (gloo) partitioning numel-5 and numel-0 parameters.
    world_size = 2
    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, "dist_init")
        torch.multiprocessing.spawn(_standalone_worker, args=(world_size, init_file), nprocs=world_size, join=True)
    print("standalone 2-process CPU check passed: partitions cover [0, numel) without overlap")


if __name__ == "__main__":
    _main()
