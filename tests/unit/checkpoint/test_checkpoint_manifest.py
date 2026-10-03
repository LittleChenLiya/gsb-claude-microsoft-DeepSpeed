# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import json
import torch
import pytest

from deepspeed.runtime.checkpoint_engine.torch_checkpoint_engine import (
    TorchCheckpointEngine,
    MANIFEST_FILENAME,
    TEMP_FILE_SUFFIX,
    MANIFEST_STATUS_PENDING,
    MANIFEST_STATUS_COMPLETE,
)
from deepspeed.runtime.checkpoint_engine.checkpoint_engine import CheckpointCommitInfo


class FailingShardEngine(TorchCheckpointEngine):
    """Test engine that raises OSError on a specified shard index."""

    def __init__(self, fail_on_shard_index=None, config_params=None):
        super().__init__(config_params)
        self._fail_on_shard_index = fail_on_shard_index
        self._shard_index = 0

    def save(self, state_dict, path: str):
        if self._fail_on_shard_index is not None and self._shard_index == self._fail_on_shard_index:
            self._shard_index += 1
            raise OSError("Simulated write failure for shard " + path)
        self._shard_index += 1
        super().save(state_dict, path)


def _read_manifest(tag_dir):
    manifest_path = os.path.join(tag_dir, MANIFEST_FILENAME)
    if not os.path.exists(manifest_path):
        return None
    with open(manifest_path, "r") as f:
        return json.load(f)


def _list_files(tag_dir):
    if not os.path.exists(tag_dir):
        return []
    return sorted(os.listdir(tag_dir))


class TestCheckpointManifest:
    """Tests for the checkpoint write manifest feature."""

    def test_successful_write_produces_complete_manifest(self, tmpdir):
        save_dir = str(tmpdir / "save_dir")
        os.makedirs(save_dir, exist_ok=True)
        tag = "global_step100"
        tag_dir = os.path.join(save_dir, tag)

        engine = TorchCheckpointEngine()
        info = CheckpointCommitInfo(tag=tag, save_dir=save_dir, save_latest=True)
        engine.create(info)

        for i in range(3):
            path = os.path.join(tag_dir, "mp_rank_%02d_model_states.pt" % i)
            engine.save({"weights": torch.randn(4, 4), "step": 100 + i}, path)

        engine.commit(info)

        manifest = _read_manifest(tag_dir)
        files = _list_files(tag_dir)
        tmp_files = [f for f in files if f.endswith(TEMP_FILE_SUFFIX)]
        final_files = [f for f in files if f.endswith("_model_states.pt")]

        assert manifest is not None
        assert manifest["status"] == MANIFEST_STATUS_COMPLETE
        assert len(manifest["shards"]) == 3
        assert len(tmp_files) == 0
        assert len(final_files) == 3

    def test_successful_load_after_commit(self, tmpdir):
        save_dir = str(tmpdir / "save_dir")
        os.makedirs(save_dir, exist_ok=True)
        tag = "global_step100"
        tag_dir = os.path.join(save_dir, tag)

        engine = TorchCheckpointEngine()
        info = CheckpointCommitInfo(tag=tag, save_dir=save_dir, save_latest=True)
        engine.create(info)

        shard_paths = []
        for i in range(3):
            path = os.path.join(tag_dir, "mp_rank_%02d_model_states.pt" % i)
            shard_paths.append(path)
            engine.save({"weights": torch.randn(4, 4), "step": 100 + i}, path)

        engine.commit(info)

        sd = engine.load(shard_paths[0])
        assert sd["step"] == 100

    def test_failed_shard_write_leaves_pending_manifest(self, tmpdir):
        save_dir = str(tmpdir / "save_dir")
        os.makedirs(save_dir, exist_ok=True)
        tag = "global_step200"
        tag_dir = os.path.join(save_dir, tag)

        engine = FailingShardEngine(fail_on_shard_index=1)
        info = CheckpointCommitInfo(tag=tag, save_dir=save_dir, save_latest=True)
        engine.create(info)

        save_error = None
        for i in range(3):
            path = os.path.join(tag_dir, "mp_rank_%02d_model_states.pt" % i)
            try:
                engine.save({"weights": torch.randn(4, 4), "step": 200 + i}, path)
            except OSError as e:
                save_error = e
                break

        manifest = _read_manifest(tag_dir)
        files = _list_files(tag_dir)
        tmp_files = [f for f in files if f.endswith(TEMP_FILE_SUFFIX)]
        final_files = [f for f in files if f.endswith("_model_states.pt")]

        assert save_error is not None
        assert manifest is not None
        assert manifest["status"] == MANIFEST_STATUS_PENDING
        assert len(tmp_files) >= 1
        assert len(final_files) == 0

    def test_failed_write_rejects_new_version_load(self, tmpdir):
        save_dir = str(tmpdir / "save_dir")
        os.makedirs(save_dir, exist_ok=True)
        tag = "global_step200"
        tag_dir = os.path.join(save_dir, tag)

        engine = FailingShardEngine(fail_on_shard_index=1)
        info = CheckpointCommitInfo(tag=tag, save_dir=save_dir, save_latest=True)
        engine.create(info)

        shard_paths = []
        for i in range(3):
            path = os.path.join(tag_dir, "mp_rank_%02d_model_states.pt" % i)
            shard_paths.append(path)
            try:
                engine.save({"weights": torch.randn(4, 4), "step": 200 + i}, path)
            except OSError:
                break

        with pytest.raises(RuntimeError, match="not complete"):
            engine.load(shard_paths[0])

    def test_failed_write_keeps_old_version(self, tmpdir):
        save_dir = str(tmpdir / "save_dir")
        os.makedirs(save_dir, exist_ok=True)

        # Write and commit an old complete checkpoint.
        old_tag = "global_step100"
        old_tag_dir = os.path.join(save_dir, old_tag)
        engine = TorchCheckpointEngine()
        old_info = CheckpointCommitInfo(tag=old_tag, save_dir=save_dir, save_latest=True)
        engine.create(old_info)
        old_shard = os.path.join(old_tag_dir, "mp_rank_00_model_states.pt")
        engine.save({"weights": torch.randn(4, 4), "step": 100}, old_shard)
        engine.commit(old_info)
        with open(os.path.join(save_dir, "latest"), "w") as f:
            f.write(old_tag)

        # Attempt to write a new checkpoint that fails on shard index 1.
        new_tag = "global_step200"
        new_tag_dir = os.path.join(save_dir, new_tag)
        engine2 = FailingShardEngine(fail_on_shard_index=1)
        new_info = CheckpointCommitInfo(tag=new_tag, save_dir=save_dir, save_latest=True)
        engine2.create(new_info)
        for i in range(3):
            path = os.path.join(new_tag_dir, "mp_rank_%02d_model_states.pt" % i)
            try:
                engine2.save({"weights": torch.randn(4, 4), "step": 200 + i}, path)
            except OSError:
                break

        # Old version must still be loadable.
        sd = engine.load(old_shard)
        assert sd["step"] == 100

        # Load selection must fall back to the old complete checkpoint.
        selected = engine.select_latest_complete(save_dir)
        assert selected == old_tag

    def test_commit_failure_on_missing_temp_file(self, tmpdir):
        save_dir = str(tmpdir / "save_dir")
        os.makedirs(save_dir, exist_ok=True)
        tag = "global_step300"
        tag_dir = os.path.join(save_dir, tag)

        engine = TorchCheckpointEngine()
        info = CheckpointCommitInfo(tag=tag, save_dir=save_dir, save_latest=True)
        engine.create(info)

        shard_paths = []
        for i in range(2):
            path = os.path.join(tag_dir, "mp_rank_%02d_model_states.pt" % i)
            shard_paths.append(path)
            engine.save({"weights": torch.randn(4, 4), "step": 300 + i}, path)

        # Remove one temp file to simulate a missing shard.
        tmp_to_remove = shard_paths[0] + TEMP_FILE_SUFFIX
        os.remove(tmp_to_remove)

        with pytest.raises(RuntimeError, match="temp file missing"):
            engine.commit(info)

        manifest = _read_manifest(tag_dir)
        assert manifest is not None
        assert manifest["status"] == MANIFEST_STATUS_PENDING

    def test_legacy_checkpoint_without_manifest_still_loads(self, tmpdir):
        save_dir = str(tmpdir / "save_dir")
        os.makedirs(save_dir, exist_ok=True)
        tag = "legacy_step"
        tag_dir = os.path.join(save_dir, tag)
        os.makedirs(tag_dir, exist_ok=True)

        shard_path = os.path.join(tag_dir, "mp_rank_00_model_states.pt")
        torch.save({"weights": torch.randn(2, 2), "step": 42}, shard_path)

        engine = TorchCheckpointEngine()
        sd = engine.load(shard_path)
        assert sd["step"] == 42

    def test_select_latest_complete_fallback(self, tmpdir):
        save_dir = str(tmpdir / "save_dir")
        os.makedirs(save_dir, exist_ok=True)

        # Old complete checkpoint.
        old_tag = "global_step100"
        old_tag_dir = os.path.join(save_dir, old_tag)
        engine = TorchCheckpointEngine()
        old_info = CheckpointCommitInfo(tag=old_tag, save_dir=save_dir, save_latest=True)
        engine.create(old_info)
        old_shard = os.path.join(old_tag_dir, "mp_rank_00_model_states.pt")
        engine.save({"weights": torch.randn(4, 4), "step": 100}, old_shard)
        engine.commit(old_info)

        # New checkpoint that fails on the first shard.
        new_tag = "global_step200"
        new_tag_dir = os.path.join(save_dir, new_tag)
        engine2 = FailingShardEngine(fail_on_shard_index=0)
        new_info = CheckpointCommitInfo(tag=new_tag, save_dir=save_dir, save_latest=True)
        engine2.create(new_info)
        try:
            engine2.save({"weights": torch.randn(4, 4), "step": 200},
                         os.path.join(new_tag_dir, "mp_rank_00_model_states.pt"))
        except OSError:
            pass

        # Point latest at the failed checkpoint.
        with open(os.path.join(save_dir, "latest"), "w") as f:
            f.write(new_tag)

        # Must fall back to the old complete checkpoint.
        selected = engine.select_latest_complete(save_dir)
        assert selected == old_tag
