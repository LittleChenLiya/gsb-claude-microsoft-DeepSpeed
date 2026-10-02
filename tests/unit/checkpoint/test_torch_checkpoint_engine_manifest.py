# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Unit tests for the commit manifest of TorchCheckpointEngine.

Behavior under test:
  * shards are written to temporary files and only become visible at commit;
  * commit() publishes a manifest marking the checkpoint complete only when
    every shard was written successfully;
  * when any shard write fails, commit() returns False, temporary files are
    discarded, the previous (complete) version is kept, and loads of the new
    version are rejected by the manifest.
"""

import json
import os

import pytest
import torch

from deepspeed.runtime.checkpoint_engine.checkpoint_engine import CheckpointCommitInfo
from deepspeed.runtime.checkpoint_engine.torch_checkpoint_engine import (
    TorchCheckpointEngine,
    CheckpointManifestError,
    MANIFEST_FILE_NAME,
    TMP_FILE_SUFFIX,
)


def _commit_info(save_dir, tag):
    return CheckpointCommitInfo(tag=tag, save_dir=str(save_dir), save_latest=True)


def _shard_path(save_dir, tag, name):
    return os.path.join(str(save_dir), tag, name)


def _save_shards(engine, save_dir, tag, names, value):
    info = _commit_info(save_dir, tag)
    engine.create(info)
    for name in names:
        engine.save({"shard": name, "value": value}, _shard_path(save_dir, tag, name))
    return info


def _read_manifest(save_dir, tag):
    with open(os.path.join(str(save_dir), tag, MANIFEST_FILE_NAME), "r") as fd:
        return json.load(fd)


def _tmp_files(save_dir, tag):
    tag_dir = os.path.join(str(save_dir), tag)
    if not os.path.isdir(tag_dir):
        return []
    return [f for f in os.listdir(tag_dir) if f.endswith(TMP_FILE_SUFFIX)]


class TestCommitManifest:

    def test_successful_save_writes_manifest_and_no_tmp_files(self, tmp_path):
        engine = TorchCheckpointEngine()
        tag = "global_step1"
        shards = ["mp_rank_00_model_states.pt", "zero_pp_rank_0_mp_rank_00_optim_states.pt"]

        info = _save_shards(engine, tmp_path, tag, shards, value=1)
        # before commit, shards exist only as temporary files
        assert sorted(_tmp_files(tmp_path, tag)) == sorted(name + TMP_FILE_SUFFIX for name in shards)
        assert not os.path.exists(_shard_path(tmp_path, tag, shards[0]))

        assert engine.commit(info) is True

        # manifest marks the checkpoint complete and lists every shard
        manifest = _read_manifest(tmp_path, tag)
        assert manifest["status"] == "complete"
        assert manifest["tag"] == tag
        assert sorted(manifest["shards"]) == sorted(shards)
        assert manifest["num_shards"] == len(shards)

        # temporary files are gone, final shards are in place
        assert _tmp_files(tmp_path, tag) == []
        for name in shards:
            assert os.path.isfile(_shard_path(tmp_path, tag, name))

        # committed shards load fine
        for name in shards:
            sd = engine.load(_shard_path(tmp_path, tag, name))
            assert sd == {"shard": name, "value": 1}

    def test_failed_shard_save_keeps_previous_version(self, tmp_path, monkeypatch):
        engine = TorchCheckpointEngine()
        tag = "global_step1"
        shards = ["mp_rank_00_model_states.pt", "zero_pp_rank_0_mp_rank_00_optim_states.pt"]

        # v1: complete checkpoint
        info_v1 = _save_shards(engine, tmp_path, tag, shards, value=1)
        assert engine.commit(info_v1) is True
        manifest_v1 = _read_manifest(tmp_path, tag)

        # v2: the second shard write fails
        info_v2 = _commit_info(tmp_path, tag)
        engine.create(info_v2)
        engine.save({"shard": shards[0], "value": 2}, _shard_path(tmp_path, tag, shards[0]))

        def failing_save(*args, **kwargs):
            raise RuntimeError("simulated shard write failure")

        monkeypatch.setattr("deepspeed.runtime.checkpoint_engine.torch_checkpoint_engine.torch.save", failing_save)
        with pytest.raises(RuntimeError, match="simulated shard write failure"):
            engine.save({"shard": shards[1], "value": 2}, _shard_path(tmp_path, tag, shards[1]))

        # commit must refuse to mark the incomplete checkpoint complete
        assert engine.commit(info_v2) is False

        # manifest is still the v1 manifest
        assert _read_manifest(tmp_path, tag) == manifest_v1

        # no temporary files are left behind
        assert _tmp_files(tmp_path, tag) == []

        # the previous version is preserved and still loadable
        for name in shards:
            sd = engine.load(_shard_path(tmp_path, tag, name))
            assert sd == {"shard": name, "value": 1}

    def test_commit_fails_when_shard_tmp_file_is_missing(self, tmp_path):
        engine = TorchCheckpointEngine()
        tag = "global_step1"
        shards = ["layer_00.pt", "layer_01.pt"]

        info = _save_shards(engine, tmp_path, tag, shards, value=1)
        # simulate losing one shard's temporary file before commit
        os.remove(_shard_path(tmp_path, tag, shards[1]) + TMP_FILE_SUFFIX)

        assert engine.commit(info) is False
        # remaining temporary files are discarded, no manifest is published
        assert _tmp_files(tmp_path, tag) == []
        assert not os.path.exists(os.path.join(str(tmp_path), tag, MANIFEST_FILE_NAME))

    def test_load_rejects_shard_not_in_manifest(self, tmp_path):
        engine = TorchCheckpointEngine()
        tag = "global_step1"
        shards = ["mp_rank_00_model_states.pt"]

        info = _save_shards(engine, tmp_path, tag, shards, value=1)
        assert engine.commit(info) is True

        # a new-version shard that was never committed (e.g. left behind by a
        # crashed save) must be rejected even though the file exists
        rogue = _shard_path(tmp_path, tag, "zero_pp_rank_0_mp_rank_00_optim_states.pt")
        torch.save({"shard": "rogue", "value": 2}, rogue)
        with pytest.raises(CheckpointManifestError, match="not listed"):
            engine.load(rogue)

    def test_load_rejects_incomplete_manifest(self, tmp_path):
        engine = TorchCheckpointEngine()
        tag = "global_step1"
        tag_dir = os.path.join(str(tmp_path), tag)
        os.makedirs(tag_dir, exist_ok=True)
        shard = _shard_path(tmp_path, tag, "mp_rank_00_model_states.pt")
        torch.save({"shard": "x", "value": 1}, shard)
        with open(os.path.join(tag_dir, MANIFEST_FILE_NAME), "w") as fd:
            json.dump({"tag": tag, "status": "in_progress", "shards": ["mp_rank_00_model_states.pt"]}, fd)

        with pytest.raises(CheckpointManifestError, match="not marked complete"):
            engine.load(shard)

    def test_load_rejects_corrupt_manifest(self, tmp_path):
        engine = TorchCheckpointEngine()
        tag = "global_step1"
        tag_dir = os.path.join(str(tmp_path), tag)
        os.makedirs(tag_dir, exist_ok=True)
        shard = _shard_path(tmp_path, tag, "mp_rank_00_model_states.pt")
        torch.save({"shard": "x", "value": 1}, shard)
        with open(os.path.join(tag_dir, MANIFEST_FILE_NAME), "w") as fd:
            fd.write("{ not json")

        with pytest.raises(CheckpointManifestError, match="unreadable"):
            engine.load(shard)

    def test_legacy_checkpoint_without_manifest_still_loads(self, tmp_path):
        engine = TorchCheckpointEngine()
        tag = "global_step1"
        tag_dir = os.path.join(str(tmp_path), tag)
        os.makedirs(tag_dir, exist_ok=True)
        shard = _shard_path(tmp_path, tag, "mp_rank_00_model_states.pt")
        torch.save({"shard": "x", "value": 1}, shard)

        sd = engine.load(shard)
        assert sd == {"shard": "x", "value": 1}
