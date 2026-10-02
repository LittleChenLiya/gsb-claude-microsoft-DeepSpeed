# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import json
import os
import time

import torch
from deepspeed.utils import log_dist
from deepspeed.runtime.checkpoint_engine.checkpoint_engine import \
    CheckpointEngine, CheckpointCommitInfo
from deepspeed.runtime.model_checkpointing import CHECKPOINT_SERIALIZATION

ENGINE_NAME = "TorchCheckpointEngine"

# Name of the commit-manifest file written into each checkpoint tag directory.
MANIFEST_FILE_NAME = "ds_checkpoint_manifest.json"
# Suffix of the temporary files shards are written to before the checkpoint is committed.
TMP_FILE_SUFFIX = ".tmp"

_MANIFEST_FORMAT = "deepspeed-checkpoint-manifest"
_MANIFEST_VERSION = 1
_MANIFEST_STATUS_COMPLETE = "complete"


class CheckpointManifestError(RuntimeError):
    """Raised when a checkpoint shard is rejected by the commit manifest.

    A shard is rejected when its checkpoint directory contains a manifest that
    is corrupt, not marked complete, or does not list the shard. This prevents
    loading a partially written (new) checkpoint version.
    """
    pass


class TorchCheckpointEngine(CheckpointEngine):
    """Torch checkpoint engine with a commit manifest.

    Shards are first written to temporary files (`<shard>.tmp`) and only moved
    to their final location when `commit()` is called. `commit()` succeeds only
    if every shard of the checkpoint was written successfully; it then atomically
    publishes a manifest (`ds_checkpoint_manifest.json`) listing the committed
    shards and marking the checkpoint complete. If any shard write failed, the
    temporary files are discarded, the previous manifest (i.e. the previous
    complete version) is left untouched, and `commit()` returns False.

    `load()` consults the manifest: a shard whose directory has a manifest that
    is missing/corrupt, incomplete, or does not list the shard is rejected with
    :class:`CheckpointManifestError`. Directories without a manifest (checkpoints
    written before this mechanism existed) are loaded as before.
    """

    def __init__(self, config_params=None):
        super().__init__(config_params)
        self.name = ENGINE_NAME
        if config_params is None:
            self.zipfile_serialization = False
        else:
            self.zipfile_serialization = config_params.checkpoint_config[CHECKPOINT_SERIALIZATION]
        # final shard path -> temporary shard path, for the in-flight checkpoint
        self._pending_shards = {}
        # set to True when any shard write of the in-flight checkpoint failed
        self._save_failed = False
        log_dist(f'[{ENGINE_NAME}] Initialized with serialization = {self.zipfile_serialization}', ranks=[0])

    def create(self, info: CheckpointCommitInfo):
        log_dist(f"[Torch] Checkpoint {info.tag} is about to be saved!", ranks=[0])
        self.makedirs(os.path.join(info.save_dir, info.tag), exist_ok=True)
        self._pending_shards = {}
        self._save_failed = False

    def save(self, state_dict, path: str):
        # Write the shard to a temporary file first. The shard only becomes
        # visible under its final name when commit() moves it into place, so a
        # failed save can never clobber a previously committed shard.
        tmp_path = path + TMP_FILE_SUFFIX
        try:
            torch.save(state_dict, tmp_path, _use_new_zipfile_serialization=self.zipfile_serialization)
        except Exception:
            self._save_failed = True
            self._pending_shards.pop(path, None)
            # Remove the partially written temp shard so it can never be
            # mistaken for a complete one.
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
        self._pending_shards[path] = tmp_path

    def load(self, path: str, map_location=None):
        self._verify_loadable(path)
        log_dist(f"[Torch] Begin Load checkpoint from {path}...", ranks=[0])
        partition = torch.load(path, map_location=map_location, weights_only=False)
        log_dist(f"[Torch] End Load checkpoint from {path}...", ranks=[0])
        return partition

    def commit(self, info: CheckpointCommitInfo):
        pending = self._pending_shards
        save_failed = self._save_failed
        self._pending_shards = {}
        self._save_failed = False

        tag_dir = os.path.join(info.save_dir, info.tag)

        # All shards must have been written successfully before the checkpoint
        # is marked complete.
        missing = [path for path, tmp_path in pending.items() if not self._is_valid_tmp_shard(tmp_path)]
        if save_failed or missing:
            log_dist(
                f"[Torch] Checkpoint {info.tag} is incomplete "
                f"(save_failed={save_failed}, missing_shards={missing}); "
                f"discarding temporary shards and keeping the previous version.",
                ranks=[0])
            self._discard_tmp_shards(pending)
            return False

        # All shards are on disk: move them into place atomically.
        for path, tmp_path in pending.items():
            os.replace(tmp_path, path)

        # Merge with the shards of an existing manifest for the same tag so
        # that shards written into the same directory by other ranks/engine
        # instances remain listed.
        shards = set(os.path.basename(path) for path in pending)
        existing = self._read_manifest(tag_dir)
        if existing is not None and existing.get("tag") == info.tag:
            shards.update(existing.get("shards", []))

        if shards:
            self._write_manifest(tag_dir, info, sorted(shards))
        return True

    @staticmethod
    def _is_valid_tmp_shard(tmp_path):
        return os.path.isfile(tmp_path) and os.path.getsize(tmp_path) > 0

    @staticmethod
    def _discard_tmp_shards(pending):
        for tmp_path in pending.values():
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass

    @staticmethod
    def _manifest_path(tag_dir):
        return os.path.join(tag_dir, MANIFEST_FILE_NAME)

    @staticmethod
    def _read_manifest(tag_dir):
        """Return the manifest dict for tag_dir, or None if absent/unreadable."""
        manifest_path = TorchCheckpointEngine._manifest_path(tag_dir)
        if not os.path.isfile(manifest_path):
            return None
        try:
            with open(manifest_path, "r") as fd:
                return json.load(fd)
        except (OSError, ValueError):
            return None

    @staticmethod
    def _write_manifest(tag_dir, info: CheckpointCommitInfo, shards):
        manifest = {
            "format": _MANIFEST_FORMAT,
            "version": _MANIFEST_VERSION,
            "tag": info.tag,
            "status": _MANIFEST_STATUS_COMPLETE,
            "num_shards": len(shards),
            "shards": list(shards),
            "saved_at": time.time(),
        }
        manifest_path = TorchCheckpointEngine._manifest_path(tag_dir)
        tmp_path = manifest_path + TMP_FILE_SUFFIX
        with open(tmp_path, "w") as fd:
            json.dump(manifest, fd, indent=2, sort_keys=True)
        os.replace(tmp_path, manifest_path)
        log_dist(
            f"[Torch] Checkpoint {info.tag} committed with {len(shards)} shard(s); "
            f"manifest written to {manifest_path}",
            ranks=[0])

    def _verify_loadable(self, path: str):
        tag_dir = os.path.dirname(path)
        manifest_path = self._manifest_path(tag_dir)
        if not os.path.isfile(manifest_path):
            # Checkpoint written before commit manifests existed: load as before.
            return
        try:
            with open(manifest_path, "r") as fd:
                manifest = json.load(fd)
        except (OSError, ValueError) as err:
            raise CheckpointManifestError(
                f"Rejecting load of {path}: checkpoint manifest {manifest_path} is unreadable ({err})")
        if manifest.get("status") != _MANIFEST_STATUS_COMPLETE:
            raise CheckpointManifestError(
                f"Rejecting load of {path}: checkpoint manifest {manifest_path} is not marked complete "
                f"(status={manifest.get('status')!r}); the checkpoint was not fully written")
        shard = os.path.basename(path)
        if shard not in manifest.get("shards", []):
            raise CheckpointManifestError(
                f"Rejecting load of {path}: shard {shard!r} is not listed in checkpoint manifest "
                f"{manifest_path}; it does not belong to the committed checkpoint version")
