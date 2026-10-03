# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import json
import time
import glob
import hashlib

import torch
from deepspeed.utils import log_dist, logger
from deepspeed.runtime.checkpoint_engine.checkpoint_engine import \
    CheckpointEngine, CheckpointCommitInfo
from deepspeed.runtime.model_checkpointing import CHECKPOINT_SERIALIZATION

ENGINE_NAME = "TorchCheckpointEngine"

# ---------------------------------------------------------------------------
# Checkpoint write manifest
# ---------------------------------------------------------------------------
# A checkpoint is only considered *complete* once every shard has been
# written to a temporary file and successfully renamed into place.  The
# manifest (``checkpoint_manifest.json``) is the single source of truth that
# loaders consult before opening any shard.  If the manifest says the
# checkpoint is still ``pending`` (or is missing for a checkpoint written by
# this engine), loading is rejected so that a partially-written new version
# can never shadow the previous, good version.
MANIFEST_FILENAME = "checkpoint_manifest.json"
TEMP_FILE_SUFFIX = ".tmp"
SHARD_LIST_PREFIX = "shards_rank_"
SHARD_LIST_SUFFIX = ".json"
MANIFEST_VERSION = 1

MANIFEST_STATUS_PENDING = "pending"
MANIFEST_STATUS_COMPLETE = "complete"


class TorchCheckpointEngine(CheckpointEngine):

    def __init__(self, config_params=None):
        super().__init__(config_params)
        self.name = ENGINE_NAME
        if config_params is None:
            self.zipfile_serialization = False
        else:
            self.zipfile_serialization = config_params.checkpoint_config[CHECKPOINT_SERIALIZATION]
        # Shards written by the current process for the in-flight checkpoint.
        # Each entry is (final_path, size_bytes).
        self._shards = []
        self._commit_info = None
        log_dist(f'[{ENGINE_NAME}] Initialized with serialization = {self.zipfile_serialization}', ranks=[0])

    # ------------------------------------------------------------------
    # CheckpointEngine interface
    # ------------------------------------------------------------------
    def create(self, info: CheckpointCommitInfo):
        log_dist(f"[Torch] Checkpoint {info.tag} is about to be saved!", ranks=[0])
        self._shards = []
        self._commit_info = info
        # Mark the checkpoint as pending *before* any shard is written.  If a
        # shard write fails, the manifest stays pending and loaders will
        # reject this version while keeping the previous one.
        self._write_manifest(info, status=MANIFEST_STATUS_PENDING, shards=[])

    def save(self, state_dict, path: str):
        # Write to a temporary file first.  The final path is only populated
        # during commit(), after every shard has succeeded.  This guarantees
        # that a failed shard write can never leave a truncated file at the
        # canonical shard path.
        tmp_path = path + TEMP_FILE_SUFFIX
        torch.save(state_dict, tmp_path, _use_new_zipfile_serialization=self.zipfile_serialization)
        size = os.path.getsize(tmp_path)
        self._shards.append((path, size))

    def load(self, path: str, map_location=None):
        # Reject loading a checkpoint whose manifest is not complete.  This is
        # the guard that prevents a partially-written new version from being
        # loaded.
        self._check_checkpoint_complete(path)
        log_dist(f"[Torch] Begin Load checkpoint from {path}...", ranks=[0])
        partition = torch.load(path, map_location=map_location, weights_only=False)
        log_dist(f"[Torch] End Load checkpoint from {path}...", ranks=[0])
        return partition

    def commit(self, info: CheckpointCommitInfo):
        # 1. Verify every shard temp file exists.  If any shard failed to
        #    write, its temp file is missing and we must not commit.
        for path, _ in self._shards:
            tmp_path = path + TEMP_FILE_SUFFIX
            if not os.path.exists(tmp_path):
                raise RuntimeError(
                    f"[{ENGINE_NAME}] Shard temp file missing: {tmp_path}. "
                    f"Checkpoint {info.tag} is incomplete; refusing to commit."
                )

        # 2. Rename temp files to their final names.  This is the point at
        #    which the new version becomes visible.
        for path, _ in self._shards:
            tmp_path = path + TEMP_FILE_SUFFIX
            os.replace(tmp_path, path)

        # 3. Record this process's shard list, then synchronize with all
        #    other processes so that rank 0 can aggregate a complete manifest.
        self._write_shard_list(info)
        self._barrier()

        # 4. Rank 0 aggregates every process's shard list into the single
        #    complete manifest.  Only after this point is the checkpoint
        #    considered loadable.
        if self._get_rank() == 0:
            self._write_complete_manifest(info)

        self._shards = []
        self._commit_info = None
        return True

    # ------------------------------------------------------------------
    # Manifest helpers
    # ------------------------------------------------------------------
    def _tag_dir(self, info: CheckpointCommitInfo):
        return os.path.join(info.save_dir, info.tag)

    def _manifest_path(self, info: CheckpointCommitInfo):
        return os.path.join(self._tag_dir(info), MANIFEST_FILENAME)

    def _write_manifest(self, info: CheckpointCommitInfo, status: str, shards):
        tag_dir = self._tag_dir(info)
        os.makedirs(tag_dir, exist_ok=True)
        manifest = {
            "version": MANIFEST_VERSION,
            "tag": info.tag,
            "status": status,
            "shards": [
                {
                    "path": os.path.basename(p),
                    "size": s,
                    "checksum": self._checksum(p + TEMP_FILE_SUFFIX) if status == MANIFEST_STATUS_COMPLETE else None,
                }
                for p, s in shards
            ],
            "timestamp": self._timestamp(),
        }
        manifest_path = self._manifest_path(info)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    def _write_shard_list(self, info: CheckpointCommitInfo):
        tag_dir = self._tag_dir(info)
        os.makedirs(tag_dir, exist_ok=True)
        rank = self._get_rank()
        shard_list_path = os.path.join(tag_dir, f"{SHARD_LIST_PREFIX}{rank}{SHARD_LIST_SUFFIX}")
        shard_list = {
            "rank": rank,
            "shards": [{"path": os.path.basename(p), "size": s} for p, s in self._shards],
        }
        with open(shard_list_path, "w") as f:
            json.dump(shard_list, f, indent=2)

    def _write_complete_manifest(self, info: CheckpointCommitInfo):
        tag_dir = self._tag_dir(info)
        all_shards = []
        shard_list_files = glob.glob(os.path.join(tag_dir, f"{SHARD_LIST_PREFIX}*{SHARD_LIST_SUFFIX}"))
        for shard_list_file in shard_list_files:
            try:
                with open(shard_list_file, "r") as f:
                    shard_list = json.load(f)
                all_shards.extend(shard_list.get("shards", []))
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"[{ENGINE_NAME}] Could not read shard list {shard_list_file}: {e}")

        # Deduplicate by path (shouldn't happen, but be safe).
        seen = set()
        unique_shards = []
        for shard in all_shards:
            key = shard.get("path")
            if key in seen:
                continue
            seen.add(key)
            # Compute checksum from the final shard file (temp files have
            # already been renamed into place at this point).
            shard = dict(shard)
            final_path = os.path.join(tag_dir, key)
            shard["checksum"] = self._checksum(final_path)
            unique_shards.append(shard)

        manifest = {
            "version": MANIFEST_VERSION,
            "tag": info.tag,
            "status": MANIFEST_STATUS_COMPLETE,
            "shards": unique_shards,
            "timestamp": self._timestamp(),
        }
        manifest_path = self._manifest_path(info)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        # Clean up per-process shard list files; the single manifest is the
        # source of truth now.
        for shard_list_file in shard_list_files:
            try:
                os.remove(shard_list_file)
            except OSError:
                pass

    def _check_checkpoint_complete(self, path: str):
        """Reject loading a checkpoint whose manifest is not complete."""
        tag_dir = os.path.dirname(path)
        manifest_path = os.path.join(tag_dir, MANIFEST_FILENAME)
        if not os.path.exists(manifest_path):
            # No manifest present.  This is either a legacy checkpoint or a
            # checkpoint written by an engine that does not use manifests;
            # assume it is complete for backward compatibility.
            return
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            raise RuntimeError(
                f"[{ENGINE_NAME}] Checkpoint manifest is unreadable: {manifest_path} ({e}). "
                f"Refusing to load."
            ) from e
        status = manifest.get("status")
        if status != MANIFEST_STATUS_COMPLETE:
            raise RuntimeError(
                f"[{ENGINE_NAME}] Checkpoint {manifest.get('tag')!r} is not complete "
                f"(status={status!r}). Refusing to load."
            )

    # ------------------------------------------------------------------
    # Load selection
    # ------------------------------------------------------------------
    def select_latest_complete(self, save_dir: str, latest_tag: str = "latest"):
        """Return the tag of the latest *complete* checkpoint.

        Reads the ``latest`` pointer and returns the tag it points to when
        that checkpoint is complete.  If the pointer targets an incomplete
        checkpoint (e.g. a shard write failed), falls back to the most
        recent complete checkpoint so the previous good version is kept.
        Returns ``None`` when no complete checkpoint exists.
        """
        latest_path = os.path.join(save_dir, latest_tag)
        if os.path.isfile(latest_path):
            with open(latest_path, "r") as f:
                tag = f.read().strip()
            if self._is_tag_complete(save_dir, tag):
                return tag

        # Fall back to the most recent complete checkpoint.
        best_tag = None
        best_time = None
        try:
            entries = os.listdir(save_dir)
        except OSError:
            return None
        for entry in entries:
            entry_path = os.path.join(save_dir, entry)
            if not os.path.isdir(entry_path):
                continue
            if not self._is_tag_complete(save_dir, entry):
                continue
            manifest_path = os.path.join(entry_path, MANIFEST_FILENAME)
            try:
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
                ts = manifest.get("timestamp", "")
            except (json.JSONDecodeError, IOError):
                continue
            if best_time is None or ts > best_time:
                best_time = ts
                best_tag = entry
        return best_tag

    def _is_tag_complete(self, save_dir: str, tag: str) -> bool:
        manifest_path = os.path.join(save_dir, tag, MANIFEST_FILENAME)
        if not os.path.exists(manifest_path):
            return False
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
        except (json.JSONDecodeError, IOError):
            return False
        return manifest.get("status") == MANIFEST_STATUS_COMPLETE

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    def _get_rank(self):
        try:
            from deepspeed import comm as dist
            if dist.is_initialized():
                return dist.get_rank()
        except Exception:
            pass
        return 0

    def _barrier(self):
        try:
            from deepspeed import comm as dist
            if dist.is_initialized():
                dist.barrier()
        except Exception:
            pass

    def _checksum(self, path: str):
        if not os.path.exists(path):
            return None
        h = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
        except OSError:
            return None
        return h.hexdigest()

    def _timestamp(self):
        return time.strftime("%Y-%m-%dT%H:%M:%S")
