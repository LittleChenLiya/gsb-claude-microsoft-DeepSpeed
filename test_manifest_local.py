# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

"""Local acceptance test for the checkpoint write manifest."""

import os
import sys
import json
import tempfile

import torch

from deepspeed.runtime.checkpoint_engine.torch_checkpoint_engine import (
    TorchCheckpointEngine,
    MANIFEST_FILENAME,
    TEMP_FILE_SUFFIX,
    MANIFEST_STATUS_PENDING,
    MANIFEST_STATUS_COMPLETE,
)
from deepspeed.runtime.checkpoint_engine.checkpoint_engine import CheckpointCommitInfo


class FailingShardEngine(TorchCheckpointEngine):
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


def read_manifest(tag_dir):
    manifest_path = os.path.join(tag_dir, MANIFEST_FILENAME)
    if not os.path.exists(manifest_path):
        return None
    with open(manifest_path, "r") as f:
        return json.load(f)


def list_files(tag_dir):
    return sorted(os.listdir(tag_dir))


def main():
    workdir = tempfile.mkdtemp(prefix="ckpt_manifest_test_")
    save_dir = os.path.join(workdir, "save_dir")
    os.makedirs(save_dir, exist_ok=True)
    results = []

    print("=" * 70)
    print("Scenario 1: successful shard write")
    print("=" * 70)
    tag1 = "global_step100"
    tag1_dir = os.path.join(save_dir, tag1)
    engine = TorchCheckpointEngine()
    info = CheckpointCommitInfo(tag=tag1, save_dir=save_dir, save_latest=True)
    engine.create(info)
    shard_paths = []
    for i in range(3):
        path = os.path.join(tag1_dir, "mp_rank_%02d_model_states.pt" % i)
        shard_paths.append(path)
        engine.save({"weights": torch.randn(4, 4), "step": 100 + i}, path)
    engine.commit(info)
    manifest = read_manifest(tag1_dir)
    files = list_files(tag1_dir)
    tmp_files = [f for f in files if f.endswith(TEMP_FILE_SUFFIX)]
    final_files = [f for f in files if f.endswith("_model_states.pt")]
    print("  manifest status :", manifest["status"] if manifest else "MISSING")
    print("  manifest shards :", len(manifest["shards"]) if manifest else 0)
    print("  files in tag dir:", files)
    print("  temp files      :", tmp_files)
    print("  final shard files:", final_files)
    ok = (manifest is not None and manifest["status"] == MANIFEST_STATUS_COMPLETE
          and len(manifest["shards"]) == 3 and len(tmp_files) == 0 and len(final_files) == 3)
    results.append(("scenario1_success", ok))
    print("  PASS" if ok else "  FAIL")
    with open(os.path.join(save_dir, "latest"), "w") as f:
        f.write(tag1)
    load_ok = True
    try:
        sd = engine.load(shard_paths[0])
        print("  load shard 0    : step=" + str(sd["step"]))
    except Exception as e:
        load_ok = False
        print("  load shard 0    : FAILED:", e)
    results.append(("scenario1_load", load_ok))
    print("  PASS" if load_ok else "  FAIL")

    print()
    print("=" * 70)
    print("Scenario 2: failed shard write (2nd shard fails)")
    print("=" * 70)
    tag2 = "global_step200"
    tag2_dir = os.path.join(save_dir, tag2)
    engine2 = FailingShardEngine(fail_on_shard_index=1)
    info2 = CheckpointCommitInfo(tag=tag2, save_dir=save_dir, save_latest=True)
    engine2.create(info2)
    shard2_paths = []
    save_error = None
    for i in range(3):
        path = os.path.join(tag2_dir, "mp_rank_%02d_model_states.pt" % i)
        shard2_paths.append(path)
        try:
            engine2.save({"weights": torch.randn(4, 4), "step": 200 + i}, path)
        except OSError as e:
            save_error = e
            print("  shard", i, "write : FAILED:", e)
            break
    manifest2 = read_manifest(tag2_dir)
    files2 = list_files(tag2_dir) if os.path.exists(tag2_dir) else []
    tmp_files2 = [f for f in files2 if f.endswith(TEMP_FILE_SUFFIX)]
    final_files2 = [f for f in files2 if f.endswith("_model_states.pt")]
    print("  manifest status :", manifest2["status"] if manifest2 else "MISSING")
    print("  files in tag dir:", files2)
    print("  temp files      :", tmp_files2)
    print("  final shard files:", final_files2)
    ok2 = (save_error is not None and manifest2 is not None
           and manifest2["status"] == MANIFEST_STATUS_PENDING
           and len(tmp_files2) >= 1 and len(final_files2) == 0)
    results.append(("scenario2_failure", ok2))
    print("  PASS" if ok2 else "  FAIL")
    reject_ok = False
    try:
        engine2.load(shard2_paths[0])
        print("  load new version: UNEXPECTEDLY SUCCEEDED")
    except RuntimeError as e:
        reject_ok = True
        print("  load new version: REJECTED:", e)
    except Exception as e:
        print("  load new version: unexpected error:", type(e).__name__, e)
    results.append(("scenario2_reject_new", reject_ok))
    print("  PASS" if reject_ok else "  FAIL")
    old_ok = False
    try:
        sd = engine.load(shard_paths[0])
        old_ok = True
        print("  load old version: step=" + str(sd["step"]))
    except Exception as e:
        print("  load old version: FAILED:", e)
    results.append(("scenario2_keep_old", old_ok))
    print("  PASS" if old_ok else "  FAIL")
    selected = engine.select_latest_complete(save_dir)
    select_ok = selected == tag1
    print("  select_latest_complete:", selected, "(expected", tag1 + ")")
    results.append(("scenario2_load_selection", select_ok))
    print("  PASS" if select_ok else "  FAIL")

    print()
    print("=" * 70)
    print("Scenario 3: all shards written but commit detects missing temp file")
    print("=" * 70)
    tag3 = "global_step300"
    tag3_dir = os.path.join(save_dir, tag3)
    engine3 = TorchCheckpointEngine()
    info3 = CheckpointCommitInfo(tag=tag3, save_dir=save_dir, save_latest=True)
    engine3.create(info3)
    shard3_paths = []
    for i in range(2):
        path = os.path.join(tag3_dir, "mp_rank_%02d_model_states.pt" % i)
        shard3_paths.append(path)
        engine3.save({"weights": torch.randn(4, 4), "step": 300 + i}, path)
    tmp_to_remove = shard3_paths[0] + TEMP_FILE_SUFFIX
    os.remove(tmp_to_remove)
    commit_error = None
    try:
        engine3.commit(info3)
    except RuntimeError as e:
        commit_error = e
        print("  commit: FAILED:", e)
    manifest3 = read_manifest(tag3_dir)
    print("  manifest status:", manifest3["status"] if manifest3 else "MISSING")
    ok3 = (commit_error is not None and manifest3 is not None
           and manifest3["status"] == MANIFEST_STATUS_PENDING)
    results.append(("scenario3_commit_failure", ok3))
    print("  PASS" if ok3 else "  FAIL")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_ok = True
    for name, ok in results:
        print("  %-30s: %s" % (name, "PASS" if ok else "FAIL"))
        if not ok:
            all_ok = False
    print()
    print("Workdir:", workdir)
    if all_ok:
        print()
        print("ALL TESTS PASSED")
    else:
        print()
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
