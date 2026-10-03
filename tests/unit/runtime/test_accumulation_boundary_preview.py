# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Ensure the CPU accelerator is selected before deepspeed is imported so the
# test runs on a local native environment without any GPU.
import os
os.environ.setdefault("DS_ACCELERATOR", "cpu")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29531")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")

# On Windows with core.symlinks=false, deepspeed/ops/op_builder is checked out
# as a text file containing the symlink target instead of a real directory.
# Copy the tracked op_builder directory so deepspeed can be imported. The
# original text file is restored after the session so the worktree stays clean.
import pathlib
import shutil

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_OP_BUILDER_LINK = _REPO_ROOT / "deepspeed" / "ops" / "op_builder"
_OP_BUILDER_DIR = _REPO_ROOT / "op_builder"
_ORIGINAL_OP_BUILDER_CONTENT = None


def _remove_op_builder():
    if _OP_BUILDER_LINK.is_dir() and not _OP_BUILDER_LINK.is_symlink():
        shutil.rmtree(_OP_BUILDER_LINK)
    elif _OP_BUILDER_LINK.exists():
        _OP_BUILDER_LINK.unlink()


def _fix_op_builder():
    global _ORIGINAL_OP_BUILDER_CONTENT
    if _OP_BUILDER_LINK.is_file() and not _OP_BUILDER_LINK.is_dir():
        _ORIGINAL_OP_BUILDER_CONTENT = _OP_BUILDER_LINK.read_text()
        _remove_op_builder()
        shutil.copytree(_OP_BUILDER_DIR, _OP_BUILDER_LINK)


def _restore_op_builder():
    if _ORIGINAL_OP_BUILDER_CONTENT is not None:
        _remove_op_builder()
        _OP_BUILDER_LINK.write_text(_ORIGINAL_OP_BUILDER_CONTENT)


_fix_op_builder()

import pytest
import torch

import deepspeed
from unit.simple_model import SimpleModel


@pytest.fixture(scope="session", autouse=True)
def _op_builder_restore():
    yield
    _restore_op_builder()


# Number of micro-batches per optimizer step (the accumulation boundary is
# reached on the 4th micro-batch).
GAS = 4
HIDDEN_DIM = 4


def _build_engine():
    model = SimpleModel(hidden_dim=HIDDEN_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ds_config = {
        "train_batch_size": GAS,  # so train_micro_batch_size_per_gpu = 1
        "gradient_accumulation_steps": GAS,
    }
    model, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config,
        model_parameters=model.parameters(),
        optimizer=optimizer,
    )
    return model, optimizer


def _run_micro_step(engine):
    x = torch.randn(1, HIDDEN_DIM)
    y = torch.randint(0, HIDDEN_DIM, (1,))
    loss = engine(x, y)
    engine.backward(loss)
    engine.step()


def _is_boundary_step(engine, global_before):
    """A step is a boundary iff global_steps incremented (optimizer update ran)."""
    return engine.global_steps > global_before


class TestAccumulationBoundaryPreview:
    """
    Verify the read-only gradient accumulation boundary preview on a CPU small
    model: four micro-batches with gradient_accumulation_steps=4, save/restore
    midway, and confirm the preview matches the actual sync/zero behavior.
    """

    def test_preview_matches_actual_and_persists_across_restore(self, tmpdir):
        engine, _ = _build_engine()
        assert engine.gradient_accumulation_steps() == GAS

        # ---- Phase 1: first two micro-batches (no boundary yet) ----
        for i in range(2):
            micro_before = engine.micro_steps
            global_before = engine.global_steps

            preview = engine.preview_gradient_accumulation_boundary()

            # Preview must be read-only: micro_steps is unchanged by the call.
            assert engine.micro_steps == micro_before
            # Calling it repeatedly must not change anything either.
            engine.preview_gradient_accumulation_boundary()
            engine.preview_gradient_accumulation_boundary()
            assert engine.micro_steps == micro_before

            _run_micro_step(engine)

            boundary = _is_boundary_step(engine, global_before)
            assert preview == boundary, (
                f"micro step {i}: preview={preview} but actual boundary={boundary}"
            )
            # After step(), micro_steps advances by exactly one.
            assert engine.micro_steps == micro_before + 1

        assert engine.micro_steps == 2
        assert engine.global_steps == 0  # no optimizer update yet

        # ---- Save checkpoint midway (after 2 micro-batches) ----
        engine.save_checkpoint(tmpdir)

        # ---- Restore into a fresh engine ----
        restored, _ = _build_engine()
        restored.load_checkpoint(tmpdir)

        # micro_steps must be restored so the boundary preview stays aligned.
        assert restored.micro_steps == 2, (
            f"micro_steps not restored: got {restored.micro_steps}, expected 2"
        )
        assert restored.global_steps == 0

        # ---- Phase 2: remaining two micro-batches ----
        for i in range(2, 4):
            micro_before = restored.micro_steps
            global_before = restored.global_steps

            preview = restored.preview_gradient_accumulation_boundary()
            assert restored.micro_steps == micro_before  # read-only

            _run_micro_step(restored)

            boundary = _is_boundary_step(restored, global_before)
            assert preview == boundary, (
                f"micro step {i}: preview={preview} but actual boundary={boundary}"
            )
            assert restored.micro_steps == micro_before + 1

        # Exactly one optimizer update (on the 4th micro-batch) across the run.
        assert restored.global_steps == 1
        assert restored.micro_steps == 4

        # The 5th micro-batch is not a boundary again.
        assert restored.preview_gradient_accumulation_boundary() is False

    def test_preview_predicts_zero_grad_on_boundary(self):
        """The preview must predict the zero_grad that follows a boundary step."""
        engine, _ = _build_engine()

        for i in range(GAS):
            global_before = engine.global_steps
            preview = engine.preview_gradient_accumulation_boundary()
            _run_micro_step(engine)
            boundary = _is_boundary_step(engine, global_before)
            assert preview == boundary

            if boundary:
                # After a boundary step, grads are zeroed (param.grad is None).
                for p in engine.module.parameters():
                    assert p.grad is None, f"param.grad not zeroed after boundary step {i}"
            else:
                # Non-boundary steps leave accumulated grads in place.
                for p in engine.module.parameters():
                    assert p.grad is not None, f"param.grad missing after non-boundary step {i}"

        assert engine.global_steps == 1
        assert engine.micro_steps == GAS

    def test_preview_read_only_at_init(self):
        """At init (micro_steps=0, gas=4) the preview is False and changes nothing."""
        engine, _ = _build_engine()
        assert engine.micro_steps == 0
        assert engine.preview_gradient_accumulation_boundary() is False
        assert engine.micro_steps == 0  # still unchanged
