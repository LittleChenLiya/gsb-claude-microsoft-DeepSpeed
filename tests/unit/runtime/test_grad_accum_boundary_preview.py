# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Tests for DeepSpeedEngine.gradient_accumulation_boundary_preview().

These tests are intentionally self-contained single-process (world_size=1, gloo,
CPU, fp32) so they are reproducible locally on any machine, including CPU-only
hosts where the forkserver-based DistributedTest harness is unavailable.

Run locally with:
    pytest tests/unit/runtime/test_grad_accum_boundary_preview.py -v
"""

import os
import socket

import torch

import deepspeed
import deepspeed.comm as dist

from unit.simple_model import SimpleModel

HIDDEN_DIM = 8
GRAD_ACCUM_STEPS = 2
NUM_MICRO_STEPS = 4


def _free_port():
    sock = socket.socket()
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _ensure_distributed():
    if not dist.is_initialized():
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', str(_free_port()))
        os.environ.setdefault('RANK', '0')
        os.environ.setdefault('LOCAL_RANK', '0')
        os.environ.setdefault('WORLD_SIZE', '1')
        deepspeed.init_distributed(dist_backend='gloo', auto_mpi_discovery=False)


def _make_engine():
    _ensure_distributed()
    config_dict = {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
        "steps_per_print": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-3,
                # CPU-only hosts cannot JIT-build the fused Adam CUDA op.
                "torch_adam": True
            }
        },
    }
    model = SimpleModel(HIDDEN_DIM)
    engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)
    return engine


def _make_batches(num_micro_steps=NUM_MICRO_STEPS):
    generator = torch.Generator().manual_seed(1234)
    batches = []
    for _ in range(num_micro_steps):
        data = torch.randn(1, HIDDEN_DIM, generator=generator)
        label = torch.empty(1, dtype=torch.long).random_(HIDDEN_DIM, generator=generator)
        batches.append((data, label))
    return batches


def _run_micro_step(engine, batch):
    loss = engine(batch[0], batch[1])
    engine.backward(loss)
    engine.step()


def _assert_preview_matches_actual(engine, batch):
    """Check the preview taken before a micro step against what step() actually does."""
    preview = engine.gradient_accumulation_boundary_preview()

    global_steps_before = engine.global_steps
    _run_micro_step(engine, batch)

    # "sync + optimizer update" happens exactly on the predicted boundary.
    took_model_step = engine.global_steps == global_steps_before + 1
    assert took_model_step == preview['is_boundary'], \
        f"preview={preview}, global_steps {global_steps_before} -> {engine.global_steps}"
    assert preview['will_sync_gradients'] == preview['is_boundary']

    # Gradients are zeroed (set to None) exactly when the preview said they would be.
    grads = [param.grad for param in engine.module.parameters()]
    if preview['will_zero_gradients']:
        assert all(grad is None for grad in grads)
    else:
        assert all(grad is not None for grad in grads)
    return preview


class TestGradientAccumulationBoundaryPreview:

    def test_preview_is_read_only_and_matches_actual(self):
        engine = _make_engine()
        batches = _make_batches()

        # gas=2: boundaries land on the 2nd and 4th micro steps.
        expected_boundaries = [False, True, False, True]
        expected_remaining = [1, 0, 1, 0]

        for step_idx, batch in enumerate(batches):
            micro_steps_before = engine.micro_steps
            preview = engine.gradient_accumulation_boundary_preview()
            # The preview is read-only: repeated calls return the same result and
            # must not advance the micro-step counter.
            assert engine.gradient_accumulation_boundary_preview() == preview
            assert engine.micro_steps == micro_steps_before

            assert preview['micro_steps'] == step_idx
            assert preview['gradient_accumulation_steps'] == GRAD_ACCUM_STEPS
            assert preview['is_boundary'] == expected_boundaries[step_idx]
            assert preview['will_sync_gradients'] == expected_boundaries[step_idx]
            assert preview['will_zero_gradients'] == expected_boundaries[step_idx]
            assert preview['micro_steps_remaining'] == expected_remaining[step_idx]

            _assert_preview_matches_actual(engine, batch)
            assert engine.micro_steps == step_idx + 1

        assert engine.global_steps == NUM_MICRO_STEPS // GRAD_ACCUM_STEPS

    def test_preview_consistent_across_checkpoint_restore(self, tmp_path):
        engine = _make_engine()
        batches = _make_batches()

        # Run the first of four micro steps (not a boundary), then save mid-cycle.
        _run_micro_step(engine, batches[0])
        assert engine.micro_steps == 1
        assert engine.global_steps == 0

        preview_at_save = engine.gradient_accumulation_boundary_preview()
        assert preview_at_save['is_boundary'] is True
        assert preview_at_save['micro_steps'] == 1

        engine.save_checkpoint(str(tmp_path), tag="mid_cycle")

        # Restore into a fresh engine: the preview must be identical to the
        # one taken before saving (micro_steps is persisted in the checkpoint).
        restored_engine = _make_engine()
        assert restored_engine.micro_steps == 0
        load_path, _ = restored_engine.load_checkpoint(str(tmp_path), tag="mid_cycle")
        assert load_path is not None

        preview_after_restore = restored_engine.gradient_accumulation_boundary_preview()
        assert preview_after_restore == preview_at_save
        assert restored_engine.micro_steps == 1

        # Run the remaining three micro steps; the preview must match actual behavior.
        expected_boundaries = [True, False, True]
        for step_idx, batch in enumerate(batches[1:]):
            preview = restored_engine.gradient_accumulation_boundary_preview()
            assert preview['is_boundary'] == expected_boundaries[step_idx]
            _assert_preview_matches_actual(restored_engine, batch)

        assert restored_engine.micro_steps == NUM_MICRO_STEPS
        assert restored_engine.global_steps == NUM_MICRO_STEPS // GRAD_ACCUM_STEPS
