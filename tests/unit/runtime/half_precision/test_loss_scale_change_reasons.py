# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Tests for dynamic loss scale change reason reporting.

Local, controllable fixtures: fake gradients are injected to drive stable
steps, overflow steps and recovery steps through DynamicLossScaler.update_scale,
and the recorded scale / reason / overflow count are verified. The reporting
must not change the numeric update rules, and repeated overflow events must
not decrease the scale more than the hysteresis/min-scale rules allow.
"""

import pytest
import torch

from deepspeed.runtime.fp16 import loss_scaler as loss_scaler_module
from deepspeed.runtime.fp16.loss_scaler import (
    DynamicLossScaler,
    LossScaler,
    LossScaleChangeReason,
    has_inf_or_nan,
)


@pytest.fixture(autouse=True)
def single_rank(monkeypatch):
    # update_scale logs through deepspeed.comm.get_rank(); stub it so the
    # fixture runs locally without initializing a process group.
    monkeypatch.setattr(loss_scaler_module.dist, "get_rank", lambda *args, **kwargs: 0)


def make_scaler(**overrides):
    kwargs = dict(init_scale=256.0,
                  scale_window=3,
                  min_scale=1.0,
                  delayed_shift=1,
                  consecutive_hysteresis=False,
                  raise_error_at_min_scale=True)
    kwargs.update(overrides)
    return DynamicLossScaler(**kwargs)


def run_step(scaler, grad_value):
    """One fake training step: inject a fake gradient, detect overflow like the
    scaler does (has_inf_or_nan), then update the scale."""
    param = torch.nn.Parameter(torch.zeros(2, 2))
    param.grad = torch.full_like(param, grad_value)
    overflow = bool(has_inf_or_nan(param.grad.data))
    scaler.update_scale(overflow)
    return overflow


def reasons_of(scaler):
    return [event.reason for event in scaler.scale_change_events]


class TestScaleChangeReasonReporting:

    def test_stable_window_growth_records_reason(self):
        scaler = make_scaler()
        assert scaler.last_scale_change is None
        assert scaler.last_scale_change_reason is None
        assert scaler.overflow_count == 0

        # scale_window=3: the 4th stable step grows the scale once.
        for _ in range(3):
            run_step(scaler, 0.5)
            assert scaler.loss_scale == 256.0
        assert scaler.scale_change_events == []

        run_step(scaler, 0.5)
        assert scaler.loss_scale == 512.0
        assert scaler.overflow_count == 0

        event = scaler.last_scale_change
        assert event.reason == LossScaleChangeReason.SCALE_WINDOW
        assert event.iteration == 3
        assert event.old_scale == 256.0
        assert event.new_scale == 512.0
        assert event.overflow_count == 0
        assert scaler.last_scale_change_reason == LossScaleChangeReason.SCALE_WINDOW

    def test_overflow_decrease_records_reason_and_count(self):
        scaler = make_scaler(scale_window=1000)

        assert run_step(scaler, float('inf'))
        assert scaler.loss_scale == 128.0
        assert scaler.overflow_count == 1

        assert run_step(scaler, float('nan'))
        assert scaler.loss_scale == 64.0
        assert scaler.overflow_count == 2

        events = scaler.scale_change_events
        assert reasons_of(scaler) == [LossScaleChangeReason.OVERFLOW, LossScaleChangeReason.OVERFLOW]
        assert (events[0].old_scale, events[0].new_scale) == (256.0, 128.0)
        assert (events[1].old_scale, events[1].new_scale) == (128.0, 64.0)
        assert [event.overflow_count for event in events] == [1, 2]
        assert [event.iteration for event in events] == [0, 1]

    def test_repeated_overflow_with_hysteresis_decreases_once(self):
        # delayed_shift=2: the first overflow only counts down hysteresis, so
        # repeated overflow events must not decrease the scale multiple times.
        scaler = make_scaler(scale_window=1000, delayed_shift=2)

        assert run_step(scaler, float('inf'))
        assert scaler.loss_scale == 256.0  # hysteresis 2 -> 1, scale untouched
        assert scaler.overflow_count == 1
        assert scaler.scale_change_events == []  # no scale change, no event

        assert run_step(scaler, float('inf'))
        assert scaler.loss_scale == 128.0  # single decrease for two overflow events
        assert scaler.overflow_count == 2
        assert reasons_of(scaler) == [LossScaleChangeReason.OVERFLOW]
        assert scaler.last_scale_change.overflow_count == 2

    def test_overflow_at_min_scale_clamps_and_reports(self):
        scaler = make_scaler(init_scale=4.0, min_scale=4.0, scale_window=1000, raise_error_at_min_scale=False)

        assert run_step(scaler, float('inf'))
        assert scaler.loss_scale == 4.0  # clamped, never below min_scale
        assert run_step(scaler, float('nan'))
        assert scaler.loss_scale == 4.0  # repeated overflow: still no decrease
        assert scaler.overflow_count == 2

        events = scaler.scale_change_events
        assert reasons_of(scaler) == [LossScaleChangeReason.MIN_SCALE, LossScaleChangeReason.MIN_SCALE]
        assert all(event.old_scale == event.new_scale == 4.0 for event in events)
        assert [event.overflow_count for event in events] == [1, 2]

    def test_overflow_at_min_scale_raise_path_records_event(self):
        scaler = make_scaler(init_scale=4.0, min_scale=4.0, scale_window=1000, raise_error_at_min_scale=True)

        with pytest.raises(Exception, match="already at minimum"):
            run_step(scaler, float('inf'))

        # Failure path: the clamped event is recorded before raising.
        assert scaler.loss_scale == 4.0
        assert scaler.overflow_count == 1
        event = scaler.last_scale_change
        assert event.reason == LossScaleChangeReason.MIN_SCALE
        assert event.old_scale == event.new_scale == 4.0
        assert event.overflow_count == 1

    def test_recovery_after_overflow(self):
        scaler = make_scaler(scale_window=2)

        assert run_step(scaler, float('inf'))
        assert scaler.loss_scale == 128.0
        assert scaler.overflow_count == 1

        # scale_window=2: growth happens on the 3rd stable step after the overflow.
        run_step(scaler, 0.5)
        run_step(scaler, 0.5)
        assert scaler.loss_scale == 128.0
        run_step(scaler, 0.5)
        assert scaler.loss_scale == 256.0

        assert reasons_of(scaler) == [LossScaleChangeReason.OVERFLOW, LossScaleChangeReason.SCALE_WINDOW]
        growth = scaler.last_scale_change
        assert growth.old_scale == 128.0
        assert growth.new_scale == 256.0
        assert growth.overflow_count == 1  # count preserved from the triggering overflow history
        assert scaler.overflow_count == 1  # stable steps do not count as overflows

    def test_static_scaler_compat_reports_nothing(self):
        scaler = LossScaler(scale=8.0)
        scaler.update_scale(False)
        scaler.update_scale(True)  # base class no-op
        assert scaler.loss_scale == 8.0
        assert scaler.overflow_count == 0
        assert scaler.scale_change_events == []
        assert scaler.last_scale_change is None
        assert scaler.last_scale_change_reason is None
