import sys
import types
import os
import logging

sys.path.insert(0, os.getcwd())

deepspeed_pkg = types.ModuleType("deepspeed")
deepspeed_pkg.__path__ = [os.path.join(os.getcwd(), "deepspeed")]
sys.modules["deepspeed"] = deepspeed_pkg

config_utils = types.ModuleType("deepspeed.runtime.config_utils")
class DeepSpeedConfigObject:
    def __init__(self):
        pass
config_utils.DeepSpeedConfigObject = DeepSpeedConfigObject
sys.modules["deepspeed.runtime.config_utils"] = config_utils

comm = types.ModuleType("deepspeed.comm")
def get_rank():
    return 0
comm.get_rank = get_rank
sys.modules["deepspeed.comm"] = comm

utils = types.ModuleType("deepspeed.utils")
logger = logging.getLogger("deepspeed")
utils.logger = logger
sys.modules["deepspeed.utils"] = utils

runtime_utils = types.ModuleType("deepspeed.runtime.utils")
def has_inf_or_nan(x):
    import torch
    return torch.isinf(x).any() or torch.isnan(x).any()
runtime_utils.has_inf_or_nan = has_inf_or_nan
sys.modules["deepspeed.runtime.utils"] = runtime_utils

import importlib.util
spec = importlib.util.spec_from_file_location(
    "deepspeed.runtime.fp16.loss_scaler",
    os.path.join(os.getcwd(), "deepspeed/runtime/fp16/loss_scaler.py"))
loss_scaler = importlib.util.module_from_spec(spec)
sys.modules["deepspeed.runtime.fp16.loss_scaler"] = loss_scaler
spec.loader.exec_module(loss_scaler)

DynamicLossScaler = loss_scaler.DynamicLossScaler
LossScaleChangeReason = loss_scaler.LossScaleChangeReason

import torch


def test_normal_and_abnormal_paths():
    print("=" * 70)
    print("Test: Dynamic Loss Scale Change Reason")
    print("=" * 70)

    init_scale = 256.0
    scale_window = 3
    min_scale = 1.0
    delayed_shift = 1
    consecutive_hysteresis = False

    scaler = DynamicLossScaler(
        init_scale=init_scale,
        scale_window=scale_window,
        min_scale=min_scale,
        delayed_shift=delayed_shift,
        consecutive_hysteresis=consecutive_hysteresis,
        raise_error_at_min_scale=False,
        dtype=torch.half,
    )

    print()
    print("Initial state:")
    print("  cur_scale =", scaler.cur_scale)
    print("  scale_window =", scaler.scale_window)
    print("  min_scale =", scaler.min_scale)
    print("  delayed_shift =", scaler.delayed_shift)

    print()
    print("--- Phase 1: Stable steps (no overflow) ---")
    for i in range(scale_window + 1):
        scaler.update_scale(overflow=False)
        info = scaler.get_change_info()
        print("  Step", i + 1, ": overflow=False, cur_scale=", scaler.cur_scale,
              ", reason=", info["reason"].value,
              ", overflow_count=", info["overflow_count"],
              ", total_overflow_count=", info["total_overflow_count"])

    expected_scale = init_scale * 2
    assert scaler.cur_scale == expected_scale
    print("  => Scale increased to", scaler.cur_scale, "after", scale_window, "stable steps")

    print()
    print("--- Phase 2: Overflow steps (with recovery between each) ---")
    for i in range(3):
        scaler.update_scale(overflow=True)
        info = scaler.get_change_info()
        print("  Overflow", i + 1, ": overflow=True, cur_scale=", scaler.cur_scale,
              ", reason=", info["reason"].value,
              ", overflow_count=", info["overflow_count"],
              ", total_overflow_count=", info["total_overflow_count"])
        if i < 2:
            scaler.update_scale(overflow=False)
            info = scaler.get_change_info()
            print("  Recovery: overflow=False, cur_scale=", scaler.cur_scale,
                  ", reason=", info["reason"].value)

    expected_scale = init_scale * 2 / (2 ** 3)
    assert scaler.cur_scale == expected_scale
    print("  => Scale decreased to", scaler.cur_scale, "after 3 overflows")

    print()
    print("--- Phase 3: Recovery steps (no overflow) ---")
    for i in range(scale_window + 1):
        scaler.update_scale(overflow=False)
        info = scaler.get_change_info()
        print("  Step", i + 1, ": overflow=False, cur_scale=", scaler.cur_scale,
              ", reason=", info["reason"].value,
              ", overflow_count=", info["overflow_count"],
              ", total_overflow_count=", info["total_overflow_count"])

    expected_scale = (init_scale * 2 / (2 ** 3)) * 2
    assert scaler.cur_scale == expected_scale
    print("  => Scale increased to", scaler.cur_scale, "after", scale_window, "stable steps")

    print()
    print("=" * 70)
    print("All normal path assertions passed!")
    print("=" * 70)


def test_duplicate_overflow():
    print()
    print("=" * 70)
    print("Test: Duplicate Overflow Prevention")
    print("=" * 70)

    scaler = DynamicLossScaler(
        init_scale=256.0,
        scale_window=3,
        min_scale=1.0,
        delayed_shift=1,
        consecutive_hysteresis=False,
        raise_error_at_min_scale=False,
        dtype=torch.half,
    )

    initial_scale = scaler.cur_scale
    print()
    print("Initial scale:", initial_scale)

    scaler.update_scale(overflow=True)
    after_first = scaler.cur_scale
    info = scaler.get_change_info()
    print("After 1st overflow: scale=", after_first,
          ", reason=", info["reason"].value,
          ", overflow_count=", info["overflow_count"])
    assert after_first == initial_scale / 2
    assert info["reason"] == LossScaleChangeReason.OVERFLOW_DECREASE

    scaler.update_scale(overflow=True)
    after_second = scaler.cur_scale
    info = scaler.get_change_info()
    print("After 2nd overflow (duplicate): scale=", after_second,
          ", reason=", info["reason"].value,
          ", overflow_count=", info["overflow_count"])
    assert after_second == after_first
    assert info["reason"] == LossScaleChangeReason.DUPLICATE_OVERFLOW

    scaler.update_scale(overflow=True)
    after_third = scaler.cur_scale
    info = scaler.get_change_info()
    print("After 3rd overflow (duplicate): scale=", after_third,
          ", reason=", info["reason"].value,
          ", overflow_count=", info["overflow_count"])
    assert after_third == after_first

    scaler.update_scale(overflow=False)
    info = scaler.get_change_info()
    print("After stable step: scale=", scaler.cur_scale,
          ", reason=", info["reason"].value,
          ", overflow_count=", info["overflow_count"])

    scaler.update_scale(overflow=True)
    after_fourth = scaler.cur_scale
    info = scaler.get_change_info()
    print("After overflow (new): scale=", after_fourth,
          ", reason=", info["reason"].value,
          ", overflow_count=", info["overflow_count"])
    assert after_fourth == after_first / 2
    assert info["reason"] == LossScaleChangeReason.OVERFLOW_DECREASE

    print()
    print("=" * 70)
    print("All duplicate overflow assertions passed!")
    print("=" * 70)


def test_min_scale_limit():
    print()
    print("=" * 70)
    print("Test: Min Scale Limit")
    print("=" * 70)

    scaler = DynamicLossScaler(
        init_scale=4.0,
        scale_window=3,
        min_scale=2.0,
        delayed_shift=1,
        consecutive_hysteresis=False,
        raise_error_at_min_scale=False,
        dtype=torch.half,
    )

    print()
    print("Initial scale:", scaler.cur_scale, ", min_scale:", scaler.min_scale)

    for i in range(5):
        scaler.update_scale(overflow=True)
        info = scaler.get_change_info()
        print("  Overflow", i + 1, ": scale=", scaler.cur_scale,
              ", reason=", info["reason"].value)

    assert scaler.cur_scale == scaler.min_scale
    print()
    print("  => Scale correctly floored at min_scale =", scaler.min_scale)

    print()
    print("=" * 70)
    print("All min scale limit assertions passed!")
    print("=" * 70)


def test_hysteresis_decrease():
    print()
    print("=" * 70)
    print("Test: Hysteresis Decrease (no scale change)")
    print("=" * 70)

    scaler = DynamicLossScaler(
        init_scale=256.0,
        scale_window=3,
        min_scale=1.0,
        delayed_shift=3,
        consecutive_hysteresis=False,
        raise_error_at_min_scale=False,
        dtype=torch.half,
    )

    print()
    print("Initial scale:", scaler.cur_scale,
          ", cur_hysteresis:", scaler.cur_hysteresis,
          ", delayed_shift:", scaler.delayed_shift)

    scaler.update_scale(overflow=True)
    info = scaler.get_change_info()
    print("After 1st overflow: scale=", scaler.cur_scale,
          ", cur_hysteresis=", scaler.cur_hysteresis,
          ", reason=", info["reason"].value)
    assert scaler.cur_scale == 256.0
    assert scaler.cur_hysteresis == 2
    assert info["reason"] == LossScaleChangeReason.HYSTERESIS_DECREASE

    scaler.update_scale(overflow=False)
    scaler.update_scale(overflow=True)
    info = scaler.get_change_info()
    print("After 2nd overflow: scale=", scaler.cur_scale,
          ", cur_hysteresis=", scaler.cur_hysteresis,
          ", reason=", info["reason"].value)
    assert scaler.cur_scale == 256.0
    assert scaler.cur_hysteresis == 1

    scaler.update_scale(overflow=False)
    scaler.update_scale(overflow=True)
    info = scaler.get_change_info()
    print("After 3rd overflow: scale=", scaler.cur_scale,
          ", cur_hysteresis=", scaler.cur_hysteresis,
          ", reason=", info["reason"].value)
    assert scaler.cur_scale == 128.0
    assert info["reason"] == LossScaleChangeReason.OVERFLOW_DECREASE

    print()
    print("=" * 70)
    print("All hysteresis decrease assertions passed!")
    print("=" * 70)


if __name__ == "__main__":
    test_normal_and_abnormal_paths()
    test_duplicate_overflow()
    test_min_scale_limit()
    test_hysteresis_decrease()
    print()
    print("=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)
