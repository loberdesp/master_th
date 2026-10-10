"""Tests for master_th.effects: memoryless synthetic clippers."""

import numpy as np
import pytest

from master_th import effects
from master_th.audio import sine

CLIPPERS = [
    pytest.param(lambda x: effects.hard_clip(x, 0.5), id="hard_clip"),
    pytest.param(lambda x: effects.soft_clip(x, 3.0), id="soft_clip"),
]


@pytest.fixture
def ramp() -> np.ndarray:
    return np.linspace(-2.0, 2.0, 401, dtype=np.float32)


# --- hard_clip -------------------------------------------------------------------


def test_hard_clip_known_values() -> None:
    x = np.linspace(-1, 1, 9, dtype=np.float32)
    expected = [-0.5, -0.5, -0.5, -0.25, 0.0, 0.25, 0.5, 0.5, 0.5]
    np.testing.assert_allclose(effects.hard_clip(x, 0.5), expected)


@pytest.mark.parametrize("threshold", [0.1, 0.5, 1.0])
def test_hard_clip_is_bounded_by_threshold(ramp: np.ndarray, threshold: float) -> None:
    assert np.max(np.abs(effects.hard_clip(ramp, threshold))) <= threshold


def test_hard_clip_leaves_small_values_unchanged(ramp: np.ndarray) -> None:
    inside = ramp[np.abs(ramp) < 0.5]
    np.testing.assert_array_equal(effects.hard_clip(inside, 0.5), inside)


@pytest.mark.parametrize("threshold", [0.0, -0.5])
def test_hard_clip_rejects_non_positive_threshold(ramp: np.ndarray, threshold: float) -> None:
    with pytest.raises(ValueError):
        effects.hard_clip(ramp, threshold)


# --- soft_clip -------------------------------------------------------------------


def test_soft_clip_known_value() -> None:
    assert float(effects.soft_clip(np.array([1.0]), 1.0)[0]) == pytest.approx(np.tanh(1.0))


@pytest.mark.parametrize("drive", [0.5, 1.0, 10.0, 100.0])
def test_soft_clip_is_bounded_by_one(ramp: np.ndarray, drive: float) -> None:
    assert np.max(np.abs(effects.soft_clip(ramp, drive))) <= 1.0


def test_soft_clip_small_drive_is_almost_linear(ramp: np.ndarray) -> None:
    x = ramp / 2  # [-1, 1]
    np.testing.assert_allclose(effects.soft_clip(x, 0.01), 0.01 * x, rtol=1e-3, atol=1e-7)


def test_soft_clip_more_drive_means_more_saturation() -> None:
    x = np.array([0.5], dtype=np.float32)
    outputs = [float(effects.soft_clip(x, d)[0]) for d in (0.5, 1.0, 3.0, 10.0)]
    assert outputs == sorted(outputs)


@pytest.mark.parametrize("drive", [0.0, -1.0])
def test_soft_clip_rejects_non_positive_drive(ramp: np.ndarray, drive: float) -> None:
    with pytest.raises(ValueError):
        effects.soft_clip(ramp, drive)


# --- properties shared by both clippers ------------------------------------------


@pytest.mark.parametrize("clip", CLIPPERS)
def test_odd_symmetry(clip, ramp: np.ndarray) -> None:
    np.testing.assert_allclose(clip(-ramp), -clip(ramp), atol=1e-7)


@pytest.mark.parametrize("clip", CLIPPERS)
def test_monotonic(clip, ramp: np.ndarray) -> None:
    assert np.all(np.diff(clip(ramp)) >= 0)


@pytest.mark.parametrize("clip", CLIPPERS)
def test_keeps_shape_and_float32(clip, ramp: np.ndarray) -> None:
    y = clip(ramp)
    assert y.shape == ramp.shape
    assert y.dtype == np.float32


@pytest.mark.parametrize("clip", CLIPPERS)
def test_does_not_modify_input(clip, ramp: np.ndarray) -> None:
    before = ramp.copy()
    clip(ramp)
    np.testing.assert_array_equal(ramp, before)


@pytest.mark.parametrize("clip", CLIPPERS)
def test_symmetric_clipping_creates_only_odd_harmonics(clip) -> None:
    # 1 s at 48 kHz -> rfft bin k is exactly k Hz
    spectrum = np.abs(np.fft.rfft(clip(sine(1000.0, 1.0, 48_000))))
    fundamental = spectrum[1000]
    for even in (2000, 4000):
        assert spectrum[even] / fundamental < 1e-3
    for odd in (3000, 5000):
        assert spectrum[odd] / fundamental > 1e-2
