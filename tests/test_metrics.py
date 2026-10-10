"""Tests for master_th.metrics: pre_emphasis, esr, esr_db, dc_error."""

import numpy as np
import pytest

from master_th import metrics
from master_th.audio import sine


@pytest.fixture
def x() -> np.ndarray:
    return sine(440.0, 1.0, 48_000)


# --- pre_emphasis ----------------------------------------------------------------


def test_pre_emphasis_impulse_response() -> None:
    y = metrics.pre_emphasis(np.array([1.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(y, [1.0, -0.85, 0.0, 0.0], atol=1e-7)


def test_pre_emphasis_coef_zero_is_identity() -> None:
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    np.testing.assert_allclose(metrics.pre_emphasis(x, coef=0.0), x)


def test_pre_emphasis_attenuates_dc() -> None:
    y = metrics.pre_emphasis(np.ones(10, dtype=np.float32))
    np.testing.assert_allclose(y[1:], 0.15, rtol=1e-6)


def test_pre_emphasis_matches_numpy_slicing() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1000).astype(np.float32)
    expected = np.concatenate([x[:1], x[1:] - 0.85 * x[:-1]])
    np.testing.assert_allclose(metrics.pre_emphasis(x), expected, rtol=1e-5, atol=1e-6)


def test_pre_emphasis_keeps_length_and_float32() -> None:
    y = metrics.pre_emphasis(np.zeros(100, dtype=np.float32))
    assert y.shape == (100,)
    assert y.dtype == np.float32


# --- esr -------------------------------------------------------------------------


def test_esr_perfect_prediction_is_zero(x: np.ndarray) -> None:
    assert metrics.esr(x, x) == 0.0


def test_esr_silent_prediction_is_one(x: np.ndarray) -> None:
    assert metrics.esr(x, np.zeros_like(x)) == pytest.approx(1.0)


def test_esr_ten_percent_amplitude_error_is_one_percent(x: np.ndarray) -> None:
    assert metrics.esr(x, 0.9 * x) == pytest.approx(0.01, rel=1e-4)


@pytest.mark.parametrize("k", [0.01, 2.0, 100.0])
def test_esr_is_scale_invariant(x: np.ndarray, k: float) -> None:
    pred = 0.9 * x + 0.05
    assert metrics.esr(k * x, k * pred) == pytest.approx(metrics.esr(x, pred), rel=1e-4)


def test_esr_is_not_symmetric(x: np.ndarray) -> None:
    assert metrics.esr(x, 0.5 * x) != pytest.approx(metrics.esr(0.5 * x, x))


def test_esr_returns_python_float(x: np.ndarray) -> None:
    assert type(metrics.esr(x, 0.9 * x)) is float


@pytest.mark.parametrize("bad", [lambda x: x[:-1], lambda x: x[:, None]])
def test_esr_shape_mismatch_raises(x: np.ndarray, bad) -> None:
    with pytest.raises(ValueError, match="shape"):
        metrics.esr(x, bad(x))


def test_esr_silent_target_is_finite() -> None:
    assert np.isfinite(metrics.esr(np.zeros(10), np.zeros(10)))
    assert np.isfinite(metrics.esr(np.zeros(10), np.ones(10)))


def test_esr_pre_emphasis_weights_high_frequency_error_more() -> None:
    low = sine(100.0, 1.0, 48_000)
    pred = low + 0.01 * sine(5000.0, 1.0, 48_000)
    plain = metrics.esr(low, pred)
    emph = metrics.esr(low, pred, pre_emph=True)
    assert plain == pytest.approx(1e-4, rel=1e-2)
    assert emph > 10 * plain


def test_esr_pre_emphasis_passes_coef(x: np.ndarray) -> None:
    pred = 0.9 * x + 0.05
    assert metrics.esr(x, pred, pre_emph=True, coef=0.0) == pytest.approx(metrics.esr(x, pred))


def test_esr_matches_auraloss(x: np.ndarray) -> None:
    torch = pytest.importorskip("torch")
    auraloss = pytest.importorskip("auraloss")
    pred = 0.9 * x + 0.01 * sine(3000.0, 1.0, 48_000)
    # auraloss signature is (input=pred, target), tensors shaped (batch, channels, samples)
    theirs = auraloss.time.ESRLoss()(torch.tensor(pred)[None, None], torch.tensor(x)[None, None])
    assert metrics.esr(x, pred) == pytest.approx(theirs.item(), rel=1e-4)


# --- esr_db ----------------------------------------------------------------------


def test_esr_db_one_percent_is_minus_20(x: np.ndarray) -> None:
    assert metrics.esr_db(x, 0.9 * x) == pytest.approx(-20.0, abs=1e-3)


def test_esr_db_silent_prediction_is_zero_db(x: np.ndarray) -> None:
    assert metrics.esr_db(x, np.zeros_like(x)) == pytest.approx(0.0, abs=1e-6)


def test_esr_db_perfect_match_is_finite(x: np.ndarray) -> None:
    result = metrics.esr_db(x, x)
    assert np.isfinite(result)
    assert result == pytest.approx(-100.0)


# --- dc_error --------------------------------------------------------------------


def test_dc_error_perfect_is_zero(x: np.ndarray) -> None:
    assert metrics.dc_error(x, x) == 0.0


def test_dc_error_offset(x: np.ndarray) -> None:
    # offset 0.1 on a unit sine: 0.1^2 / 0.5
    assert metrics.dc_error(x, x + 0.1) == pytest.approx(0.02, rel=1e-3)
    assert metrics.dc_error(x, x - 0.1) == pytest.approx(0.02, rel=1e-3)


def test_dc_error_ignores_pure_scaling(x: np.ndarray) -> None:
    assert metrics.dc_error(x, 0.9 * x) == pytest.approx(0.0, abs=1e-8)


def test_dc_error_is_part_of_esr(x: np.ndarray) -> None:
    pred = 0.9 * x + 0.1
    assert metrics.esr(x, pred) == pytest.approx(0.03, rel=1e-3)
    assert metrics.dc_error(x, pred) == pytest.approx(0.02, rel=1e-3)


def test_dc_error_shape_mismatch_raises(x: np.ndarray) -> None:
    with pytest.raises(ValueError, match="shape"):
        metrics.dc_error(x, x[:, None])
