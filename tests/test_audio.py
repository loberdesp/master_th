"""Tests for master_th.audio: dB helpers, rms, peak_normalize, sine, save/load."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from master_th import audio

# --- db_to_gain / gain_to_db ---------------------------------------------------


@pytest.mark.parametrize(
    ("db", "gain"),
    [(0.0, 1.0), (-6.0206, 0.5), (-20.0, 0.1), (20.0, 10.0), (-60.0, 0.001)],
)
def test_db_to_gain_known_values(db: float, gain: float) -> None:
    assert float(audio.db_to_gain(db)) == pytest.approx(gain, rel=1e-4)


@pytest.mark.parametrize(
    ("gain", "db"),
    [(1.0, 0.0), (0.5, -6.0206), (0.1, -20.0), (10.0, 20.0)],
)
def test_gain_to_db_known_values(gain: float, db: float) -> None:
    assert float(audio.gain_to_db(gain)) == pytest.approx(db, abs=1e-3)


def test_db_gain_roundtrip_array() -> None:
    dbs = np.linspace(-80.0, 20.0, 11)
    np.testing.assert_allclose(audio.gain_to_db(audio.db_to_gain(dbs)), dbs, atol=1e-9)


def test_gain_to_db_returns_number_not_list() -> None:
    assert not isinstance(audio.gain_to_db(1.0), list)


def test_gain_to_db_of_zero_is_finite() -> None:
    result = float(audio.gain_to_db(0.0))
    assert np.isfinite(result)
    assert result < -200.0


# --- rms -----------------------------------------------------------------------


def test_rms_of_constant() -> None:
    assert audio.rms(np.full(100, -0.5)) == pytest.approx(0.5)


def test_rms_of_silence() -> None:
    assert audio.rms(np.zeros(100)) == 0.0


def test_rms_of_full_scale_sine() -> None:
    x = audio.sine(1000.0, 1.0, 48_000)
    assert audio.rms(x) == pytest.approx(1 / np.sqrt(2), rel=1e-3)


# --- peak_normalize --------------------------------------------------------------


@pytest.mark.parametrize("peak_db", [-1.0, -6.0, 0.0])
def test_peak_normalize_hits_target(peak_db: float) -> None:
    x = np.array([0.1, -0.3, 0.2], dtype=np.float32)
    y = audio.peak_normalize(x, peak_db=peak_db)
    assert float(audio.gain_to_db(np.max(np.abs(y)))) == pytest.approx(peak_db, abs=1e-4)


def test_peak_normalize_keeps_shape_of_waveform() -> None:
    x = np.array([0.1, -0.3, 0.2], dtype=np.float32)
    y = audio.peak_normalize(x)
    np.testing.assert_allclose(y / y[1], x / x[1], rtol=1e-6)


def test_peak_normalize_keeps_float32() -> None:
    x = np.array([0.1, -0.3, 0.2], dtype=np.float32)
    assert audio.peak_normalize(x).dtype == np.float32


def test_peak_normalize_does_not_modify_input() -> None:
    x = np.array([0.1, -0.3, 0.2], dtype=np.float32)
    before = x.copy()
    audio.peak_normalize(x)
    np.testing.assert_array_equal(x, before)


def test_peak_normalize_silence_stays_silent() -> None:
    y = audio.peak_normalize(np.zeros(10, dtype=np.float32))
    np.testing.assert_array_equal(y, 0.0)


# --- sine ----------------------------------------------------------------------


def test_sine_length_dtype_shape() -> None:
    x = audio.sine(440.0, 1.0, 48_000)
    assert x.shape == (48_000,)
    assert x.dtype == np.float32


def test_sine_starts_at_zero_and_peaks_at_amp() -> None:
    x = audio.sine(1000.0, 0.1, 48_000, amp=0.5)
    assert x[0] == pytest.approx(0.0, abs=1e-7)
    assert np.max(np.abs(x)) == pytest.approx(0.5, rel=1e-3)


def test_sine_above_nyquist_raises() -> None:
    with pytest.raises(ValueError):
        audio.sine(30_000.0, 1.0, 48_000)


# --- save / load ---------------------------------------------------------------


def test_save_load_roundtrip(tmp_path: Path) -> None:
    x = audio.sine(440.0, 0.5, 48_000, amp=0.8)
    path = tmp_path / "tone.wav"
    audio.save(path, x, 48_000)

    y, sr = audio.load(path)
    assert sr == 48_000
    assert y.dtype == np.float32
    assert y.shape == x.shape
    np.testing.assert_allclose(y, x, atol=1e-6)


def test_save_creates_missing_folders(tmp_path: Path) -> None:
    path = tmp_path / "runs" / "nested" / "out.wav"
    audio.save(path, np.zeros(100, dtype=np.float32), 48_000)
    assert path.exists()


def test_save_keeps_values_above_full_scale(tmp_path: Path) -> None:
    x = np.array([0.0, 1.5, -2.0], dtype=np.float32)
    path = tmp_path / "hot.wav"
    audio.save(path, x, 48_000)
    y, _ = audio.load(path)
    np.testing.assert_allclose(y, x, atol=1e-6)


def test_load_stereo_returns_1d_channel(tmp_path: Path) -> None:
    left = np.full(100, 0.25, dtype=np.float32)
    right = np.full(100, -0.5, dtype=np.float32)
    path = tmp_path / "stereo.wav"
    sf.write(path, np.stack([left, right], axis=1), 48_000, subtype="FLOAT")

    y, _ = audio.load(path)
    assert y.ndim == 1
    np.testing.assert_allclose(y, left)

    y_right, _ = audio.load(path, channel=1)
    np.testing.assert_allclose(y_right, right)


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(Exception):  # noqa: B017 - soundfile raises its own error type
        audio.load(tmp_path / "nope.wav")
