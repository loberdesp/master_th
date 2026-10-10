"""Audio I/O and basic signal helpers.

Conventions used across the project:
- signals are mono, 1-D numpy arrays of shape (n,)
- dtype is float32, nominal range [-1.0, 1.0] (1.0 = 0 dBFS)
- the project-wide sample rate is SR = 48 kHz
"""

from pathlib import Path

import numpy as np
import soundfile as sf

SR = 48_000


def load(path: str | Path, channel: int = 0) -> tuple[np.ndarray, int]:
    """Read an audio file as a mono float32 signal and its sample rate."""
    data, samplerate = sf.read(path, dtype="float32", always_2d=True)

    # data has shape (n, channels); keep only the requested channel
    n_channels = data.shape[1]
    if not 0 <= channel < n_channels:
        raise ValueError(f"channel {channel} out of range, file has {n_channels} channel(s)")
    return data[:, channel], int(samplerate)


def save(path: str | Path, x: np.ndarray, sr: int) -> None:
    """Write a mono signal to a 32-bit float WAV file, creating parent folders."""
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"expected a mono 1-D signal, got shape {x.shape}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, x, sr, subtype="FLOAT")


def db_to_gain(db: float | np.ndarray) -> np.ndarray:
    """Convert decibels to linear amplitude."""
    db = np.asarray(db)
    return np.power(10, (db / 20))


def gain_to_db(g: float | np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Convert linear amplitude to decibels; zero is clamped to eps (-240 dB)."""
    g = np.asarray(g)
    g = np.maximum(g, eps)
    return 20 * np.log10(g)


def rms(x: np.ndarray) -> float:
    """Root-mean-square level of a signal (linear)."""
    x = np.asarray(x)
    if x.size == 0:
        raise ValueError("rms of an empty signal is undefined")
    return float(np.sqrt(np.mean(np.power(x, 2))))


def peak_normalize(x: np.ndarray, peak_db: float = -1.0) -> np.ndarray:
    """Scale x so its largest absolute sample sits at peak_db dBFS."""
    x = np.asarray(x)
    peak = np.max(np.abs(x)) if x.size else 0.0

    if peak == 0:
        return x.copy()

    target = db_to_gain(peak_db)
    return (x * (target / peak)).astype(x.dtype)


def sine(freq: float, dur: float, sr: int = SR, amp: float = 1.0) -> np.ndarray:
    """Generate a float32 sine tone of dur seconds."""
    if freq >= sr / 2:
        raise ValueError(f"freq {freq} Hz is at or above Nyquist ({sr / 2} Hz)")
    if dur <= 0:
        raise ValueError(f"dur must be positive, got {dur}")

    t = np.arange(round(dur * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
