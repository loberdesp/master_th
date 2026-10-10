"""Objective error metrics for comparing a model output with the target effect output.

All functions take (target, pred) in that order: the target is the reference
(real effect) and sits in the denominator, so the metrics are not symmetric.
"""

import numpy as np
from scipy.signal import lfilter


def pre_emphasis(x: np.ndarray, coef: float = 0.85) -> np.ndarray:
    """Apply the first-order pre-emphasis filter 1 - coef*z^-1 (Wright et al. 2019)."""
    x = np.asarray(x)

    a = [1]
    b = [1, -coef]

    filtered = np.asarray(lfilter(b, a, x))
    return filtered.astype(np.float32)


def _check_shapes(target: np.ndarray, pred: np.ndarray) -> None:
    if target.shape != pred.shape:
        raise ValueError(f"target shape {target.shape} doesn't match pred shape {pred.shape}")


def esr(
    target: np.ndarray,
    pred: np.ndarray,
    pre_emph: bool = False,
    coef: float = 0.85,
    eps: float = 1e-10,
) -> float:
    """Error-to-signal ratio: error energy divided by target energy (0 = perfect)."""
    target = np.asarray(target)
    pred = np.asarray(pred)
    _check_shapes(target, pred)
    if pre_emph:
        target = pre_emphasis(target, coef)
        pred = pre_emphasis(pred, coef)
    error = np.subtract(target, pred)
    esr_num = np.mean(np.power(error, 2))
    esr_den = np.mean(np.power(target, 2)) + eps
    return float(esr_num / esr_den)


def esr_db(
    target: np.ndarray,
    pred: np.ndarray,
    pre_emph: bool = False,
    coef: float = 0.85,
    floor: float = 1e-10,
) -> float:
    """ESR in dB (power ratio, 10*log10); a perfect match is clamped to floor (-100 dB)."""
    esr_value = esr(target, pred, pre_emph, coef)
    return float(10 * np.log10(np.maximum(esr_value, floor)))


def dc_error(target: np.ndarray, pred: np.ndarray, eps: float = 1e-10) -> float:
    """DC offset error: squared mean of the error divided by target power (Wright et al.)."""
    target = np.asarray(target)
    pred = np.asarray(pred)
    _check_shapes(target, pred)
    dc = np.power(np.mean(target - pred), 2) / (np.mean(np.power(target, 2)) + eps)
    return float(dc)
