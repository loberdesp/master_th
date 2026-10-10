import numpy as np


def hard_clip(x, threshold=0.5):
    if threshold <= 0:
        raise ValueError(f"Threshold must be positive, got {threshold}.")
    x = np.asarray(x)
    return np.clip(x, -threshold, threshold)


def soft_clip(x, drive=1.0):
    if drive <= 0:
        raise ValueError(f"Drive must be positive, got {drive}.")
    x = np.asarray(x)
    return np.tanh(drive * x)
