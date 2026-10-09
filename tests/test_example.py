"""Template showing common pytest patterns. Copy what you need, then delete this file.

Run: `make test` (fast tests only) or `make test-all` (includes @slow).
"""

import json
from pathlib import Path

import numpy as np
import pytest


# 1. Plain assert: pytest rewrites asserts to show both sides on failure.
def test_plain_assert() -> None:
    assert sum([1, 2, 3]) == 6


# 2. Fixture: reusable setup, injected by argument name.
@pytest.fixture
def sine() -> np.ndarray:
    sr = 48_000
    t = np.arange(sr) / sr
    return np.sin(2 * np.pi * 440.0 * t)


def test_fixture(sine: np.ndarray) -> None:
    assert sine.shape == (48_000,)
    assert np.max(np.abs(sine)) <= 1.0


# 3. Parametrize: one test function, several input/expected cases.
@pytest.mark.parametrize(
    ("db", "expected_gain"),
    [(0.0, 1.0), (-6.0, 0.501), (20.0, 10.0)],
)
def test_parametrize(db: float, expected_gain: float) -> None:
    assert 10 ** (db / 20) == pytest.approx(expected_gain, rel=1e-3)


# 4. tmp_path: a fresh temporary directory per test (pathlib.Path).
def test_tmp_path(tmp_path: Path) -> None:
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"lr": 1e-3}))
    assert json.loads(cfg.read_text())["lr"] == 1e-3


# 5. NumPy comparison: tolerance-based, reports mismatching elements.
def test_numpy_allclose() -> None:
    x = np.linspace(0.0, 1.0, 5)
    np.testing.assert_allclose(np.sqrt(x) ** 2, x, rtol=1e-12, atol=1e-12)


# 6. pytest.raises: assert that an exception is raised (optionally matching a message).
def test_raises() -> None:
    with pytest.raises(ValueError, match="could not be broadcast"):
        _ = np.zeros(3) + np.zeros(4)


# 7. Custom marker: excluded by `make test` and CI; run with `make test-all`.
@pytest.mark.slow
def test_slow() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1_000_000)
    assert abs(x.mean()) < 0.01
