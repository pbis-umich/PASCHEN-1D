"""Nonfinite transport data must not appear to satisfy stability limits."""

import numpy as np
import pytest

from numerics import compute_diffusion_cfl, compute_drift_cfl
from numerics_jit import (
    compute_diffusion_cfl_numba,
    compute_drift_cfl_numba,
    is_numba_available,
)


requires_numba = pytest.mark.skipif(not is_numba_available(), reason="Numba is not installed")
drift_backends = [compute_drift_cfl, pytest.param(compute_drift_cfl_numba, marks=requires_numba)]
diffusion_backends = [
    compute_diffusion_cfl,
    pytest.param(compute_diffusion_cfl_numba, marks=requires_numba),
]


@pytest.mark.parametrize("drift", drift_backends)
@pytest.mark.parametrize("profile", [0, 1, 2])
def test_drift_cfl_propagates_nan(drift, profile):
    values = [np.ones(3, dtype=np.float32) for _ in range(3)]
    values[profile][1] = np.nan
    assert np.isnan(drift(*values, dt=0.1, dx=1.0))


@pytest.mark.parametrize("diffusion", diffusion_backends)
@pytest.mark.parametrize("profile", [0, 1])
def test_diffusion_cfl_propagates_nan(diffusion, profile):
    values = [np.ones(3, dtype=np.float32) for _ in range(2)]
    values[profile][1] = np.nan
    assert np.isnan(diffusion(*values, dt=0.1, dx=1.0))


@pytest.mark.parametrize("drift", drift_backends)
def test_drift_cfl_preserves_infinite_stability_number(drift):
    ones = np.ones(3, dtype=np.float32)
    field = np.array([1.0, np.inf, 1.0], dtype=np.float32)
    assert np.isposinf(drift(ones, ones, field, dt=0.1, dx=1.0))


@pytest.mark.parametrize("diffusion", diffusion_backends)
def test_diffusion_cfl_preserves_infinite_stability_number(diffusion):
    electrons = np.ones(3, dtype=np.float32)
    ions = np.array([1.0, np.inf, 1.0], dtype=np.float32)
    assert np.isposinf(diffusion(electrons, ions, dt=0.1, dx=1.0))
