"""
numerics_jit.py

Optional Numba-accelerated hot loops for PASCHEN-1D.

This module provides JIT kernels for the linear KT+RK4 updates used in
drift-diffusion continuity equations:

    f(n, u) = adv_coeff * n * u

with explicit diffusion and source terms on a uniform 1D grid.

All entry points here are optional runtime accelerators. The reference
NumPy path in ``numerics.py`` remains the baseline solver behavior.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - executed only when numba is absent
    NUMBA_AVAILABLE = False
    njit = None  # type: ignore[assignment]
    prange = range  # type: ignore[assignment]


def is_numba_available() -> bool:
    """Return True when Numba is importable in the current environment."""
    return bool(NUMBA_AVAILABLE)


def create_numba_linear_rk4_workspace(
    Nx: int,
    dtype=np.float32,
) -> dict[str, np.ndarray]:
    """
    Allocate reusable buffers for Numba linear KT+RK4 kernels.
    """
    if Nx < 3:
        raise ValueError("Nx must be >= 3.")
    n_face = Nx - 2
    return {
        "slope": np.zeros(Nx, dtype=dtype),
        "nL_p": np.empty(n_face, dtype=dtype),
        "nR_p": np.empty(n_face, dtype=dtype),
        "nL_m": np.empty(n_face, dtype=dtype),
        "nR_m": np.empty(n_face, dtype=dtype),
        "u_face_p": np.empty(n_face, dtype=dtype),
        "u_face_m": np.empty(n_face, dtype=dtype),
        "a_p": np.empty(n_face, dtype=dtype),
        "a_m": np.empty(n_face, dtype=dtype),
        "H_p": np.empty(n_face, dtype=dtype),
        "H_m": np.empty(n_face, dtype=dtype),
        "grad_p": np.empty(n_face, dtype=dtype),
        "grad_m": np.empty(n_face, dtype=dtype),
        "D_p": np.empty(n_face, dtype=dtype),
        "D_m": np.empty(n_face, dtype=dtype),
        "Fd_p": np.empty(n_face, dtype=dtype),
        "Fd_m": np.empty(n_face, dtype=dtype),
        "k1": np.empty(Nx, dtype=dtype),
        "k2": np.empty(Nx, dtype=dtype),
        "k3": np.empty(Nx, dtype=dtype),
        "k4": np.empty(Nx, dtype=dtype),
        "n_tmp": np.empty(Nx, dtype=dtype),
        "accum": np.empty(Nx, dtype=dtype),
    }


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _minmod_scalar(a: float, b: float, c: float) -> float:
        if (a > 0.0) and (b > 0.0) and (c > 0.0):
            m = a if a < b else b
            return m if m < c else c
        if (a < 0.0) and (b < 0.0) and (c < 0.0):
            m = a if a > b else b
            return m if m > c else c
        return 0.0


    @njit(cache=True)
    def _kt_rhs_linear_serial(
        n: np.ndarray,
        u: np.ndarray,
        D: np.ndarray,
        S: np.ndarray,
        dx: float,
        theta: float,
        adv_coeff: float,
        rhs_out: np.ndarray,
        slope: np.ndarray,
        nL_p: np.ndarray,
        nR_p: np.ndarray,
        nL_m: np.ndarray,
        nR_m: np.ndarray,
        u_face_p: np.ndarray,
        u_face_m: np.ndarray,
        a_p: np.ndarray,
        a_m: np.ndarray,
        H_p: np.ndarray,
        H_m: np.ndarray,
        grad_p: np.ndarray,
        grad_m: np.ndarray,
        D_p: np.ndarray,
        D_m: np.ndarray,
        Fd_p: np.ndarray,
        Fd_m: np.ndarray,
        boundary_flux_left: float,
        boundary_flux_right: float,
        replace_boundary_flux_left: bool,
        replace_boundary_flux_right: bool,
    ) -> None:
        Nx = n.shape[0]
        rhs_out[0] = 0.0
        rhs_out[Nx - 1] = 0.0
        slope[0] = 0.0
        slope[Nx - 1] = 0.0

        for i in range(1, Nx - 1):
            dn_bwd = (n[i] - n[i - 1]) / dx
            dn_ctr = 0.5 * (n[i + 1] - n[i - 1]) / dx
            dn_fwd = (n[i + 1] - n[i]) / dx
            slope[i] = _minmod_scalar(theta * dn_bwd, dn_ctr, theta * dn_fwd)

        for j in range(Nx - 2):
            i = j + 1
            nL_p[j] = n[i] + 0.5 * dx * slope[i]
            nR_p[j] = n[i + 1] - 0.5 * dx * slope[i + 1]
            nL_m[j] = n[i - 1] + 0.5 * dx * slope[i - 1]
            nR_m[j] = n[i] - 0.5 * dx * slope[i]

            u_face_p[j] = 0.5 * (u[i] + u[i + 1])
            u_face_m[j] = 0.5 * (u[i] + u[i - 1])
            a_p[j] = abs(adv_coeff * u_face_p[j])
            a_m[j] = abs(adv_coeff * u_face_m[j])

            H_p[j] = (
                0.5 * adv_coeff * (nL_p[j] + nR_p[j]) * u_face_p[j]
                - 0.5 * a_p[j] * (nR_p[j] - nL_p[j])
            )
            H_m[j] = (
                0.5 * adv_coeff * (nL_m[j] + nR_m[j]) * u_face_m[j]
                - 0.5 * a_m[j] * (nR_m[j] - nL_m[j])
            )

            grad_p[j] = (n[i + 1] - n[i]) / dx
            grad_m[j] = (n[i] - n[i - 1]) / dx
            D_p[j] = 0.5 * (D[i] + D[i + 1])
            D_m[j] = 0.5 * (D[i - 1] + D[i])
            Fd_p[j] = -D_p[j] * grad_p[j]
            Fd_m[j] = -D_m[j] * grad_m[j]

            rhs_out[i] = (
                -(H_p[j] - H_m[j]) / dx
                - (Fd_p[j] - Fd_m[j]) / dx
                + S[i]
            )

        if replace_boundary_flux_left:
            old_left = H_m[0] + Fd_m[0]
            drift_left = adv_coeff * n[1] * u[1]
            outflow_left = drift_left if drift_left < 0.0 else 0.0
            new_left = outflow_left + boundary_flux_left
            rhs_out[1] += (new_left - old_left) / dx
        elif boundary_flux_left != 0.0:
            rhs_out[1] += boundary_flux_left / dx

        if replace_boundary_flux_right:
            old_right = H_p[Nx - 3] + Fd_p[Nx - 3]
            drift_right = adv_coeff * n[Nx - 2] * u[Nx - 2]
            outflow_right = drift_right if drift_right > 0.0 else 0.0
            new_right = outflow_right + boundary_flux_right
            rhs_out[Nx - 2] += (old_right - new_right) / dx
        elif boundary_flux_right != 0.0:
            rhs_out[Nx - 2] -= boundary_flux_right / dx


    @njit(cache=True, parallel=True)
    def _kt_rhs_linear_parallel(
        n: np.ndarray,
        u: np.ndarray,
        D: np.ndarray,
        S: np.ndarray,
        dx: float,
        theta: float,
        adv_coeff: float,
        rhs_out: np.ndarray,
        slope: np.ndarray,
        nL_p: np.ndarray,
        nR_p: np.ndarray,
        nL_m: np.ndarray,
        nR_m: np.ndarray,
        u_face_p: np.ndarray,
        u_face_m: np.ndarray,
        a_p: np.ndarray,
        a_m: np.ndarray,
        H_p: np.ndarray,
        H_m: np.ndarray,
        grad_p: np.ndarray,
        grad_m: np.ndarray,
        D_p: np.ndarray,
        D_m: np.ndarray,
        Fd_p: np.ndarray,
        Fd_m: np.ndarray,
        boundary_flux_left: float,
        boundary_flux_right: float,
        replace_boundary_flux_left: bool,
        replace_boundary_flux_right: bool,
    ) -> None:
        Nx = n.shape[0]
        rhs_out[0] = 0.0
        rhs_out[Nx - 1] = 0.0
        slope[0] = 0.0
        slope[Nx - 1] = 0.0

        for i in prange(1, Nx - 1):
            dn_bwd = (n[i] - n[i - 1]) / dx
            dn_ctr = 0.5 * (n[i + 1] - n[i - 1]) / dx
            dn_fwd = (n[i + 1] - n[i]) / dx
            slope[i] = _minmod_scalar(theta * dn_bwd, dn_ctr, theta * dn_fwd)

        for j in prange(Nx - 2):
            i = j + 1
            nL_p[j] = n[i] + 0.5 * dx * slope[i]
            nR_p[j] = n[i + 1] - 0.5 * dx * slope[i + 1]
            nL_m[j] = n[i - 1] + 0.5 * dx * slope[i - 1]
            nR_m[j] = n[i] - 0.5 * dx * slope[i]

            u_face_p[j] = 0.5 * (u[i] + u[i + 1])
            u_face_m[j] = 0.5 * (u[i] + u[i - 1])
            a_p[j] = abs(adv_coeff * u_face_p[j])
            a_m[j] = abs(adv_coeff * u_face_m[j])

            H_p[j] = (
                0.5 * adv_coeff * (nL_p[j] + nR_p[j]) * u_face_p[j]
                - 0.5 * a_p[j] * (nR_p[j] - nL_p[j])
            )
            H_m[j] = (
                0.5 * adv_coeff * (nL_m[j] + nR_m[j]) * u_face_m[j]
                - 0.5 * a_m[j] * (nR_m[j] - nL_m[j])
            )

            grad_p[j] = (n[i + 1] - n[i]) / dx
            grad_m[j] = (n[i] - n[i - 1]) / dx
            D_p[j] = 0.5 * (D[i] + D[i + 1])
            D_m[j] = 0.5 * (D[i - 1] + D[i])
            Fd_p[j] = -D_p[j] * grad_p[j]
            Fd_m[j] = -D_m[j] * grad_m[j]

            rhs_out[i] = (
                -(H_p[j] - H_m[j]) / dx
                - (Fd_p[j] - Fd_m[j]) / dx
                + S[i]
            )

        if replace_boundary_flux_left:
            old_left = H_m[0] + Fd_m[0]
            drift_left = adv_coeff * n[1] * u[1]
            outflow_left = drift_left if drift_left < 0.0 else 0.0
            new_left = outflow_left + boundary_flux_left
            rhs_out[1] += (new_left - old_left) / dx
        elif boundary_flux_left != 0.0:
            rhs_out[1] += boundary_flux_left / dx

        if replace_boundary_flux_right:
            old_right = H_p[Nx - 3] + Fd_p[Nx - 3]
            drift_right = adv_coeff * n[Nx - 2] * u[Nx - 2]
            outflow_right = drift_right if drift_right > 0.0 else 0.0
            new_right = outflow_right + boundary_flux_right
            rhs_out[Nx - 2] += (old_right - new_right) / dx
        elif boundary_flux_right != 0.0:
            rhs_out[Nx - 2] -= boundary_flux_right / dx


    @njit(cache=True)
    def _rk4_linear_serial(
        n: np.ndarray,
        u: np.ndarray,
        D: np.ndarray,
        S: np.ndarray,
        dx: float,
        dt: float,
        theta: float,
        adv_coeff: float,
        n_out: np.ndarray,
        k1: np.ndarray,
        k2: np.ndarray,
        k3: np.ndarray,
        k4: np.ndarray,
        n_tmp: np.ndarray,
        accum: np.ndarray,
        slope: np.ndarray,
        nL_p: np.ndarray,
        nR_p: np.ndarray,
        nL_m: np.ndarray,
        nR_m: np.ndarray,
        u_face_p: np.ndarray,
        u_face_m: np.ndarray,
        a_p: np.ndarray,
        a_m: np.ndarray,
        H_p: np.ndarray,
        H_m: np.ndarray,
        grad_p: np.ndarray,
        grad_m: np.ndarray,
        D_p: np.ndarray,
        D_m: np.ndarray,
        Fd_p: np.ndarray,
        Fd_m: np.ndarray,
        boundary_flux_left: float,
        boundary_flux_right: float,
        replace_boundary_flux_left: bool,
        replace_boundary_flux_right: bool,
    ) -> None:
        Nx = n.shape[0]
        _kt_rhs_linear_serial(
            n, u, D, S, dx, theta, adv_coeff, k1,
            slope, nL_p, nR_p, nL_m, nR_m, u_face_p, u_face_m, a_p, a_m,
            H_p, H_m, grad_p, grad_m, D_p, D_m, Fd_p, Fd_m,
            boundary_flux_left, boundary_flux_right,
            replace_boundary_flux_left, replace_boundary_flux_right
        )
        for i in range(Nx):
            n_tmp[i] = n[i] + 0.5 * dt * k1[i]

        _kt_rhs_linear_serial(
            n_tmp, u, D, S, dx, theta, adv_coeff, k2,
            slope, nL_p, nR_p, nL_m, nR_m, u_face_p, u_face_m, a_p, a_m,
            H_p, H_m, grad_p, grad_m, D_p, D_m, Fd_p, Fd_m,
            boundary_flux_left, boundary_flux_right,
            replace_boundary_flux_left, replace_boundary_flux_right
        )
        for i in range(Nx):
            n_tmp[i] = n[i] + 0.5 * dt * k2[i]

        _kt_rhs_linear_serial(
            n_tmp, u, D, S, dx, theta, adv_coeff, k3,
            slope, nL_p, nR_p, nL_m, nR_m, u_face_p, u_face_m, a_p, a_m,
            H_p, H_m, grad_p, grad_m, D_p, D_m, Fd_p, Fd_m,
            boundary_flux_left, boundary_flux_right,
            replace_boundary_flux_left, replace_boundary_flux_right
        )
        for i in range(Nx):
            n_tmp[i] = n[i] + dt * k3[i]

        _kt_rhs_linear_serial(
            n_tmp, u, D, S, dx, theta, adv_coeff, k4,
            slope, nL_p, nR_p, nL_m, nR_m, u_face_p, u_face_m, a_p, a_m,
            H_p, H_m, grad_p, grad_m, D_p, D_m, Fd_p, Fd_m,
            boundary_flux_left, boundary_flux_right,
            replace_boundary_flux_left, replace_boundary_flux_right
        )
        for i in range(Nx):
            accum[i] = k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]
            val = n[i] + (dt / 6.0) * accum[i]
            n_out[i] = 0.0 if val < 0.0 else val


    @njit(cache=True, parallel=True)
    def _rk4_linear_parallel(
        n: np.ndarray,
        u: np.ndarray,
        D: np.ndarray,
        S: np.ndarray,
        dx: float,
        dt: float,
        theta: float,
        adv_coeff: float,
        n_out: np.ndarray,
        k1: np.ndarray,
        k2: np.ndarray,
        k3: np.ndarray,
        k4: np.ndarray,
        n_tmp: np.ndarray,
        accum: np.ndarray,
        slope: np.ndarray,
        nL_p: np.ndarray,
        nR_p: np.ndarray,
        nL_m: np.ndarray,
        nR_m: np.ndarray,
        u_face_p: np.ndarray,
        u_face_m: np.ndarray,
        a_p: np.ndarray,
        a_m: np.ndarray,
        H_p: np.ndarray,
        H_m: np.ndarray,
        grad_p: np.ndarray,
        grad_m: np.ndarray,
        D_p: np.ndarray,
        D_m: np.ndarray,
        Fd_p: np.ndarray,
        Fd_m: np.ndarray,
        boundary_flux_left: float,
        boundary_flux_right: float,
        replace_boundary_flux_left: bool,
        replace_boundary_flux_right: bool,
    ) -> None:
        Nx = n.shape[0]
        _kt_rhs_linear_parallel(
            n, u, D, S, dx, theta, adv_coeff, k1,
            slope, nL_p, nR_p, nL_m, nR_m, u_face_p, u_face_m, a_p, a_m,
            H_p, H_m, grad_p, grad_m, D_p, D_m, Fd_p, Fd_m,
            boundary_flux_left, boundary_flux_right,
            replace_boundary_flux_left, replace_boundary_flux_right
        )
        for i in prange(Nx):
            n_tmp[i] = n[i] + 0.5 * dt * k1[i]

        _kt_rhs_linear_parallel(
            n_tmp, u, D, S, dx, theta, adv_coeff, k2,
            slope, nL_p, nR_p, nL_m, nR_m, u_face_p, u_face_m, a_p, a_m,
            H_p, H_m, grad_p, grad_m, D_p, D_m, Fd_p, Fd_m,
            boundary_flux_left, boundary_flux_right,
            replace_boundary_flux_left, replace_boundary_flux_right
        )
        for i in prange(Nx):
            n_tmp[i] = n[i] + 0.5 * dt * k2[i]

        _kt_rhs_linear_parallel(
            n_tmp, u, D, S, dx, theta, adv_coeff, k3,
            slope, nL_p, nR_p, nL_m, nR_m, u_face_p, u_face_m, a_p, a_m,
            H_p, H_m, grad_p, grad_m, D_p, D_m, Fd_p, Fd_m,
            boundary_flux_left, boundary_flux_right,
            replace_boundary_flux_left, replace_boundary_flux_right
        )
        for i in prange(Nx):
            n_tmp[i] = n[i] + dt * k3[i]

        _kt_rhs_linear_parallel(
            n_tmp, u, D, S, dx, theta, adv_coeff, k4,
            slope, nL_p, nR_p, nL_m, nR_m, u_face_p, u_face_m, a_p, a_m,
            H_p, H_m, grad_p, grad_m, D_p, D_m, Fd_p, Fd_m,
            boundary_flux_left, boundary_flux_right,
            replace_boundary_flux_left, replace_boundary_flux_right
        )
        for i in prange(Nx):
            accum[i] = k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]
            val = n[i] + (dt / 6.0) * accum[i]
            n_out[i] = 0.0 if val < 0.0 else val


    @njit(cache=True)
    def _compute_drift_cfl_numba(
        mu_e: np.ndarray,
        mu_i: np.ndarray,
        E: np.ndarray,
        dt: float,
        dx: float,
    ) -> float:
        a_max = 0.0
        Nx = E.shape[0]
        for i in range(Nx):
            a_e = abs(mu_e[i] * E[i])
            a_i = abs(mu_i[i] * E[i])
            if np.isnan(a_e) or np.isnan(a_i):
                return np.nan
            if a_e > a_max:
                a_max = a_e
            if a_i > a_max:
                a_max = a_i
        return a_max * dt / dx


    @njit(cache=True)
    def _compute_diffusion_cfl_numba(
        D_e: np.ndarray,
        D_i: np.ndarray,
        dt: float,
        dx: float,
    ) -> float:
        D_max = 0.0
        Nx = D_e.shape[0]
        for i in range(Nx):
            D_e_abs = abs(D_e[i])
            D_i_abs = abs(D_i[i])
            if np.isnan(D_e_abs) or np.isnan(D_i_abs):
                return np.nan
            if D_e_abs > D_max:
                D_max = D_e_abs
            if D_i_abs > D_max:
                D_max = D_i_abs
        return D_max * dt / (dx * dx)


def rk4_step_linear_numba_reuse(
    n: np.ndarray,
    u: np.ndarray,
    D: np.ndarray,
    S: np.ndarray,
    dx: float,
    dt: float,
    kt_limiter_theta: float,
    adv_coeff: float,
    ws: dict[str, np.ndarray],
    n_out: np.ndarray,
    *,
    parallel: bool = False,
    boundary_flux_left: float = 0.0,
    boundary_flux_right: float = 0.0,
    replace_boundary_flux_left: bool = False,
    replace_boundary_flux_right: bool = False,
) -> np.ndarray:
    """
    Run Numba-accelerated linear KT+RK4 update with reusable buffers.
    """
    if not NUMBA_AVAILABLE:
        raise RuntimeError("Numba backend requested but numba is not available.")

    kernel = _rk4_linear_parallel if parallel else _rk4_linear_serial
    kernel(
        n,
        u,
        D,
        S,
        float(dx),
        float(dt),
        float(kt_limiter_theta),
        float(adv_coeff),
        n_out,
        ws["k1"],
        ws["k2"],
        ws["k3"],
        ws["k4"],
        ws["n_tmp"],
        ws["accum"],
        ws["slope"],
        ws["nL_p"],
        ws["nR_p"],
        ws["nL_m"],
        ws["nR_m"],
        ws["u_face_p"],
        ws["u_face_m"],
        ws["a_p"],
        ws["a_m"],
        ws["H_p"],
        ws["H_m"],
        ws["grad_p"],
        ws["grad_m"],
        ws["D_p"],
        ws["D_m"],
        ws["Fd_p"],
        ws["Fd_m"],
        float(boundary_flux_left),
        float(boundary_flux_right),
        bool(replace_boundary_flux_left),
        bool(replace_boundary_flux_right),
    )
    return n_out


def compute_drift_cfl_numba(
    mu_e: np.ndarray,
    mu_i: np.ndarray,
    E: np.ndarray,
    dt: float,
    dx: float,
) -> float:
    """
    Numba-accelerated drift CFL utility.
    """
    if not NUMBA_AVAILABLE:
        raise RuntimeError("Numba backend requested but numba is not available.")
    return float(
        _compute_drift_cfl_numba(
            mu_e,
            mu_i,
            E,
            float(dt),
            float(dx),
        )
    )


def compute_diffusion_cfl_numba(
    D_e: np.ndarray,
    D_i: np.ndarray,
    dt: float,
    dx: float,
) -> float:
    """
    Numba-accelerated explicit-diffusion stability diagnostic.
    """
    if not NUMBA_AVAILABLE:
        raise RuntimeError("Numba backend requested but numba is not available.")
    return float(
        _compute_diffusion_cfl_numba(
            D_e,
            D_i,
            float(dt),
            float(dx),
        )
    )
