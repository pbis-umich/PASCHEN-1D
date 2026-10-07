"""
numerics.py

Numerical kernels used by PASCHEN-1D.

Included components:
1) 1D Poisson solve with Dirichlet boundaries (banded Laplacian).
2) Minmod slope limiter.
3) KT central-upwind drift flux + explicit diffusion + source RHS.
4) RK4 density update wrapper.
5) Reusable linear KT+RK4 workspaces (low-allocation hot path).
6) Boundary-condition helpers for electron/ion densities.
7) Drift- and diffusion-based explicit-stability diagnostics.

All routines assume a uniform 1D spatial grid.
"""

import numpy as np
from typing import Callable
from scipy.linalg import solve_banded

from physical_constants import e, eps0
from physics import (
    boundary_zero_density,
    boundary_cathode_ion_implicit_drift_density,
    boundary_anode_electron_implicit_drift_density,
)


# ============================================================
# Poisson solver (banded Laplacian with Dirichlet BCs)
# ============================================================


def build_poisson_tridiag_interior(Nx: int, dtype=np.float64) -> np.ndarray:
    """
    Build banded interior matrix for 1D Laplacian with Dirichlet endpoints.

    We discretize the Laplacian on a uniform grid as:

        (phi_{j+1} - 2 phi_j + phi_{j-1}) / dx² ≈ d²phi/dx²

    For `scipy.linalg.solve_banded` with (l, u) = (1, 1), the banded
    matrix layout is:

        ab[0, 1:]  = upper diagonal  (1)
        ab[1, :]   = main  diagonal  (-2)
        ab[2, :-1] = lower diagonal  (1)

    so the interior system (Nx-2 unknowns) reads:

        A_int phi_int = b_int,

    with phi_0 and phi_{Nx-1} enforced separately through b_int.

    Parameters
    ----------
    Nx : int
        Total number of grid points in x (must be >= 3).
    dtype : data-type, optional
        Data type for the banded matrix. Default is np.float64.

    Returns
    -------
    ab : np.ndarray
        Banded matrix representation with shape (3, Nx - 2).
    """
    Nint = Nx - 2
    if Nint <= 0:
        raise ValueError("Nx must be >= 3.")
    ab = np.zeros((3, Nint), dtype=dtype)
    ab[0, 1:] = 1.0
    ab[1, :] = -2.0
    ab[2, :-1] = 1.0
    return ab


def poisson_1d_dirichlet_interior(
    n_i: np.ndarray,
    n_e: np.ndarray,
    dx: float,
    phi_left: float,
    phi_right: float,
    ab_int: np.ndarray,
    phi_out: np.ndarray,
    E_out: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve the 1D Poisson equation

        d²phi/dx² = -(e/eps0) (n_i - n_e)

    on a grid j = 0..Nx-1 with Dirichlet boundary conditions:

        phi(0)      = phi_left
        phi(Nx - 1) = phi_right.

    The solution and derived electric field are written in-place to
    `phi_out` and `E_out`.

    Parameters
    ----------
    n_i, n_e : np.ndarray
        Ion and electron number densities [m⁻³], shape (Nx,).
    dx : float
        Grid spacing [m].
    phi_left, phi_right : float
        Dirichlet boundary potentials at x = 0 and x = L [V].
    ab_int : np.ndarray
        Banded Laplacian for interior nodes, as produced by
        `build_poisson_tridiag_interior`, shape (3, Nx - 2).
    phi_out : np.ndarray
        Output array for potential phi(x), shape (Nx,). Will be overwritten.
    E_out : np.ndarray
        Output array for electric field E(x) = -dphi/dx, shape (Nx,).
        Will be overwritten.

    Returns
    -------
    phi_out, E_out : np.ndarray
        References to the same arrays passed in (for convenience).

    Notes
    -----
    * The banded solver is applied only on interior nodes (1..Nx-2).
    * Boundary values are set directly from phi_left, phi_right.
    * E is computed via second-order central differences in the interior
      and first-order one-sided differences at the boundaries.
    """
    # RHS for interior equations. Multiply by dx^2 so the discrete operator
    # remains [-2, 1, 1] without extra dx scaling inside A.
    b = -(e / eps0) * (n_i.astype(np.float64) - n_e.astype(np.float64)) * (dx * dx)
    b_int = b[1:-1].copy()
    b_int[0] -= phi_left
    b_int[-1] -= phi_right

    # Solve for interior phi using banded solver
    phi_int = solve_banded(
        (1, 1),
        ab_int,
        b_int,
        overwrite_ab=False,
        overwrite_b=False,
        check_finite=False,
    )

    # Assemble full potential.
    phi_out[0] = float(phi_left)
    phi_out[-1] = float(phi_right)
    phi_out[1:-1] = phi_int.astype(phi_out.dtype, copy=False)

    # Electric field E = -dphi/dx.
    inv_dx = 1.0 / dx
    # Interior: second-order central
    E_out[1:-1] = -(phi_out[2:] - phi_out[:-2]) * (0.5 * inv_dx)
    # Boundaries: one-sided first-order
    E_out[0] = -(phi_out[1] - phi_out[0]) * inv_dx
    E_out[-1] = -(phi_out[-1] - phi_out[-2]) * inv_dx

    return phi_out, E_out


# ============================================================
# Minmod limiter
# ============================================================


def minmod(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Vectorized three-argument minmod slope limiter.

    For each component i:

        minmod(a_i, b_i, c_i) =
            min(a_i, b_i, c_i)  if all > 0,
            max(a_i, b_i, c_i)  if all < 0,
            0                   otherwise.

    Parameters
    ----------
    a, b, c : np.ndarray
        Arrays of the same shape containing candidate slopes.

    Returns
    -------
    np.ndarray
        Limited slopes with the same shape as the inputs.

    Notes
    -----
    * This function is used to compute limited gradients for the KT
      central-upwind scheme.
    * TVD-style limiter to suppress spurious oscillations near steep gradients.
    """
    result = np.zeros_like(a)

    pos_mask = (a > 0) & (b > 0) & (c > 0)
    neg_mask = (a < 0) & (b < 0) & (c < 0)

    result[pos_mask] = np.minimum(np.minimum(a[pos_mask], b[pos_mask]), c[pos_mask])
    result[neg_mask] = np.maximum(np.maximum(a[neg_mask], b[neg_mask]), c[neg_mask])

    return result


def minmod_inplace(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    out: np.ndarray,
) -> np.ndarray:
    """
    In-place three-argument minmod limiter.

    This variant writes directly into ``out`` and avoids allocating an
    intermediate result array at each call.
    """
    out.fill(0.0)
    pos_mask = (a > 0) & (b > 0) & (c > 0)
    neg_mask = (a < 0) & (b < 0) & (c < 0)
    out[pos_mask] = np.minimum(np.minimum(a[pos_mask], b[pos_mask]), c[pos_mask])
    out[neg_mask] = np.maximum(np.maximum(a[neg_mask], b[neg_mask]), c[neg_mask])
    return out


def create_linear_rk4_workspace(Nx: int, dtype=np.float32) -> dict[str, np.ndarray]:
    """
    Allocate reusable arrays for the linear KT+RK4 hot path.

    The returned workspace is intended for repeated calls to
    ``rk4_step_linear_reuse`` with fixed grid size ``Nx``.
    """
    if Nx < 3:
        raise ValueError("Nx must be >= 3.")
    n_face = Nx - 2
    return {
        "slope": np.zeros(Nx, dtype=dtype),
        "dn_bwd": np.empty(n_face, dtype=dtype),
        "dn_central": np.empty(n_face, dtype=dtype),
        "dn_fwd": np.empty(n_face, dtype=dtype),
        "limiter_a": np.empty(n_face, dtype=dtype),
        "limiter_c": np.empty(n_face, dtype=dtype),
        "limiter_out": np.empty(n_face, dtype=dtype),
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


# ============================================================
# Kurganov-Tadmor flux update with diffusion and sources
# ============================================================


def kt_flux_update(
    n: np.ndarray,
    f: Callable[[np.ndarray, np.ndarray], np.ndarray],
    df: Callable[[np.ndarray, np.ndarray], np.ndarray],
    E: np.ndarray,
    D: float | np.ndarray,
    S: np.ndarray,
    dx: float,
    kt_limiter_theta: float = 1.1,
) -> np.ndarray:
    """
    Compute finite-volume RHS for scalar transport with drift, diffusion, source.

        dn/dt = - (H_{j+1/2} - H_{j-1/2}) / dx
                - (F_d{j+1/2} - F_d{j-1/2}) / dx
                + S_j,

    using the Kurganov-Tadmor (KT) central-upwind flux for drift
    and an explicit diffusion flux.

    Parameters
    ----------
    n : np.ndarray
        Cell-centered number density, shape (Nx,).
    f : callable
        Drift/convection flux function:

            f(n_face, E_face) -> Gamma

        where Gamma is the particle flux at a face.
    df : callable
        Derivative of f with respect to n, i.e. df/dn, used to estimate
        local wave speeds |a_{j±1/2}|.
    E : np.ndarray
        Cell-centered electric field [V/m], shape (Nx,).
    D : float or np.ndarray
        Diffusion coefficient [m²/s], scalar or cell-centered array
        of shape (Nx,).
    S : np.ndarray
        Cell-centered source term (e.g. ionization, recombination),
        shape (Nx,).
    dx : float
        Grid spacing [m].
    kt_limiter_theta : float, optional
        KT slope-limiter parameter (theta >= 1). Larger values reduce
        limiting; default is 1.1.

    Returns
    -------
    rhs : np.ndarray
        Time derivative dn/dt at each cell center, shape (Nx,).

    Notes
    -----
    * Boundary cells j=0 and j=N-1 are left with rhs=0; enforce BCs externally.
    * Limiter strength is controlled by kt_limiter_theta.
    """
    rhs = np.zeros_like(n)

    # Limited slopes (cell-centered).
    # theta >= 1. Larger theta gives less limiting.
    theta = kt_limiter_theta
    dn_bwd = (n[1:-1] - n[:-2]) / dx
    dn_central = 0.5 * (n[2:] - n[:-2]) / dx
    dn_fwd = (n[2:] - n[1:-1]) / dx

    slope = np.zeros_like(n)
    slope[1:-1] = minmod(theta * dn_bwd, dn_central, theta * dn_fwd)

    # Reconstruct left/right states at faces.
    # face j+1/2 uses cells j (left) and j+1 (right)
    nL_p = n[1:-1] + 0.5 * dx * slope[1:-1]  # left state at j+1/2
    nR_p = n[2:] - 0.5 * dx * slope[2:]      # right state at j+1/2

    # face j-1/2 uses cells j-1 (left) and j (right)
    nL_m = n[:-2] + 0.5 * dx * slope[:-2]    # left state at j-1/2
    nR_m = n[1:-1] - 0.5 * dx * slope[1:-1]  # right state at j-1/2

    # E-field at faces (neighbor average).
    Ei_p = 0.5 * (E[1:-1] + E[2:])   # at j+1/2
    Ei_m = 0.5 * (E[1:-1] + E[:-2])  # at j-1/2

    # Local speeds a_{j+/-1/2} = max |df/dn| from left/right states.
    a_p = np.maximum(np.abs(df(nL_p, Ei_p)), np.abs(df(nR_p, Ei_p)))  # j+1/2
    a_m = np.maximum(np.abs(df(nL_m, Ei_m)), np.abs(df(nR_m, Ei_m)))  # j-1/2

    # KT convective fluxes at faces.
    H_p = 0.5 * (f(nL_p, Ei_p) + f(nR_p, Ei_p)) - 0.5 * a_p * (nR_p - nL_p)
    H_m = 0.5 * (f(nL_m, Ei_m) + f(nR_m, Ei_m)) - 0.5 * a_m * (nR_m - nL_m)

    # Diffusion fluxes at faces: F_d = -D_face * (dn/dx)_face.
    grad_p = (n[2:] - n[1:-1]) / dx
    grad_m = (n[1:-1] - n[:-2]) / dx

    if np.isscalar(D):
        D_p = D
        D_m = D
    else:
        # Arithmetic average to face centers.
        D_p = 0.5 * (D[1:-1] + D[2:])
        D_m = 0.5 * (D[:-2] + D[1:-1])

    Fd_p = -D_p * grad_p
    Fd_m = -D_m * grad_m

    # Finite-volume update on interior cells.
    rhs[1:-1] = -(H_p - H_m) / dx - (Fd_p - Fd_m) / dx + S[1:-1]

    # Boundaries: rhs[0], rhs[-1] remain zero; BCs enforced elsewhere.
    return rhs


def rk4_step(
    n: np.ndarray,
    f: Callable[[np.ndarray, np.ndarray], np.ndarray],
    df: Callable[[np.ndarray, np.ndarray], np.ndarray],
    E: np.ndarray,
    D: float | np.ndarray,
    S: np.ndarray,
    dx: float,
    dt: float,
    kt_limiter_theta: float = 1.1,
) -> np.ndarray:
    """
    Advance density `n` by one time step with RK4 using KT+diffusion+source RHS.

    Parameters
    ----------
    n : np.ndarray
        Current cell-centered density, shape (Nx,).
    f, df : callable
        Flux and its derivative with respect to n (see `kt_flux_update`).
    E : np.ndarray
        Electric field [V/m], shape (Nx,).
    D : float or np.ndarray
        Diffusion coefficient [m²/s], scalar or array of shape (Nx,).
    S : np.ndarray
        Source term, shape (Nx,).
    dx : float
        Grid spacing [m].
    dt : float
        Time step [s].
    kt_limiter_theta : float, optional
        KT slope-limiter parameter passed to `kt_flux_update`.

    Returns
    -------
    n_new : np.ndarray
        Updated density after one RK4 step, shape (Nx,). Negative
        values are projected to zero via:

            n_new = max(n_new, 0).

    Notes
    -----
    * Boundary conditions are not applied inside this routine.
    """
    k1 = kt_flux_update(n,               f, df, E, D, S, dx, kt_limiter_theta)
    k2 = kt_flux_update(n + 0.5 * dt * k1, f, df, E, D, S, dx, kt_limiter_theta)
    k3 = kt_flux_update(n + 0.5 * dt * k2, f, df, E, D, S, dx, kt_limiter_theta)
    k4 = kt_flux_update(n +       dt * k3, f, df, E, D, S, dx, kt_limiter_theta)

    n_new = n + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # Enforce non-negativity after combining RK stages.
    np.maximum(n_new, 0.0, out=n_new)
    return n_new


def kt_flux_update_linear_reuse(
    n: np.ndarray,
    u: np.ndarray,
    D: float | np.ndarray,
    S: np.ndarray,
    dx: float,
    kt_limiter_theta: float,
    adv_coeff: float,
    rhs_out: np.ndarray,
    ws: dict[str, np.ndarray],
    boundary_flux_left: float = 0.0,
    boundary_flux_right: float = 0.0,
    replace_boundary_flux_left: bool = False,
    replace_boundary_flux_right: bool = False,
) -> np.ndarray:
    """
    Low-allocation KT RHS for linear fluxes of the form:

        f(n, u) = adv_coeff * n * u

    Parameters mirror ``kt_flux_update`` for the linear case but write the
    result directly to ``rhs_out`` using reusable workspace arrays.
    """
    rhs_out.fill(0.0)
    slope = ws["slope"]
    slope.fill(0.0)

    dn_bwd = ws["dn_bwd"]
    dn_central = ws["dn_central"]
    dn_fwd = ws["dn_fwd"]

    np.subtract(n[1:-1], n[:-2], out=dn_bwd)
    dn_bwd /= dx
    np.subtract(n[2:], n[:-2], out=dn_central)
    dn_central *= (0.5 / dx)
    np.subtract(n[2:], n[1:-1], out=dn_fwd)
    dn_fwd /= dx

    limiter_a = ws["limiter_a"]
    limiter_c = ws["limiter_c"]
    np.multiply(dn_bwd, kt_limiter_theta, out=limiter_a)
    np.multiply(dn_fwd, kt_limiter_theta, out=limiter_c)
    minmod_inplace(
        limiter_a,
        dn_central,
        limiter_c,
        ws["limiter_out"],
    )
    slope[1:-1] = ws["limiter_out"]

    nL_p = ws["nL_p"]
    nR_p = ws["nR_p"]
    nL_m = ws["nL_m"]
    nR_m = ws["nR_m"]

    np.multiply(slope[1:-1], 0.5 * dx, out=nL_p)
    nL_p += n[1:-1]
    np.multiply(slope[2:], 0.5 * dx, out=nR_p)
    nR_p *= -1.0
    nR_p += n[2:]

    np.multiply(slope[:-2], 0.5 * dx, out=nL_m)
    nL_m += n[:-2]
    np.multiply(slope[1:-1], 0.5 * dx, out=nR_m)
    nR_m *= -1.0
    nR_m += n[1:-1]

    u_face_p = ws["u_face_p"]
    u_face_m = ws["u_face_m"]
    np.add(u[1:-1], u[2:], out=u_face_p)
    u_face_p *= 0.5
    np.add(u[1:-1], u[:-2], out=u_face_m)
    u_face_m *= 0.5

    a_p = ws["a_p"]
    a_m = ws["a_m"]
    np.multiply(u_face_p, adv_coeff, out=a_p)
    np.abs(a_p, out=a_p)
    np.multiply(u_face_m, adv_coeff, out=a_m)
    np.abs(a_m, out=a_m)

    H_p = ws["H_p"]
    H_m = ws["H_m"]
    np.add(nL_p, nR_p, out=H_p)
    np.multiply(H_p, 0.5 * adv_coeff, out=H_p)
    H_p *= u_face_p
    np.subtract(nR_p, nL_p, out=ws["grad_p"])
    ws["grad_p"] *= 0.5
    ws["grad_p"] *= a_p
    H_p -= ws["grad_p"]

    np.add(nL_m, nR_m, out=H_m)
    np.multiply(H_m, 0.5 * adv_coeff, out=H_m)
    H_m *= u_face_m
    np.subtract(nR_m, nL_m, out=ws["grad_m"])
    ws["grad_m"] *= 0.5
    ws["grad_m"] *= a_m
    H_m -= ws["grad_m"]

    grad_p = ws["grad_p"]
    grad_m = ws["grad_m"]
    np.subtract(n[2:], n[1:-1], out=grad_p)
    grad_p /= dx
    np.subtract(n[1:-1], n[:-2], out=grad_m)
    grad_m /= dx

    D_p = ws["D_p"]
    D_m = ws["D_m"]
    if np.isscalar(D):
        D_p.fill(float(D))
        D_m.fill(float(D))
    else:
        np.add(D[1:-1], D[2:], out=D_p)
        D_p *= 0.5
        np.add(D[:-2], D[1:-1], out=D_m)
        D_m *= 0.5

    Fd_p = ws["Fd_p"]
    Fd_m = ws["Fd_m"]
    np.multiply(D_p, grad_p, out=Fd_p)
    Fd_p *= -1.0
    np.multiply(D_m, grad_m, out=Fd_m)
    Fd_m *= -1.0

    np.subtract(H_p, H_m, out=ws["accum"][1:-1])
    ws["accum"][1:-1] *= (-1.0 / dx)
    np.subtract(Fd_p, Fd_m, out=rhs_out[1:-1])
    rhs_out[1:-1] *= (-1.0 / dx)
    rhs_out[1:-1] += ws["accum"][1:-1]
    rhs_out[1:-1] += S[1:-1]

    # Signed boundary-face particle fluxes. Positive flux is in +x.
    # By default these are additive corrections. When replace_* is enabled,
    # replace the KT+diffusion wall face with outflow-only drift plus the
    # supplied signed surface flux. Interior KT faces are untouched.
    if replace_boundary_flux_left:
        old_left = float(H_m[0] + Fd_m[0])
        drift_left = float(adv_coeff * n[1] * u[1])
        outflow_left = min(drift_left, 0.0)
        new_left = outflow_left + float(boundary_flux_left)
        rhs_out[1] += (new_left - old_left) / dx
    elif boundary_flux_left != 0.0:
        rhs_out[1] += float(boundary_flux_left) / dx

    if replace_boundary_flux_right:
        old_right = float(H_p[-1] + Fd_p[-1])
        drift_right = float(adv_coeff * n[-2] * u[-2])
        outflow_right = max(drift_right, 0.0)
        new_right = outflow_right + float(boundary_flux_right)
        rhs_out[-2] += (old_right - new_right) / dx
    elif boundary_flux_right != 0.0:
        rhs_out[-2] -= float(boundary_flux_right) / dx

    return rhs_out


def rk4_step_linear_reuse(
    n: np.ndarray,
    u: np.ndarray,
    D: float | np.ndarray,
    S: np.ndarray,
    dx: float,
    dt: float,
    kt_limiter_theta: float,
    adv_coeff: float,
    ws: dict[str, np.ndarray],
    n_out: np.ndarray,
    boundary_flux_left: float = 0.0,
    boundary_flux_right: float = 0.0,
    replace_boundary_flux_left: bool = False,
    replace_boundary_flux_right: bool = False,
) -> np.ndarray:
    """
    Low-allocation RK4 update for linear fluxes using reusable workspace.

    This is the optimized path for:
    - ions:      adv_coeff = +1
    - electrons: adv_coeff = -1
    """
    k1 = ws["k1"]
    k2 = ws["k2"]
    k3 = ws["k3"]
    k4 = ws["k4"]
    n_tmp = ws["n_tmp"]
    accum = ws["accum"]

    kt_flux_update_linear_reuse(
        n=n,
        u=u,
        D=D,
        S=S,
        dx=dx,
        kt_limiter_theta=kt_limiter_theta,
        adv_coeff=adv_coeff,
        rhs_out=k1,
        ws=ws,
        boundary_flux_left=boundary_flux_left,
        boundary_flux_right=boundary_flux_right,
        replace_boundary_flux_left=replace_boundary_flux_left,
        replace_boundary_flux_right=replace_boundary_flux_right,
    )

    np.multiply(k1, 0.5 * dt, out=n_tmp)
    n_tmp += n
    kt_flux_update_linear_reuse(
        n=n_tmp,
        u=u,
        D=D,
        S=S,
        dx=dx,
        kt_limiter_theta=kt_limiter_theta,
        adv_coeff=adv_coeff,
        rhs_out=k2,
        ws=ws,
        boundary_flux_left=boundary_flux_left,
        boundary_flux_right=boundary_flux_right,
        replace_boundary_flux_left=replace_boundary_flux_left,
        replace_boundary_flux_right=replace_boundary_flux_right,
    )

    np.multiply(k2, 0.5 * dt, out=n_tmp)
    n_tmp += n
    kt_flux_update_linear_reuse(
        n=n_tmp,
        u=u,
        D=D,
        S=S,
        dx=dx,
        kt_limiter_theta=kt_limiter_theta,
        adv_coeff=adv_coeff,
        rhs_out=k3,
        ws=ws,
        boundary_flux_left=boundary_flux_left,
        boundary_flux_right=boundary_flux_right,
        replace_boundary_flux_left=replace_boundary_flux_left,
        replace_boundary_flux_right=replace_boundary_flux_right,
    )

    np.multiply(k3, dt, out=n_tmp)
    n_tmp += n
    kt_flux_update_linear_reuse(
        n=n_tmp,
        u=u,
        D=D,
        S=S,
        dx=dx,
        kt_limiter_theta=kt_limiter_theta,
        adv_coeff=adv_coeff,
        rhs_out=k4,
        ws=ws,
        boundary_flux_left=boundary_flux_left,
        boundary_flux_right=boundary_flux_right,
        replace_boundary_flux_left=replace_boundary_flux_left,
        replace_boundary_flux_right=replace_boundary_flux_right,
    )

    # accum = k1 + 2*k2 + 2*k3 + k4
    np.multiply(k2, 2.0, out=accum)
    accum += k1
    accum += k4
    n_out[:] = n
    n_out += (dt / 6.0) * accum
    n_out += (dt / 3.0) * k3

    np.maximum(n_out, 0.0, out=n_out)
    return n_out


def rk4_step_rhs(
    n: np.ndarray,
    rhs_func: Callable[[np.ndarray], np.ndarray],
    dt: float,
) -> np.ndarray:
    """
    Advance state ``n`` by one RK4 step for a user-supplied RHS callback.

    This helper is intended for nonlocal closures where the RHS cannot be
    expressed through ``kt_flux_update`` with local coefficients alone.

    Parameters
    ----------
    n : np.ndarray
        Current state vector, shape (Nx,).
    rhs_func : callable
        Function returning ``dn/dt`` for a supplied state:
            rhs = rhs_func(n_state)
    dt : float
        Time step [s].

    Returns
    -------
    np.ndarray
        Updated state after one RK4 step with non-negativity projection.
    """
    k1 = rhs_func(n)
    k2 = rhs_func(n + 0.5 * dt * k1)
    k3 = rhs_func(n + 0.5 * dt * k2)
    k4 = rhs_func(n + dt * k3)

    n_new = n + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    np.maximum(n_new, 0.0, out=n_new)
    return n_new


# ============================================================
# Boundary conditions for n_e and n_i
# ============================================================


def set_boundary_condition_implicit(
    ne_next: np.ndarray,
    ni_next: np.ndarray,
    ne_curr: np.ndarray,
    ni_curr: np.ndarray,
    phi_curr: np.ndarray,
    gamma: float,
    anode_electron_induced_yield: float,
    T_e_eV: float,
    mu_i_anode: float,
    mu_i_cathode: float,
    mu_e_anode: float,
    mu_e_cathode: float,
    dx: float,
    dt: float,
    Gamma_ext_anode: float = 0.0,
    Gamma_ext_cathode: float = 0.0,
    use_vaughan_sey: bool = False,
    vaughan_Emax0_eV: float = 400.0,
    vaughan_dmax0: float = 3.2,
    vaughan_ks: float = 1.0,
    vaughan_z: float = 0.0,
    vaughan_E0: float = 0.0,
    anode_ion_boundary: str = "zero_density",
    anode_electron_boundary: str = "implicit_drift_closure",
    cathode_ion_boundary: str = "implicit_drift_closure",
    cathode_electron_boundary: str = "electron_emission",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Mode-driven boundary-condition update for anode/cathode and both species.

    Geometry / labels:
        x = 0   → anode
        x = L   → cathode

    Supported per-boundary modes:
        - "zero_density"
        - "electron_emission" (electrons only)
        - "implicit_drift_closure" (implemented for anode electrons and cathode ions)

    Parameters
    ----------
    ne_next, ni_next : np.ndarray
        Electron/ion densities at the next time step, shape (Nx,).
        These arrays are modified in-place at the boundaries.
    ne_curr, ni_curr : np.ndarray
        Electron/ion densities at the current time step, shape (Nx,).
    phi_curr : np.ndarray
        Electric potential at the current time step, shape (Nx,).
    gamma : float
        Cathode ion-induced secondary electron yield coefficient.
    anode_electron_induced_yield : float
        Anode electron-induced secondary electron emission yield.
    T_e_eV : float
        Electron temperature proxy in eV used for anode impact-energy proxy.
    mu_i_anode, mu_i_cathode : float
        Local ion mobilities at the anode and cathode boundaries [m²/(V·s)].
    mu_e_anode, mu_e_cathode : float
        Local electron mobilities at the anode and cathode boundaries
        [m²/(V·s)].
    dx : float
        Spatial step [m].
    dt : float
        Time step [s].
    Gamma_ext_anode, Gamma_ext_cathode : float, optional
        Externally driven electron-emission number flux magnitudes [m^-2 s^-1].
        These arguments are retained for API compatibility; conservative
        surface-emission flux coupling is performed in ``paschen_1d.py`` inside
        the electron continuity update before this boundary closure is called.
    use_vaughan_sey : bool, optional
        Enable Vaughan-model anode electron-induced SEY. Cathode SEE remains
        the constant-gamma model.
    vaughan_Emax0_eV, vaughan_dmax0, vaughan_ks, vaughan_z, vaughan_E0 : float
        Vaughan-model parameters used when use_vaughan_sey=True for the
        anode electron-emission boundary.

    Returns
    -------
    ne_next, ni_next : np.ndarray
        Updated next-step densities with BCs enforced and clipped to be
        non-negative.
    """
    # -----------------------
    # Ion boundary conditions
    # -----------------------
    if anode_ion_boundary == "zero_density":
        ni_next[0] = boundary_zero_density()
    elif anode_ion_boundary == "implicit_drift_closure":
        raise ValueError(
            "Unsupported boundary combination: anode_ion_boundary='implicit_drift_closure'. "
            "Current implementation supports implicit ion drift closure only at the cathode."
        )
    elif anode_ion_boundary == "electron_emission":
        raise ValueError("Invalid ion boundary mode: 'electron_emission'.")
    else:
        raise ValueError(f"Unknown anode_ion_boundary mode: {anode_ion_boundary}")

    if cathode_ion_boundary == "implicit_drift_closure":
        ni_next[-1] = boundary_cathode_ion_implicit_drift_density(
            ni_curr_right=float(ni_curr[-1]),
            ni_next_inner=float(ni_next[-2]),
            phi_right=float(phi_curr[-1]),
            phi_inner=float(phi_curr[-2]),
            phi_inner2=float(phi_curr[-3]),
            gamma=gamma,
            mu_i=mu_i_cathode,
            dx=dx,
            dt=dt,
        )
    elif cathode_ion_boundary == "zero_density":
        ni_next[-1] = boundary_zero_density()
    elif cathode_ion_boundary == "electron_emission":
        raise ValueError("Invalid ion boundary mode: 'electron_emission'.")
    else:
        raise ValueError(f"Unknown cathode_ion_boundary mode: {cathode_ion_boundary}")

    # ----------------------------
    # Electron boundary conditions
    # ----------------------------
    if anode_electron_boundary == "implicit_drift_closure":
        ne_next[0] = boundary_anode_electron_implicit_drift_density(
            ne_curr_left=float(ne_curr[0]),
            ne_next_inner=float(ne_next[1]),
            phi_left=float(phi_curr[0]),
            phi_inner=float(phi_curr[1]),
            phi_inner2=float(phi_curr[2]),
            mu_e=mu_e_anode,
            dx=dx,
            dt=dt,
        )
    elif anode_electron_boundary == "zero_density":
        ne_next[0] = boundary_zero_density()
    elif anode_electron_boundary == "electron_emission":
        # Surface-emitted electrons enter through a boundary-face flux in
        # paschen_1d.py. Keep the wall node absorbing so this boundary mode
        # cannot act as a density reservoir.
        ne_next[0] = boundary_zero_density()
    else:
        raise ValueError(
            f"Unknown anode_electron_boundary mode: {anode_electron_boundary}"
        )

    if cathode_electron_boundary == "electron_emission":
        # Surface-emitted electrons enter through a boundary-face flux in
        # paschen_1d.py. Keep the wall node absorbing so this boundary mode
        # cannot act as a density reservoir.
        ne_next[-1] = boundary_zero_density()
    elif cathode_electron_boundary == "zero_density":
        ne_next[-1] = boundary_zero_density()
    elif cathode_electron_boundary == "implicit_drift_closure":
        raise ValueError(
            "Unsupported boundary combination: cathode_electron_boundary='implicit_drift_closure'. "
            "Current implementation supports implicit electron drift closure only at the anode."
        )
    else:
        raise ValueError(
            f"Unknown cathode_electron_boundary mode: {cathode_electron_boundary}"
        )

    return np.clip(ne_next, 0.0, None), np.clip(ni_next, 0.0, None)


# ============================================================
# CFL utilities / diagnostics
# ============================================================


def compute_drift_cfl(
    mu_e: float | np.ndarray,
    mu_i: float | np.ndarray,
    E: np.ndarray,
    dt: float,
    dx: float,
) -> float:
    """
    Return drift-based CFL number without side effects.

    This is the pure utility counterpart of ``CFL_test`` and is intended for
    control logic (e.g., adaptive substepping) where printing is undesirable.

    CFL definition:

        CFL = a_max * dt / dx

    with

        a_max = max_x max(|mu_e E|, |mu_i E|).
    """
    a_e = np.abs(mu_e * E)
    a_i = np.abs(mu_i * E)
    a_max = float(np.maximum(np.max(a_e), np.max(a_i)))
    return float(a_max * dt / dx)


def compute_diffusion_cfl(
    D_e: float | np.ndarray,
    D_i: float | np.ndarray,
    dt: float,
    dx: float,
) -> float:
    """
    Return the explicit-diffusion stability number without side effects.

    The KT/RK4 density update treats diffusion explicitly. A useful parabolic
    stability diagnostic is therefore

        C_D = D_max * dt / dx^2,

    where

        D_max = max_x max(|D_e|, |D_i|).

    For simple explicit central-difference diffusion, values near or below
    0.5 are typically expected. PASCHEN-1D exposes the exact target through
    ``cfg.numerics.target_diffusion_cfl_substep`` so users can choose a more
    conservative value when needed.
    """
    D_e_arr = np.asarray(D_e)
    D_i_arr = np.asarray(D_i)
    D_e_abs = np.abs(D_e_arr)
    D_i_abs = np.abs(D_i_arr)
    D_max = float(np.maximum(np.max(D_e_abs), np.max(D_i_abs)))
    return float(D_max * dt / (dx * dx))


def CFL_test(
    mu_e: float | np.ndarray,
    mu_i: float | np.ndarray,
    E_next: np.ndarray,
    dt: float,
    dx: float,
    time: np.ndarray,
    n_idx: int,
) -> float:
    """
    Compute drift-based CFL number and print a warning if CFL > 1.

    We define:

        CFL = a_max * dt / dx,

    where

        a_max = max_x max(|mu_e E|, |mu_i E|).

    Parameters
    ----------
    mu_e : float or np.ndarray
        Electron mobility [m²/(V·s)]. May be a scalar or a spatial profile
        with shape matching `E_next`.
    mu_i : float or np.ndarray
        Ion mobility [m²/(V·s)]. May be a scalar or a spatial profile
        with shape matching `E_next`.
    E_next : np.ndarray
        Electric field at the next time step [V/m], shape (Nx,).
    dt : float
        Time step [s].
    dx : float
        Spatial step [m].
    time : np.ndarray
        Time array, shape (Nt,).
    n_idx : int
        Current time-step index (n). The function will reference
        time[n_idx + 1] when printing warnings, if in range.

    Returns
    -------
    C : float
        CFL number for the current step.

    Notes
    -----
    * This diagnostic uses only the drift term; diffusion/source impose
      separate constraints.
    * If CFL > 1, you are in a regime where an explicit drift update
      would be unstable; however, the KT + RK4 combo may still be
      more tolerant than a simple upwind scheme.
    """
    C = compute_drift_cfl(
        mu_e=mu_e,
        mu_i=mu_i,
        E=E_next,
        dt=dt,
        dx=dx,
    )

    # Optional diagnostic print (with bounds guard).
    if C > 1.0:
        if (n_idx + 1) < len(time):
            t_str = f"{time[n_idx + 1]:.4e}"
        else:
            t_str = "unknown"
        print(f"CFL condition violated at time {t_str}, CFL = {C:.3f}")

    return C
