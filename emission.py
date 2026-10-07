"""
emission.py

Emission models for the PASCHEN-1D 1D drift-diffusion-Poisson solver.

This module centralizes externally driven electrode emission behavior behind a single,
high-level interface:

    J_emit = EmissionModel.current_density(t, V_gap, dt, E_surface, electrode)

which returns an emission current density [A/m²] at a given simulation time.

Supported external emission components (enabled via SimulationConfig booleans):

    - "none"
        → No emission (handled by returning None from build_emission_model).

    - "constant_J"
        → Time-windowed constant current density:
            J_emit = emission_J_const [A/m²]
          for emission_t_start <= t <= emission_t_end (or open-ended after
          emission_t_start if emission_t_end <= emission_t_start).
          With a positive timestep, returns the average over [t, t + dt].

    - "fn"
        → Fowler–Nordheim field emission:
            J_FN(E, W) = (A_FN * E² / W) * exp(-B_FN W^(3/2) / |E|)
          where:
            E : effective local surface field [V/m]
            W : work function [eV]
            A_FN, B_FN are standard FN constants.
          Here, E is taken from the local surface field passed by the solver.

    - "mg"
        → cold Murphy-Good field emission with Schottky-Nordheim (SN)
          image-force barrier correction:
            J_MG = t(f)^(-2) * (A_FN * E² / W) * exp(-v(f) * B_FN W^(3/2) / |E|)
          where v(f), t(f) are SN correction functions of scaled field f.

    - "rd"
        → Richardson–Dushman thermionic emission:
            J_RD(T, W) = A_R T² exp(-W / (k_B T))
          where:
            T : emitter temperature [K]
            W : work function [eV]
            A_R : Richardson constant [A/(m²·K²)].

    - "quantum_pulse"
        → Quantum photoemission model based on a multi-photon / field-emission
          formulation. We:
            1) Build a fine time grid in picoseconds around the laser center.
            2) Compute J(t) on that grid using the quantum model.
            3) Area-average the local laser-spot current density over the
               modeled 1D cross section.
            4) Wrap it into a runtime emitter via piecewise-constant binning.

Design goals:
-------------
- Keep the PASCHEN-1D main driver agnostic to emission details:
    - It only calls EmissionModel.current_density(t, V_gap, dt, E_surface, electrode).
- Allow easily switching between different emission physics by changing
  SimulationConfig only, without touching the numerics or circuit code.
- Amortize expensive quantum integrals by precomputing them once on a
  dedicated emission time grid.

Current coupling note:
----------------------
    The solver can inject external emission on anode and/or cathode
    through the boundary emission closure, controlled by:
      - enable_anode_external_emission
      - enable_cathode_external_emission
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from numpy.lib.scimath import sqrt as csqrt  # complex-safe sqrt for Airy-branch
from scipy import special

from physical_constants import e, eps0, c, kB, m_e, hbar
from config import SimulationConfig

from tqdm.auto import tqdm


# ============================================================
# High-level emission interface
# ============================================================


@dataclass
class EmissionModel:
    """
    High-level wrapper for emission: J_emit(t, V_gap, dt, E_surface, electrode).

    This class stores per-electrode emitter functions with a fixed signature:

        emitter_func(t, V_gap, dt, E_surface) -> J [A/m²]

    Attributes
    ----------
    cfg : SimulationConfig
        Full simulation configuration used to construct this model.
        (Mostly kept for provenance / future configuration-aware models.)
    emitter_func_anode, emitter_func_cathode : callable or None
        Per-electrode functions returning emission current density [A/m²].
        If a function is None, that electrode contributes zero external emission.
    """
    cfg: SimulationConfig
    emitter_func_anode: Optional[Callable[[float, float, float, Optional[float]], float]] = None
    emitter_func_cathode: Optional[Callable[[float, float, float, Optional[float]], float]] = None

    def current_density(
        self,
        t: float,
        V_gap: float,
        dt: float,
        E_surface: float | None = None,
        electrode: str = "cathode",
    ) -> float:
        """
        Return per-electrode external emission current density J_emit [A/m²].

        Parameters
        ----------
        t : float
            Global simulation time [s].
        V_gap : float
            Instantaneous plasma gap voltage [V].
            - Used by field-dependent models (e.g. FN).
            - Ignored by models that depend only on time or temperature.
        dt : float
            Simulation time step [s].
            - Used by constant_J and quantum pulse models to average J over
              [t, t + dt].
            - Ignored by purely instantaneous models (FN, MG, RD).
        E_surface : float or None, optional
            Local electric field at the emitting surface [V/m].
            Used by FN emission in the current implementation.

        electrode : str, optional
            "anode" or "cathode".

        Returns
        -------
        J_emit : float
            Emission current density [A/m²] for the requested electrode.
            Returns 0 if the electrode emitter is not configured.
        """
        if electrode == "anode":
            emitter = self.emitter_func_anode
        elif electrode == "cathode":
            emitter = self.emitter_func_cathode
        else:
            raise ValueError(f"Unknown electrode: {electrode}")

        if emitter is None:
            return 0.0
        return float(emitter(t, V_gap, dt, E_surface))


# ============================================================
# 1. Helper: trapezoidal weights
# ============================================================


def _trapz_weights(x: np.ndarray) -> np.ndarray:
    """
    Compute trapezoidal-rule weights for a (possibly non-uniform) 1D grid.

    For a sorted grid x[0..N-1], we build weights w such that:

        ∫ f(x) dx ≈ Σ_j w[j] * f(x[j])

    Parameters
    ----------
    x : np.ndarray
        1D array of grid points, assumed sorted in ascending order.

    Returns
    -------
    w : np.ndarray
        Trapezoidal weights corresponding to x, same shape as x.
    """
    x = np.asarray(x, dtype=float)

    w = np.empty_like(x, dtype=float)
    # interior points: average of neighboring spacings
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    # boundary points: half of adjacent spacing
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])

    return w


# ============================================================
# 1a. Simple constant-J emitter
# ============================================================


def make_constant_J_emitter(
    J_const: float,
    t_start: float = 0.0,
    t_end: Optional[float] = None,
) -> Callable[[float, float, Optional[float], Optional[float]], float]:
    """
    Build a simple emitter that returns a constant J_const [A/m²]
    over a user-specified time window.

    Instantaneous behavior when dt_run is absent or nonpositive:
        J_emit(t) = J_const     for t_start <= t <= t_end
                  = 0           otherwise

    If t_end is None:
        J_emit(t) = J_const     for t >= t_start

    Parameters
    ----------
    J_const : float
        Constant emission current density [A/m²].
    t_start : float, optional
        Time [s] at which emission turns on. Default is 0.
    t_end : float or None, optional
        Time [s] at which emission turns off. If None, emission stays on
        indefinitely for t >= t_start.

    Returns
    -------
    emitter : callable
        emitter(t_run, V_gap=None, dt_run=None, E_surface=None) -> J [A/m²]

    Notes
    -----
    - Instantaneous sampling is inclusive at both ends:
          t_start <= t <= t_end.
    - For dt_run > 0, return J_const times the overlap duration of
      [t_run, t_run + dt_run] with the emission window, divided by dt_run.
      An endpoint alone has zero duration and contributes no charge.
    - `V_gap` and `E_surface` are accepted for interface consistency but
      are not used by this emitter.
    """
    def emitter(
        t_run: float,
        V_gap: float = 0.0,
        dt_run: Optional[float] = None,
        E_surface: float | None = None,
    ) -> float:
        if dt_run is not None and dt_run > 0.0:
            start = max(t_run, t_start)
            end = t_run + dt_run
            if t_end is not None:
                end = min(end, t_end)
            return J_const * (max(0.0, end - start) / dt_run)

        # Before the emission window: no emission
        if t_run < t_start:
            return 0.0
        # After t_end (if specified): no emission
        if (t_end is not None) and (t_run > t_end):
            return 0.0
        # Within the active window: constant current density
        return J_const

    return emitter


# ============================================================
# 1b. Fowler–Nordheim (FN) field emission helper
# ============================================================


def fowler_nordheim_J(
    E: float,
    work_function_eV: float,
) -> float:
    """
    Fowler–Nordheim emission current density J [A/m²] for a given local field.

    Parameters
    ----------
    E : float
        Local surface electric field [V/m].
        The magnitude |E| is used; the sign does not matter for FN magnitude.
    work_function_eV : float
        Material work function [eV].

    Returns
    -------
    J : float
        FN emission current density [A/m²].

    Notes
    -----
    Uses a standard FN-like parametrization:

        J = (A_FN * E^2 / W) * exp(-B_FN * W^(3/2) / |E|),

    where:
        - W is the work function in eV,
        - E is in V/m,
        - A_FN, B_FN are empirical constants.

    This is a simplified, textbook FN expression. It neglects image-force
    corrections, non-planar geometry, and more detailed band-structure
    effects. It is primarily intended as a qualitative field-emission model.
    It assumes work_function_eV > 0.
    """
    # Standard (approximate) FN constants:
    A_FN = 1.54e-6  # [A·eV·V^-2]
    B_FN = 6.83e9   # [eV^(-3/2)·V/m]

    E_abs = float(abs(E))
    if E_abs <= 0.0:
        # No field → no field emission
        return 0.0

    work_function = float(work_function_eV)
    if work_function <= 0.0:
        # Non-physical work function -> disable emission safely.
        return 0.0

    # Exponential factor can be extremely steep at low fields; guard against
    # underflow to avoid floating-point issues.
    exponent = -B_FN * (work_function ** 1.5) / E_abs
    if exponent < -700.0:  # exp(-700) ~ 5e-305 (effectively zero)
        return 0.0

    J = (A_FN * E_abs * E_abs / work_function) * np.exp(exponent)
    return float(J)


# ============================================================
# 1c. cold Murphy-Good (MG) field emission helper
# ============================================================


def _sn_v_f(f: float) -> float:
    """
    Schottky-Nordheim v(f) correction using Forbes' simple approximation.

    v(f) ~= 1 - f + (f/6) ln(f), valid for 0 < f <= 1.
    """
    f_eff = float(max(f, 1.0e-30))
    return 1.0 - f_eff + (f_eff / 6.0) * np.log(f_eff)


def _sn_t_f(f: float) -> float:
    """
    Schottky-Nordheim t(f) from t = v - (4/3) f dv/df.

    With v(f) ~= 1 - f + (f/6)ln(f), this gives:
        t(f) ~= 1 + f/9 - (f/18) ln(f).
    """
    f_eff = float(max(f, 1.0e-30))
    return 1.0 + (f_eff / 9.0) - (f_eff / 18.0) * np.log(f_eff)


def murphy_good_cold_J(
    E: float,
    work_function_eV: float,
    f_clip_min: float = 1.0e-9,
    f_clip_max: float = 0.99,
) -> float:
    """
    Cold Murphy-Good FE current density J [A/m²] with SN image-force correction.

    Parameters
    ----------
    E : float
        Local surface electric field [V/m].
    work_function_eV : float
        Material work function [eV].
    f_clip_min : float, optional
        Lower clamp for scaled field f used in v(f), t(f).
    f_clip_max : float, optional
        Upper clamp for scaled field f, typically < 1 for SN barrier validity.

    Returns
    -------
    J : float
        Emission current density [A/m²].

    Notes
    -----
    Uses zero-temperature Murphy-Good form:
        J = t(f)^(-2) * (A_FN * E^2 / W) * exp(-v(f) * B_FN * W^(3/2) / |E|)

    with scaled field
        f = c_S^2 * E[V/nm] / W^2,
    where c_S^2 ~= 1.439965 (eV^2)/(V/nm).
    """
    A_FN = 1.541434e-6   # [A·eV·V^-2]
    B_FN = 6.830890e9    # [eV^(-3/2)·V/m]
    c_s_sq = 1.439965    # [eV^2/(V/nm)]

    E_abs = float(abs(E))
    if E_abs <= 0.0:
        return 0.0

    work_function = float(work_function_eV)
    if work_function <= 0.0:
        return 0.0

    f_raw = c_s_sq * (E_abs * 1.0e-9) / (work_function * work_function)
    f_lo = float(max(f_clip_min, 1.0e-30))
    f_hi = float(min(max(f_clip_max, f_lo), 0.999999))
    f_eff = float(np.clip(f_raw, f_lo, f_hi))

    v_f = _sn_v_f(f_eff)
    t_f = _sn_t_f(f_eff)
    if t_f <= 0.0:
        return 0.0

    exponent = -v_f * B_FN * (work_function ** 1.5) / E_abs
    if exponent < -700.0:
        return 0.0

    kernel = (A_FN * E_abs * E_abs / work_function) / (t_f * t_f)
    return float(kernel * np.exp(exponent))


# ============================================================
# 1d. Richardson–Dushman (RD) thermionic emission helper
# ============================================================


def richardson_dushman_J(
    T_K: float,
    work_function_eV: float,
    A_R: float = 1.2e6,
) -> float:
    """
    Richardson–Dushman thermionic emission current density J [A/m²].

    Parameters
    ----------
    T_K : float
        Cathode temperature [K].
    work_function_eV : float
        Work function [eV].
    A_R : float, optional
        Richardson constant [A/(m²·K²)].
        Defaults to 1.2e6, a typical value for many metals.

    Returns
    -------
    J : float
        Thermionic emission current density [A/m²].

    Notes
    -----
    Uses the standard RD formula:

        J = A_R * T^2 * exp(-W / (k_B T)),

    where:
        - T is in K,
        - W is the work function,
        - k_B is Boltzmann's constant.

    Here:
        - W is provided in eV and converted internally to J,
        - k_B is used in SI units [J/K].
    """
    if T_K <= 0.0:
        # Non-physical / zero temperature → no emission
        return 0.0

    work_function = float(work_function_eV)
    if work_function <= 0.0:
        # Non-physical work function -> disable emission safely.
        return 0.0

    A_R_eff = float(A_R)
    if A_R_eff < 0.0:
        # Non-physical Richardson constant -> disable emission safely.
        return 0.0

    # Convert work function from eV to joules
    work_function_J = work_function * e
    exponent = -work_function_J / (kB * T_K)

    # Again, guard against extreme underflow in the exponential
    if exponent < -700.0:
        return 0.0

    return float(A_R_eff * T_K * T_K * np.exp(exponent))


# ============================================================
# 2. Quantum photoemission model helpers
# ============================================================


def parameters(
    F11: float,
    F01: float,
    Ef1_eV: float,
    W1_eV: float,
    lambda_m: float,
):
    """
    Map dimensional fields + material properties into dimensionless
    parameters for the quantum emission formulas.

    Parameters
    ----------
    F11 : float
        AC/laser field amplitude [GV/m].
        (Caller is responsible for converting from V/m to GV/m.)
    F01 : float
        DC field amplitude [GV/m].
    Ef1_eV : float
        Fermi level [eV].
    W1_eV : float
        Work function [eV].
    lambda_m : float
        Laser wavelength [m].

    Returns
    -------
    F0 : float
        Dimensionless DC field.
    F1 : float
        Dimensionless AC field.
    Ef : float
        Dimensionless Fermi level.
    up : float
        Dimensionless ponderomotive energy.
    omega : float
        Dimensionless laser frequency.
    W : float
        Effective (Schottky-corrected) work function [J].

    Notes
    -----
    This follows the normalization used by the quantum photoemission helper:
    - Fields are converted to V/m for intermediate steps,
    - A Schottky lowering is applied to the DC work function,
    - A characteristic length λ₀ ~ sqrt(ħ² / (2 m_e W)) is introduced,
    - All energies are scaled by W to form dimensionless quantities.
    - The physically valid regime requires W > 0 after Schottky correction.
    """
    Ef1 = Ef1_eV   # [eV]
    W1 = W1_eV     # [eV]

    # Convert input fields from GV/m → V/m
    F02 = F01 * 1e9
    F12 = F11 * 1e9

    # Laser frequency [rad/s]
    lambd = lambda_m
    omega_l = 2.0 * np.pi * c / lambd

    # Classical ponderomotive (quiver) energy [J]
    up1 = e**2 * F12**2 / (4.0 * m_e * omega_l**2)

    # Schottky lowering of the work function due to DC field F02
    W = W1 * e - 2.0 * np.sqrt(e**3 * F02 / (16.0 * np.pi * eps0))
    # Alternative, without Schottky effect:
    # W = W1 * e

    # Characteristic length scale [m]
    lambd0 = np.sqrt(hbar**2 / (2.0 * m_e * W))

    # Dimensionless fields and energies
    F0 = F02 * e * lambd0 / W
    F1 = F12 * e * lambd0 / W
    Ef = Ef1 * e / W
    up = up1 / W
    omega = omega_l * hbar / W

    return F0, F1, Ef, up, omega, W


def emission_current_density(
    k: int,
    wt: np.ndarray,
    F0: float,
    F1: float,
    Ef: float,
    up: float,
    omega: float,
    W: float,
    epsilon_0_eV: float,
    T: float,
    eps_points: int = 100,
) -> complex:
    """
    Quantum photo-/field-emission current density J [A/m²] (complex).

    This routine evaluates the quantum photoemission formulation using the
    parameter set from `parameters(...)` and the PASCHEN-1D physical constants.

    Parameters
    ----------
    k : int
        Truncation index for photon sidebands (n = -k..k).
    wt : np.ndarray
        Phase grid over [0, 2π], shape (Nw,).
    F0, F1 : float
        Dimensionless DC and AC fields.
    Ef : float
        Dimensionless Fermi level.
    up : float
        Dimensionless ponderomotive energy.
    omega : float
        Dimensionless laser frequency.
    W : float
        Effective (Schottky-corrected) work function [J].
    epsilon_0_eV : float
        Upper energy limit in eV for the epsilon integration in the quantum
        photoemission normalization.
    T : float
        Emitter temperature [K].
    eps_points : int, optional
        Number of points in the epsilon energy grid.

    Returns
    -------
    J : complex
        Emission current density [A/m²]. Typically Re(J) is taken as physical.

    Notes
    -----
    - When F1 != 0 (AC + optional DC), the algorithm uses Airy functions and
      a Fourier expansion over the phase variable wt.
    - When F1 == 0 (pure static field), it reduces to a simpler static-field
      Airy function evaluation.
    - We use complex-safe sqrt (`csqrt`) to handle branch cuts correctly.
    """
    F0_13 = F0 ** (1.0 / 3.0)

    # Build epsilon grid in the quantum photoemission normalization.  The
    # original implementation used 100 right-endpoint bins; keep that convention but make
    # the bin width consistent with the requested epsilon resolution.
    eps_points = int(eps_points)
    if eps_points < 1:
        raise ValueError("eps_points must be at least 1")

    epsilon_eV = np.linspace(
        epsilon_0_eV / eps_points,
        epsilon_0_eV,
        eps_points,
        dtype=float,
    )
    epsilon_arr = epsilon_eV * e / W  # convert to dimensionless via division by W
    dE_J = (epsilon_0_eV / eps_points) * e

    if F1 != 0.0:
        # =====================================================
        # AC (and possibly DC) case
        # =====================================================
        wt = np.asarray(wt, dtype=float)
        two_pi = 2.0 * np.pi

        sin_wt = np.sin(wt)
        cos_wt = np.cos(wt)

        # Photon sideband indices n in [-k, k]
        n_vals = np.arange(-k, k + 1)
        Nn = n_vals.size

        # Fourier index m in [-2k, 2k]
        mlist = np.arange(-2 * k, 2 * k + 1)
        E_mat = np.exp(-1j * np.outer(wt, mlist))   # (Nw, 4k + 1)

        # Map (n2, n1) → index for m = n1 - n2, used to build the system matrix
        diff_idx = (n_vals[None, :] - n_vals[:, None]) + 2 * k   # (Nn, Nn)
        I_cols = np.broadcast_to(np.arange(Nn), (Nn, Nn))

        # Trapezoidal weights on wt, normalized by 2π
        w = _trapz_weights(wt) / two_pi

        # q(wt) phase factor from AC + DC fields
        qwt_phase = (
            -2j * F0 * F1 * sin_wt / (omega ** 3)
            + 1j * (F1 ** 2) * np.sin(2.0 * wt) / (4.0 * omega ** 3)
        )
        qwt = np.exp(qwt_phase)

        J = 0.0 + 0.0j

        for eps in epsilon_arr:
            # en(n) = eps + n*omega - Ef - up - 1
            en_n1 = eps + n_vals * omega - Ef - up - 1.0  # shape (Nn,)

            if F0 == 0.0:
                # ---------------------------------------------
                # Pure AC case (no DC)
                # ---------------------------------------------
                PN = qwt[None, :] * np.exp(
                    2j * F1 * csqrt(en_n1[:, None]) * cos_wt[None, :] / (omega ** 2)
                )
                QN = PN * (csqrt(en_n1[:, None]) + F1 * sin_wt[None, :] / omega)

                # Fourier coefficients over wt:
                c_mat = (PN * w[None, :]) @ E_mat   # (Nn, 4k + 1)
                d_mat = (QN * w[None, :]) @ E_mat   # (Nn, 4k + 1)

                Cdiff = c_mat[I_cols, diff_idx]   # (Nn, Nn)
                Ddiff = d_mat[I_cols, diff_idx]   # (Nn, Nn)

                # g(n) factor
                g = csqrt(eps + omega * n_vals)   # (Nn,)
                pnlznl = g[:, None] * Cdiff + Ddiff  # system matrix

                # Right-hand side delta_n,k term (only central sideband)
                delta = np.zeros(Nn, dtype=np.complex128)
                delta[k] = 2.0 * csqrt(eps)

                # Solve linear system; fall back to least squares if needed
                try:
                    tn = np.linalg.solve(pnlznl, delta)
                except np.linalg.LinAlgError:
                    tn, *_ = np.linalg.lstsq(pnlznl, delta, rcond=None)

                wn = (1j * csqrt(en_n1) / csqrt(eps) * np.abs(tn) ** 2).imag

            else:
                # ---------------------------------------------
                # DC + AC case: Airy-based formulation
                # ---------------------------------------------
                alpha = -(
                    (en_n1[:, None] / F0)
                    + (2.0 * F1 * cos_wt[None, :]) / (omega ** 2)
                ) * F0_13   # (Nn, Nw)

                # Airy functions
                ai, aip, bi, bip = special.airy(alpha)
                r = ai - 1j * bi
                s = 1j * aip + bip

                # Build P and Q matrices
                PN = qwt[None, :] * r
                QN = qwt[None, :] * (
                    (F1 * sin_wt[None, :] * r) / omega + F0_13 * s
                )

                # Fourier coefficients
                c_mat = (PN * w[None, :]) @ E_mat
                d_mat = (QN * w[None, :]) @ E_mat

                # Difference-indexed matrices
                Cdiff = c_mat[I_cols, diff_idx]
                Ddiff = d_mat[I_cols, diff_idx]

                g = csqrt(eps + omega * n_vals)
                pnlznl = g[:, None] * Cdiff + Ddiff

                delta = np.zeros(Nn, dtype=np.complex128)
                delta[k] = 2.0 * csqrt(eps)

                try:
                    tn = np.linalg.solve(pnlznl, delta)
                except np.linalg.LinAlgError:
                    tn, *_ = np.linalg.lstsq(pnlznl, delta, rcond=None)

                wn = F0_13 / (np.pi * csqrt(eps)) * (np.abs(tn) ** 2)

            # Sum over sidebands
            D_sum = wn.sum()

            # Occupation factor N(eps) with log1p for numerical stability
            N_val = (
                m_e * kB * T
                * np.log1p(np.exp((Ef - eps) * W / (kB * T)))
                / (2.0 * (np.pi ** 2) * (hbar ** 3))
            )

            J += e * D_sum * N_val * dE_J

    else:
        # =====================================================
        # Static (F1 == 0) DC field case
        # =====================================================
        J = 0.0 + 0.0j
        for eps in epsilon_arr:
            alpha = -(eps - Ef - 1.0) / (F0 ** (2.0 / 3.0))

            ai, aip, bi, bip = special.airy(alpha)

            term1 = csqrt(eps) * ai + F0_13 * bip
            term2 = F0_13 * aip - csqrt(eps) * bi

            D_sum = 4.0 * F0_13 * csqrt(eps) / (np.pi * (term1**2 + term2**2))

            N_val = (
                m_e * kB * T
                * np.log1p(np.exp((Ef - eps) * W / (kB * T)))
                / (2.0 * (np.pi ** 2) * (hbar ** 3))
            )

            J += e * D_sum * N_val * dE_J

    return J


# ============================================================
# 3. Laser field envelope helper
# ============================================================


def calc_F11_arr(
    t: np.ndarray,
    U: float,
    wx: float,
    wy: float,
    tau_p: float,
    theta: float,
    theta_in_degrees: bool = True,
):
    """
    Compute the (scalar) laser field envelope F11(t) at the emitter surface.

    This routine assumes:
        - Temporally Gaussian laser pulse with total energy U,
        - Elliptical spot with radii (wx, wy) on the cathode.

    Parameters
    ----------
    t : np.ndarray
        Time array [s] at which to evaluate the field envelope.
    U : float
        Laser pulse energy [J].
    wx, wy : float
        Effective spot radii on the cathode [m]. These define an area
            A ≈ π wx wy / 4
        consistent with an elliptical Gaussian beam (1/e radii).
    tau_p : float
        Pulse FWHM duration [s].
    theta : float
        Incidence angle between beam and surface normal. If
        theta_in_degrees is True, this is in degrees.
    theta_in_degrees : bool, optional
        If True, interpret `theta` as degrees and convert internally to
        radians. If False, assume `theta` is already in radians.

    Returns
    -------
    A : float
        Effective illuminated area on the cathode [m²].
    F11_arr : np.ndarray
        Time-dependent field amplitude [V/m] at the surface, same shape as t.

    Notes
    -----
    Peak field magnitude is estimated from energy flux and pulse duration:

        F_peak ≈ sqrt(2 U / (eps0 c A τ_p)) cos(theta),

    and the temporal envelope is Gaussian:

        F(t) = F_peak * exp(-2 ln 2 (t / tau_p)²).
    """
    th = np.deg2rad(theta) if theta_in_degrees else theta

    # Effective illuminated area
    A = np.pi * wx * wy / 4.0

    # Energy-flux-derived peak field (keep argument of sqrt non-negative)
    val = (2.0 * U) / (eps0 * c * A * tau_p)
    inside = np.maximum(val, 0.0)
    F_peak = np.sqrt(inside) * np.cos(th)

    # Gaussian temporal profile with FWHM tau_p
    F11_arr = F_peak * np.exp(-2.0 * np.log(2.0) * (t / tau_p) ** 2)
    return A, F11_arr


def area_average_scale_for_1d(source_area: float, model_area: float) -> float:
    """
    Return the source-area fraction used to convert local emission to 1D emission.

    PASCHEN-1D treats boundary emission as laterally uniform over the modeled
    cross-sectional area. If a source only covers a finite illuminated spot, the
    current density supplied to the 1D boundary must be area-averaged:

        J_1D = J_local * min(source_area, model_area) / model_area

    The cap at one handles beams or sources larger than the modeled area.
    """
    source_area = float(source_area)
    model_area = float(model_area)
    if (
        not np.isfinite(source_area)
        or not np.isfinite(model_area)
        or source_area <= 0.0
        or model_area <= 0.0
    ):
        return 1.0
    return min(source_area, model_area) / model_area


# ============================================================
# 4. Wrapper: precomputed J(t) window → runtime emitter
# ============================================================


def make_emitter_ps_window(
    J_series: np.ndarray,
    t_ps: np.ndarray,
    runtime_unit: str = "ns",
):
    """
    Wrap a precomputed J_series(t_ps) into an emitter(t_run, V_gap, dt_run, E_surface).

    Concept:
    --------
    - Precompute emission current density on a fine *picosecond* grid:
          t_ps[k] [ps], J_series[k] [A/m²].
    - At runtime, the plasma simulation uses times expressed in units
      specified by `runtime_unit` ("ps", "ns", or "s").
    - We map t_run → t_ps and interpret J_series as *piecewise constant*
      in each bin [t0_ps + k*dt_ps, t0_ps + (k+1)*dt_ps).

    Parameters
    ----------
    J_series : np.ndarray
        1D array of emission current density samples [A/m²].
    t_ps : np.ndarray
        1D array of times [ps] corresponding to J_series.
    runtime_unit : {"ps", "ns", "s"}, optional
        Units of the runtime arguments `t_run` and `dt_run` in the returned
        emitter. For example:
            runtime_unit = "s"  → t_run is interpreted in seconds.

    Returns
    -------
    emitter : callable
        emitter(t_run, V_gap=None, dt_run=None, E_surface=None) -> J [A/m²].

        Behavior:
        - If dt_run is None or <= 0:
              returns a *point sample* at t_run.
        - If dt_run > 0:
              returns the average emission over [t_run, t_run + dt_run].

    Notes
    -----
    - The internal grid is assumed to be approximately uniform; dt_ps is taken
      from t_ps[1] - t_ps[0].
    - Outside the precomputed time window, the emitter returns 0.
    - The averaging branch computes a piecewise-constant bin overlap average.
    """
    J_series = np.ravel(J_series).astype(float)
    t_ps = np.ravel(t_ps).astype(float)

    if J_series.size != t_ps.size:
        raise ValueError("J_series and t_ps must have same length")
    N = J_series.size

    if N < 2:
        # Degenerate case: only one sample; assume 1 ps bin width
        dt_ps = 1.0
        t0_ps = t_ps[0]
        t_end = t0_ps + dt_ps
    else:
        # Nominal spacing from first two points
        dt_ps = t_ps[1] - t_ps[0]
        # Optional near-uniform-spacing checks can be added by callers that need
        # stricter validation of the precomputed emission grid.
        t0_ps = t_ps[0]
        t_end = t_ps[-1] + dt_ps

    # Conversion factors from runtime units → picoseconds
    to_ps = {"ps": 1.0, "ns": 1e3, "s": 1e12}[runtime_unit]

    def emitter(t_run: float, V_gap: float = 0.0, dt_run: Optional[float] = None) -> float:
        """
        Local emitter closure.

        Parameters
        ----------
        t_run : float
            Runtime time in `runtime_unit` (ps, ns, or s).
        V_gap : float, optional
            Gap voltage [V]. Unused here; included for signature consistency.
        dt_run : float or None, optional
            Averaging interval length in `runtime_unit`.
            - If None or <= 0: point sample at t_run.
            - If > 0: average over [t_run, t_run + dt_run].

        Returns
        -------
        J : float
            Emission current density [A/m²].
        """
        # Convert runtime time to picoseconds
        tg = t_run * to_ps  # [ps]

        # ------------- Point sample -------------
        if dt_run is None or dt_run <= 0.0:
            if tg < t0_ps or tg >= t_end:
                # Outside precomputed window
                return 0.0
            # Map to bin index
            k = int((tg - t0_ps) // dt_ps)
            k = max(0, min(N - 1, k))
            return float(J_series[k])

        # ------------- Average over [t_run, t_run + dt_run] -------------
        tg1 = tg + dt_run * to_ps  # end of interval in ps
        a = max(tg, t0_ps)
        b = min(tg1, t_end)

        if b <= a:
            # No overlap with precomputed window
            return 0.0

        total = 0.0

        # Identify first and last potentially overlapping bins
        k0 = int(np.floor((a - t0_ps) / dt_ps))
        k1 = int(np.floor((b - t0_ps) / dt_ps))

        for k in range(max(0, k0), min(N - 1, k1) + 1):
            bin_a = t0_ps + k * dt_ps
            bin_b = bin_a + dt_ps

            left = max(a, bin_a)
            right = min(b, bin_b)

            if right > left:
                # Add contribution of this bin: J_k * (overlap length)
                total += J_series[k] * (right - left)

        # Normalize by total interval length in ps; result is still [A/m²]
        return float(total / (tg1 - tg))

    return emitter


# ============================================================
# 5. Factory: build an EmissionModel from SimulationConfig
# ============================================================


def build_emission_model(cfg: SimulationConfig) -> Optional[EmissionModel]:
    """
    Factory function to construct per-electrode composite external emission models.

    For each electrode ("anode", "cathode"), all enabled emission components
    are summed. If no component is enabled on an electrode, that side emits zero.
    """
    emission_cfg = cfg.emission

    if not emission_cfg.enable_external_emission:
        return None

    def _resolve_electrode_param(
        electrode: str,
        base_name: str,
        default: float | int,
    ) -> float | int:
        """
        Resolve emission/material parameter for an electrode using config mode.

        Priority:
          1) electrode_material_mode="separate": <electrode>_<base_name>
          2) shared_<base_name>
          3) default
        """
        mode = emission_cfg.electrode_material_mode
        if mode == "separate":
            key_sep = f"{electrode}_{base_name}"
            if hasattr(emission_cfg, key_sep):
                return getattr(emission_cfg, key_sep)

        key_shared = f"shared_{base_name}"
        if hasattr(emission_cfg, key_shared):
            return getattr(emission_cfg, key_shared)

        return default

    def _build_components_for_electrode(
        electrode: str,
    ) -> list[Callable[[float, float, float, float | None], float]]:
        emitters: list[Callable[[float, float, float, float | None], float]] = []
        prefix = "anode" if electrode == "anode" else "cathode"

        if not getattr(emission_cfg, f"enable_{prefix}_external_emission", False):
            return emitters

        # 1) Constant-J component.
        if getattr(emission_cfg, f"{prefix}_enable_constant_J_emission", False):
            J_const = float(
                _resolve_electrode_param(
                    electrode,
                    "emission_J_const",
                    0.0,
                )
            )
            t_start = float(
                _resolve_electrode_param(
                    electrode,
                    "emission_t_start",
                    0.0,
                )
            )
            t_end_raw = float(
                _resolve_electrode_param(
                    electrode,
                    "emission_t_end",
                    0.0,
                )
            )
            t_end = None if (t_end_raw <= t_start) else t_end_raw
            emitters.append(
                make_constant_J_emitter(J_const=J_const, t_start=t_start, t_end=t_end)
            )

        # 2) FN component.
        if getattr(emission_cfg, f"{prefix}_enable_fn_emission", False):
            fn_work_function_eV = float(
                _resolve_electrode_param(
                    electrode,
                    "fn_work_function_eV",
                    4.5,
                )
            )
            fn_field_scale = float(
                _resolve_electrode_param(
                    electrode,
                    "fn_field_scale_factor",
                    1.0,
                )
            )

            def emitter_fn(
                t_run: float,
                V_gap: float,
                dt_run: Optional[float] = None,
                E_surface: float | None = None,
            ) -> float:
                if E_surface is None:
                    raise ValueError(f"FN emission requires E_surface at {electrode}.")
                E_eff = fn_field_scale * float(E_surface)
                return fowler_nordheim_J(E_eff, work_function_eV=fn_work_function_eV)

            emitters.append(emitter_fn)

        # 3) MG component.
        if getattr(emission_cfg, f"{prefix}_enable_mg_emission", False):
            mg_work_function_eV = float(
                _resolve_electrode_param(
                    electrode,
                    "mg_work_function_eV",
                    4.5,
                )
            )
            mg_field_scale = float(
                _resolve_electrode_param(
                    electrode,
                    "mg_field_scale_factor",
                    1.0,
                )
            )
            f_clip_min = float(
                _resolve_electrode_param(
                    electrode,
                    "mg_f_clip_min",
                    1.0e-9,
                )
            )
            f_clip_max = float(
                _resolve_electrode_param(
                    electrode,
                    "mg_f_clip_max",
                    0.99,
                )
            )

            def emitter_mg(
                t_run: float,
                V_gap: float,
                dt_run: Optional[float] = None,
                E_surface: float | None = None,
            ) -> float:
                if E_surface is None:
                    raise ValueError(f"MG emission requires E_surface at {electrode}.")
                E_eff = mg_field_scale * float(E_surface)
                return murphy_good_cold_J(
                    E=E_eff,
                    work_function_eV=mg_work_function_eV,
                    f_clip_min=f_clip_min,
                    f_clip_max=f_clip_max,
                )

            emitters.append(emitter_mg)

        # 4) RD component.
        if getattr(emission_cfg, f"{prefix}_enable_rd_emission", False):
            A_R = float(
                _resolve_electrode_param(
                    electrode,
                    "rd_A_R",
                    1.2e6,
                )
            )
            T_cath = float(
                _resolve_electrode_param(
                    electrode,
                    "rd_emitter_K",
                    300.0,
                )
            )
            rd_work_function_eV = float(
                _resolve_electrode_param(
                    electrode,
                    "rd_work_function_eV",
                    4.1,
                )
            )

            def emitter_rd(
                t_run: float,
                V_gap: float,
                dt_run: Optional[float] = None,
                E_surface: float | None = None,
            ) -> float:
                return richardson_dushman_J(
                    T_K=T_cath,
                    work_function_eV=rd_work_function_eV,
                    A_R=A_R,
                )

            emitters.append(emitter_rd)

        # 5) Quantum-pulse component.
        if getattr(emission_cfg, f"{prefix}_enable_quantum_pulse_emission", False):
            half_window_ps = float(
                _resolve_electrode_param(
                    electrode,
                    "laser_t_window_ps",
                    200.0,
                )
            )
            dt_ps = float(
                _resolve_electrode_param(
                    electrode,
                    "emission_dt_ps",
                    2.0,
                )
            )
            N_ps = int(2.0 * half_window_ps / dt_ps) + 1
            t_ps = np.linspace(-half_window_ps, +half_window_ps, N_ps)
            t_emit = t_ps * 1e-12

            laser_spot_area, F11_arr = calc_F11_arr(
                t_emit,
                U=float(_resolve_electrode_param(electrode, "laser_U_J", 150e-6)),
                wx=float(_resolve_electrode_param(electrode, "laser_wx_m", 8.3e-3)),
                wy=float(_resolve_electrode_param(electrode, "laser_wy_m", 3.0e-3)),
                tau_p=float(_resolve_electrode_param(electrode, "laser_tau_p_s", 30e-12)),
                theta=float(_resolve_electrode_param(electrode, "laser_theta_deg", 19.0)),
            )
            model_area = float(getattr(getattr(cfg, "geometry", None), "A", 0.0))
            area_scale_1d = area_average_scale_for_1d(laser_spot_area, model_area)

            T0 = float(
                _resolve_electrode_param(
                    electrode,
                    "emission_T",
                    300.0,
                )
            )
            eps0_eV = float(
                _resolve_electrode_param(
                    electrode,
                    "emission_epsilon0_eV",
                    12.0,
                )
            )
            k_ph = int(
                _resolve_electrode_param(
                    electrode,
                    "emission_k_ph",
                    14,
                )
            )
            wt_points = int(
                _resolve_electrode_param(
                    electrode,
                    "emission_wt_points",
                    200,
                )
            )
            wt = np.linspace(0.0, 2.0 * np.pi, wt_points)
            Ef1_eV = float(
                _resolve_electrode_param(
                    electrode,
                    "emission_Ef_eV",
                    11.7,
                )
            )
            W1_eV = float(
                _resolve_electrode_param(
                    electrode,
                    "emission_W_eV",
                    4.1,
                )
            )
            lambda_m = float(
                _resolve_electrode_param(
                    electrode,
                    "emission_lambda_m",
                    230e-9,
                )
            )
            eps_points = int(
                _resolve_electrode_param(
                    electrode,
                    "emission_eps_points",
                    40,
                )
            )

            print(
                f"Quantum photoemission area averaging ({electrode}): "
                f"A_spot={laser_spot_area:.6e} m^2, "
                f"A_1D={model_area:.6e} m^2, "
                f"J_scale={area_scale_1d:.6e}"
            )

            J_em_series = np.empty_like(F11_arr, dtype=float)
            print(f"Precomputing quantum photoemission pulse ({electrode})...")
            for kk, F10 in enumerate(
                tqdm(F11_arr, desc=f"Building photoemission pulse ({electrode})")
            ):
                F0, F1, Ef, up, omega, W = parameters(
                    F11=F10 * 1e-9,
                    F01=0.0,
                    Ef1_eV=Ef1_eV,
                    W1_eV=W1_eV,
                    lambda_m=lambda_m,
                )
                J_result = emission_current_density(
                    k=k_ph,
                    wt=wt,
                    F0=F0,
                    F1=F1,
                    Ef=Ef,
                    up=up,
                    omega=omega,
                    W=W,
                    epsilon_0_eV=eps0_eV,
                    T=T0,
                    eps_points=eps_points,
                )
                J_em_series[kk] = float(np.real(J_result))

            J_em_series *= area_scale_1d
            emitter_raw = make_emitter_ps_window(J_em_series, t_ps, runtime_unit="s")

            def shifted_emitter(
                t_run: float,
                V_gap: float = 0.0,
                dt_run: Optional[float] = None,
                E_surface: float | None = None,
            ) -> float:
                t0 = float(_resolve_electrode_param(electrode, "laser_t0", 10e-6))
                t_local = t_run - t0
                return emitter_raw(t_local, V_gap=V_gap, dt_run=dt_run)

            emitters.append(shifted_emitter)

        return emitters

    anode_emitters = _build_components_for_electrode("anode")
    cathode_emitters = _build_components_for_electrode("cathode")

    if (not anode_emitters) and (not cathode_emitters):
        return None

    def _sum_emitters(
        emitter_list: list[Callable[[float, float, float, float | None], float]]
    ) -> Optional[Callable[[float, float, float, float | None], float]]:
        if not emitter_list:
            return None

        def composite(
            t_run: float,
            V_gap: float,
            dt_run: Optional[float] = None,
            E_surface: float | None = None,
        ) -> float:
            return float(sum(em(t_run, V_gap, dt_run, E_surface) for em in emitter_list))

        return composite

    return EmissionModel(
        cfg=cfg,
        emitter_func_anode=_sum_emitters(anode_emitters),
        emitter_func_cathode=_sum_emitters(cathode_emitters),
    )
