"""
circuit_implicit_euler.py

Implicit-Euler stepping for external-circuit states in PASCHEN-1D.

This module mirrors the circuit topology set in `circuit.py` and uses the
same coupling interface (transport current in, gap voltage out). The implicit
stepper is selected through `SimulationConfig.circuit_time_scheme =
"implicit_euler"` and is intended for stiff circuit parameter sets.

Only the lumped circuit states are advanced implicitly here. Plasma species
time integration remains in the plasma solver (`paschen_1d.py` + `numerics.py`).
"""

from typing import Callable, Optional

import numpy as np

from physical_constants import e, eps0


def _compute_transport_current(
    Gamma_i: np.ndarray,
    Gamma_e: np.ndarray,
    dx: float,
    A: float,
    L: float,
) -> float:
    """Compute the area-integrated plasma transport current from flux profiles."""
    flux_diff = Gamma_i - Gamma_e
    integral_flux = 0.5 * dx * (
        flux_diff[0] + flux_diff[-1] + 2.0 * np.add.reduce(flux_diff[1:-1])
    )
    # Cast to Python float so downstream circuit algebra stays in float64 even
    # when Gamma_i/Gamma_e are float32 plasma work arrays.
    return float((A * e / L) * integral_flux)


def _dielectric_coeffs(l: float, eps_r: float, L: float) -> tuple[float, float]:
    """Return the dielectric mapping coefficients used by coated-electrode circuits."""
    alpha_d = 1.0 + 2.0 * l / (eps_r * L)
    beta_d = (2.0 * e * l) / (eps0 * eps_r * L)
    return alpha_d, beta_d


def step_circuit_implicit_euler(
    circuit_type: str,
    V_app_func: Callable[[float], float],
    t: float,
    dt: float,
    V_gap_prev: float,
    Gamma_i: np.ndarray,
    Gamma_e: np.ndarray,
    dx: float,
    A: float,
    L: float,
    l: float,
    eps_r: float,
    R0: float,
    C_s: float,
    C_p: float,
    R_m: float,
    L_s: float,
    L_p: float,
    V_d_prev: Optional[float],
    V_n_prev: Optional[float],
    V_Cs_prev: Optional[float],
    I_s_prev: Optional[float],
    I_Lp_prev: Optional[float],
    C_ext: float = 0.0,
) -> tuple[
    float,
    float,
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
]:
    """
    Implicit-Euler analog of `circuit.step_circuit(...)` for external-circuit
    state variables.

    Return order:
        (V_gap_new, I_discharge, V_d_new, V_n_new, V_Cs_new, I_s_new, I_Lp_new)
    """
    if V_d_prev is None:
        raise ValueError("Implicit circuit step requires V_d_prev.")

    I_transport = _compute_transport_current(Gamma_i, Gamma_e, dx, A, L)
    C_gap = eps0 * A / L
    Phi = I_transport * L / (A * e)
    alpha_d, beta_d = _dielectric_coeffs(l, eps_r, L)
    Vs_now = float(V_app_func(t))
    Vs_next = float(V_app_func(t + dt))

    dV_d = dt * beta_d * Phi
    V_d_new = float(V_d_prev + dV_d)

    def i_dis(Vg_new: float) -> float:
        return I_transport + C_gap * (Vg_new - V_gap_prev) / dt

    # ------------------------------------------------------------------
    # R0_Rm_Cext
    # unknown: [V_gap_new]; V_n and V_load are algebraic
    # ------------------------------------------------------------------
    if circuit_type == "R0_Rm_Cext":
        if R0 < 0.0:
            raise ValueError("R0_Rm_Cext requires R0 >= 0.")
        if R_m < 0.0:
            raise ValueError("R0_Rm_Cext requires R_m >= 0.")
        if C_ext < 0.0:
            raise ValueError("R0_Rm_Cext requires C_ext >= 0.")

        R_total = R0 + R_m
        if R_total <= 0.0:
            raise ValueError("R0_Rm_Cext requires R0 + R_m > 0.")

        C_gap_eff = C_gap + alpha_d * C_ext
        if C_gap_eff <= 0.0:
            raise ZeroDivisionError("R0_Rm_Cext requires positive gap-node capacitance.")

        V_load_prev = alpha_d * V_gap_prev + V_d_prev
        denom = C_gap_eff / dt + alpha_d / R_total
        rhs = (
            (C_gap / dt) * V_gap_prev
            + (C_ext / dt) * (V_load_prev - V_d_new)
            + (Vs_now - V_d_new) / R_total
            - I_transport
        )
        V_gap_new = rhs / denom
        V_load_new = alpha_d * V_gap_new + V_d_new
        V_n_new = (R_m * Vs_now + R0 * V_load_new) / R_total
        I_series = (Vs_now - V_load_new) / R_total
        return V_gap_new, I_series, V_d_new, V_n_new, None, None, None

    # ------------------------------------------------------------------
    # dielectric_plasma
    # ------------------------------------------------------------------
    if circuit_type == "dielectric_plasma":
        if l <= 0.0:
            raise ValueError("dielectric_plasma requires l > 0.")
        V_gap_new = V_gap_prev + (Vs_next - Vs_now - dV_d) / alpha_d
        return V_gap_new, i_dis(V_gap_new), V_d_new, None, None, None, None

    # ------------------------------------------------------------------
    # R0_Cp
    # unknowns: [V_n_new, V_gap_new]
    # ------------------------------------------------------------------
    if circuit_type == "R0_Cp":
        if V_n_prev is None:
            raise ValueError("R0_Cp requires V_n_prev.")
        A_mat = np.array(
            [
                [-(1.0 / R0 + C_p / dt), -(C_gap / dt)],
                [1.0, -alpha_d],
            ],
            dtype=np.float64,
        )
        b_vec = np.array(
            [
                -(Vs_now / R0 + (C_p / dt) * V_n_prev - I_transport + (C_gap / dt) * V_gap_prev),
                V_n_prev - alpha_d * V_gap_prev + dV_d,
            ],
            dtype=np.float64,
        )
        V_n_new, V_gap_new = np.linalg.solve(A_mat, b_vec)
        return V_gap_new, i_dis(V_gap_new), V_d_new, V_n_new, None, None, None

    # ------------------------------------------------------------------
    # R0_Cp_Rm
    # unknowns: [V_n_new, V_gap_new]
    # ------------------------------------------------------------------
    if circuit_type == "R0_Cp_Rm":
        if V_n_prev is None:
            raise ValueError("R0_Cp_Rm requires V_n_prev.")
        A_mat = np.array(
            [
                [-(1.0 / R0 + C_p / dt), -(C_gap / dt)],
                [1.0 / R_m, -(alpha_d / R_m + C_gap / dt)],
            ],
            dtype=np.float64,
        )
        b_vec = np.array(
            [
                -(Vs_now / R0 + (C_p / dt) * V_n_prev - I_transport + (C_gap / dt) * V_gap_prev),
                V_d_new / R_m + I_transport - (C_gap / dt) * V_gap_prev,
            ],
            dtype=np.float64,
        )
        V_n_new, V_gap_new = np.linalg.solve(A_mat, b_vec)
        return V_gap_new, i_dis(V_gap_new), V_d_new, V_n_new, None, None, None

    # ------------------------------------------------------------------
    # R0_Cs_Cp
    # unknowns: [V_n_new, V_gap_new, V_Cs_new, I_s_new]
    # ------------------------------------------------------------------
    if circuit_type == "R0_Cs_Cp":
        if V_n_prev is None or V_Cs_prev is None:
            raise ValueError("R0_Cs_Cp requires V_n_prev and V_Cs_prev.")
        A_mat = np.array(
            [
                [1.0 / R0, 0.0, 1.0 / R0, 1.0],
                [0.0, 0.0, 1.0, -(dt / C_s)],
                [-(C_p / dt), -(C_gap / dt), 0.0, 1.0],
                [1.0, -alpha_d, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        b_vec = np.array(
            [
                Vs_now / R0,
                V_Cs_prev,
                I_transport - (C_p / dt) * V_n_prev - (C_gap / dt) * V_gap_prev,
                V_n_prev - alpha_d * V_gap_prev + dV_d,
            ],
            dtype=np.float64,
        )
        V_n_new, V_gap_new, V_Cs_new, I_s_new = np.linalg.solve(A_mat, b_vec)
        return V_gap_new, i_dis(V_gap_new), V_d_new, V_n_new, V_Cs_new, I_s_new, None

    # ------------------------------------------------------------------
    # R0_Cs_Cp_Rm
    # unknowns: [V_n_new, V_gap_new, V_Cs_new, I_s_new]
    # ------------------------------------------------------------------
    if circuit_type == "R0_Cs_Cp_Rm":
        if V_n_prev is None or V_Cs_prev is None:
            raise ValueError("R0_Cs_Cp_Rm requires V_n_prev and V_Cs_prev.")
        A_mat = np.array(
            [
                [1.0 / R0, 0.0, 1.0 / R0, 1.0],
                [0.0, 0.0, 1.0, -(dt / C_s)],
                [1.0 / R_m, -(alpha_d / R_m + C_gap / dt), 0.0, 0.0],
                [-(C_p / dt), -(C_gap / dt), 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        b_vec = np.array(
            [
                Vs_now / R0,
                V_Cs_prev,
                V_d_new / R_m + I_transport - (C_gap / dt) * V_gap_prev,
                I_transport - (C_p / dt) * V_n_prev - (C_gap / dt) * V_gap_prev,
            ],
            dtype=np.float64,
        )
        V_n_new, V_gap_new, V_Cs_new, I_s_new = np.linalg.solve(A_mat, b_vec)
        return V_gap_new, i_dis(V_gap_new), V_d_new, V_n_new, V_Cs_new, I_s_new, None

    # ------------------------------------------------------------------
    # R0_Cs_Ls_Cp
    # unknowns: [V_n_new, V_gap_new, V_Cs_new, I_s_new]
    # ------------------------------------------------------------------
    if circuit_type == "R0_Cs_Ls_Cp":
        if V_n_prev is None or V_Cs_prev is None or I_s_prev is None:
            raise ValueError("R0_Cs_Ls_Cp requires V_n_prev, V_Cs_prev, I_s_prev.")
        A_mat = np.array(
            [
                [dt / L_s, 0.0, dt / L_s, 1.0 + dt * R0 / L_s],
                [0.0, 0.0, 1.0, -(dt / C_s)],
                [-(C_p / dt), -(C_gap / dt), 0.0, 1.0],
                [1.0, -alpha_d, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        b_vec = np.array(
            [
                I_s_prev + (dt / L_s) * Vs_now,
                V_Cs_prev,
                I_transport - (C_p / dt) * V_n_prev - (C_gap / dt) * V_gap_prev,
                V_n_prev - alpha_d * V_gap_prev + dV_d,
            ],
            dtype=np.float64,
        )
        V_n_new, V_gap_new, V_Cs_new, I_s_new = np.linalg.solve(A_mat, b_vec)
        return V_gap_new, i_dis(V_gap_new), V_d_new, V_n_new, V_Cs_new, I_s_new, None

    # ------------------------------------------------------------------
    # R0_Cs_Ls_Cp_Rm
    # unknowns: [V_n_new, V_gap_new, V_Cs_new, I_s_new]
    # ------------------------------------------------------------------
    if circuit_type == "R0_Cs_Ls_Cp_Rm":
        if V_n_prev is None or V_Cs_prev is None or I_s_prev is None:
            raise ValueError("R0_Cs_Ls_Cp_Rm requires V_n_prev, V_Cs_prev, I_s_prev.")
        A_mat = np.array(
            [
                [dt / L_s, 0.0, dt / L_s, 1.0 + dt * R0 / L_s],
                [0.0, 0.0, 1.0, -(dt / C_s)],
                [1.0 / R_m, -(alpha_d / R_m + C_gap / dt), 0.0, 0.0],
                [-(C_p / dt), -(C_gap / dt), 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        b_vec = np.array(
            [
                I_s_prev + (dt / L_s) * Vs_now,
                V_Cs_prev,
                V_d_new / R_m + I_transport - (C_gap / dt) * V_gap_prev,
                I_transport - (C_p / dt) * V_n_prev - (C_gap / dt) * V_gap_prev,
            ],
            dtype=np.float64,
        )
        V_n_new, V_gap_new, V_Cs_new, I_s_new = np.linalg.solve(A_mat, b_vec)
        return V_gap_new, i_dis(V_gap_new), V_d_new, V_n_new, V_Cs_new, I_s_new, None

    # ------------------------------------------------------------------
    # R0_Cs_Ls_Cp_Lp
    # unknowns: [V_n_new, V_gap_new, V_Cs_new, I_s_new, I_Lp_new]
    # ------------------------------------------------------------------
    if circuit_type == "R0_Cs_Ls_Cp_Lp":
        if any(v is None for v in (V_n_prev, V_Cs_prev, I_s_prev, I_Lp_prev)):
            raise ValueError("R0_Cs_Ls_Cp_Lp requires V_n_prev, V_Cs_prev, I_s_prev, I_Lp_prev.")
        A_mat = np.array(
            [
                [dt / L_s, 0.0, dt / L_s, 1.0 + dt * R0 / L_s, 0.0],
                [0.0, 0.0, 1.0, -(dt / C_s), 0.0],
                [-(dt / L_p), 0.0, 0.0, 0.0, 1.0],
                [-(C_p / dt), -(C_gap / dt), 0.0, 1.0, -1.0],
                [1.0, -alpha_d, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        b_vec = np.array(
            [
                I_s_prev + (dt / L_s) * Vs_now,
                V_Cs_prev,
                I_Lp_prev,
                I_transport - (C_p / dt) * V_n_prev - (C_gap / dt) * V_gap_prev,
                V_n_prev - alpha_d * V_gap_prev + dV_d,
            ],
            dtype=np.float64,
        )
        V_n_new, V_gap_new, V_Cs_new, I_s_new, I_Lp_new = np.linalg.solve(A_mat, b_vec)
        return V_gap_new, i_dis(V_gap_new), V_d_new, V_n_new, V_Cs_new, I_s_new, I_Lp_new

    # ------------------------------------------------------------------
    # R0_Cs_Ls_Cp_Lp_Rm_Cext
    # unknowns: [V_n_new, V_gap_new, V_Cs_new, I_s_new, I_Lp_new]
    # ------------------------------------------------------------------
    if circuit_type == "R0_Cs_Ls_Cp_Lp_Rm_Cext":
        if any(v is None for v in (V_n_prev, V_Cs_prev, I_s_prev, I_Lp_prev)):
            raise ValueError("R0_Cs_Ls_Cp_Lp_Rm_Cext requires V_n_prev, V_Cs_prev, I_s_prev, I_Lp_prev.")
        if C_ext < 0.0:
            raise ValueError("R0_Cs_Ls_Cp_Lp_Rm_Cext requires C_ext >= 0.")
        V_load_prev = alpha_d * V_gap_prev + V_d_prev
        A_mat = np.array(
            [
                [dt / L_s, 0.0, dt / L_s, 1.0 + dt * R0 / L_s, 0.0],
                [0.0, 0.0, 1.0, -(dt / C_s), 0.0],
                [-(dt / L_p), 0.0, 0.0, 0.0, 1.0],
                [
                    1.0 / R_m,
                    -(alpha_d / R_m + C_gap / dt + alpha_d * C_ext / dt),
                    0.0,
                    0.0,
                    0.0,
                ],
                [-(C_p / dt + 1.0 / R_m), alpha_d / R_m, 0.0, 1.0, -1.0],
            ],
            dtype=np.float64,
        )
        b_vec = np.array(
            [
                I_s_prev + (dt / L_s) * Vs_now,
                V_Cs_prev,
                I_Lp_prev,
                V_d_new / R_m
                + I_transport
                - (C_gap / dt) * V_gap_prev
                + (C_ext / dt) * (V_d_new - V_load_prev),
                -(C_p / dt) * V_n_prev - V_d_new / R_m,
            ],
            dtype=np.float64,
        )
        V_n_new, V_gap_new, V_Cs_new, I_s_new, I_Lp_new = np.linalg.solve(A_mat, b_vec)
        V_load_new = alpha_d * V_gap_new + V_d_new
        I_Rm = (V_n_new - V_load_new) / R_m
        return V_gap_new, I_Rm, V_d_new, V_n_new, V_Cs_new, I_s_new, I_Lp_new

    raise ValueError(f"Unknown or unsupported circuit_type for implicit step: {circuit_type}")
