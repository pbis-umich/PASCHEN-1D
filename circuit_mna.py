"""
circuit_mna.py

Dielectric-aware modified-nodal-analysis backend for the unified PASCHEN-1D
external circuit.

The intended user-facing topology is

    R0_Cs_Ls_Cp_Lp_Rm_Cext

with the following conceptual network:

    Vs(t) -- R0 -- Cs -- Ls -- V_n -- Rm -- V_load
                                      |             |
                                      Cp            C_ext
                                      |             |
                                    ground        ground
                                      |
                                      Lp
                                      |
                                    ground

The plasma/dielectric branch is connected from V_load to ground. With
dielectric-coated electrodes, the external load-terminal voltage and internal
plasma-gap voltage are related by

    V_load = alpha_d * V_gap + V_d
    alpha_d = 1 + 2*l/(eps_r*L)
    dV_d/dt = beta_d*Phi,  beta_d = 2*e*l/(eps0*eps_r*L)

Neutral values used to remove elements from the maximum topology:

    R0    = 0.0     -> short source resistor
    C_s   = np.inf  -> short series capacitor
    L_s   = 0.0     -> short series inductor
    C_p   = 0.0     -> open shunt capacitor
    L_p   = np.inf  -> open shunt inductor
    R_m   = 0.0     -> short measurement/load resistor
    C_ext = 0.0     -> open load-side capacitance

The solver uses backward-Euler companion models for capacitors and inductors,
and merges nodes connected by exact short-circuit neutral values. This avoids
the stiffness and conditioning problems that come from replacing removed
elements with arbitrary tiny or huge finite values.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from physical_constants import e, eps0


UNIFIED_CIRCUIT_TYPE = "R0_Cs_Ls_Cp_Lp_Rm_Cext"


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
    return float((A * e / L) * integral_flux)


def _dielectric_coeffs(l: float, eps_r: float, L: float) -> tuple[float, float]:
    """Return dielectric mapping coefficients alpha_d and beta_d."""
    alpha_d = 1.0 + 2.0 * l / (eps_r * L)
    beta_d = (2.0 * e * l) / (eps0 * eps_r * L)
    return alpha_d, beta_d


def _is_inf(value: float) -> bool:
    return bool(np.isinf(float(value)))


def _is_zero(value: float) -> bool:
    return float(value) == 0.0


class _UnionFind:
    def __init__(self, nodes: tuple[str, ...]) -> None:
        self.parent = {node: node for node in nodes}

    def find(self, node: str) -> str:
        parent = self.parent[node]
        if parent != node:
            self.parent[node] = self.find(parent)
        return self.parent[node]

    def union(self, a: str, b: str) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _validate_unified_values(
    R0: float,
    C_s: float,
    L_s: float,
    C_p: float,
    L_p: float,
    R_m: float,
    C_ext: float,
) -> None:
    """Validate component values and neutral-value conventions."""
    if R0 < 0.0:
        raise ValueError("R0 must be >= 0; use R0 = 0 to remove it as a short.")
    if R_m < 0.0:
        raise ValueError("R_m must be >= 0; use R_m = 0 to remove it as a short.")
    if C_s < 0.0:
        raise ValueError("C_s must be >= 0 or np.inf; use C_s = np.inf to remove it as a short.")
    if L_s < 0.0:
        raise ValueError("L_s must be >= 0; use L_s = 0 to remove it as a short.")
    if C_p < 0.0 or _is_inf(C_p):
        raise ValueError("C_p must be finite and >= 0; use C_p = 0 to remove it as an open.")
    if L_p < 0.0:
        raise ValueError("L_p must be >= 0 or np.inf; use L_p = np.inf to remove it as an open.")
    if C_ext < 0.0 or _is_inf(C_ext):
        raise ValueError("C_ext must be finite and >= 0; use C_ext = 0 to remove it as an open.")


def _require_state(name: str, value: Optional[float]) -> float:
    if value is None:
        raise ValueError(f"{name} is required for the active unified-circuit element.")
    return float(value)


def step_circuit_mna(
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
    Advance the unified maximum circuit by one backward-Euler MNA step.

    The return order matches the existing PASCHEN-1D circuit interface:

        (V_gap_new, I_discharge, V_d_new, V_n_new, V_Cs_new, I_s_new, I_Lp_new)

    `I_discharge` is the total load-side current into the plasma/dielectric
    terminal, including the external load-side capacitance C_ext when present.
    """
    if circuit_type != UNIFIED_CIRCUIT_TYPE:
        raise ValueError(
            f"circuit_mna only supports {UNIFIED_CIRCUIT_TYPE}; got {circuit_type!r}."
        )

    if dt <= 0.0:
        raise ValueError("dt must be > 0.")
    if A <= 0.0 or L <= 0.0:
        raise ValueError("A and L must be > 0.")
    if eps_r <= 0.0:
        raise ValueError("eps_r must be > 0.")

    R0 = float(R0)
    C_s = float(C_s)
    L_s = float(L_s)
    C_p = float(C_p)
    R_m = float(R_m)
    L_p = float(L_p)
    C_ext = float(C_ext)
    _validate_unified_values(R0, C_s, L_s, C_p, L_p, R_m, C_ext)

    V_d_prev_f = 0.0 if V_d_prev is None else float(V_d_prev)
    V_gap_prev_f = float(V_gap_prev)
    Vs = float(V_app_func(t))

    I_transport = _compute_transport_current(Gamma_i, Gamma_e, dx, A, L)
    C_gap = eps0 * A / L
    Phi = I_transport * L / (A * e)
    alpha_d, beta_d = _dielectric_coeffs(l, eps_r, L)
    dV_d = dt * beta_d * Phi
    V_d_new = V_d_prev_f + dV_d
    V_load_prev = alpha_d * V_gap_prev_f + V_d_prev_f

    # Nodes:
    # src is the ideal source node, a/b are internal series nodes, n is the
    # node before Rm, d is the load/plasma terminal, and gnd is ground.
    nodes = ("src", "a", "b", "n", "d", "gnd")
    uf = _UnionFind(nodes)

    if _is_zero(R0):
        uf.union("src", "a")
    if _is_inf(C_s):
        uf.union("a", "b")
    if _is_zero(L_s):
        uf.union("b", "n")
    if _is_zero(R_m):
        uf.union("n", "d")
    if _is_zero(L_p):
        uf.union("n", "gnd")

    src_rep = uf.find("src")
    gnd_rep = uf.find("gnd")
    if src_rep == gnd_rep and abs(Vs) > 1.0e-12:
        raise ValueError("Unified circuit shorts the voltage source directly to ground.")

    known: dict[str, float] = {src_rep: Vs, gnd_rep: 0.0}
    if src_rep == gnd_rep:
        known[src_rep] = 0.0

    active_node_reps: set[str] = set()
    branch_specs: list[tuple[str, str, float, float, str]] = []

    def add_active(*node_names: str) -> None:
        for node_name in node_names:
            rep = uf.find(node_name)
            if rep not in known:
                active_node_reps.add(rep)

    def finite_resistance(value: float) -> bool:
        return (not _is_zero(value)) and (not _is_inf(value))

    def finite_capacitance(value: float) -> bool:
        return value > 0.0 and not _is_inf(value)

    def finite_inductance(value: float) -> bool:
        return value > 0.0 and not _is_inf(value)

    if finite_resistance(R0):
        add_active("src", "a")
    if finite_capacitance(C_s):
        add_active("a", "b")
    if finite_inductance(L_s):
        I_s_prev_f = _require_state("I_s_prev", I_s_prev)
        add_active("b", "n")
        branch_specs.append(("b", "n", L_s, I_s_prev_f, "Ls"))
    if finite_capacitance(C_p):
        _require_state("V_n_prev", V_n_prev)
        add_active("n", "gnd")
    if finite_inductance(L_p):
        I_Lp_prev_f = _require_state("I_Lp_prev", I_Lp_prev)
        add_active("n", "gnd")
        branch_specs.append(("n", "gnd", L_p, I_Lp_prev_f, "Lp"))
    if finite_resistance(R_m):
        add_active("n", "d")
    if finite_capacitance(C_ext):
        add_active("d", "gnd")

    # The plasma/dielectric branch always references the load node.
    add_active("d", "gnd")

    node_reps = sorted(active_node_reps)
    node_index = {rep: i for i, rep in enumerate(node_reps)}
    n_node = len(node_reps)
    n_branch = len(branch_specs)
    size = n_node + n_branch

    mat = np.zeros((size, size), dtype=np.float64)
    rhs = np.zeros(size, dtype=np.float64)

    def rep_of(node_name: str) -> str:
        return uf.find(node_name)

    def known_voltage(rep: str) -> Optional[float]:
        return known.get(rep)

    def idx_of(rep: str) -> Optional[int]:
        return node_index.get(rep)

    def stamp_conductance(p: str, q: str, conductance: float) -> None:
        if conductance == 0.0:
            return
        rp = rep_of(p)
        rq = rep_of(q)
        if rp == rq:
            return
        ip = idx_of(rp)
        iq = idx_of(rq)
        vp = known_voltage(rp)
        vq = known_voltage(rq)

        if ip is not None:
            mat[ip, ip] += conductance
            if iq is not None:
                mat[ip, iq] -= conductance
            else:
                rhs[ip] += conductance * float(vq)
        if iq is not None:
            mat[iq, iq] += conductance
            if ip is not None:
                mat[iq, ip] -= conductance
            else:
                rhs[iq] += conductance * float(vp)

    def stamp_capacitor(p: str, q: str, capacitance: float, v_prev: float) -> None:
        if capacitance <= 0.0:
            return
        conductance = capacitance / dt
        stamp_conductance(p, q, conductance)
        rp = rep_of(p)
        rq = rep_of(q)
        if rp == rq:
            return
        ip = idx_of(rp)
        iq = idx_of(rq)
        if ip is not None:
            rhs[ip] += conductance * v_prev
        if iq is not None:
            rhs[iq] -= conductance * v_prev

    def stamp_current_leaving(p: str, current: float) -> None:
        rp = rep_of(p)
        ip = idx_of(rp)
        if ip is not None:
            rhs[ip] -= current

    # Static/dynamic passive elements.
    if finite_resistance(R0):
        stamp_conductance("src", "a", 1.0 / R0)
    if finite_capacitance(C_s):
        V_Cs_prev_f = _require_state("V_Cs_prev", V_Cs_prev)
        stamp_capacitor("a", "b", C_s, V_Cs_prev_f)
    if finite_capacitance(C_p):
        V_n_prev_f = _require_state("V_n_prev", V_n_prev)
        stamp_capacitor("n", "gnd", C_p, V_n_prev_f)
    if finite_resistance(R_m):
        stamp_conductance("n", "d", 1.0 / R_m)
    if finite_capacitance(C_ext):
        stamp_capacitor("d", "gnd", C_ext, V_load_prev)

    # Plasma/dielectric branch at V_load:
    # I_plasma = I_transport + C_gap*((V_load - V_d_new)/alpha_d - V_gap_prev)/dt
    #          = G_plasma*V_load + I_const
    g_plasma = C_gap / (alpha_d * dt)
    i_const = I_transport - (C_gap / dt) * (V_gap_prev_f + V_d_new / alpha_d)
    stamp_conductance("d", "gnd", g_plasma)
    stamp_current_leaving("d", i_const)

    # Inductor branch-current unknowns.
    branch_current_indices: dict[str, int] = {}
    for bnum, (p, q, inductance, i_prev, name) in enumerate(branch_specs):
        row = n_node + bnum
        branch_current_indices[name] = row
        rp = rep_of(p)
        rq = rep_of(q)
        ip = idx_of(rp)
        iq = idx_of(rq)
        vp = known_voltage(rp)
        vq = known_voltage(rq)

        # KCL: branch current is oriented from p to q.
        if ip is not None:
            mat[ip, row] += 1.0
        if iq is not None:
            mat[iq, row] -= 1.0

        # Branch equation: Vp - Vq - (L/dt) * I_new = -(L/dt) * I_prev.
        if ip is not None:
            mat[row, ip] += 1.0
        else:
            rhs[row] -= float(vp)
        if iq is not None:
            mat[row, iq] -= 1.0
        else:
            rhs[row] += float(vq)
        impedance = inductance / dt
        mat[row, row] -= impedance
        rhs[row] -= impedance * i_prev

    if size > 0:
        try:
            sol = np.linalg.solve(mat, rhs)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "Unified MNA circuit is singular for the selected component values. "
                "Check for floating islands or contradictory short circuits."
            ) from exc
    else:
        sol = np.zeros(0, dtype=np.float64)

    previous_node_voltage = {
        "src": Vs,
        "a": Vs,
        "b": Vs,
        "n": V_load_prev if V_n_prev is None else float(V_n_prev),
        "d": V_load_prev,
        "gnd": 0.0,
    }

    def voltage(node_name: str) -> float:
        rep = rep_of(node_name)
        if rep in known:
            return known[rep]
        idx = idx_of(rep)
        if idx is not None:
            return float(sol[idx])
        return previous_node_voltage[node_name]

    Va_new = voltage("a")
    Vb_new = voltage("b")
    V_n_new = voltage("n")
    V_load_new = voltage("d")
    V_gap_new = (V_load_new - V_d_new) / alpha_d

    dV_gap_dt = (V_gap_new - V_gap_prev_f) / dt
    dV_load_dt = (V_load_new - V_load_prev) / dt
    I_plasma = I_transport + C_gap * dV_gap_dt
    I_load = I_plasma + C_ext * dV_load_dt

    if finite_capacitance(C_s) or _is_zero(C_s):
        V_Cs_new = Va_new - Vb_new
    else:
        V_Cs_new = 0.0

    if "Lp" in branch_current_indices:
        I_Lp_new = float(sol[branch_current_indices["Lp"]])
    else:
        I_Lp_new = 0.0

    I_Cp = C_p * (V_n_new - _require_state("V_n_prev", V_n_prev)) / dt if finite_capacitance(C_p) else 0.0
    if "Ls" in branch_current_indices:
        I_s_new = float(sol[branch_current_indices["Ls"]])
    elif _is_zero(L_p) and finite_resistance(R0):
        I_s_new = (Vs - Va_new) / R0
    elif _is_zero(L_p) and finite_capacitance(C_s):
        I_s_new = C_s * (V_Cs_new - _require_state("V_Cs_prev", V_Cs_prev)) / dt
    elif _is_zero(L_p) and (_is_zero(C_s) or _is_inf(R0) or _is_inf(L_s)):
        I_s_new = 0.0
    else:
        I_s_new = I_Cp + I_Lp_new + I_load

    if _is_zero(L_p) and not (_is_zero(R0) and _is_inf(C_s) and _is_zero(L_s)):
        # Node merging removes the short's current unknown. Recover it by KCL.
        # Parallel ideal source/shunt shorts have an undetermined split; retain
        # the existing I_Lp = 0 convention for that degenerate case.
        I_Lp_new = I_s_new - I_Cp - I_load

    return (
        float(V_gap_new),
        float(I_load),
        float(V_d_new),
        float(V_n_new),
        float(V_Cs_new),
        float(I_s_new),
        float(I_Lp_new),
    )
