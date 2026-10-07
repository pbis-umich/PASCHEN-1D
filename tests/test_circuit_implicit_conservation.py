from __future__ import annotations

import numpy as np
import pytest

from circuit_implicit_euler import step_circuit_implicit_euler
from circuit_mna import step_circuit_mna
from physical_constants import e


TOPOLOGIES = (
    "R0_Cs_Cp",
    "R0_Cs_Cp_Rm",
    "R0_Cs_Ls_Cp",
    "R0_Cs_Ls_Cp_Rm",
    "R0_Cs_Ls_Cp_Lp",
)


@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("dielectric_thickness", [0.0, 1.0e-4])
@pytest.mark.parametrize(
    "initial_gap, transport_current", [(10.0, 0.0), (0.0, 1.0e-3)]
)
def test_reduced_implicit_circuits_conserve_current(
    topology, dielectric_thickness, initial_gap, transport_current
):
    """Check capacitor history and transport signs independently against KCL."""
    area = 1.0e-4
    length = 1.0e-3
    alpha = 1.0 + 2.0 * dielectric_thickness / (4.0 * length)
    initial_node = alpha * initial_gap
    arguments = dict(
        circuit_type=topology,
        V_app_func=lambda t: initial_node,
        t=1.0e-9,
        dt=1.0e-9,
        V_gap_prev=initial_gap,
        Gamma_i=np.full(5, transport_current / (area * e)),
        Gamma_e=np.zeros(5),
        dx=length / 4.0,
        A=area,
        L=length,
        l=dielectric_thickness,
        eps_r=4.0,
        R0=1.0e5,
        C_s=1.0e-9,
        C_p=1.0e-10,
        R_m=2.0e5,
        L_s=1.0e-6,
        L_p=2.0e-6,
        V_d_prev=0.0,
        V_n_prev=initial_node,
        V_Cs_prev=0.0,
        I_s_prev=0.0,
        I_Lp_prev=0.0,
        C_ext=0.0,
    )
    actual = step_circuit_implicit_euler(**arguments)
    _, discharge_current, _, node, _, series_current, shunt_current = actual
    capacitor_current = arguments["C_p"] * (node - initial_node) / arguments["dt"]
    assert series_current == pytest.approx(
        capacitor_current + discharge_current + (shunt_current or 0.0),
        rel=1.0e-10,
        abs=1.0e-11,
    )

    # An independently stamped MNA network represents the same physical circuit.
    mna_arguments = dict(arguments, circuit_type="R0_Cs_Ls_Cp_Lp_Rm_Cext")
    if "Ls" not in topology:
        mna_arguments["L_s"] = 0.0
    if "Lp" not in topology:
        mna_arguments["L_p"] = np.inf
    if "Rm" not in topology:
        mna_arguments["R_m"] = 0.0
    expected = step_circuit_mna(**mna_arguments)
    for reduced_value, mna_value in zip(actual, expected):
        if reduced_value is not None:
            assert reduced_value == pytest.approx(mna_value, rel=1.0e-10, abs=1.0e-11)
