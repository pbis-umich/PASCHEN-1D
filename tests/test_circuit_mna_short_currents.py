from __future__ import annotations

import numpy as np
import pytest

from circuit_mna import step_circuit_mna
from physical_constants import e


def shorted_shunt_arguments(**overrides):
    arguments = dict(
        circuit_type="R0_Cs_Ls_Cp_Lp_Rm_Cext",
        V_app_func=lambda t: 100.0,
        t=1.0e-6,
        dt=1.0e-6,
        V_gap_prev=0.0,
        Gamma_i=np.zeros(5),
        Gamma_e=np.zeros(5),
        dx=2.5e-4,
        A=1.0e-4,
        L=1.0e-3,
        l=0.0,
        eps_r=4.0,
        R0=0.0,
        C_s=np.inf,
        C_p=0.0,
        R_m=0.0,
        L_s=0.0,
        L_p=0.0,
        V_d_prev=0.0,
        V_n_prev=0.0,
        V_Cs_prev=0.0,
        I_s_prev=0.0,
        I_Lp_prev=0.0,
        C_ext=0.0,
    )
    arguments.update(overrides)
    return arguments


@pytest.mark.parametrize(
    "components",
    [
        {"R0": 1.0e5},
        {"R0": 1.0e5, "L_s": 1.0e-3, "I_s_prev": 2.0e-3},
        {"C_s": 1.0e-9, "V_Cs_prev": 5.0},
        {"R0": 1.0e5, "C_s": 1.0e-9, "V_Cs_prev": 5.0},
    ],
)
@pytest.mark.parametrize("transport_current", [0.0, 5.0e-4])
def test_shunt_short_currents_satisfy_series_branch_law_and_kcl(
    components, transport_current
):
    arguments = shorted_shunt_arguments(**components)
    arguments["Gamma_i"][:] = transport_current / (arguments["A"] * e)
    result = step_circuit_mna(**arguments)
    gap, load_current, _, node, capacitor_voltage, series_current, shunt_current = result

    # With the node grounded by Lp=0, solve the remaining series branch directly.
    dt = arguments["dt"]
    capacitor_impedance = dt / arguments["C_s"]
    inductor_impedance = arguments["L_s"] / dt
    expected_series = (
        100.0 - arguments["V_Cs_prev"] + inductor_impedance * arguments["I_s_prev"]
    ) / (arguments["R0"] + capacitor_impedance + inductor_impedance)
    assert node == 0.0
    assert gap == 0.0
    assert series_current == pytest.approx(expected_series, rel=1.0e-12, abs=1.0e-14)
    assert load_current == pytest.approx(transport_current, rel=1.0e-12, abs=1.0e-14)
    assert shunt_current == pytest.approx(
        expected_series - transport_current, rel=1.0e-12, abs=1.0e-14
    )
    assert capacitor_voltage == pytest.approx(
        arguments["V_Cs_prev"] + capacitor_impedance * expected_series, abs=1.0e-12
    )


def test_open_series_capacitor_isolates_source_from_shunt_short():
    arguments = shorted_shunt_arguments(C_s=0.0)
    transport_current = 5.0e-4
    arguments["Gamma_i"][:] = transport_current / (arguments["A"] * e)
    result = step_circuit_mna(**arguments)
    assert result[5] == pytest.approx(0.0, abs=1.0e-14)
    assert result[6] == pytest.approx(-transport_current, abs=1.0e-14)


def test_shunt_short_current_includes_parallel_capacitor_discharge():
    arguments = shorted_shunt_arguments(R0=1.0e5, C_p=1.0e-9, V_n_prev=10.0)
    result = step_circuit_mna(**arguments)
    capacitor_current = arguments["C_p"] * (result[3] - arguments["V_n_prev"]) / arguments["dt"]
    assert result[5] == pytest.approx(1.0e-3, abs=1.0e-14)
    assert result[6] == pytest.approx(1.0e-3 - capacitor_current, abs=1.0e-14)


def test_parallel_ideal_source_and_shunt_shorts_preserve_current_convention():
    arguments = shorted_shunt_arguments(V_app_func=lambda t: 0.0)
    transport_current = 5.0e-4
    arguments["Gamma_i"][:] = transport_current / (arguments["A"] * e)
    result = step_circuit_mna(**arguments)
    # Two parallel ideal shorts cannot determine the current split uniquely.
    assert result[5] == pytest.approx(transport_current, abs=1.0e-14)
    assert result[6] == 0.0
