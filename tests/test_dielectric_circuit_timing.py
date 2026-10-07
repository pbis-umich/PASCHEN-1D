from __future__ import annotations

import contextlib
import io
from unittest.mock import patch

import numpy as np
import pytest

from circuit_implicit_euler import step_circuit_implicit_euler
from config_case_nitrogen_pulsed_discharge import SimulationConfig
from paschen_1d import run_simulation
from physical_constants import e
from release_test_utils import make_miniature_config


@pytest.mark.parametrize("transport_current", [0.0, 1.0e-3])
def test_implicit_dielectric_preserves_source_mapping_at_step_endpoint(transport_current):
    """The caller supplies the new time, including when substep sizes change."""
    area = 1.0e-4
    length = 1.0e-3
    thickness = 1.0e-4
    alpha = 1.0 + 2.0 * thickness / (4.0 * length)
    source = lambda t: 100.0 * (t / 1.0e-8) ** 2
    gap = dielectric = time = 0.0
    for dt in (1.0e-9, 0.5e-9, 2.0e-9, 3.0e-9, 0.25e-9, 3.25e-9):
        time += dt
        gap, _, dielectric, *_ = step_circuit_implicit_euler(
            circuit_type="dielectric_plasma",
            V_app_func=source,
            t=time,
            dt=dt,
            V_gap_prev=gap,
            Gamma_i=np.full(5, transport_current / (area * e)),
            Gamma_e=np.zeros(5),
            dx=length / 4.0,
            A=area,
            L=length,
            l=thickness,
            eps_r=4.0,
            R0=0.0,
            C_s=0.0,
            C_p=0.0,
            R_m=0.0,
            L_s=0.0,
            L_p=0.0,
            V_d_prev=dielectric,
            V_n_prev=None,
            V_Cs_prev=None,
            I_s_prev=None,
            I_Lp_prev=None,
        )
        assert alpha * gap + dielectric == pytest.approx(source(time), abs=1.0e-12)


def test_solver_implicit_dielectric_tracks_nonlinear_source(tmp_path):
    cfg = make_miniature_config(SimulationConfig(), run_name="dielectric_timing")
    cfg.plasma_state.n0 = 0.0
    cfg.circuit.circuit_time_scheme = "implicit_euler"
    source_times = []

    def source(time):
        source_times.append(float(time))
        return 100.0 * (time / cfg.run.T_total) ** 2

    with (
        contextlib.chdir(tmp_path),
        contextlib.redirect_stdout(io.StringIO()),
        patch("paschen_1d.make_voltage_waveform", return_value=source),
        patch("paschen_1d.tqdm", lambda iterable, **kwargs: iterable),
    ):
        state = run_simulation(cfg)

    alpha = 1.0 + 2.0 * cfg.geometry.l / (cfg.geometry.eps_r * cfg.geometry.L)
    expected_source = 100.0 * (state.time / cfg.run.T_total) ** 2
    np.testing.assert_allclose(alpha * state.V_gap, expected_source, rtol=1.0e-12, atol=1.0e-12)
    assert max(source_times) <= cfg.run.T_total * (1.0 + 1.0e-12)
