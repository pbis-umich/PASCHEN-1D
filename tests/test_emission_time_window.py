from __future__ import annotations

import json

import numpy as np
import pytest

import paschen_1d
from emission import EmissionModel, make_constant_J_emitter, make_emitter_ps_window
from release_test_utils import make_miniature_config


@pytest.mark.parametrize("electrode", ("anode", "cathode"))
@pytest.mark.parametrize("pulse_start_ps", (0.0, 1.0, 2.0))
@pytest.mark.parametrize("adaptive", (False, True))
def test_solver_injects_pulse_charge_only_during_advanced_intervals(
    tmp_path, monkeypatch, electrode: str, pulse_start_ps: float, adaptive: bool
) -> None:
    cfg = make_miniature_config(run_name="emission_window")
    cfg.run.T_total = 2.0e-12
    cfg.numerics.Nt = 3
    cfg.numerics.use_adaptive_substepping = adaptive
    # The initial diffusion CFL is about 5.36e-5, requiring two substeps.
    cfg.numerics.target_diffusion_cfl_substep = 3.0e-5
    cfg.plasma_state.n0 = 0.0
    cfg.waveform.V_peak = 0.0
    cfg.boundary.enable_volume_sources = False
    cfg.boundary.enable_ionization_source = False
    cfg.boundary.enable_recombination_sink = False
    cfg.emission.gamma = 0.0
    cfg.emission.anode_electron_induced_yield = 0.0
    cfg.emission.enable_external_emission = True
    setattr(cfg.emission, f"enable_{electrode}_external_emission", True)
    setattr(cfg.emission, f"{electrode}_enable_quantum_pulse_emission", True)
    setattr(cfg.boundary, f"{electrode}_electron_boundary", "electron_emission")

    # A 1 A/m² square pulse lasting 1 ps has a known charge independent of
    # plasma evolution. The final case lies entirely after this simulation.
    pulse = make_emitter_ps_window(
        np.array([1.0, 0.0]),
        np.array([pulse_start_ps, pulse_start_ps + 1.0]),
        runtime_unit="s",
    )

    def emitter(t, V_gap, dt, E_surface):
        return pulse(t, V_gap, dt)

    model = EmissionModel(cfg, **{f"emitter_func_{electrode}": emitter})
    monkeypatch.setattr(paschen_1d, "build_emission_model", lambda cfg: model)
    monkeypatch.setattr(paschen_1d, "tqdm", lambda iterable, **kwargs: iterable)
    monkeypatch.chdir(tmp_path)
    state = paschen_1d.run_simulation(cfg)
    np.testing.assert_array_equal(state.adaptive_substeps[1:], 2.0 if adaptive else 1.0)

    stats = json.loads(
        (tmp_path / cfg.run.run_name / "surface_emission_charge_stats.json").read_text()
    )
    expected_charge = cfg.geometry.A * 1.0e-12 if pulse_start_ps < 2.0 else 0.0
    sign = 1.0 if electrode == "cathode" else -1.0
    np.testing.assert_allclose(
        stats["Q_emit_external_signed_C"], sign * expected_charge,
        rtol=1.0e-13, atol=0.0,
    )
    np.testing.assert_allclose(
        stats[f"Q_injected_surface_{electrode}_abs_C"], expected_charge,
        rtol=1.0e-13, atol=0.0,
    )
    np.testing.assert_allclose(
        stats["Q_injected_surface_total_abs_C"], expected_charge,
        rtol=1.0e-13, atol=0.0,
    )
    expected_current = np.zeros(2)
    if pulse_start_ps < 2.0:
        expected_current[int(pulse_start_ps)] = sign * cfg.geometry.A
    np.testing.assert_allclose(
        state.I_emission_area[:-1], expected_current, rtol=1.0e-13, atol=0.0
    )
    assert np.all(np.isfinite(state.ne_final))
    assert np.any(state.ne_final > 0.0) == (expected_charge > 0.0)


@pytest.mark.parametrize(
    "kwargs", [{}, {"dt_run": None}, {"dt_run": 0.0}, {"dt_run": -1.0}],
    ids=["omitted", "none", "zero", "negative"],
)
def test_constant_emitter_retains_inclusive_point_sampling(kwargs) -> None:
    emitter = make_constant_J_emitter(2.0, t_start=1.0, t_end=2.0)
    assert [emitter(t, **kwargs) for t in (0.0, 1.0, 2.0, 3.0)] == [0.0, 2.0, 2.0, 0.0]


@pytest.mark.parametrize(
    "t_end, time, dt, expected",
    [
        (2.0, 0.0, 0.5, 0.0),
        (2.0, 0.0, 1.0, 0.0),
        (2.0, 0.0, 1.5, 2.0 / 3.0),
        (2.0, 1.0, 0.25, 2.0),
        (2.0, 1.0, 1.0, 2.0),
        (2.0, 1.5, 1.0, 1.0),
        (2.0, 2.0, 1.0, 0.0),
        (2.0, 0.0, 3.0, 2.0 / 3.0),
        (2.0, 3.0, 1.0, 0.0),
        (None, 0.0, 2.0, 1.0),
        (None, 1.0, 2.0, 2.0),
        (None, 2.0, 1.0, 2.0),
    ],
)
def test_constant_emitter_averages_interval_overlap(t_end, time, dt, expected) -> None:
    emitter = make_constant_J_emitter(2.0, t_start=1.0, t_end=t_end)
    assert emitter(time, dt_run=dt) == pytest.approx(expected)


@pytest.mark.parametrize("electrode", ("anode", "cathode"))
@pytest.mark.parametrize("adaptive", (False, True))
@pytest.mark.parametrize(
    "start_ps, end_ps, duration_ps, fixed_current, adaptive_current",
    [
        (0.0, 1.0, 1.0, [1.0, 0.0], [1.0, 0.0]),
        (1.0, 2.0, 1.0, [0.0, 1.0], [0.0, 1.0]),
        (0.25, 1.25, 1.0, [0.75, 0.25], [1.0, 0.0]),
        (0.25, 0.75, 0.5, [0.5, 0.0], [0.5, 0.0]),
        (0.25, None, 1.75, [0.75, 1.0], [1.0, 1.0]),
        (2.0, 3.0, 0.0, [0.0, 0.0], [0.0, 0.0]),
    ],
    ids=["exact-end", "exact-start", "off-grid", "short", "open-ended", "after-run"],
)
def test_solver_integrates_constant_emission_window(
    tmp_path, monkeypatch, electrode, adaptive,
    start_ps, end_ps, duration_ps, fixed_current, adaptive_current,
) -> None:
    cfg = make_miniature_config(run_name="constant_emission_window")
    cfg.run.T_total = 2.0e-12
    cfg.numerics.Nt = 3
    cfg.numerics.use_adaptive_substepping = adaptive
    cfg.numerics.target_diffusion_cfl_substep = 3.0e-5
    cfg.plasma_state.n0 = 0.0
    cfg.waveform.V_peak = 0.0
    cfg.boundary.enable_volume_sources = False
    cfg.boundary.enable_ionization_source = False
    cfg.boundary.enable_recombination_sink = False
    cfg.emission.gamma = 0.0
    cfg.emission.anode_electron_induced_yield = 0.0
    cfg.emission.enable_external_emission = True
    setattr(cfg.emission, f"enable_{electrode}_external_emission", True)
    setattr(cfg.emission, f"{electrode}_enable_constant_J_emission", True)
    setattr(cfg.boundary, f"{electrode}_electron_boundary", "electron_emission")
    cfg.emission.shared_emission_J_const = 1.0
    cfg.emission.shared_emission_t_start = start_ps * 1.0e-12
    # The factory treats end <= start as an open-ended emission window.
    cfg.emission.shared_emission_t_end = (start_ps if end_ps is None else end_ps) * 1.0e-12

    monkeypatch.setattr(paschen_1d, "tqdm", lambda iterable, **kwargs: iterable)
    monkeypatch.chdir(tmp_path)
    state = paschen_1d.run_simulation(cfg)
    np.testing.assert_array_equal(state.adaptive_substeps[1:], 2.0 if adaptive else 1.0)
    stats = json.loads(
        (tmp_path / cfg.run.run_name / "surface_emission_charge_stats.json").read_text()
    )
    expected_charge = cfg.geometry.A * duration_ps * 1.0e-12
    sign = 1.0 if electrode == "cathode" else -1.0
    np.testing.assert_allclose(
        stats["Q_emit_external_signed_C"], sign * expected_charge,
        rtol=1.0e-13, atol=0.0,
    )
    np.testing.assert_allclose(
        stats[f"Q_injected_surface_{electrode}_abs_C"], expected_charge,
        rtol=1.0e-13, atol=0.0,
    )
    np.testing.assert_allclose(
        stats["Q_injected_external_signed_C"], sign * expected_charge,
        rtol=1.0e-13, atol=0.0,
    )
    # This diagnostic stores the final substep's current for each macrostep.
    expected_current = sign * cfg.geometry.A * np.array(
        adaptive_current if adaptive else fixed_current
    )
    np.testing.assert_allclose(
        state.I_emission_area[:-1], expected_current, rtol=1.0e-13, atol=0.0
    )
