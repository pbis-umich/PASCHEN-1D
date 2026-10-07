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


def test_constant_emitter_retains_inclusive_point_sampling() -> None:
    emitter = make_constant_J_emitter(2.0, t_start=1.0, t_end=2.0)
    assert [emitter(t, dt_run=3.0) for t in (0.0, 1.0, 2.0, 3.0)] == [0.0, 2.0, 2.0, 0.0]


@pytest.mark.parametrize("electrode", ("anode", "cathode"))
@pytest.mark.parametrize("pulse_start", (0.0, 2.0e-12))
def test_solver_samples_constant_emission_at_substep_start(
    tmp_path, monkeypatch, electrode: str, pulse_start: float
) -> None:
    cfg = make_miniature_config(run_name="constant_emission_window")
    cfg.run.T_total = 2.0e-12
    cfg.numerics.Nt = 3
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
    cfg.emission.shared_emission_t_start = pulse_start
    # The emitter includes its endpoint; place it just before the next step.
    cfg.emission.shared_emission_t_end = np.nextafter(pulse_start + 1.0e-12, pulse_start)

    monkeypatch.setattr(paschen_1d, "tqdm", lambda iterable, **kwargs: iterable)
    monkeypatch.chdir(tmp_path)
    state = paschen_1d.run_simulation(cfg)
    stats = json.loads(
        (tmp_path / cfg.run.run_name / "surface_emission_charge_stats.json").read_text()
    )
    expected_charge = cfg.geometry.A * 1.0e-12 if pulse_start == 0.0 else 0.0
    sign = 1.0 if electrode == "cathode" else -1.0
    np.testing.assert_allclose(
        stats["Q_emit_external_signed_C"], sign * expected_charge,
        rtol=1.0e-13, atol=0.0,
    )
    np.testing.assert_allclose(
        stats[f"Q_injected_surface_{electrode}_abs_C"], expected_charge,
        rtol=1.0e-13, atol=0.0,
    )
    expected_current = [sign * cfg.geometry.A if pulse_start == 0.0 else 0.0, 0.0]
    np.testing.assert_allclose(
        state.I_emission_area[:-1], expected_current, rtol=1.0e-13, atol=0.0
    )
