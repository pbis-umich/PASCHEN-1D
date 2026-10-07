"""Authoritative fail-fast validation for PASCHEN-1D configurations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from config import SimulationConfig
from data_paths import resolve_ion_swarm_data_file
from electron_transport import load_electron_swarm_table
from ion_transport import (
    load_ion_transport_table,
    validate_table_identity,
    validate_table_pair,
)
from numerics_jit import is_numba_available


ELECTRON_KINETICS_MODELS = {
    "user_defined_electron_kinetics",
    "local_field_approximation",
}
ION_KINETICS_MODELS = {"user_defined_ion_kinetics", "local_field_ion_kinetics"}
ELECTRON_SOURCES = {"user_defined_equation", "swarm_data_table_interpolation"}
ION_MOBILITY_SOURCES = {"user_defined_equation", "swarm_data_table_interpolation"}
ION_DIFFUSION_SOURCES = {
    "user_defined_equation",
    "swarm_data_table_interpolation",
    "einstein_relation",
}
IMPACT_MODELS = {"from_townsend_alpha", "from_ionization_frequency"}
TOWNSEND_SOURCES = {"user_defined_equation", "interpolate_from_e_over_n_table"}
RECOMBINATION_MODELS = {"user_defined_constant_coefficient"}
WAVEFORMS = {"step", "gaussian", "dc", "rf", "table", "tabulated", "measured_table"}
CIRCUIT_TYPES = {
    "dielectric_plasma",
    "R0_Cp",
    "R0_Cp_Rm",
    "R0_Rm_Cext",
    "R0_Cs_Cp",
    "R0_Cs_Cp_Rm",
    "R0_Cs_Ls_Cp",
    "R0_Cs_Ls_Cp_Rm",
    "R0_Cs_Ls_Cp_Lp",
    "R0_Cs_Ls_Cp_Lp_Rm_Cext",
}
CIRCUIT_SCHEMES = {"explicit_euler", "implicit_euler", "mna"}
OUT_OF_RANGE_POLICIES = {"clip", "error"}
VAUGHAN_EFFECTIVE_TEMPERATURE_MODES = {"fixed", "local_field_approximation"}
BOUNDARY_MODES = {"zero_density", "implicit_drift_closure", "electron_emission"}
TEMPORAL_QUANTITIES = {
    "V_app", "V_node", "V_source", "V_gap", "I_discharge",
    "I_transport_plasma", "I_transport_circuit", "I_emission_circuit",
    "I_emission_area", "I_displacement_gap", "cfl", "diffusion_cfl",
    "picard_iterations", "adaptive_substeps", "adaptive_dt_sub",
    "adaptive_cfl_est", "adaptive_diffusion_cfl_est", "particle_inventory",
}
SPATIAL_QUANTITIES = {
    "ne", "ni", "phi", "E", "Gamma_i", "Gamma_e", "townsend_alpha",
    "nu_i", "S_ion", "S", "mu_e", "D_e", "mu_i", "D_i",
}


def _append_choice(errors: list[str], label: str, value: Any, allowed: set[str]) -> None:
    if value not in allowed:
        errors.append(f"{label} must be one of {sorted(allowed)}; got {value!r}")


def _append_finite(errors: list[str], label: str, value: Any) -> None:
    try:
        valid = np.isfinite(float(value))
    except (TypeError, ValueError):
        valid = False
    if not valid:
        errors.append(f"{label} must be finite; got {value!r}")


def _append_positive(errors: list[str], label: str, value: Any) -> None:
    try:
        valid = np.isfinite(float(value)) and float(value) > 0.0
    except (TypeError, ValueError):
        valid = False
    if not valid:
        errors.append(f"{label} must be finite and > 0; got {value!r}")


def _append_nonnegative(errors: list[str], label: str, value: Any) -> None:
    try:
        valid = np.isfinite(float(value)) and float(value) >= 0.0
    except (TypeError, ValueError):
        valid = False
    if not valid:
        errors.append(f"{label} must be finite and >= 0; got {value!r}")


def _append_int_at_least(errors: list[str], label: str, value: Any, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        errors.append(f"{label} must be an integer >= {minimum}; got {value!r}")


def _validate_table_file(
    errors: list[str],
    cfg: SimulationConfig,
    value: str | None,
    label: str,
) -> None:
    if value is None or not str(value).strip():
        errors.append(f"{label} is required")
        return
    try:
        load_electron_swarm_table(
            value,
            configured_gas=cfg.plasma_state.gas,
            gas_temperature_K=cfg.plasma_state.T_i,
            temperature_tolerance_K=cfg.electron_swarm_data.gas_temperature_tolerance_K,
        )
    except (FileNotFoundError, ValueError) as exc:
        errors.append(f"{label}: {exc}")


def _validate_diagnostics(errors: list[str], cfg: SimulationConfig) -> None:
    temporal = cfg.diagnostics.temporal
    spatial = cfg.diagnostics.spatial
    averaged = cfg.diagnostics.averaged_spatial
    unknown_temporal = sorted(set(temporal.quantities) - TEMPORAL_QUANTITIES)
    unknown_spatial = sorted(set(spatial.quantities) - SPATIAL_QUANTITIES)
    unknown_averaged = sorted(set(averaged.quantities) - SPATIAL_QUANTITIES)
    if unknown_temporal:
        errors.append(f"diagnostics.temporal.quantities contains {unknown_temporal}")
    if unknown_spatial:
        errors.append(f"diagnostics.spatial.quantities contains {unknown_spatial}")
    if unknown_averaged:
        errors.append(f"diagnostics.averaged_spatial.quantities contains {unknown_averaged}")
    for label, value in (
        ("diagnostics.temporal.t_start", temporal.t_start),
        ("diagnostics.temporal.t_end", temporal.t_end),
        ("diagnostics.averaged_spatial.t_avg_start", averaged.t_avg_start),
        ("diagnostics.averaged_spatial.t_avg_end", averaged.t_avg_end),
    ):
        if value is not None:
            _append_finite(errors, label, value)
    if temporal.t_start is not None and temporal.t_end is not None and temporal.t_end < temporal.t_start:
        errors.append("diagnostics.temporal.t_end must be >= t_start")
    if (
        averaged.t_avg_start is not None
        and averaged.t_avg_end is not None
        and averaged.t_avg_end < averaged.t_avg_start
    ):
        errors.append("diagnostics.averaged_spatial.t_avg_end must be >= t_avg_start")
    if spatial.x_unit not in {"m", "cm", "mm"}:
        errors.append("diagnostics.spatial.x_unit must be 'm', 'cm', or 'mm'")
    if averaged.x_unit not in {"m", "cm", "mm"}:
        errors.append("diagnostics.averaged_spatial.x_unit must be 'm', 'cm', or 'mm'")
    _append_choice(
        errors,
        "diagnostics.averaged_spatial.mode",
        averaged.mode,
        {"time_window", "last_n_cycles"},
    )
    _append_int_at_least(
        errors, "diagnostics.averaged_spatial.N_cycle_avg", averaged.N_cycle_avg, 1
    )


def _enabled_components(cfg: SimulationConfig, electrode: str) -> tuple[str, ...]:
    names = ("constant_J", "fn", "mg", "rd", "quantum_pulse")
    return tuple(
        name
        for name in names
        if bool(getattr(cfg.emission, f"{electrode}_enable_{name}_emission"))
    )


def _validate_emission(errors: list[str], cfg: SimulationConfig) -> None:
    emission = cfg.emission
    _append_choice(
        errors,
        "emission.electrode_material_mode",
        emission.electrode_material_mode,
        {"shared", "separate"},
    )
    _append_nonnegative(errors, "emission.gamma", emission.gamma)
    _append_nonnegative(
        errors, "emission.anode_electron_induced_yield", emission.anode_electron_induced_yield
    )
    _append_choice(
        errors,
        "emission.vaughan_effective_temperature_mode",
        emission.vaughan_effective_temperature_mode,
        VAUGHAN_EFFECTIVE_TEMPERATURE_MODES,
    )
    _append_positive(errors, "emission.vaughan_Emax0_eV", emission.vaughan_Emax0_eV)
    _append_nonnegative(errors, "emission.vaughan_dmax0", emission.vaughan_dmax0)
    _append_positive(errors, "emission.vaughan_ks", emission.vaughan_ks)

    any_external = False
    for electrode in ("anode", "cathode"):
        components = _enabled_components(cfg, electrode)
        electrode_enabled = bool(
            getattr(emission, f"enable_{electrode}_external_emission")
        )
        any_external = any_external or electrode_enabled or bool(components)
        if components and not electrode_enabled:
            errors.append(
                f"{electrode} emission components {components} are enabled while "
                f"enable_{electrode}_external_emission is False"
            )
        if electrode_enabled and not components:
            errors.append(
                f"enable_{electrode}_external_emission is True but no {electrode} "
                "external-emission component is enabled"
            )
        if electrode_enabled and (
            getattr(cfg.boundary, f"{electrode}_electron_boundary") != "electron_emission"
        ):
            errors.append(
                f"{electrode} external emission requires boundary.{electrode}_electron_boundary="
                "'electron_emission'"
            )

    if any_external and not emission.enable_external_emission:
        errors.append(
            "emission.enable_external_emission is False while electrode external-emission "
            "toggles or components are enabled"
        )
    if emission.enable_external_emission and not any(
        bool(getattr(emission, f"enable_{side}_external_emission"))
        for side in ("anode", "cathode")
    ):
        errors.append("emission.enable_external_emission is True but both electrodes are disabled")

    prefixes: Iterable[str]
    prefixes = ("shared",) if emission.electrode_material_mode == "shared" else ("anode", "cathode")
    for prefix in prefixes:
        for suffix in (
            "fn_work_function_eV", "fn_field_scale_factor", "mg_work_function_eV",
            "mg_field_scale_factor", "rd_emitter_K", "rd_work_function_eV",
            "emission_T", "emission_W_eV", "emission_Ef_eV", "emission_epsilon0_eV",
            "emission_lambda_m", "laser_tau_p_s", "laser_t_window_ps",
            "emission_dt_ps", "laser_wx_m", "laser_wy_m",
        ):
            _append_positive(errors, f"emission.{prefix}_{suffix}", getattr(emission, f"{prefix}_{suffix}"))
        for suffix in ("rd_A_R", "emission_J_const", "laser_U_J"):
            _append_nonnegative(errors, f"emission.{prefix}_{suffix}", getattr(emission, f"{prefix}_{suffix}"))
        for suffix in ("emission_t_start", "emission_t_end", "laser_t0"):
            _append_finite(errors, f"emission.{prefix}_{suffix}", getattr(emission, f"{prefix}_{suffix}"))
        if getattr(emission, f"{prefix}_emission_t_end") < getattr(emission, f"{prefix}_emission_t_start"):
            errors.append(f"emission.{prefix}_emission_t_end must be >= emission_t_start")
        f_min = float(getattr(emission, f"{prefix}_mg_f_clip_min"))
        f_max = float(getattr(emission, f"{prefix}_mg_f_clip_max"))
        if not (0.0 < f_min < f_max < 1.0):
            errors.append(f"emission.{prefix}_mg_f_clip values must satisfy 0 < min < max < 1")
        _append_int_at_least(errors, f"emission.{prefix}_emission_k_ph", getattr(emission, f"{prefix}_emission_k_ph"), 1)
        _append_int_at_least(errors, f"emission.{prefix}_emission_eps_points", getattr(emission, f"{prefix}_emission_eps_points"), 1)
        _append_int_at_least(errors, f"emission.{prefix}_emission_wt_points", getattr(emission, f"{prefix}_emission_wt_points"), 2)


def _validate_circuit(errors: list[str], cfg: SimulationConfig) -> None:
    circuit = cfg.circuit
    _append_choice(errors, "circuit.circuit_type", circuit.circuit_type, CIRCUIT_TYPES)
    _append_choice(
        errors, "circuit.circuit_time_scheme", circuit.circuit_time_scheme, CIRCUIT_SCHEMES
    )
    if circuit.circuit_time_scheme == "mna" and circuit.circuit_type != "R0_Cs_Ls_Cp_Lp_Rm_Cext":
        errors.append(
            "circuit_time_scheme='mna' requires circuit_type='R0_Cs_Ls_Cp_Lp_Rm_Cext'"
        )
    if circuit.circuit_time_scheme == "mna":
        for label in ("R0", "L_s", "C_p", "R_m", "C_ext"):
            _append_nonnegative(errors, f"circuit.{label}", getattr(circuit, label))
        for label in ("C_s", "L_p"):
            value = float(getattr(circuit, label))
            if value < 0.0 or np.isnan(value):
                errors.append(f"circuit.{label} must be >= 0 or infinity")
        return

    for label in ("R0", "C_s", "L_s", "C_p", "L_p", "R_m", "C_ext"):
        _append_nonnegative(errors, f"circuit.{label}", getattr(circuit, label))
    required_positive = {
        "R0_Cp": ("R0",),
        "R0_Cp_Rm": ("R0", "C_p", "R_m"),
        "R0_Cs_Cp": ("R0", "C_s", "C_p"),
        "R0_Cs_Cp_Rm": ("R0", "C_s", "C_p", "R_m"),
        "R0_Cs_Ls_Cp": ("R0", "C_s", "L_s", "C_p"),
        "R0_Cs_Ls_Cp_Rm": ("R0", "C_s", "L_s", "C_p", "R_m"),
        "R0_Cs_Ls_Cp_Lp": ("R0", "C_s", "L_s", "C_p", "L_p"),
        "R0_Cs_Ls_Cp_Lp_Rm_Cext": (
            "R0", "C_s", "L_s", "C_p", "L_p", "R_m"
        ),
    }
    for label in required_positive.get(circuit.circuit_type, ()):
        _append_positive(errors, f"circuit.{label}", getattr(circuit, label))
    if circuit.circuit_type == "R0_Rm_Cext" and circuit.R0 + circuit.R_m <= 0.0:
        errors.append("circuit.R0_Rm_Cext requires R0 + R_m > 0")
    if circuit.circuit_type == "dielectric_plasma" and cfg.geometry.l <= 0.0:
        errors.append("circuit.dielectric_plasma requires geometry.l > 0")


def validate_simulation_config(cfg: SimulationConfig) -> None:
    """Validate the complete public configuration before any output is allocated."""
    errors: list[str] = []
    run_name = str(cfg.run.run_name).strip()
    run_path = Path(run_name)
    if not run_name or run_path.is_absolute() or ".." in run_path.parts:
        errors.append("run.run_name must be a non-empty portable relative directory name")
    _append_positive(errors, "run.T_total", cfg.run.T_total)

    _append_int_at_least(errors, "numerics.Nt", cfg.numerics.Nt, 2)
    _append_int_at_least(errors, "numerics.Nx", cfg.numerics.Nx, 3)
    _append_finite(errors, "numerics.kt_limiter_theta", cfg.numerics.kt_limiter_theta)
    if not (1.0 <= float(cfg.numerics.kt_limiter_theta) <= 2.0):
        errors.append("numerics.kt_limiter_theta must be between 1 and 2")
    _append_choice(errors, "numerics.hotloop_backend", cfg.numerics.hotloop_backend, {"numpy", "numba"})
    if cfg.numerics.hotloop_backend == "numba" and not is_numba_available():
        errors.append("numerics.hotloop_backend='numba' requested but Numba is unavailable")
    _append_positive(errors, "numerics.target_cfl_substep", cfg.numerics.target_cfl_substep)
    _append_positive(
        errors,
        "numerics.target_diffusion_cfl_substep",
        cfg.numerics.target_diffusion_cfl_substep,
    )
    _append_int_at_least(errors, "numerics.max_substeps", cfg.numerics.max_substeps, 1)
    _append_choice(
        errors,
        "numerics.adaptive_substep_overflow_policy",
        cfg.numerics.adaptive_substep_overflow_policy,
        {"warn_and_cap", "error"},
    )
    _append_int_at_least(
        errors, "numerics.adaptive_substep_warn_every", cfg.numerics.adaptive_substep_warn_every, 1
    )
    _append_int_at_least(
        errors, "numerics.bc_poisson_picard_min_iter", cfg.numerics.bc_poisson_picard_min_iter, 1
    )
    _append_int_at_least(
        errors, "numerics.bc_poisson_picard_max_iter", cfg.numerics.bc_poisson_picard_max_iter, 1
    )
    if cfg.numerics.bc_poisson_picard_min_iter > cfg.numerics.bc_poisson_picard_max_iter:
        errors.append("numerics.bc_poisson_picard_min_iter must be <= max_iter")
    _append_positive(errors, "numerics.bc_poisson_picard_tol", cfg.numerics.bc_poisson_picard_tol)

    _append_positive(errors, "geometry.L", cfg.geometry.L)
    _append_positive(errors, "geometry.A", cfg.geometry.A)
    _append_nonnegative(errors, "geometry.l", cfg.geometry.l)
    _append_positive(errors, "geometry.eps_r", cfg.geometry.eps_r)
    if not str(cfg.plasma_state.gas).strip():
        errors.append("plasma_state.gas must be non-empty")
    _append_positive(errors, "plasma_state.p_Torr", cfg.plasma_state.p_Torr)
    _append_positive(errors, "plasma_state.T_e", cfg.plasma_state.T_e)
    _append_positive(errors, "plasma_state.T_i", cfg.plasma_state.T_i)
    _append_nonnegative(errors, "plasma_state.n0", cfg.plasma_state.n0)

    _append_choice(
        errors, "plasma.electron_kinetics_model", cfg.plasma.electron_kinetics_model,
        ELECTRON_KINETICS_MODELS,
    )
    _append_choice(errors, "plasma.ion_kinetics_model", cfg.plasma.ion_kinetics_model, ION_KINETICS_MODELS)
    _append_choice(errors, "plasma.impact_ionization_model", cfg.plasma.impact_ionization_model, IMPACT_MODELS)
    _append_choice(errors, "plasma.recombination_model", cfg.plasma.recombination_model, RECOMBINATION_MODELS)
    _append_choice(
        errors,
        "local_field_approximation.electron_transport_source",
        cfg.local_field_approximation.electron_transport_source,
        ELECTRON_SOURCES,
    )
    _append_choice(
        errors, "electron_swarm_data.out_of_range_policy",
        cfg.electron_swarm_data.out_of_range_policy, OUT_OF_RANGE_POLICIES,
    )
    _append_nonnegative(
        errors, "electron_swarm_data.gas_temperature_tolerance_K",
        cfg.electron_swarm_data.gas_temperature_tolerance_K,
    )
    _append_choice(
        errors, "townsend_coefficient.townsend_alpha_source_mode",
        cfg.townsend_coefficient.townsend_alpha_source_mode, TOWNSEND_SOURCES,
    )
    _append_choice(
        errors, "ionization_frequency_source.ionization_frequency_source_mode",
        cfg.ionization_frequency_source.ionization_frequency_source_mode, TOWNSEND_SOURCES,
    )
    _append_nonnegative(errors, "recombination.recombination_coefficient", cfg.recombination.recombination_coefficient)

    empirical_gases = {"argon", "nitrogen"}
    gas_key = str(cfg.plasma_state.gas).strip().lower()
    uses_empirical_electron_transport = (
        cfg.plasma.electron_kinetics_model == "user_defined_electron_kinetics"
        or (
            cfg.plasma.electron_kinetics_model == "local_field_approximation"
            and cfg.local_field_approximation.electron_transport_source == "user_defined_equation"
        )
    )
    if uses_empirical_electron_transport and gas_key not in empirical_gases:
        errors.append("user-defined electron transport is implemented only for argon and nitrogen")
    uses_electron_transport_table = (
        cfg.plasma.electron_kinetics_model == "local_field_approximation"
        and cfg.local_field_approximation.electron_transport_source == "swarm_data_table_interpolation"
    )
    if uses_electron_transport_table:
        _validate_table_file(
            errors, cfg, cfg.local_field_approximation.electron_swarm_data_path,
            "local_field_approximation.electron_swarm_data_path",
        )
    if (
        cfg.emission.use_vaughan_sey
        and cfg.emission.vaughan_effective_temperature_mode
        == "local_field_approximation"
        and not uses_electron_transport_table
    ):
        _validate_table_file(
            errors,
            cfg,
            cfg.local_field_approximation.electron_swarm_data_path,
            "local_field_approximation.electron_swarm_data_path",
        )
    if (
        cfg.plasma.impact_ionization_model == "from_townsend_alpha"
        and cfg.townsend_coefficient.townsend_alpha_source_mode == "interpolate_from_e_over_n_table"
    ):
        _validate_table_file(
            errors, cfg, cfg.townsend_coefficient.townsend_alpha_swarm_data_path,
            "townsend_coefficient.townsend_alpha_swarm_data_path",
        )
    elif cfg.plasma.impact_ionization_model == "from_townsend_alpha" and gas_key not in empirical_gases:
        errors.append("user-defined Townsend alpha is implemented only for argon and nitrogen")
    if (
        cfg.plasma.impact_ionization_model == "from_ionization_frequency"
        and cfg.ionization_frequency_source.ionization_frequency_source_mode == "interpolate_from_e_over_n_table"
    ):
        _validate_table_file(
            errors, cfg,
            cfg.ionization_frequency_source.ionization_frequency_swarm_data_path,
            "ionization_frequency_source.ionization_frequency_swarm_data_path",
        )
    elif cfg.plasma.impact_ionization_model == "from_ionization_frequency" and gas_key not in empirical_gases:
        errors.append("user-defined ionization frequency is implemented only for argon and nitrogen")

    ion = cfg.ion_transport
    if not str(ion.positive_ion).strip() or "+" not in str(ion.positive_ion) or "-" in str(ion.positive_ion):
        errors.append("ion_transport.positive_ion must identify a positive ion")
    _append_choice(errors, "ion_transport.mobility_source_mode", ion.mobility_source_mode, ION_MOBILITY_SOURCES)
    _append_choice(errors, "ion_transport.diffusion_source_mode", ion.diffusion_source_mode, ION_DIFFUSION_SOURCES)
    _append_choice(errors, "ion_transport.out_of_range_policy", ion.out_of_range_policy, OUT_OF_RANGE_POLICIES)
    _append_nonnegative(errors, "ion_transport.gas_temperature_tolerance_K", ion.gas_temperature_tolerance_K)
    if cfg.plasma.ion_kinetics_model == "user_defined_ion_kinetics":
        if ion.mobility_source_mode != "user_defined_equation" or ion.diffusion_source_mode != "user_defined_equation":
            errors.append("user_defined_ion_kinetics requires both ion sources to be user_defined_equation")
        if gas_key not in empirical_gases:
            errors.append("user-defined ion transport is implemented only for argon and nitrogen")
    else:
        mobility_table = None
        diffusion_table = None
        if ion.mobility_source_mode == "user_defined_equation" and gas_key not in empirical_gases:
            errors.append("user-defined ion mobility is implemented only for argon and nitrogen")
        if ion.diffusion_source_mode == "user_defined_equation" and gas_key not in empirical_gases:
            errors.append("user-defined ion diffusion is implemented only for argon and nitrogen")
        if ion.mobility_source_mode == "swarm_data_table_interpolation":
            try:
                mobility_table = load_ion_transport_table(resolve_ion_swarm_data_file(ion.mobility_table_path or ""))
                validate_table_identity(
                    mobility_table, expected_quantity="reduced_mobility",
                    configured_ion=ion.positive_ion, configured_neutral=cfg.plasma_state.gas,
                    gas_temperature_K=cfg.plasma_state.T_i,
                    temperature_tolerance_K=ion.gas_temperature_tolerance_K,
                )
            except (FileNotFoundError, ValueError) as exc:
                errors.append(f"ion_transport.mobility_table_path: {exc}")
        if ion.diffusion_source_mode == "swarm_data_table_interpolation":
            try:
                diffusion_table = load_ion_transport_table(resolve_ion_swarm_data_file(ion.diffusion_table_path or ""))
                validate_table_identity(
                    diffusion_table, expected_quantity="reduced_longitudinal_diffusion",
                    configured_ion=ion.positive_ion, configured_neutral=cfg.plasma_state.gas,
                    gas_temperature_K=cfg.plasma_state.T_i,
                    temperature_tolerance_K=ion.gas_temperature_tolerance_K,
                )
            except (FileNotFoundError, ValueError) as exc:
                errors.append(f"ion_transport.diffusion_table_path: {exc}")
        if mobility_table is not None and diffusion_table is not None:
            try:
                validate_table_pair(mobility_table, diffusion_table)
            except (FileNotFoundError, ValueError) as exc:
                errors.append(f"ion_transport table pair: {exc}")

    _append_choice(errors, "waveform.waveform_type", cfg.waveform.waveform_type, WAVEFORMS)
    for label in ("V_peak", "tV_start", "tV_end", "tau", "t_peak", "f_rf", "V_dc", "phi_rf", "table_time_scale", "table_time_offset", "table_voltage_scale", "table_voltage_offset"):
        _append_finite(errors, f"waveform.{label}", getattr(cfg.waveform, label))
    if cfg.waveform.waveform_type == "step" and cfg.waveform.tV_end < cfg.waveform.tV_start:
        errors.append("waveform.tV_end must be >= tV_start for a step waveform")
    if cfg.waveform.waveform_type == "gaussian" and cfg.waveform.tau <= 0.0:
        errors.append("waveform.tau must be > 0 for a Gaussian waveform")
    if cfg.waveform.waveform_type == "rf" and cfg.waveform.f_rf <= 0.0:
        errors.append("waveform.f_rf must be > 0 for an RF waveform")
    if cfg.waveform.waveform_type in {"table", "tabulated", "measured_table"}:
        table_path = Path(cfg.waveform.table_path)
        if table_path.is_absolute() or ".." in table_path.parts:
            errors.append("waveform.table_path must be portable and relative to the project")
        if cfg.waveform.table_time_column < 0 or cfg.waveform.table_voltage_column < 0:
            errors.append("waveform table column indices must be >= 0")
        if cfg.waveform.table_time_column == cfg.waveform.table_voltage_column:
            errors.append("waveform table time and voltage columns must differ")
        if cfg.waveform.table_time_scale == 0.0:
            errors.append("waveform.table_time_scale must be nonzero")

    boundaries = (
        ("anode_ion_boundary", cfg.boundary.anode_ion_boundary),
        ("anode_electron_boundary", cfg.boundary.anode_electron_boundary),
        ("cathode_ion_boundary", cfg.boundary.cathode_ion_boundary),
        ("cathode_electron_boundary", cfg.boundary.cathode_electron_boundary),
    )
    for label, value in boundaries:
        _append_choice(errors, f"boundary.{label}", value, BOUNDARY_MODES)
    if cfg.boundary.anode_ion_boundary != "zero_density":
        errors.append("boundary.anode_ion_boundary currently supports only 'zero_density'")
    if cfg.boundary.cathode_ion_boundary not in {"zero_density", "implicit_drift_closure"}:
        errors.append("boundary.cathode_ion_boundary must be zero_density or implicit_drift_closure")
    if cfg.boundary.cathode_electron_boundary not in {"zero_density", "electron_emission"}:
        errors.append("boundary.cathode_electron_boundary must be zero_density or electron_emission")
    if not cfg.boundary.enable_volume_sources and (
        cfg.boundary.enable_ionization_source or cfg.boundary.enable_recombination_sink
    ):
        errors.append("volume-source sub-toggles must be False when enable_volume_sources is False")

    _validate_circuit(errors, cfg)
    _validate_emission(errors, cfg)
    _append_int_at_least(errors, "output.save_every", cfg.output.save_every, 1)
    _validate_diagnostics(errors, cfg)

    if errors:
        raise ValueError("Invalid PASCHEN-1D configuration:\n- " + "\n- ".join(errors))
