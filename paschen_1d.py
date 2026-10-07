"""
paschen_1d.py

High-level simulation driver for the 1D drift-diffusion-Poisson model.

This module orchestrates:
- grid/time construction,
- waveform and external-circuit coupling,
- transport setup and initial conditions,
- KT+RK4 continuity updates for n_e and n_i,
- Poisson solve with Dirichlet boundaries (phi_left=V_gap, phi_right=0),
- optional anode/cathode external emission models,
- drift/diffusion stability diagnostics and snapshot output.

Typical usage (from command line):

    $ python paschen_1d.py

or from another script:

    from config_case_argon_dc_discharge import SimulationConfig
    from paschen_1d import run_simulation
    cfg = SimulationConfig()
    state = run_simulation(cfg)

For notebook-driven case selection, see run_paschen_1d.ipynb. The notebook uses
config_loader.py to prepare the selected case config before importing solver
modules, so case-specific configs do not require a separate config.py file.

The __main__ block provides a simple default test configuration.

Run-flow map (quick onboarding):
1. Build grid/time arrays from cfg.numerics.Nx, cfg.numerics.Nt, cfg.geometry.L, cfg.run.T_total.
2. Build V_app(t), transport coefficients, emission model, BC handler.
3. Allocate memmapped outputs in cfg.run.run_name.
4. Build initial fields and initialize circuit states.
5. For each time step: compute drift-diffusion fluxes.
6. Advance external circuit (step_circuit) to obtain new V_gap.
7. Build volumetric sources (Townsend ionization/recombination, optional toggles).
8. Advance n_i and n_e with KT + RK4, then enforce BC + Poisson fixed-point.
9. Apply optional adaptive substepping (if enabled), compute drift/diffusion
   stability diagnostics, and write snapshots every cfg.output.save_every.
10. Return SimulationState; inspect saved outputs with the diagnostics notebooks
    or optional quick-look plotting in run_paschen_1d.ipynb.
"""

import time as pytime
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

from config import SimulationConfig, SimulationState
from validation import validate_simulation_config
from outputs import allocate_outputs, write_snapshot, write_run_metadata
from physics import (
    make_voltage_waveform,
    build_transport_reference_state,
    build_swarm_interpolation_cache,
    build_ion_mobility_profile,
    build_ion_diffusion_profile,
    build_initial_conditions,
    compute_user_defined_ionization_frequency,
    compute_anode_electron_induced_yield,
)
from numerics import (
    build_poisson_tridiag_interior,
    poisson_1d_dirichlet_interior,
    create_linear_rk4_workspace,
    rk4_step_linear_reuse,
    set_boundary_condition_implicit,
    compute_drift_cfl,
    compute_diffusion_cfl,
)
from numerics_jit import (
    is_numba_available,
    create_numba_linear_rk4_workspace,
    rk4_step_linear_numba_reuse,
    compute_drift_cfl_numba,
    compute_diffusion_cfl_numba,
)
from circuit import step_circuit
from circuit_implicit_euler import step_circuit_implicit_euler
from circuit_mna import step_circuit_mna
from physical_constants import e, eps0
from emission import build_emission_model


def _transport_current_from_fluxes(
    Gamma_i: np.ndarray,
    Gamma_e: np.ndarray,
    dx: float,
    A: float,
    L: float,
) -> float:
    """Compute area-integrated transport current from 1D ion/electron fluxes."""
    flux_diff = Gamma_i - Gamma_e
    integral_flux = 0.5 * dx * (
        flux_diff[0] + flux_diff[-1] + 2.0 * np.add.reduce(flux_diff[1:-1])
    )
    return float((A * e / L) * integral_flux)


def _format_waveform_summary(cfg: SimulationConfig) -> str:
    """Return a compact human-readable summary of the configured voltage waveform."""
    if cfg.waveform.waveform_type == "dc":
        return f"type=dc, V_peak={cfg.waveform.V_peak:.6g} V"
    if cfg.waveform.waveform_type == "step":
        return (
            f"type=step, V_peak={cfg.waveform.V_peak:.6g} V, "
            f"t_on={cfg.waveform.tV_start:.6g} s, t_off={cfg.waveform.tV_end:.6g} s"
        )
    if cfg.waveform.waveform_type == "gaussian":
        return (
            f"type=gaussian, V_peak={cfg.waveform.V_peak:.6g} V, "
            f"t_peak={cfg.waveform.t_peak:.6g} s, tau={cfg.waveform.tau:.6g} s"
        )
    if cfg.waveform.waveform_type == "rf":
        return (
            f"type=rf, V_peak={cfg.waveform.V_peak:.6g} V, f_rf={cfg.waveform.f_rf:.6g} Hz, "
            f"V_dc={cfg.waveform.V_dc:.6g} V, phi_rf={cfg.waveform.phi_rf:.6g} rad"
        )
    if cfg.waveform.waveform_type in ("table", "tabulated", "measured_table"):
        table_path = getattr(cfg.waveform, "table_path", "<unset>")
        return f"type={cfg.waveform.waveform_type}, table='{table_path}'"
    return f"type={cfg.waveform.waveform_type}"


def _enabled_external_emission_components_for_electrode(
    cfg: SimulationConfig,
    electrode: str,
) -> list[str]:
    """Return the enabled externally driven emission component labels for one electrode."""
    prefix = "anode" if electrode == "anode" else "cathode"
    emission_cfg = cfg.emission
    enabled = []
    if getattr(emission_cfg, f"{prefix}_enable_constant_J_emission", False):
        enabled.append("constant_J")
    if getattr(emission_cfg, f"{prefix}_enable_fn_emission", False):
        enabled.append("fn")
    if getattr(emission_cfg, f"{prefix}_enable_mg_emission", False):
        enabled.append("mg")
    if getattr(emission_cfg, f"{prefix}_enable_rd_emission", False):
        enabled.append("rd")
    if getattr(emission_cfg, f"{prefix}_enable_quantum_pulse_emission", False):
        enabled.append("quantum_pulse")
    return enabled


def _format_electron_transport_summary(cfg: SimulationConfig) -> str:
    """Summarize the configured electron transport-coefficient source."""
    kinetics_mode = cfg.plasma.electron_kinetics_model
    if kinetics_mode == "user_defined_electron_kinetics":
        source = "user_defined_equation"
        e_swarm_path = cfg.local_field_approximation.electron_swarm_data_path
    elif kinetics_mode == "local_field_approximation":
        source = cfg.local_field_approximation.electron_transport_source
        e_swarm_path = cfg.local_field_approximation.electron_swarm_data_path
    else:
        source = "unknown"
        e_swarm_path = cfg.local_field_approximation.electron_swarm_data_path

    if source == "user_defined_equation":
        return "source=user_defined_equation (transport formulas in physics.py)"
    if source == "swarm_data_table_interpolation":
        return (
            "source=swarm_data_table_interpolation, "
            f"file='{e_swarm_path}', "
            f"gas={cfg.plasma_state.gas}"
        )
    return f"source={source}"


def _format_ion_transport_summary(cfg: SimulationConfig) -> str:
    """Summarize the configured ion transport-coefficient source."""
    ion_mode = cfg.plasma.ion_kinetics_model
    if ion_mode == "user_defined_ion_kinetics":
        return (
            f"ion={cfg.ion_transport.positive_ion}, "
            "mobility=user_defined_equation, diffusion=user_defined_equation"
        )
    if ion_mode == "local_field_ion_kinetics":
        return (
            f"ion={cfg.ion_transport.positive_ion}, "
            f"neutral={cfg.plasma_state.gas}, "
            f"mobility={cfg.ion_transport.mobility_source_mode}, "
            f"diffusion={cfg.ion_transport.diffusion_source_mode}"
        )
    return f"unsupported ion_kinetics_model={ion_mode!r}"


def _format_townsend_alpha_summary(cfg: SimulationConfig) -> str:
    """Summarize the configured Townsend-ionization coefficient source."""
    model = str(cfg.plasma.impact_ionization_model).strip().lower()
    if model != "from_townsend_alpha":
        return (
            "not used (impact-ionization model does not use Townsend alpha)"
        )

    source = str(cfg.townsend_coefficient.townsend_alpha_source_mode).strip().lower()
    alpha_path = (
        cfg.townsend_coefficient.townsend_alpha_swarm_data_path
        if cfg.townsend_coefficient.townsend_alpha_swarm_data_path is not None
        else cfg.local_field_approximation.electron_swarm_data_path
    )
    if source == "user_defined_equation":
        return "source=user_defined_equation (alpha(E, p) formulas in physics.py)"
    if source in ("interpolate_from_e_over_n_table",):
        return (
            f"source={source}, "
            f"file='{alpha_path}', "
            f"gas={cfg.plasma_state.gas}"
        )
    return f"source={source}"


def _resolve_ionization_mode(cfg: SimulationConfig) -> str:
    """
    Resolve ionization-source mode from explicit model selectors.
    """
    model = str(cfg.plasma.impact_ionization_model).strip().lower()

    if model == "from_townsend_alpha":
        source = str(cfg.townsend_coefficient.townsend_alpha_source_mode).strip().lower()
        if source in {"user_defined_equation"}:
            return "townsend_alpha_user_defined"
        if source in {"interpolate_from_e_over_n_table"}:
            return "townsend_alpha_e_over_n"
        raise ValueError(
            "Unknown townsend_alpha_source_mode: "
            f"{cfg.townsend_coefficient.townsend_alpha_source_mode}"
        )

    if model == "from_ionization_frequency":
        source = str(
            cfg.ionization_frequency_source.ionization_frequency_source_mode
        ).strip().lower()
        if source in {"user_defined_equation"}:
            return "nu_i_user_defined"
        if source in {"interpolate_from_e_over_n_table", "swarm_data_table_interpolation"}:
            return "nu_i_table_e_over_n"
        raise ValueError(
            "Unknown ionization_frequency_source_mode: "
            f"{cfg.ionization_frequency_source.ionization_frequency_source_mode}"
        )

    raise ValueError(f"Unknown impact_ionization_model: {cfg.plasma.impact_ionization_model}")


def _format_ionization_summary(cfg: SimulationConfig) -> str:
    """Summarize the resolved impact-ionization frequency source path."""
    mode = _resolve_ionization_mode(cfg)
    if mode.startswith("townsend_alpha"):
        src = str(cfg.townsend_coefficient.townsend_alpha_source_mode).strip().lower()
        if src == "user_defined_equation":
            return "nu_i=alpha*|u_e| with alpha from user_defined_equation"
        alpha_path = (
            cfg.townsend_coefficient.townsend_alpha_swarm_data_path
            if cfg.townsend_coefficient.townsend_alpha_swarm_data_path is not None
            else cfg.local_field_approximation.electron_swarm_data_path
        )
        return (
            "nu_i=alpha*|u_e| with alpha from "
            f"table(axis=e_over_n, file='{alpha_path}')"
        )
    if mode == "nu_i_user_defined":
        return "nu_i from user_defined_equation in physics.py"
    if mode == "nu_i_table_e_over_n":
        nu_path = (
            cfg.ionization_frequency_source.ionization_frequency_swarm_data_path
            if cfg.ionization_frequency_source.ionization_frequency_swarm_data_path is not None
            else cfg.local_field_approximation.electron_swarm_data_path
        )
        return f"nu_i from table(axis=e_over_n, file='{nu_path}')"
    return f"mode={mode}"


def _format_anode_secondary_emission_summary(cfg: SimulationConfig) -> str:
    """Summarize the active anode electron-induced secondary-emission model."""
    base = f"delta_ae={cfg.emission.anode_electron_induced_yield:.6g}"
    if not cfg.emission.use_vaughan_sey:
        return f"model=constant, {base}"
    return (
        "model=vaughan, "
        f"{base}, "
        f"effective_temperature_mode="
        f"{cfg.emission.vaughan_effective_temperature_mode}, "
        f"Emax0={cfg.emission.vaughan_Emax0_eV:.6g} eV, "
        f"dmax0={cfg.emission.vaughan_dmax0:.6g}, "
        f"ks={cfg.emission.vaughan_ks:.6g}, "
        f"z={cfg.emission.vaughan_z:.6g}, "
        f"E0={cfg.emission.vaughan_E0:.6g} eV"
    )


def _print_run_config_summary(cfg: SimulationConfig, dt: float, dx: float) -> None:
    """Print the resolved startup configuration summary shown before time stepping."""
    temporal = cfg.diagnostics.temporal
    spatial = cfg.diagnostics.spatial
    averaged_spatial = cfg.diagnostics.averaged_spatial

    temporal_plot_mode = (
        f"grouped ({len(temporal.plot_groups)} groups)"
        if temporal.plot_groups is not None
        else f"separate ({len(temporal.quantities)} quantities)"
    )
    spatial_plot_mode = (
        f"grouped ({len(spatial.plot_groups)} groups)"
        if spatial.plot_groups is not None
        else f"separate ({len(spatial.quantities)} quantities)"
    )
    t_samples_text = (
        "final_time_only"
        if spatial.t_samples is None
        else ", ".join(f"{t:.3e}" for t in spatial.t_samples)
    )
    averaged_spatial_plot_mode = (
        f"grouped ({len(averaged_spatial.plot_groups)} groups)"
        if averaged_spatial.plot_groups is not None
        else f"separate ({len(averaged_spatial.quantities)} quantities)"
    )

    print("\n=== PASCHEN-1D RUN SUMMARY ===")
    print(f"run_name: {cfg.run.run_name}")
    print(
        f"geometry: L={cfg.geometry.L:.6g} m, A={cfg.geometry.A:.6g} m^2, "
        f"l={cfg.geometry.l:.6g} m, eps_r={cfg.geometry.eps_r:.6g}"
    )
    print(
        f"plasma: gas={cfg.plasma_state.gas}, p_Torr={cfg.plasma_state.p_Torr:.6g}, "
        f"T_e={cfg.plasma_state.T_e:.6g} K, T_i={cfg.plasma_state.T_i:.6g} K, "
        f"n0={cfg.plasma_state.n0:.6g} m^-3"
    )
    print(
        f"transport-e: {_format_electron_transport_summary(cfg)}"
    )
    print(
        f"transport-i: {_format_ion_transport_summary(cfg)}"
    )
    print(
        f"townsend-alpha: {_format_townsend_alpha_summary(cfg)}"
    )
    print(
        "electron-kinetics: "
        f"mode={cfg.plasma.electron_kinetics_model}, "
        f"ionization-mode={_resolve_ionization_mode(cfg)}"
    )
    print(f"ionization-source: {_format_ionization_summary(cfg)}")
    print(
        f"grid/time: Nx={cfg.numerics.Nx}, Nt={cfg.numerics.Nt}, T_total={cfg.run.T_total:.6g} s, "
        f"dx={dx:.6g} m, dt={dt:.6g} s"
    )
    print(f"waveform: {_format_waveform_summary(cfg)}")
    print(
        f"circuit: type={cfg.circuit.circuit_type}, scheme={cfg.circuit.circuit_time_scheme}, "
        f"R0={cfg.circuit.R0:.6g}, C_s={cfg.circuit.C_s:.6g}, "
        f"L_s={cfg.circuit.L_s:.6g}, C_p={cfg.circuit.C_p:.6g}, "
        f"L_p={cfg.circuit.L_p:.6g}, R_m={cfg.circuit.R_m:.6g}, "
        f"C_ext={getattr(cfg.circuit, 'C_ext', 0.0):.6g}"
    )
    print(
        "numerics: "
        f"kt_limiter_theta={cfg.numerics.kt_limiter_theta:.6g}, "
        f"hotloop_backend={cfg.numerics.hotloop_backend}, "
        f"numba_parallel={cfg.numerics.numba_parallel}, "
        f"adaptive_substepping={cfg.numerics.use_adaptive_substepping}, "
        f"target_cfl_substep={cfg.numerics.target_cfl_substep:.6g}, "
        f"target_diffusion_cfl_substep={cfg.numerics.target_diffusion_cfl_substep:.6g}, "
        f"max_substeps={cfg.numerics.max_substeps}, "
        f"overflow_policy={cfg.numerics.adaptive_substep_overflow_policy}, "
        f"warn_every={cfg.numerics.adaptive_substep_warn_every}"
    )
    print(
        "bc-poisson-picard: "
        f"min_iter={cfg.numerics.bc_poisson_picard_min_iter}, "
        f"max_iter={cfg.numerics.bc_poisson_picard_max_iter}, "
        f"tol={cfg.numerics.bc_poisson_picard_tol:.3e}"
    )
    print(
        "boundary-modes: "
        f"anode(i={cfg.boundary.anode_ion_boundary}, e={cfg.boundary.anode_electron_boundary}), "
        f"cathode(i={cfg.boundary.cathode_ion_boundary}, e={cfg.boundary.cathode_electron_boundary})"
    )
    print(
        "secondary-emission: "
        f"cathode_gamma={cfg.emission.gamma:.6g}, "
        f"anode={_format_anode_secondary_emission_summary(cfg)}"
    )
    print(
        "sources: "
        f"volume={cfg.boundary.enable_volume_sources}, ionization={cfg.boundary.enable_ionization_source}, "
        f"recombination={cfg.boundary.enable_recombination_sink}"
    )
    anode_components = _enabled_external_emission_components_for_electrode(cfg, "anode")
    cathode_components = _enabled_external_emission_components_for_electrode(cfg, "cathode")
    print(
        "emission: "
        f"enabled={cfg.emission.enable_external_emission}, "
        f"material_mode={cfg.emission.electrode_material_mode}, "
        f"anode={anode_components}, cathode={cathode_components}, "
        f"electrodes=(anode={cfg.emission.enable_anode_external_emission}, cathode={cfg.emission.enable_cathode_external_emission})"
    )
    print(
        "output: "
        f"save_every={cfg.output.save_every}, "
        f"log_intermediate={cfg.output.log_intermediate}, "
        f"print_run_summary={cfg.output.print_run_summary}"
    )
    print(
        "diagnostics-temporal: "
        f"enabled={temporal.enabled}, mode={temporal_plot_mode}, "
        f"window=[{temporal.t_start},{temporal.t_end}]"
    )
    print(
        "diagnostics-spatial: "
        f"enabled={spatial.enabled}, mode={spatial_plot_mode}, "
        f"t_samples={t_samples_text}, x_unit={spatial.x_unit}"
    )
    if averaged_spatial.mode == "time_window":
        avg_window_text = (
            f"window=[{averaged_spatial.t_avg_start},{averaged_spatial.t_avg_end}]"
        )
    else:
        avg_window_text = f"last_n_cycles={averaged_spatial.N_cycle_avg}"
    print(
        "diagnostics-averaged-spatial: "
        f"enabled={averaged_spatial.enabled}, mode={averaged_spatial_plot_mode}, "
        f"avg_mode={averaged_spatial.mode}, {avg_window_text}, x_unit={averaged_spatial.x_unit}"
    )
    print("==============================\n")


def run_simulation(cfg: SimulationConfig) -> SimulationState:
    """
    Run one PASCHEN-1D simulation for the provided configuration.

    Parameters
    ----------
    cfg : SimulationConfig
        Physical, numerical, and circuit parameters for the run.

    Returns
    -------
    SimulationState
        Container holding:
            - cfg           : the input SimulationConfig (for provenance)
            - time          : 1D time array [s], length Nt
            - x             : 1D spatial grid [m], length Nx
            - V_gap         : gap voltage history [V], shape (Nt,)
            - I_discharge   : discharge current history [A], shape (Nt,)
            - I_transport_plasma  : physical plasma transport current [A], shape (Nt,)
            - I_transport_circuit : transport current recovered from circuit current [A], shape (Nt,)
            - I_emission_circuit  : direct-emission residual diagnostic [A], shape (Nt,)
            - I_emission_area     : represented surface-emission current over 1D area [A], shape (Nt,)
            - I_displacement_gap  : gap displacement current [A], shape (Nt,)
            - c_cfl         : drift-CFL number history, shape (Nt,)
            - diffusion_cfl : explicit diffusion stability-number history, shape (Nt,)
            - adaptive_substeps : adaptive substep count per macro step, shape (Nt,)
            - adaptive_dt_sub   : effective substep dt per macro step [s], shape (Nt,)
            - adaptive_cfl_est  : pre-step drift-CFL estimate (macro dt), shape (Nt,)
            - adaptive_diffusion_cfl_est : pre-step diffusion stability estimate
              using macro dt, shape (Nt,)
            - ne_final      : final electron density profile [m^-3], shape (Nx,)
            - ni_final      : final ion    density profile [m^-3], shape (Nx,)
            - phi_final     : final potential profile [V], shape (Nx,)
            - E_final       : final electric field profile [V/m], shape (Nx,)
            - mu_e_final    : final electron mobility profile [m^2/(V s)], shape (Nx,)
            - D_e_final     : final electron diffusion profile [m^2/s], shape (Nx,)
            - picard_iterations : Picard iterations used per macro step, shape (Nt,)

    Notes
    -----
    - Drift-diffusion update uses KT flux + explicit diffusion + RK4 in time.
      Macro time step `dt` can be split into adaptive substeps when enabled.
    - Poisson equation is solved at each time step on interior nodes with a
      banded tridiagonal solver and Dirichlet BCs.
    - Boundary conditions for n_e and n_i are selected per electrode/species
      using config boundary modes:
      anode_ion_boundary, anode_electron_boundary,
      cathode_ion_boundary, cathode_electron_boundary.
    - External circuit dynamics are handled by `step_circuit` or
      `step_circuit_implicit_euler` and include R0, Cp, Cs, Ls, Lp, Rm,
      optional load-side C_ext, and optional dielectric layers using the
      Adamovic convention.
    - Optional external emission is provided through `build_emission_model`
      and can be configured independently for anode/cathode.
      In the current formulation, emitted electrons enter the density update
      through conservative boundary-face fluxes. They are not added directly
      to the circuit-current integral; separate diagnostics record represented
      emission current and current-decomposition residuals.

    Onboarding map
    --------------
    The implementation body below follows numbered section headers:
    (1) discretization, (2) physics helpers, (3) circuit setup,
    (4) output allocation, (5) initialization, (6) initial snapshot,
    (7) Poisson matrix, (8) local flux functions, (9) time loop,
    (10) pack and return state.
    """
    validate_simulation_config(cfg)

    # ------------------------------------------------------------
    # 1) Basic discretization setup
    # ------------------------------------------------------------
    Nt = cfg.numerics.Nt
    Nx = cfg.numerics.Nx
    T  = cfg.run.T_total

    dt = T / (Nt - 1)            # uniform time step [s]
    dx = cfg.geometry.L / (Nx - 1)        # uniform spatial step [m]

    if cfg.output.print_run_summary:
        _print_run_config_summary(cfg, dt=dt, dx=dx)

    time    = np.linspace(0.0, T, Nt, dtype=np.float64)       # 0 .. T
    x_array = np.linspace(0.0, cfg.geometry.L, Nx, dtype=np.float64)   # 0 .. L

    # ------------------------------------------------------------
    # 2) Physics helpers: waveform, transport, emission, BCs
    # ------------------------------------------------------------
    # Applied voltage waveform Vs(t)
    V_app_func = make_voltage_waveform(cfg)

    # Mobilities, diffusion, recombination, reduced pressure, etc.
    transport = build_transport_reference_state(cfg)
    swarm_interp_cache = build_swarm_interpolation_cache(cfg)

    # Emission model (may be None if emission is disabled in cfg)
    emission_model = build_emission_model(cfg)

    # Boundary conditions are applied via per-electrode/per-species mode knobs.
    bc_func = set_boundary_condition_implicit

    # Frequently used scalars.
    L      = cfg.geometry.L
    A      = cfg.geometry.A
    l      = cfg.geometry.l
    eps_r_ = cfg.geometry.eps_r
    C_gap_ = eps0 * A / L
    gamma_ = cfg.emission.gamma
    anode_electron_induced_yield_ = cfg.emission.anode_electron_induced_yield
    use_vaughan_sey_ = cfg.emission.use_vaughan_sey
    vaughan_effective_temperature_mode_ = (
        cfg.emission.vaughan_effective_temperature_mode
    )
    vaughan_Emax0_eV_ = cfg.emission.vaughan_Emax0_eV
    vaughan_dmax0_ = cfg.emission.vaughan_dmax0
    vaughan_ks_ = cfg.emission.vaughan_ks
    vaughan_z_ = cfg.emission.vaughan_z
    vaughan_E0_ = cfg.emission.vaughan_E0
    p_Torr = cfg.plasma_state.p_Torr
    pr_    = transport.pr
    ionization_mode = _resolve_ionization_mode(cfg)

    if (
        ionization_mode == "nu_i_table_e_over_n"
        and swarm_interp_cache.nu_over_N_eovern_interp is None
    ):
        raise RuntimeError(
            "Ionization mode 'nu_i_table_e_over_n' requires preloaded E/N-axis "
            "nu_i/N interpolation data."
        )

    # Density-update hotloop backend selection.
    requested_hotloop_backend = str(cfg.numerics.hotloop_backend).strip().lower()
    use_numba_hotloop = (
        requested_hotloop_backend == "numba" and is_numba_available()
    )
    if (requested_hotloop_backend == "numba") and (not use_numba_hotloop):
        print(
            "Numerics warning: hotloop_backend='numba' requested but numba is unavailable. "
            "Falling back to hotloop_backend='numpy'."
        )
    effective_hotloop_backend = "numba" if use_numba_hotloop else "numpy"
    use_numba_parallel = bool(use_numba_hotloop and cfg.numerics.numba_parallel)

    neutral_density = float(transport.neutral_density)

    # ------------------------------------------------------------
    # 3) External-circuit setup (topology + state arrays)
    # ------------------------------------------------------------
    R0_val       = cfg.circuit.R0
    C_p          = cfg.circuit.C_p
    C_s          = cfg.circuit.C_s
    L_s          = cfg.circuit.L_s
    L_p          = cfg.circuit.L_p
    R_m          = cfg.circuit.R_m
    C_ext        = float(getattr(cfg.circuit, "C_ext", 0.0))
    circuit_type = cfg.circuit.circuit_type
    circuit_time_scheme = cfg.circuit.circuit_time_scheme
    C_ext_active = (
        C_ext
        if circuit_type in ("R0_Rm_Cext", "R0_Cs_Ls_Cp_Lp_Rm_Cext")
        else 0.0
    )

    if circuit_time_scheme == "implicit_euler":
        circuit_stepper = step_circuit_implicit_euler
    elif circuit_time_scheme == "explicit_euler":
        circuit_stepper = step_circuit
    elif circuit_time_scheme == "mna":
        circuit_stepper = step_circuit_mna
    else:
        raise ValueError(f"Unknown circuit_time_scheme: {circuit_time_scheme}")

    # Circuit state arrays (allocated only when needed by topology).
    # Circuit state variables are intentionally float64. With small dt and
    # large capacitances, valid circuit updates can be below float32 spacing at
    # hundreds of volts, which freezes slow recovery.
    V_d = np.zeros(Nt, dtype=np.float64)  # dielectric mapping voltage state
    V_n = None      # node voltage
    V_Cs = None     # series capacitor voltage
    I_s = None      # series-branch current (R0-Cs-Ls)
    I_Lp = None     # shunt inductor current at node

    # Circuits that maintain a node voltage
    if circuit_type in (
        "R0_Cp",
        "R0_Cp_Rm",
        "R0_Rm_Cext",
        "R0_Cs_Cp",
        "R0_Cs_Ls_Cp",
        "R0_Cs_Cp_Rm",
        "R0_Cs_Ls_Cp_Rm",
        "R0_Cs_Ls_Cp_Lp",
        "R0_Cs_Ls_Cp_Lp_Rm_Cext",
    ):
        V_n = np.zeros(Nt, dtype=np.float64)

    # Circuits that include a series capacitor Cs
    if circuit_type in (
        "R0_Cs_Cp",
        "R0_Cs_Ls_Cp",
        "R0_Cs_Cp_Rm",
        "R0_Cs_Ls_Cp_Rm",
        "R0_Cs_Ls_Cp_Lp",
        "R0_Cs_Ls_Cp_Lp_Rm_Cext",
    ) and C_s > 0.0:
        V_Cs = np.zeros(Nt, dtype=np.float64)

    # Circuits with a series inductor L_s (requires a current state I_s)
    if circuit_type in (
        "R0_Cs_Ls_Cp",
        "R0_Cs_Ls_Cp_Rm",
        "R0_Cs_Ls_Cp_Lp",
        "R0_Cs_Ls_Cp_Lp_Rm_Cext",
    ) and L_s > 0.0:
        I_s = np.zeros(Nt, dtype=np.float64)

    # Circuits with a parallel inductor L_p at the node
    if circuit_type in (
        "R0_Cs_Ls_Cp_Lp",
        "R0_Cs_Ls_Cp_Lp_Rm_Cext",
    ) and L_p > 0.0:
        I_Lp = np.zeros(Nt, dtype=np.float64)

    # Convenience aliases.
    SAVE_EVERY       = cfg.output.save_every
    LOG_INTERMEDIATE = cfg.output.log_intermediate

    # ------------------------------------------------------------
    # 4) Allocate memory-mapped outputs
    # ------------------------------------------------------------
    outputs = allocate_outputs(cfg, Nt, Nx)
    phi_sampled = outputs.phi_sampled
    E_sampled = outputs.E_sampled
    n_e_sampled = outputs.n_e_sampled
    n_i_sampled = outputs.n_i_sampled
    Gamma_i_sampled = outputs.Gamma_i_sampled
    Gamma_e_sampled = outputs.Gamma_e_sampled
    townsend_alpha_sampled = outputs.townsend_alpha_sampled
    nu_i_sampled = outputs.nu_i_sampled
    S_ion_sampled = outputs.S_ion_sampled
    S_sampled = outputs.S_sampled
    mu_e_sampled = outputs.mu_e_sampled
    D_e_sampled = outputs.D_e_sampled
    mu_i_sampled = outputs.mu_i_sampled
    D_i_sampled = outputs.D_i_sampled
    V_gap = outputs.V_gap
    V_node_history = outputs.V_node
    c_cfl = outputs.c_cfl
    I_discharge = outputs.I_discharge
    I_transport_plasma = outputs.I_transport_plasma
    I_transport_circuit = outputs.I_transport_circuit
    I_emission_circuit = outputs.I_emission_circuit
    I_emission_area = outputs.I_emission_area
    I_displacement_gap = outputs.I_displacement_gap
    picard_iterations = outputs.picard_iterations
    adaptive_substeps = outputs.adaptive_substeps
    adaptive_dt_sub = outputs.adaptive_dt_sub
    adaptive_cfl_est = outputs.adaptive_cfl_est
    diffusion_cfl = outputs.diffusion_cfl
    adaptive_diffusion_cfl_est = outputs.adaptive_diffusion_cfl_est

    # Write run metadata so plots can be regenerated without rerunning simulation.
    write_run_metadata(
        cfg,
        Nt=Nt,
        Nx=Nx,
        dt=dt,
        dx=dx,
        hotloop_stats={
            "requested_backend": str(requested_hotloop_backend),
            "effective_backend": str(effective_hotloop_backend),
            "numba_available": bool(is_numba_available()),
            "numba_parallel": bool(use_numba_parallel),
        },
        electron_transport_provenance=swarm_interp_cache.electron_transport_provenance(),
        ion_transport_provenance=swarm_interp_cache.ion_transport_provenance(),
    )

    # ------------------------------------------------------------
    # 5) Initial conditions (phi, E, n_e, n_i, V_gap)
    # ------------------------------------------------------------
    phi0, E0, ne0, ni0, V0 = build_initial_conditions(cfg, x_array, V_app_func)
    V_gap[0] = V0  # initial gap voltage (usually Vs(t=0))

    # Initialize circuit-state arrays where applicable.
    initial_source_voltage = float(V_app_func(0.0))

    if V_n is not None:
        # Start with node voltage equal to source at t=0.
        V_n[0] = initial_source_voltage

    if V_Cs is not None:
        V_Cs[0] = 0.0

    if I_s is not None:
        I_s[0] = 0.0

    if I_Lp is not None:
        I_Lp[0] = 0.0

    if V_node_history is not None and V_n is not None:
        V_node_history[0] = V_n[0]

    # Initialize dielectric mapping voltage from algebraic mapping.
    alpha_d = 1.0 + 2.0 * l / (eps_r_ * L)
    if V_n is not None:
        V_d[0] = float(V_n[0] - alpha_d * V_gap[0])
    else:
        V_d[0] = float(V_app_func(0.0) - alpha_d * V_gap[0])

    # Current/next-step fields (reused in-place).
    phi_curr = phi0.copy()
    E_curr   = E0.copy()
    ne_curr  = ne0.copy()
    ni_curr  = ni0.copy()

    phi_next = np.empty_like(phi_curr)
    E_next   = np.empty_like(E_curr)
    ne_next  = np.empty_like(ne_curr)
    ni_next  = np.empty_like(ni_curr)

    # Temporary arrays for gradients, fluxes, coefficients, and sources.
    grad_i      = np.empty(Nx, dtype=np.float32)
    grad_e      = np.empty(Nx, dtype=np.float32)
    mu_i_row    = np.empty(Nx, dtype=np.float32)
    D_i_row     = np.empty(Nx, dtype=np.float32)
    u_i_row     = np.empty(Nx, dtype=np.float32)
    mu_e_row    = np.empty(Nx, dtype=np.float32)
    D_e_row     = np.empty(Nx, dtype=np.float32)
    u_e_row     = np.empty(Nx, dtype=np.float32)
    Gamma_i_row = np.empty(Nx, dtype=np.float32)
    Gamma_e_row = np.empty(Nx, dtype=np.float32)
    townsend_alpha_row   = np.empty(Nx, dtype=np.float32)
    nu_row      = np.empty(Nx, dtype=np.float32)
    S_ion_row   = np.empty(Nx, dtype=np.float32)
    recomb_row  = np.empty(Nx, dtype=np.float32)
    S_i_row     = np.zeros(Nx, dtype=np.float32)
    S_e_row     = np.zeros(Nx, dtype=np.float32)
    S_row       = np.zeros(Nx, dtype=np.float32)

    # Reusable RK4 workspaces for linear KT+diffusion+source updates.
    rk4_workspace_numpy = create_linear_rk4_workspace(Nx, dtype=np.float32)
    rk4_workspace_numba = (
        create_numba_linear_rk4_workspace(Nx, dtype=np.float32)
        if use_numba_hotloop
        else None
    )

    if use_numba_hotloop:

        def rk4_linear_step(
            *,
            n: np.ndarray,
            u: np.ndarray,
            D: np.ndarray,
            S: np.ndarray,
            dt_local: float,
            adv_coeff: float,
            n_out: np.ndarray,
            boundary_flux_left: float = 0.0,
            boundary_flux_right: float = 0.0,
            replace_boundary_flux_left: bool = False,
            replace_boundary_flux_right: bool = False,
        ) -> np.ndarray:
            return rk4_step_linear_numba_reuse(
                n=n,
                u=u,
                D=D,
                S=S,
                dx=dx,
                dt=dt_local,
                kt_limiter_theta=cfg.numerics.kt_limiter_theta,
                adv_coeff=adv_coeff,
                ws=rk4_workspace_numba,
                n_out=n_out,
                parallel=use_numba_parallel,
                boundary_flux_left=boundary_flux_left,
                boundary_flux_right=boundary_flux_right,
                replace_boundary_flux_left=replace_boundary_flux_left,
                replace_boundary_flux_right=replace_boundary_flux_right,
            )

        drift_cfl_func = compute_drift_cfl_numba
        diffusion_cfl_func = compute_diffusion_cfl_numba
    else:

        def rk4_linear_step(
            *,
            n: np.ndarray,
            u: np.ndarray,
            D: np.ndarray,
            S: np.ndarray,
            dt_local: float,
            adv_coeff: float,
            n_out: np.ndarray,
            boundary_flux_left: float = 0.0,
            boundary_flux_right: float = 0.0,
            replace_boundary_flux_left: bool = False,
            replace_boundary_flux_right: bool = False,
        ) -> np.ndarray:
            return rk4_step_linear_reuse(
                n=n,
                u=u,
                D=D,
                S=S,
                dx=dx,
                dt=dt_local,
                kt_limiter_theta=cfg.numerics.kt_limiter_theta,
                adv_coeff=adv_coeff,
                ws=rk4_workspace_numpy,
                n_out=n_out,
                boundary_flux_left=boundary_flux_left,
                boundary_flux_right=boundary_flux_right,
                replace_boundary_flux_left=replace_boundary_flux_left,
                replace_boundary_flux_right=replace_boundary_flux_right,
            )

        drift_cfl_func = compute_drift_cfl
        diffusion_cfl_func = compute_diffusion_cfl

    mu_i_row[:] = build_ion_mobility_profile(
        cfg=cfg,
        x_array=x_array,
        E_column=E_curr,
        neutral_density=neutral_density,
        interpolation_cache=swarm_interp_cache,
    ).astype(mu_i_row.dtype, copy=False)
    D_i_row[:] = build_ion_diffusion_profile(
        cfg=cfg,
        x_array=x_array,
        E_column=E_curr,
        neutral_density=neutral_density,
        ion_mobility=mu_i_row,
        interpolation_cache=swarm_interp_cache,
    ).astype(D_i_row.dtype, copy=False)
    mu_e_row[:] = swarm_interp_cache.electron_mobility_from_field(
        cfg=cfg,
        x_array=x_array,
        E_column=E_curr,
        neutral_density=neutral_density,
    ).astype(mu_e_row.dtype, copy=False)
    D_e_row[:] = swarm_interp_cache.electron_diffusion_from_field(
        cfg=cfg,
        x_array=x_array,
        E_column=E_curr,
        neutral_density=neutral_density,
    ).astype(D_e_row.dtype, copy=False)

    # Initial adaptive-substepping diagnostics at t=0.
    cfl_initial = drift_cfl_func(
        mu_e=mu_e_row,
        mu_i=mu_i_row,
        E=E_curr,
        dt=dt,
        dx=dx,
    )
    diffusion_cfl_initial = diffusion_cfl_func(
        D_e=D_e_row,
        D_i=D_i_row,
        dt=dt,
        dx=dx,
    )
    c_cfl[0] = cfl_initial
    diffusion_cfl[0] = diffusion_cfl_initial
    picard_iterations[0] = 0.0
    adaptive_substeps[0] = 1.0
    adaptive_dt_sub[0] = dt
    adaptive_cfl_est[0] = cfl_initial
    adaptive_diffusion_cfl_est[0] = diffusion_cfl_initial

    surface_Q_emit_external_signed = 0.0
    surface_Q_injected_external_signed = 0.0
    surface_Q_injected_total_abs = 0.0
    surface_Q_injected_anode_abs = 0.0
    surface_Q_injected_cathode_abs = 0.0
    surface_max_equivalent_dn_per_substep_m3 = 0.0

    # ------------------------------------------------------------
    # 6) Store initial snapshot (k = 0)
    # ------------------------------------------------------------
    k0 = 0
    phi_sampled[k0, :] = phi_curr
    E_sampled[k0,   :] = E_curr
    n_e_sampled[k0, :] = ne_curr
    n_i_sampled[k0, :] = ni_curr

    if LOG_INTERMEDIATE:
        Gamma_i_sampled[k0, :] = 0.0
        Gamma_e_sampled[k0, :] = 0.0
        townsend_alpha_sampled[k0, :]   = 0.0
        nu_i_sampled[k0, :]    = 0.0
        S_ion_sampled[k0, :]   = 0.0
        S_sampled[k0,   :]     = 0.0
        if mu_e_sampled is not None:
            mu_e_sampled[k0, :] = mu_e_row
        if D_e_sampled is not None:
            D_e_sampled[k0, :] = D_e_row
        if mu_i_sampled is not None:
            mu_i_sampled[k0, :] = mu_i_row
        if D_i_sampled is not None:
            D_i_sampled[k0, :] = D_i_row

    # ------------------------------------------------------------
    # 7) Pre-build Poisson band matrix (interior Laplacian)
    # ------------------------------------------------------------
    ab_int = build_poisson_tridiag_interior(Nx, dtype=np.float64)

    # ------------------------------------------------------------
    # 8) Local transport and source updates
    # ------------------------------------------------------------
    def update_electron_transport_profiles() -> None:
        mu_e_row[:] = swarm_interp_cache.electron_mobility_from_field(
            cfg=cfg,
            x_array=x_array,
            E_column=E_curr,
            neutral_density=neutral_density,
        ).astype(mu_e_row.dtype, copy=False)
        D_e_row[:] = swarm_interp_cache.electron_diffusion_from_field(
            cfg=cfg,
            x_array=x_array,
            E_column=E_curr,
            neutral_density=neutral_density,
        ).astype(D_e_row.dtype, copy=False)

    if ionization_mode in (
        "townsend_alpha_user_defined",
        "townsend_alpha_e_over_n",
    ):
        def update_ionization_terms() -> None:
            townsend_alpha_row[:] = swarm_interp_cache.townsend_alpha_from_field(
                E_column=E_curr,
                p_Torr=p_Torr,
                pr=pr_,
                gas=cfg.plasma_state.gas,
                neutral_density=float(transport.neutral_density),
            ).astype(np.float32, copy=False)
            nu_row[:] = townsend_alpha_row * np.abs(u_e_row)
            S_ion_row[:] = nu_row * ne_curr

    elif ionization_mode == "nu_i_table_e_over_n":

        def update_ionization_terms() -> None:
            townsend_alpha_row.fill(0.0)
            nu_row[:] = swarm_interp_cache.ionization_frequency_from_field(
                E_column=E_curr,
                neutral_density=neutral_density,
            )
            S_ion_row[:] = nu_row * ne_curr

    elif ionization_mode == "nu_i_user_defined":

        def update_ionization_terms() -> None:
            townsend_alpha_row.fill(0.0)
            nu_row[:] = compute_user_defined_ionization_frequency(
                cfg=cfg,
                x_array=x_array,
                E_column=E_curr,
            )
            S_ion_row[:] = nu_row * ne_curr

    else:
        raise ValueError(f"Unknown resolved ionization mode: {ionization_mode}")

    # ------------------------------------------------------------
    # 9) Main time-integration loop
    # ------------------------------------------------------------
    start = pytime.perf_counter()
    use_adaptive_substepping = bool(cfg.numerics.use_adaptive_substepping)
    target_cfl_substep = float(cfg.numerics.target_cfl_substep)
    target_diffusion_cfl_substep = float(cfg.numerics.target_diffusion_cfl_substep)
    max_substeps = int(cfg.numerics.max_substeps)
    overflow_policy = cfg.numerics.adaptive_substep_overflow_policy
    warn_every = int(cfg.numerics.adaptive_substep_warn_every)
    bc_poisson_picard_min_iter = max(1, int(cfg.numerics.bc_poisson_picard_min_iter))
    bc_poisson_picard_max_iter = max(1, int(cfg.numerics.bc_poisson_picard_max_iter))
    if bc_poisson_picard_min_iter > bc_poisson_picard_max_iter:
        bc_poisson_picard_min_iter = bc_poisson_picard_max_iter
    bc_poisson_picard_tol = max(float(cfg.numerics.bc_poisson_picard_tol), 1.0e-14)
    adaptive_total_substeps = 0
    adaptive_steps_with_split = 0
    adaptive_overflow_events = 0
    adaptive_steps_limited_by_drift = 0
    adaptive_steps_limited_by_diffusion = 0
    adaptive_steps_limited_by_both = 0
    adaptive_max_required_substeps = 1
    adaptive_max_used_substeps = 1
    adaptive_max_pre_step_cfl = float(adaptive_cfl_est[0])
    adaptive_max_realized_substep_cfl = float(c_cfl[0])
    adaptive_max_pre_step_diffusion_cfl = float(adaptive_diffusion_cfl_est[0])
    adaptive_max_realized_substep_diffusion_cfl = float(diffusion_cfl[0])
    bc_poisson_picard_total_solves = 0
    bc_poisson_picard_total_iterations = 0
    bc_poisson_picard_max_iterations = 0
    bc_poisson_picard_nonconverged_solves = 0
    bc_poisson_picard_max_residual = 0.0

    for n_idx in tqdm(range(Nt - 1), mininterval=2, desc="Time stepping"):
        # Pre-step transport estimate at macro-step start for adaptive control.
        mu_i_row[:] = build_ion_mobility_profile(
            cfg=cfg,
            x_array=x_array,
            E_column=E_curr,
            neutral_density=neutral_density,
            interpolation_cache=swarm_interp_cache,
        ).astype(mu_i_row.dtype, copy=False)
        D_i_row[:] = build_ion_diffusion_profile(
            cfg=cfg,
            x_array=x_array,
            E_column=E_curr,
            neutral_density=neutral_density,
            ion_mobility=mu_i_row,
            interpolation_cache=swarm_interp_cache,
        ).astype(D_i_row.dtype, copy=False)
        update_electron_transport_profiles()

        cfl_est_macro = drift_cfl_func(
            mu_e=mu_e_row,
            mu_i=mu_i_row,
            E=E_curr,
            dt=dt,
            dx=dx,
        )
        diffusion_cfl_est_macro = diffusion_cfl_func(
            D_e=D_e_row,
            D_i=D_i_row,
            dt=dt,
            dx=dx,
        )
        if not (np.isfinite(cfl_est_macro) and np.isfinite(diffusion_cfl_est_macro)):
            if use_adaptive_substepping and (overflow_policy == "error"):
                raise RuntimeError(
                    "Adaptive substepping encountered a non-finite stability estimate "
                    f"at t={time[n_idx + 1]:.4e} s: "
                    f"drift_cfl={cfl_est_macro}, "
                    f"diffusion_cfl={diffusion_cfl_est_macro}."
                )
            if not np.isfinite(cfl_est_macro):
                cfl_est_macro = np.inf
            if not np.isfinite(diffusion_cfl_est_macro):
                diffusion_cfl_est_macro = np.inf
        adaptive_max_pre_step_cfl = max(adaptive_max_pre_step_cfl, cfl_est_macro)
        adaptive_max_pre_step_diffusion_cfl = max(
            adaptive_max_pre_step_diffusion_cfl, diffusion_cfl_est_macro
        )

        if use_adaptive_substepping:
            if np.isfinite(cfl_est_macro):
                drift_substeps_required = max(
                    1,
                    int(np.ceil(cfl_est_macro / max(target_cfl_substep, 1.0e-12))),
                )
            else:
                drift_substeps_required = max_substeps + 1
            if np.isfinite(diffusion_cfl_est_macro):
                diffusion_substeps_required = max(
                    1,
                    int(
                        np.ceil(
                            diffusion_cfl_est_macro
                            / max(target_diffusion_cfl_substep, 1.0e-12)
                        )
                    ),
                )
            else:
                diffusion_substeps_required = max_substeps + 1
            n_sub_required = max(drift_substeps_required, diffusion_substeps_required)
            adaptive_max_required_substeps = max(adaptive_max_required_substeps, n_sub_required)
            if n_sub_required > 1:
                if (drift_substeps_required > 1) and (diffusion_substeps_required > 1):
                    adaptive_steps_limited_by_both += 1
                elif drift_substeps_required > 1:
                    adaptive_steps_limited_by_drift += 1
                else:
                    adaptive_steps_limited_by_diffusion += 1
            if n_sub_required > max_substeps:
                if overflow_policy == "error":
                    raise RuntimeError(
                        "Adaptive substepping overflow: required n_sub="
                        f"{n_sub_required} exceeds max_substeps={max_substeps} "
                        f"at t={time[n_idx + 1]:.4e} s "
                        f"(required_drift={drift_substeps_required}, "
                        f"required_diffusion={diffusion_substeps_required}, "
                        f"drift_cfl_est={cfl_est_macro:.3f}, "
                        f"diffusion_cfl_est={diffusion_cfl_est_macro:.3f})."
                    )
                adaptive_overflow_events += 1
                if (adaptive_overflow_events == 1) or (n_idx % warn_every == 0):
                    print(
                        "Adaptive substepping capped at max_substeps: "
                        f"t={time[n_idx + 1]:.4e} s, required={n_sub_required}, "
                        f"required_drift={drift_substeps_required}, "
                        f"required_diffusion={diffusion_substeps_required}, "
                        f"max_substeps={max_substeps}, drift_cfl_est={cfl_est_macro:.3f}, "
                        f"diffusion_cfl_est={diffusion_cfl_est_macro:.3f}"
                    )
                n_sub = max_substeps
            else:
                n_sub = n_sub_required
        else:
            drift_substeps_required = 1
            diffusion_substeps_required = 1
            n_sub_required = 1
            n_sub = 1

        adaptive_total_substeps += n_sub
        if n_sub > 1:
            adaptive_steps_with_split += 1
        adaptive_max_used_substeps = max(adaptive_max_used_substeps, n_sub)

        dt_sub = dt / float(n_sub)
        adaptive_substeps[n_idx + 1] = float(n_sub)
        adaptive_dt_sub[n_idx + 1] = float(dt_sub)
        adaptive_cfl_est[n_idx + 1] = float(cfl_est_macro)
        adaptive_diffusion_cfl_est[n_idx + 1] = float(diffusion_cfl_est_macro)

        t_macro_start = float(time[n_idx])
        V_gap_local = float(V_gap[n_idx])
        V_n_local = float(V_n[n_idx]) if V_n is not None else None
        V_d_local = float(V_d[n_idx])
        V_Cs_local = float(V_Cs[n_idx]) if V_Cs is not None else None
        I_s_local = float(I_s[n_idx]) if I_s is not None else None
        I_Lp_local = float(I_Lp[n_idx]) if I_Lp is not None else None
        I_local = float(I_discharge[n_idx]) if n_idx < len(I_discharge) else 0.0
        I_transport_plasma_local = 0.0
        I_transport_circuit_local = 0.0
        I_emission_circuit_local = 0.0
        I_emission_area_local = 0.0
        I_displacement_gap_local = 0.0
        max_substep_cfl_step = 0.0
        max_substep_diffusion_cfl_step = 0.0
        picard_iters_macro_max = 0
        for sub_idx in range(n_sub):
            t_sub_start = t_macro_start + sub_idx * dt_sub
            t_next = t_macro_start + (sub_idx + 1) * dt_sub

            mu_i_row[:] = build_ion_mobility_profile(
                cfg=cfg,
                x_array=x_array,
                E_column=E_curr,
                neutral_density=neutral_density,
                interpolation_cache=swarm_interp_cache,
            ).astype(mu_i_row.dtype, copy=False)
            D_i_row[:] = build_ion_diffusion_profile(
                cfg=cfg,
                x_array=x_array,
                E_column=E_curr,
                neutral_density=neutral_density,
                ion_mobility=mu_i_row,
                interpolation_cache=swarm_interp_cache,
            ).astype(D_i_row.dtype, copy=False)
            u_i_row[:] = mu_i_row * E_curr

            update_electron_transport_profiles()
            u_e_row[:] = mu_e_row * E_curr

            # Density gradients and drift-diffusion fluxes.
            grad_i[:] = np.gradient(ni_curr, dx, edge_order=1).astype(
                grad_i.dtype, copy=False
            )
            grad_e[:] = np.gradient(ne_curr, dx, edge_order=1).astype(
                grad_e.dtype, copy=False
            )
            Gamma_i_row[:] = -D_i_row * grad_i + ni_curr * u_i_row
            Gamma_e_row[:] = -D_e_row * grad_e - ne_curr * u_e_row

            # External emission averages cover [t_sub_start, t_next].
            if emission_model is not None:
                J_emit_anode = (
                    emission_model.current_density(
                        t=t_sub_start,
                        V_gap=V_gap_local,
                        dt=dt_sub,
                        E_surface=float(E_curr[0]),
                        electrode="anode",
                    )
                    if cfg.emission.enable_anode_external_emission
                    else 0.0
                )
                J_emit_cathode = (
                    emission_model.current_density(
                        t=t_sub_start,
                        V_gap=V_gap_local,
                        dt=dt_sub,
                        E_surface=float(E_curr[-1]),
                        electrode="cathode",
                    )
                    if cfg.emission.enable_cathode_external_emission
                    else 0.0
                )
                Gamma_ext_anode = J_emit_anode / e
                Gamma_ext_cathode = J_emit_cathode / e
            else:
                J_emit_anode = 0.0
                J_emit_cathode = 0.0
                Gamma_ext_anode = 0.0
                Gamma_ext_cathode = 0.0

            Gamma_surface_anode = (
                max(float(Gamma_ext_anode), 0.0)
                if cfg.boundary.anode_electron_boundary == "electron_emission"
                else 0.0
            )
            Gamma_surface_cathode = (
                max(float(Gamma_ext_cathode), 0.0)
                if cfg.boundary.cathode_electron_boundary == "electron_emission"
                else 0.0
            )
            if (
                cfg.boundary.anode_electron_boundary == "electron_emission"
                and (use_vaughan_sey_ or anode_electron_induced_yield_ > 0.0)
            ):
                Gamma_e_incident_anode = max(-float(Gamma_e_row[0]), 0.0)
                T_e_for_vaughan_eV = float(transport.T_e_eV)
                if (
                    vaughan_effective_temperature_mode_
                    == "local_field_approximation"
                ):
                    T_e_for_vaughan_eV = (
                        swarm_interp_cache.vaughan_effective_temperature_eV_from_field(
                            E_anode=float(E_curr[0]),
                            neutral_density=neutral_density,
                        )
                    )
                delta_ae = compute_anode_electron_induced_yield(
                    incident_electron_flux=Gamma_e_incident_anode,
                    electron_density_inner=float(ne_curr[1]),
                    T_e_eV=T_e_for_vaughan_eV,
                    constant_yield=anode_electron_induced_yield_,
                    use_vaughan_sey=use_vaughan_sey_,
                    vaughan_Emax0_eV=vaughan_Emax0_eV_,
                    vaughan_dmax0=vaughan_dmax0_,
                    vaughan_ks=vaughan_ks_,
                    vaughan_z=vaughan_z_,
                    vaughan_E0=vaughan_E0_,
                )
                Gamma_surface_anode += (
                    delta_ae * Gamma_e_incident_anode
                )
            if (
                cfg.boundary.cathode_electron_boundary == "electron_emission"
                and gamma_ > 0.0
            ):
                Gamma_i_incident_cathode = max(float(Gamma_i_row[-1]), 0.0)
                Gamma_surface_cathode += max(gamma_, 0.0) * Gamma_i_incident_cathode

            Gamma_e_boundary_left = 0.0
            Gamma_e_boundary_right = 0.0
            replace_electron_boundary_flux_left = (
                cfg.boundary.anode_electron_boundary == "electron_emission"
            )
            replace_electron_boundary_flux_right = (
                cfg.boundary.cathode_electron_boundary == "electron_emission"
            )
            if cfg.boundary.anode_electron_boundary == "electron_emission":
                Gamma_e_boundary_left = float(Gamma_surface_anode)
            if cfg.boundary.cathode_electron_boundary == "electron_emission":
                Gamma_e_boundary_right = -float(Gamma_surface_cathode)

            if Gamma_e_boundary_left != 0.0 or Gamma_e_boundary_right != 0.0:
                dn_equiv = (
                    max(abs(Gamma_e_boundary_left), abs(Gamma_e_boundary_right))
                    * dt_sub
                    / dx
                )
                if np.isfinite(dn_equiv):
                    surface_max_equivalent_dn_per_substep_m3 = max(
                        surface_max_equivalent_dn_per_substep_m3,
                        float(dn_equiv),
                    )

            # Electron-emission boundaries replace the KT+diffusion wall face
            # with outflow-only drift plus signed surface-emission flux. Do not
            # add surface emission to Gamma_e for the circuit integral; that
            # would treat a zero-thickness surface source as volume current.

            # Passive current-decomposition diagnostics. These values are saved
            # after the circuit step and do not feed back into the algorithm.
            I_transport_plasma_step = _transport_current_from_fluxes(
                Gamma_i_row,
                Gamma_e_row,
                dx,
                A,
                L,
            )
            V_gap_before_circuit = V_gap_local
            I_emission_area_step = A * (float(J_emit_cathode) - float(J_emit_anode))
            surface_Q_emit_external_signed += float(I_emission_area_step) * dt_sub
            surface_Q_injected_external_signed += (
                e * A
                * (max(float(Gamma_ext_cathode), 0.0) - max(float(Gamma_ext_anode), 0.0))
                * dt_sub
            )
            surface_Q_injected_anode_abs += e * A * Gamma_surface_anode * dt_sub
            surface_Q_injected_cathode_abs += e * A * Gamma_surface_cathode * dt_sub
            surface_Q_injected_total_abs += (
                e * A * (Gamma_surface_anode + Gamma_surface_cathode) * dt_sub
            )

            # Circuit step with substep dt.
            V_gap_new, I_new, V_d_new, V_n_new, V_Cs_new, I_s_new, I_Lp_new = circuit_stepper(
                circuit_type=circuit_type,
                V_app_func=V_app_func,
                t=t_next,
                dt=dt_sub,
                V_gap_prev=V_gap_local,
                Gamma_i=Gamma_i_row,
                Gamma_e=Gamma_e_row,
                dx=dx,
                A=A,
                L=L,
                l=l,
                eps_r=eps_r_,
                R0=R0_val,
                C_s=C_s,
                C_p=C_p,
                R_m=R_m,
                L_s=L_s,
                L_p=L_p,
                V_d_prev=V_d_local,
                V_n_prev=V_n_local,
                V_Cs_prev=V_Cs_local,
                I_s_prev=I_s_local,
                I_Lp_prev=I_Lp_local,
                C_ext=C_ext,
            )
            if V_d_new is None:
                raise RuntimeError("Circuit step did not return V_d_new.")
            V_load_before_circuit = alpha_d * float(V_gap_before_circuit) + float(V_d_local)
            V_load_after_circuit = alpha_d * float(V_gap_new) + float(V_d_new)
            I_displacement_gap_step = (
                C_gap_ * (float(V_gap_new) - float(V_gap_before_circuit))
                + C_ext_active * (V_load_after_circuit - V_load_before_circuit)
            ) / dt_sub
            # With direct surface-emission-to-circuit coupling removed,
            # I_emission_circuit_step is a residual consistency diagnostic and
            # should stay near zero. The represented emitted surface current is
            # recorded separately as I_emission_area_step. For C_ext topologies,
            # the reported capacitive term includes geometric gap displacement
            # plus load-side C_ext charging.
            I_transport_circuit_step = float(I_new) - I_displacement_gap_step
            I_emission_circuit_step = I_transport_circuit_step - I_transport_plasma_step
            I_transport_plasma_local = float(I_transport_plasma_step)
            I_transport_circuit_local = float(I_transport_circuit_step)
            I_emission_circuit_local = float(I_emission_circuit_step)
            I_emission_area_local = float(I_emission_area_step)
            I_displacement_gap_local = float(I_displacement_gap_step)

            V_gap_local = float(V_gap_new)
            I_local = float(I_new)
            V_d_local = float(V_d_new)
            if V_n is not None:
                V_n_local = float(V_n_new) if V_n_new is not None else V_n_local
            if V_Cs is not None:
                V_Cs_local = float(V_Cs_new) if V_Cs_new is not None else V_Cs_local
            if I_s is not None:
                I_s_local = float(I_s_new) if I_s_new is not None else I_s_local
            if I_Lp is not None:
                I_Lp_local = float(I_Lp_new) if I_Lp_new is not None else I_Lp_local

            # Ionization/recombination source terms.
            update_ionization_terms()

            if cfg.boundary.enable_volume_sources:
                if cfg.boundary.enable_ionization_source:
                    S_i_row[:] = S_ion_row
                    S_e_row[:] = S_ion_row
                else:
                    S_i_row.fill(0.0)
                    S_e_row.fill(0.0)
                if cfg.boundary.enable_recombination_sink:
                    np.multiply(ni_curr, np.float32(transport.beta), out=recomb_row)
                    recomb_row *= ne_curr
                    S_i_row -= recomb_row
                    S_e_row -= recomb_row
            else:
                S_i_row.fill(0.0)
                S_e_row.fill(0.0)
            S_row[:] = S_e_row

            # Drift-diffusion update for ions/electrons (KT + RK4).
            ni_next[:] = rk4_linear_step(
                n=ni_curr,
                u=u_i_row,
                D=D_i_row,
                S=S_i_row,
                dt_local=dt_sub,
                adv_coeff=1.0,
                n_out=ni_next,
            )
            ne_next[:] = rk4_linear_step(
                n=ne_curr,
                u=u_e_row,
                D=D_e_row,
                S=S_e_row,
                dt_local=dt_sub,
                adv_coeff=-1.0,
                n_out=ne_next,
                boundary_flux_left=Gamma_e_boundary_left,
                boundary_flux_right=Gamma_e_boundary_right,
                replace_boundary_flux_left=replace_electron_boundary_flux_left,
                replace_boundary_flux_right=replace_electron_boundary_flux_right,
            )

            np.nan_to_num(ne_next, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

            # BC + Poisson fixed-point (Picard) iteration.
            phi_iter = phi_curr.copy()
            T_e_for_bc_eV = float(transport.T_e_eV)
            bc_poisson_picard_total_solves += 1
            bc_picard_iter_used = 0
            bc_picard_residual = np.inf
            bc_picard_converged = False
            for picard_iter in range(bc_poisson_picard_max_iter):
                ne_next, ni_next = bc_func(
                    ne_next,
                    ni_next,
                    ne_curr,
                    ni_curr,
                    phi_iter,
                    gamma_,
                    anode_electron_induced_yield_,
                    T_e_for_bc_eV,
                    float(mu_i_row[0]),
                    float(mu_i_row[-1]),
                    float(mu_e_row[0]),
                    float(mu_e_row[-1]),
                    dx,
                    dt_sub,
                    Gamma_ext_anode=Gamma_ext_anode,
                    Gamma_ext_cathode=Gamma_ext_cathode,
                    use_vaughan_sey=use_vaughan_sey_,
                    vaughan_Emax0_eV=vaughan_Emax0_eV_,
                    vaughan_dmax0=vaughan_dmax0_,
                    vaughan_ks=vaughan_ks_,
                    vaughan_z=vaughan_z_,
                    vaughan_E0=vaughan_E0_,
                    anode_ion_boundary=cfg.boundary.anode_ion_boundary,
                    anode_electron_boundary=cfg.boundary.anode_electron_boundary,
                    cathode_ion_boundary=cfg.boundary.cathode_ion_boundary,
                    cathode_electron_boundary=cfg.boundary.cathode_electron_boundary,
                )
                poisson_1d_dirichlet_interior(
                    n_i=ni_next,
                    n_e=ne_next,
                    dx=dx,
                    phi_left=V_gap_local,
                    phi_right=0.0,
                    ab_int=ab_int,
                    phi_out=phi_next,
                    E_out=E_next,
                )
                bc_picard_iter_used = picard_iter + 1
                bc_picard_residual = float(np.max(np.abs(phi_next - phi_iter)))
                if not np.isfinite(bc_picard_residual):
                    bc_picard_residual = np.inf
                if (
                    bc_picard_iter_used >= bc_poisson_picard_min_iter
                    and bc_picard_residual < bc_poisson_picard_tol
                ):
                    bc_picard_converged = True
                    break
                phi_iter[:] = phi_next
            bc_poisson_picard_total_iterations += int(bc_picard_iter_used)
            bc_poisson_picard_max_iterations = max(
                bc_poisson_picard_max_iterations, int(bc_picard_iter_used)
            )
            bc_poisson_picard_max_residual = max(
                bc_poisson_picard_max_residual, float(bc_picard_residual)
            )
            if not bc_picard_converged:
                bc_poisson_picard_nonconverged_solves += 1
                if (
                    bc_poisson_picard_nonconverged_solves == 1
                    or (n_idx % warn_every == 0)
                ):
                    print(
                        "BC+Poisson Picard nonconvergence: "
                        f"t={t_next:.4e} s, iterations={bc_picard_iter_used}, "
                        f"residual={bc_picard_residual:.3e}, tol={bc_poisson_picard_tol:.3e}. "
                        "Continuing with last iterate."
                    )
            picard_iters_macro_max = max(picard_iters_macro_max, bc_picard_iter_used)

            substep_cfl = drift_cfl_func(
                mu_e=mu_e_row,
                mu_i=mu_i_row,
                E=E_next,
                dt=dt_sub,
                dx=dx,
            )
            substep_diffusion_cfl = diffusion_cfl_func(
                D_e=D_e_row,
                D_i=D_i_row,
                dt=dt_sub,
                dx=dx,
            )
            max_substep_cfl_step = max(max_substep_cfl_step, substep_cfl)
            max_substep_diffusion_cfl_step = max(
                max_substep_diffusion_cfl_step, substep_diffusion_cfl
            )

            # Roll arrays: next -> current (substep-to-substep progression).
            ni_curr, ni_next = ni_next, ni_curr
            ne_curr, ne_next = ne_next, ne_curr
            phi_curr, phi_next = phi_next, phi_curr
            E_curr, E_next = E_next, E_curr

        adaptive_max_realized_substep_cfl = max(
            adaptive_max_realized_substep_cfl, max_substep_cfl_step
        )
        adaptive_max_realized_substep_diffusion_cfl = max(
            adaptive_max_realized_substep_diffusion_cfl,
            max_substep_diffusion_cfl_step,
        )

        # Store updated macro-step circuit quantities.
        V_gap[n_idx + 1] = V_gap_local
        I_discharge[n_idx] = I_local  # last entry remains default
        I_transport_plasma[n_idx] = I_transport_plasma_local
        I_transport_circuit[n_idx] = I_transport_circuit_local
        I_emission_circuit[n_idx] = I_emission_circuit_local
        I_emission_area[n_idx] = I_emission_area_local
        I_displacement_gap[n_idx] = I_displacement_gap_local
        V_d[n_idx + 1] = V_d_local
        if (V_n is not None) and (V_n_local is not None):
            V_n[n_idx + 1] = V_n_local
            if V_node_history is not None:
                V_node_history[n_idx + 1] = V_n_local
        if (V_Cs is not None) and (V_Cs_local is not None):
            V_Cs[n_idx + 1] = V_Cs_local
        if (I_s is not None) and (I_s_local is not None):
            I_s[n_idx + 1] = I_s_local
        if (I_Lp is not None) and (I_Lp_local is not None):
            I_Lp[n_idx + 1] = I_Lp_local

        c_cfl[n_idx + 1] = max_substep_cfl_step
        diffusion_cfl[n_idx + 1] = max_substep_diffusion_cfl_step
        picard_iterations[n_idx + 1] = float(picard_iters_macro_max)
        if (not use_adaptive_substepping) and (max_substep_cfl_step > 1.0):
            print(
                f"CFL condition violated at time {time[n_idx + 1]:.4e}, "
                f"CFL = {max_substep_cfl_step:.3f}"
            )
        if (
            not use_adaptive_substepping
            and max_substep_diffusion_cfl_step > target_diffusion_cfl_substep
        ):
            print(
                "Diffusion stability condition violated at time "
                f"{time[n_idx + 1]:.4e}, diffusion CFL = "
                f"{max_substep_diffusion_cfl_step:.3f}, target = "
                f"{target_diffusion_cfl_substep:.3f}"
            )

        # Save snapshots at configured interval.
        if (n_idx + 1) % SAVE_EVERY == 0:
            k = (n_idx + 1) // SAVE_EVERY
            Gamma_e_to_save = Gamma_e_row

            snapshot_kwargs = dict(
                k=k,
                n_i_sampled=n_i_sampled,
                n_e_sampled=n_e_sampled,
                phi_sampled=phi_sampled,
                E_sampled=E_sampled,
                ni=ni_curr,
                ne=ne_curr,
                phi=phi_curr,
                E=E_curr,
                log_intermediate=LOG_INTERMEDIATE,
                Gamma_i_sampled=Gamma_i_sampled,
                Gamma_e_sampled=Gamma_e_sampled,
                townsend_alpha_sampled=townsend_alpha_sampled,
                nu_i_sampled=nu_i_sampled,
                S_ion_sampled=S_ion_sampled,
                S_sampled=S_sampled,
                mu_e_sampled=mu_e_sampled,
                D_e_sampled=D_e_sampled,
                mu_i_sampled=mu_i_sampled,
                D_i_sampled=D_i_sampled,
                Gamma_i=Gamma_i_row,
                Gamma_e=Gamma_e_to_save,
                townsend_alpha=townsend_alpha_row,
                nu=nu_row,
                S_ion=S_ion_row,
                S=S_row,
                mu_e=mu_e_row,
                D_e=D_e_row,
                mu_i=mu_i_row,
                D_i=D_i_row,
            )
            write_snapshot(**snapshot_kwargs)

    elapsed = pytime.perf_counter() - start
    print(f"Elapsed time: {elapsed:.6f} s")

    macro_steps = max(Nt - 1, 1)
    adaptive_stats = {
        "enabled": bool(use_adaptive_substepping),
        "macro_steps": int(Nt - 1),
        "total_substeps": int(adaptive_total_substeps),
        "mean_substeps_per_macro": float(adaptive_total_substeps / macro_steps),
        "steps_with_substepping": int(adaptive_steps_with_split),
        "steps_limited_by_drift": int(adaptive_steps_limited_by_drift),
        "steps_limited_by_diffusion": int(adaptive_steps_limited_by_diffusion),
        "steps_limited_by_both": int(adaptive_steps_limited_by_both),
        "max_substeps_used": int(adaptive_max_used_substeps),
        "max_required_substeps_estimate": int(adaptive_max_required_substeps),
        "overflow_events": int(adaptive_overflow_events),
        "target_cfl_substep": float(target_cfl_substep),
        "target_diffusion_cfl_substep": float(target_diffusion_cfl_substep),
        "max_pre_step_cfl_estimate": float(adaptive_max_pre_step_cfl),
        "max_realized_substep_cfl": float(adaptive_max_realized_substep_cfl),
        "max_pre_step_diffusion_cfl_estimate": float(adaptive_max_pre_step_diffusion_cfl),
        "max_realized_substep_diffusion_cfl": float(
            adaptive_max_realized_substep_diffusion_cfl
        ),
    }
    hotloop_stats = {
        "requested_backend": str(requested_hotloop_backend),
        "effective_backend": str(effective_hotloop_backend),
        "numba_available": bool(is_numba_available()),
        "numba_parallel": bool(use_numba_parallel),
    }
    bc_poisson_picard_stats = {
        "enabled": True,
        "total_solves": int(bc_poisson_picard_total_solves),
        "total_iterations": int(bc_poisson_picard_total_iterations),
        "mean_iterations_per_solve": float(
            bc_poisson_picard_total_iterations / max(bc_poisson_picard_total_solves, 1)
        ),
        "max_iterations": int(bc_poisson_picard_max_iterations),
        "nonconverged_solves": int(bc_poisson_picard_nonconverged_solves),
        "max_residual": float(bc_poisson_picard_max_residual),
        "min_iter": int(bc_poisson_picard_min_iter),
        "max_iter": int(bc_poisson_picard_max_iter),
        "tol": float(bc_poisson_picard_tol),
    }
    surface_emission_stats = {
        "scheme": "rk4_boundary_face_flux_replace_wall_flux",
        "wall_flux_closure": "outflow_only_drift_plus_signed_surface_flux",
        "anode_signed_electron_flux_convention": "+Gamma_surface_anode_at_left_face",
        "cathode_signed_electron_flux_convention": "-Gamma_surface_cathode_at_right_face",
        "boundary_density_for_electron_emission": "zero_density_non_reservoir",
        "Q_emit_external_signed_C": float(surface_Q_emit_external_signed),
        "Q_injected_external_signed_C": float(surface_Q_injected_external_signed),
        "Q_injected_surface_total_abs_C": float(surface_Q_injected_total_abs),
        "Q_injected_surface_anode_abs_C": float(surface_Q_injected_anode_abs),
        "Q_injected_surface_cathode_abs_C": float(surface_Q_injected_cathode_abs),
        "max_equivalent_boundary_flux_dn_per_substep_m3": float(
            surface_max_equivalent_dn_per_substep_m3
        ),
    }
    surface_stats_path = Path(cfg.run.run_name) / "surface_emission_charge_stats.json"
    surface_stats_path.write_text(
        json.dumps(surface_emission_stats, indent=2),
        encoding="utf-8",
    )
    write_run_metadata(
        cfg,
        Nt=Nt,
        Nx=Nx,
        dt=dt,
        dx=dx,
        adaptive_stats=adaptive_stats,
        hotloop_stats=hotloop_stats,
        bc_poisson_picard_stats=bc_poisson_picard_stats,
        electron_transport_provenance=swarm_interp_cache.electron_transport_provenance(),
        ion_transport_provenance=swarm_interp_cache.ion_transport_provenance(),
    )

    if use_adaptive_substepping:
        print(
            "Adaptive substepping stats: "
            f"total_substeps={adaptive_stats['total_substeps']}, "
            f"mean_per_macro={adaptive_stats['mean_substeps_per_macro']:.3f}, "
            f"max_used={adaptive_stats['max_substeps_used']}, "
            f"overflow_events={adaptive_stats['overflow_events']}, "
            f"limited_by=(drift={adaptive_stats['steps_limited_by_drift']}, "
            f"diffusion={adaptive_stats['steps_limited_by_diffusion']}, "
            f"both={adaptive_stats['steps_limited_by_both']}), "
            f"target_drift_cfl={adaptive_stats['target_cfl_substep']:.3f}, "
            f"target_diffusion_cfl={adaptive_stats['target_diffusion_cfl_substep']:.3f}, "
            f"max_pre_step_cfl={adaptive_stats['max_pre_step_cfl_estimate']:.3f}, "
            f"max_realized_substep_cfl={adaptive_stats['max_realized_substep_cfl']:.3f}, "
            f"max_pre_step_diffusion_cfl="
            f"{adaptive_stats['max_pre_step_diffusion_cfl_estimate']:.3f}, "
            f"max_realized_substep_diffusion_cfl="
            f"{adaptive_stats['max_realized_substep_diffusion_cfl']:.3f}"
        )
    print(
        "BC+Poisson Picard stats: "
        f"total_solves={bc_poisson_picard_stats['total_solves']}, "
        f"mean_iter={bc_poisson_picard_stats['mean_iterations_per_solve']:.3f}, "
        f"max_iter={bc_poisson_picard_stats['max_iterations']}, "
        f"nonconverged={bc_poisson_picard_stats['nonconverged_solves']}, "
        f"max_residual={bc_poisson_picard_stats['max_residual']:.3e}"
    )
    print(
        "Surface emission charge stats: "
        f"Q_emit_ext_signed={surface_emission_stats['Q_emit_external_signed_C']:.6e} C, "
        f"Q_injected_ext_signed={surface_emission_stats['Q_injected_external_signed_C']:.6e} C, "
        f"Q_injected_total_abs={surface_emission_stats['Q_injected_surface_total_abs_C']:.6e} C"
    )
    print(
        "Hotloop backend: "
        f"requested={hotloop_stats['requested_backend']}, "
        f"effective={hotloop_stats['effective_backend']}, "
        f"numba_parallel={hotloop_stats['numba_parallel']}"
    )

    # ------------------------------------------------------------
    # 10) Pack final state and return.
    # ------------------------------------------------------------
    return SimulationState(
        cfg=cfg,
        time=time,
        x=x_array,
        V_gap=np.array(V_gap),
        I_discharge=np.array(I_discharge),
        c_cfl=np.array(c_cfl),
        diffusion_cfl=np.array(diffusion_cfl),
        I_transport_plasma=np.array(I_transport_plasma),
        I_transport_circuit=np.array(I_transport_circuit),
        I_emission_circuit=np.array(I_emission_circuit),
        I_emission_area=np.array(I_emission_area),
        I_displacement_gap=np.array(I_displacement_gap),
        V_node=np.array(V_n) if V_n is not None else None,
        V_source=None,
        ne_final=ne_curr.copy(),
        ni_final=ni_curr.copy(),
        phi_final=phi_curr.copy(),
        E_final=E_curr.copy(),
        mu_e_final=mu_e_row.copy(),
        D_e_final=D_e_row.copy(),
        mu_i_final=mu_i_row.copy(),
        D_i_final=D_i_row.copy(),
        picard_iterations=np.array(picard_iterations),
        adaptive_substeps=np.array(adaptive_substeps),
        adaptive_dt_sub=np.array(adaptive_dt_sub),
        adaptive_cfl_est=np.array(adaptive_cfl_est),
        adaptive_diffusion_cfl_est=np.array(adaptive_diffusion_cfl_est),
    )


def main() -> None:
    """Run a selected case module from the command line."""
    import argparse

    from config_loader import load_simulation_case
    from version import __version__

    parser = argparse.ArgumentParser(description="Run a PASCHEN-1D case.")
    parser.add_argument(
        "--config",
        default="config",
        help="Case-module name or .py filename (default: config).",
    )
    parser.add_argument(
        "--run-name",
        help="Optional portable output-directory override.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args()
    config_type, runner = load_simulation_case(args.config)
    cfg = config_type()
    if args.run_name:
        cfg.run.run_name = args.run_name
    runner(cfg)


if __name__ == "__main__":
    main()
