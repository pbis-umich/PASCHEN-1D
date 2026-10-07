"""
Notebook-facing postprocessing utilities for PASCHEN-1D.

This module keeps the public postprocessing notebooks compact and consistent.
It reads saved memmap outputs from a run folder and provides small plotting
helpers for temporal histories, instantaneous spatial snapshots, averaged
spatial profiles, grouped diagnostics, and derived sheath diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

import diagnostics_plotting as _plotting
from physical_constants import e
from derived_diagnostics import _v_app_from_metadata, compute_sheath_diagnostics


TEMPORAL_FILES = {
    "V_gap": "Vgap_mm.dat",
    "V_node": "Vnode_mm.dat",
    "V_source": "Vsource_mm.dat",
    "I_discharge": "Idischarge_mm.dat",
    "I_transport_plasma": "I_transport_plasma_mm.dat",
    "I_transport_circuit": "I_transport_circuit_mm.dat",
    "I_emission_circuit": "I_emission_circuit_mm.dat",
    "I_emission_area": "I_emission_area_mm.dat",
    "I_displacement_gap": "I_displacement_gap_mm.dat",
    "cfl": "c_cfl_mm.dat",
    "diffusion_cfl": "diffusion_cfl_mm.dat",
    "picard_iterations": "picard_iterations_mm.dat",
    "adaptive_substeps": "adaptive_substeps_mm.dat",
    "adaptive_dt_sub": "adaptive_dt_sub_mm.dat",
    "adaptive_cfl_est": "adaptive_cfl_est_mm.dat",
    "adaptive_diffusion_cfl_est": "adaptive_diffusion_cfl_est_mm.dat",
}

SPATIAL_FILES = {
    "ne": "ne_sampled_mm.dat",
    "ni": "ni_sampled_mm.dat",
    "phi": "phi_sampled_mm.dat",
    "E": "E_sampled_mm.dat",
    "Gamma_i": "Gamma_i_sampled_mm.dat",
    "Gamma_e": "Gamma_e_sampled_mm.dat",
    "townsend_alpha": "townsend_alpha_sampled_mm.dat",
    "nu_i": "nu_i_sampled_mm.dat",
    "S_ion": "S_ion_sampled_mm.dat",
    "S": "S_sampled_mm.dat",
    "mu_e": "mu_e_sampled_mm.dat",
    "D_e": "D_e_sampled_mm.dat",
    "mu_i": "mu_i_sampled_mm.dat",
    "D_i": "D_i_sampled_mm.dat",
}

SHEATH_METRIC_KEYS = {
    "edge_x_m": "edge_x_m",
    "width_m": "width_m",
    "voltage_drop_abs_V": "voltage_drop_abs_V",
    "peak_abs_E_V_m": "peak_abs_E_V_m",
    "space_charge_sigma_C_m2": "space_charge_sigma_C_m2",
    "space_charge_Q_C": "space_charge_Q_C",
}

DEFAULT_YLABELS = _plotting.DEFAULT_YLABELS
DISPLAY_LABELS = _plotting.DISPLAY_LABELS
SHEATH_DISPLAY_LABELS = _plotting.SHEATH_DISPLAY_LABELS
SHEATH_DEFAULT_YLABELS = _plotting.SHEATH_DEFAULT_YLABELS


def display_label(name: str) -> str:
    """Return a Matplotlib-mathtext display label for a saved diagnostic name."""
    return _plotting.display_label(name)


def sheath_metric_label(metric: str) -> str:
    """Return a Matplotlib-mathtext display label for a derived sheath metric."""
    return _plotting.sheath_metric_label(metric)


def title_from_quantity(quantity: str, *, suffix: str = "") -> str:
    """Build a title from a diagnostic name while preserving math formatting."""
    return _plotting.title_from_quantity(quantity, suffix=suffix)


def title_from_sheath_metric(metric: str, *, prefix: str = "") -> str:
    """Build a title from a sheath metric name while preserving math formatting."""
    return _plotting.title_from_sheath_metric(metric, prefix=prefix)


def resolved_quantity_title(title, quantity: str, *, default_suffix: str = "") -> str:
    """Use a math label when a notebook passes an internal diagnostic name as title."""
    return _plotting.resolved_quantity_title(
        title, quantity, default_suffix=default_suffix
    )


def resolved_sheath_title(title, metric: str, *, prefix: str = "") -> str:
    """Use a math label when a notebook passes an internal sheath metric as title."""
    return _plotting.resolved_sheath_title(title, metric, prefix=prefix)


def resolved_context_title(title, profile_quantity: str) -> str:
    """Resolve sheath-context titles that may contain raw diagnostic names."""
    return _plotting.resolved_context_title(title, profile_quantity)


@dataclass(frozen=True)
class RunContext:
    """Saved-run metadata and grids used by notebook plotting helpers."""

    project_dir: Path
    run_name: str
    run_dir: Path
    meta: dict
    Nt: int
    Nx: int
    T_total: float
    L: float
    A: float
    save_every: int
    time: np.ndarray
    x: np.ndarray


def set_notebook_plot_style(
    *,
    font_size: float = 11,
    figure_dpi: int = 150,
    savefig_dpi: int = 600,
) -> None:
    """Set conservative publication-oriented Matplotlib defaults."""
    _plotting.set_notebook_plot_style(
        font_size=font_size,
        figure_dpi=figure_dpi,
        savefig_dpi=savefig_dpi,
    )


def discover_runs(project_dir: str | Path = ".") -> list[str]:
    """Return folders in project_dir that contain run_metadata.json."""
    project = Path(project_dir)
    return sorted(
        p.name for p in project.iterdir() if p.is_dir() and (p / "run_metadata.json").exists()
    )


def load_run_context(run_name: str, project_dir: str | Path = ".") -> RunContext:
    """Load metadata and construct uniform time/space grids for a saved run."""
    project = Path(project_dir)
    run_dir = project / run_name
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    Nt = int(meta["Nt"])
    Nx = int(meta["Nx"])
    T_total = float(meta["T_total"])
    L = float(meta["L"])
    A = float(meta.get("A", 1.0))
    save_every = int(meta["save_every"])
    return RunContext(
        project_dir=project,
        run_name=run_name,
        run_dir=run_dir,
        meta=meta,
        Nt=Nt,
        Nx=Nx,
        T_total=T_total,
        L=L,
        A=A,
        save_every=save_every,
        time=np.linspace(0.0, T_total, Nt, dtype=np.float64),
        x=np.linspace(0.0, L, Nx, dtype=np.float64),
    )


def print_run_summary(ctx: RunContext) -> None:
    """Print a compact summary for the selected run."""
    print(f"Selected run_name: {ctx.run_name}")
    print(
        f"Nt={ctx.Nt:,}, Nx={ctx.Nx}, T_total={ctx.T_total:.6g} s, "
        f"save_every={ctx.save_every}"
    )


def available_temporal(ctx: RunContext) -> list[str]:
    """Return scalar temporal diagnostics available for this run."""
    names = ["V_app"]
    names.extend(q for q, filename in TEMPORAL_FILES.items() if (ctx.run_dir / filename).exists())
    return names


def available_spatial(ctx: RunContext) -> list[str]:
    """Return sampled spatial diagnostics available for this run."""
    return [q for q, filename in SPATIAL_FILES.items() if (ctx.run_dir / filename).exists()]


def print_available_diagnostics(ctx: RunContext) -> None:
    """Print available scalar and spatial diagnostic names."""
    print("Available temporal diagnostics:")
    print(", ".join(available_temporal(ctx)))
    print("\nAvailable spatial diagnostics:")
    print(", ".join(available_spatial(ctx)))


def trapz_integral(y, *, x=None, axis=-1):
    """NumPy 1.x/2.x compatible trapezoidal integration."""
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:
        trapezoid = np.trapz
    return trapezoid(y, x=x, axis=axis)


def infer_scalar_dtype(path: Path, expected_count: int):
    """Infer float32/float64 dtype from a scalar memmap file size."""
    size = path.stat().st_size
    if size == expected_count * np.dtype(np.float64).itemsize:
        return np.float64
    if size == expected_count * np.dtype(np.float32).itemsize:
        return np.float32
    raise ValueError(f"Cannot infer dtype for {path}; file size is inconsistent.")


def sampled_count(ctx: RunContext, path: Path, dtype=np.float32) -> int:
    """Return number of saved snapshots in a sampled spatial memmap."""
    row_bytes = np.dtype(dtype).itemsize * ctx.Nx
    size = path.stat().st_size
    if size % row_bytes != 0:
        raise ValueError(f"Unexpected sampled file size for {path}")
    return size // row_bytes


def saved_times_for(ctx: RunContext, path: Path) -> np.ndarray:
    """Return saved snapshot times corresponding to a sampled spatial memmap."""
    nsave = sampled_count(ctx, path)
    saved_indices = np.arange(nsave, dtype=np.int64) * ctx.save_every
    saved_indices = np.minimum(saved_indices, ctx.Nt - 1)
    return ctx.time[saved_indices]


def reconstruct_v_app(ctx: RunContext, t: np.ndarray) -> np.ndarray:
    """Reconstruct applied voltage from saved waveform metadata."""
    meta = ctx.meta
    waveform_type = meta["waveform_type"]
    if waveform_type == "dc":
        return float(meta["V_peak"]) * np.ones_like(t)
    if waveform_type == "step":
        return (
            float(meta["V_peak"])
            * ((t >= float(meta["tV_start"])) & (t <= float(meta["tV_end"])))
        ).astype(np.float64)
    if waveform_type == "gaussian":
        return float(meta["V_peak"]) * np.exp(
            -((t - float(meta["t_peak"])) / float(meta["tau"])) ** 2
        )
    if waveform_type == "rf":
        omega = 2.0 * np.pi * float(meta["f_rf"])
        return float(meta["V_dc"]) + float(meta["V_peak"]) * np.sin(
            omega * t + float(meta["phi_rf"])
        )
    if waveform_type in ("table", "tabulated", "measured_table"):
        return _v_app_from_metadata(t, meta, project_dir=ctx.project_dir)
    raise ValueError(f"Unknown waveform_type: {waveform_type}")


def read_temporal(ctx: RunContext, quantity: str) -> tuple[np.ndarray, np.ndarray]:
    """Read or reconstruct one scalar temporal diagnostic."""
    if quantity == "V_app":
        return ctx.time, reconstruct_v_app(ctx, ctx.time)
    filename = TEMPORAL_FILES.get(quantity)
    if filename is None:
        raise KeyError(f"Unknown temporal quantity: {quantity}")
    path = ctx.run_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing temporal file for {quantity}: {path}")
    dtype = infer_scalar_dtype(path, ctx.Nt)
    values = np.memmap(path, mode="r", dtype=dtype, shape=(ctx.Nt,))
    return ctx.time, np.asarray(values, dtype=np.float64)


def read_spatial(ctx: RunContext, quantity: str):
    """Read one sampled spatial diagnostic as (saved_times, memmap array)."""
    filename = SPATIAL_FILES.get(quantity)
    if filename is None:
        raise KeyError(f"Unknown spatial quantity: {quantity}")
    path = ctx.run_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing spatial file for {quantity}: {path}")
    nsave = sampled_count(ctx, path)
    arr = np.memmap(path, mode="r", dtype=np.float32, shape=(nsave, ctx.Nx))
    return saved_times_for(ctx, path), arr


def apply_value_mode(y, mode: str):
    """Apply common signed/absolute transformations to plotted values."""
    return _plotting.apply_value_mode(y, mode)


def window_mask(t, start=None, end=None) -> np.ndarray:
    """Return a boolean mask for an optional time window."""
    return _plotting.window_mask(t, start, end)


def reduce_xy_for_plot(xv, yv, max_points=30000):
    """Downsample dense lines while preserving local minima and maxima."""
    return _plotting.reduce_xy_for_plot(xv, yv, max_points=max_points)


def scale_time(t, unit: str) -> tuple[np.ndarray, str]:
    """Scale time array and return axis label."""
    return _plotting.scale_time(t, unit)


def scale_x(x, unit: str) -> tuple[np.ndarray, str]:
    """Scale position array and return axis label."""
    return _plotting.scale_x(x, unit)


def save_figure(ctx: RunContext, fig, *, fig_name=None, outdir=None, save=False) -> None:
    """Save a figure as PDF and PNG when save=True."""
    if not save:
        return
    if outdir is None:
        outdir = ctx.run_dir / "postprocess_figures"
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    base = outdir / (fig_name or "figure")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    print(f"Saved: {base.with_suffix('.pdf')}")
    print(f"Saved: {base.with_suffix('.png')}")


def show_figure() -> None:
    """Show figures in notebooks while staying quiet in headless script tests."""
    _plotting.show_figure()


def summarize_spatial(ctx: RunContext, arr, summary="max", value_mode="raw"):
    """Reduce sampled spatial snapshots into one time trace."""
    y = apply_value_mode(arr, value_mode)
    if summary == "max":
        return np.nanmax(y, axis=1), "max"
    if summary == "min":
        return np.nanmin(y, axis=1), "min"
    if summary == "mean":
        return trapz_integral(y, x=ctx.x, axis=1) / ctx.L, "mean"
    if summary == "integral":
        return ctx.A * trapz_integral(y, x=ctx.x, axis=1), "integral over gap"
    if summary == "left":
        return y[:, 0], "left boundary"
    if summary == "right":
        return y[:, -1], "right boundary"
    raise ValueError(f"Unknown summary: {summary}")


def _apply_log_filter(y, yscale: str):
    return _plotting._apply_log_filter(y, yscale)


def plot_temporal_quantity(
    ctx: RunContext,
    quantity: str,
    *,
    t_start=None,
    t_end=None,
    t_unit="us",
    value_mode="raw",
    y_scale_factor=1.0,
    ylabel=None,
    title=None,
    xscale="linear",
    yscale="linear",
    figsize=(5.0, 3.5),
    color="C0",
    linestyle="-",
    linewidth=2.0,
    marker=None,
    label=None,
    grid=True,
    legend=True,
    legend_loc="best",
    max_points=30000,
    ylim=None,
    xlim=None,
    save=False,
    fig_name=None,
):
    """Plot one scalar temporal diagnostic."""
    t, y = read_temporal(ctx, quantity)
    fig, ax = _plotting.plot_temporal_quantity_from_arrays(
        t,
        y,
        quantity,
        t_start=t_start,
        t_end=t_end,
        t_unit=t_unit,
        value_mode=value_mode,
        y_scale_factor=y_scale_factor,
        ylabel=ylabel,
        title=title,
        xscale=xscale,
        yscale=yscale,
        figsize=figsize,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        marker=marker,
        label=label,
        grid=grid,
        legend=legend,
        legend_loc=legend_loc,
        max_points=max_points,
        ylim=ylim,
        xlim=xlim,
        show=False,
    )
    save_figure(ctx, fig, fig_name=fig_name or f"{quantity}_temporal", save=save)
    show_figure()
    return fig, ax


def plot_spatial_temporal_summary(
    ctx: RunContext,
    quantity: str,
    *,
    summary="max",
    t_start=None,
    t_end=None,
    t_unit="us",
    value_mode="raw",
    y_scale_factor=1.0,
    ylabel=None,
    title=None,
    xscale="linear",
    yscale="linear",
    figsize=(5.0, 3.5),
    color="C0",
    linestyle="-",
    linewidth=2.0,
    marker=None,
    grid=True,
    legend=True,
    legend_loc="best",
    max_points=30000,
    ylim=None,
    xlim=None,
    save=False,
    fig_name=None,
):
    """Plot a temporal summary derived from saved spatial snapshots."""
    ts, arr = read_spatial(ctx, quantity)

    if summary == "minmax":
        items = [("min", "C1"), ("max", color)]
    else:
        items = [(summary, color)]

    series = []
    for subsummary, subcolor in items:
        vals, label_suffix = summarize_spatial(ctx, arr, summary=subsummary, value_mode=value_mode)
        series.append(
            {
                "time": ts,
                "values": vals,
                "quantity": quantity,
                "scale": y_scale_factor,
                "label": f"{display_label(quantity)} {label_suffix}",
                "color": subcolor,
                "linestyle": linestyle,
                "marker": marker,
            }
        )

    summary_title = f"{title_from_quantity(quantity)}: {summary} vs time"
    fig, ax = _plotting.plot_temporal_group_from_series(
        series,
        t_start=t_start,
        t_end=t_end,
        t_unit=t_unit,
        ylabel=ylabel or DEFAULT_YLABELS.get(quantity, "Value"),
        title=summary_title if title is None or title == f"{quantity}: {summary} vs time" else title,
        xscale=xscale,
        yscale=yscale,
        figsize=figsize,
        linewidth=linewidth,
        grid=grid,
        legend=legend,
        legend_loc=legend_loc,
        max_points=max_points,
        ylim=ylim,
        xlim=xlim,
        show=False,
    )
    save_figure(ctx, fig, fig_name=fig_name or f"{quantity}_temporal_summary", save=save)
    show_figure()
    return fig, ax


def nearest_snapshot_indices(ts: np.ndarray, t_samples) -> np.ndarray:
    """Return unique nearest saved snapshot indices for requested times."""
    if t_samples is None:
        return np.array([len(ts) - 1], dtype=int)
    return np.array(
        sorted({int(np.argmin(np.abs(ts - float(t)))) for t in t_samples}),
        dtype=int,
    )


def plot_spatial_profiles(
    ctx: RunContext,
    quantity: str,
    *,
    t_samples=None,
    x_unit="cm",
    value_mode="raw",
    y_scale_factor=1.0,
    ylabel=None,
    title=None,
    xscale="linear",
    yscale="linear",
    figsize=(5.0, 3.5),
    cmap="viridis",
    linewidth=2.0,
    linestyle="-",
    grid=True,
    legend=True,
    legend_loc="best",
    colorbar=True,
    time_unit_for_labels="us",
    ylim=None,
    xlim=None,
    save=False,
    fig_name=None,
):
    """Plot one spatial diagnostic at selected saved snapshot times."""
    ts, arr = read_spatial(ctx, quantity)
    idx = nearest_snapshot_indices(ts, t_samples)
    fig, ax = _plotting.plot_spatial_profiles_from_arrays(
        ctx.x,
        np.asarray(arr[idx], dtype=np.float64),
        ts[idx],
        quantity,
        x_unit=x_unit,
        value_mode=value_mode,
        y_scale_factor=y_scale_factor,
        ylabel=ylabel,
        title=title,
        xscale=xscale,
        yscale=yscale,
        figsize=figsize,
        cmap=cmap,
        linewidth=linewidth,
        linestyle=linestyle,
        grid=grid,
        legend=legend,
        legend_loc=legend_loc,
        colorbar=colorbar,
        time_unit_for_labels=time_unit_for_labels,
        ylim=ylim,
        xlim=xlim,
        show=False,
    )
    save_figure(ctx, fig, fig_name=fig_name or f"{quantity}_spatial_profiles", save=save)
    show_figure()
    return fig, ax


def averaging_indices(
    ctx: RunContext,
    ts: np.ndarray,
    *,
    averaged_mode="time_window",
    t_avg_start=None,
    t_avg_end=None,
    N_cycle_avg=1,
) -> tuple[np.ndarray, str]:
    """Return snapshot mask and a human-readable averaging label."""
    if averaged_mode == "time_window":
        if t_avg_start is None:
            t_avg_start = float(ts[0])
        if t_avg_end is None:
            t_avg_end = float(ts[-1])
        if t_avg_start > t_avg_end:
            t_avg_start, t_avg_end = t_avg_end, t_avg_start
        mask = (ts >= float(t_avg_start)) & (ts <= float(t_avg_end))
        label = f"average: {float(t_avg_start)*1e6:.4g}-{float(t_avg_end)*1e6:.4g} us"
        return mask, label

    if averaged_mode == "last_n_cycles":
        f_rf = float(ctx.meta.get("f_rf", 0.0))
        if f_rf <= 0.0:
            raise ValueError("last_n_cycles averaging requires positive f_rf in run metadata.")
        period = 1.0 / f_rf
        duration = max(int(N_cycle_avg), 1) * period
        t_avg_end = float(ts[-1]) if t_avg_end is None else float(t_avg_end)
        t_avg_start = t_avg_end - duration
        mask = (ts >= t_avg_start) & (ts <= t_avg_end)
        label = f"average: last {max(int(N_cycle_avg), 1)} cycle(s)"
        return mask, label

    raise ValueError(f"Unknown averaged_mode: {averaged_mode}")


def average_profiles(profiles: np.ndarray, *, statistic="mean") -> np.ndarray:
    """Average sampled profiles using a selected statistic."""
    y = np.asarray(profiles, dtype=np.float64)
    if statistic == "mean":
        return np.nanmean(y, axis=0)
    if statistic == "mean_abs":
        return np.nanmean(np.abs(y), axis=0)
    if statistic == "rms":
        return np.sqrt(np.nanmean(y * y, axis=0))
    if statistic == "max":
        return np.nanmax(y, axis=0)
    if statistic == "min":
        return np.nanmin(y, axis=0)
    raise ValueError(f"Unknown statistic: {statistic}")


def plot_averaged_spatial_profile(
    ctx: RunContext,
    quantity: str,
    *,
    averaged_mode="time_window",
    t_avg_start=None,
    t_avg_end=None,
    N_cycle_avg=1,
    statistic="mean",
    value_mode="raw",
    x_unit="cm",
    y_scale_factor=1.0,
    ylabel=None,
    title=None,
    xscale="linear",
    yscale="linear",
    figsize=(5.0, 3.5),
    color="C0",
    linestyle="-",
    linewidth=2.0,
    grid=True,
    legend=True,
    legend_loc="best",
    ylim=None,
    xlim=None,
    save=False,
    fig_name=None,
):
    """Plot a time-window or last-N-cycle averaged spatial profile."""
    ts, arr = read_spatial(ctx, quantity)
    mask, averaging_label = averaging_indices(
        ctx,
        ts,
        averaged_mode=averaged_mode,
        t_avg_start=t_avg_start,
        t_avg_end=t_avg_end,
        N_cycle_avg=N_cycle_avg,
    )
    if not np.any(mask):
        raise RuntimeError("Averaging window contains no saved spatial snapshots.")
    profiles = apply_value_mode(arr[mask], value_mode)
    profile = average_profiles(profiles, statistic=statistic) * y_scale_factor
    fig, ax = _plotting.plot_averaged_spatial_profile_from_array(
        ctx.x,
        profile,
        quantity,
        averaging_label=averaging_label,
        x_unit=x_unit,
        value_mode="raw",
        y_scale_factor=1.0,
        ylabel=ylabel,
        title=title,
        xscale=xscale,
        yscale=yscale,
        figsize=figsize,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        grid=grid,
        legend=legend,
        legend_loc=legend_loc,
        ylim=ylim,
        xlim=xlim,
        show=False,
    )
    save_figure(ctx, fig, fig_name=fig_name or f"{quantity}_averaged_spatial", save=save)
    show_figure()
    return fig, ax


def compute_sheath_rows(
    ctx: RunContext,
    *,
    quasineutrality_tol=0.05,
    density_floor_fraction=1.0e-8,
    density_floor_m3=None,
    save=True,
):
    """Compute derived sheath rows using the public postprocess implementation."""
    return compute_sheath_diagnostics(
        str(ctx.run_dir),
        quasineutrality_tol=quasineutrality_tol,
        density_floor_fraction=density_floor_fraction,
        density_floor_m3=density_floor_m3,
        save=save,
    )


def sheath_row_values(sheath_rows, name: str) -> np.ndarray:
    """Extract a float array from sheath diagnostic rows."""
    return np.array(
        [np.nan if row.get(name) is None else float(row[name]) for row in sheath_rows],
        dtype=np.float64,
    )


def sheath_metric(sheath_rows, side: str, metric: str) -> np.ndarray:
    """Extract a side-specific sheath metric array."""
    key = SHEATH_METRIC_KEYS.get(metric, metric)
    return sheath_row_values(sheath_rows, f"{side}_{key}")


def print_sheath_validity(sheath_rows) -> None:
    """Print valid-snapshot ranges for derived sheath rows."""
    print(f"Computed sheath diagnostics for {len(sheath_rows)} saved snapshots.")
    for side in ("anode", "cathode"):
        n_valid = sum(row.get(f"{side}_edge_found") == "yes" for row in sheath_rows)
        if n_valid == 0:
            print(f"  {side}: no valid sheath edges found")
            continue
        t_valid = np.array(
            [row["time_s"] for row in sheath_rows if row.get(f"{side}_edge_found") == "yes"],
            dtype=float,
        )
        print(
            f"  {side}: {n_valid} valid snapshots, "
            f"t = {t_valid.min()*1e6:.4g} to {t_valid.max()*1e6:.4g} us"
        )


def nearest_sheath_indices(sheath_rows, t_samples) -> np.ndarray:
    """Return nearest valid sheath-row indices for requested times."""
    t = sheath_row_values(sheath_rows, "time_s")
    finite = np.isfinite(t)
    if not np.any(finite):
        return np.array([], dtype=int)
    if t_samples is None:
        return np.array([int(np.flatnonzero(finite)[-1])], dtype=int)
    finite_idx = np.flatnonzero(finite)
    t_finite = t[finite_idx]
    out = [int(finite_idx[np.argmin(np.abs(t_finite - float(ts)))]) for ts in t_samples]
    return np.array(sorted(set(out)), dtype=int)


def plot_sheath_temporal_metric(
    ctx: RunContext,
    sheath_rows,
    metric: str,
    *,
    sides=("anode", "cathode"),
    t_start=None,
    t_end=None,
    t_unit="us",
    y_scale_factor=1.0,
    ylabel=None,
    title=None,
    xscale="linear",
    yscale="linear",
    figsize=(5.0, 3.5),
    linewidth=2.0,
    linestyle="-",
    marker=None,
    grid=True,
    legend=True,
    legend_loc="best",
    xlim=None,
    ylim=None,
    save=False,
    fig_name=None,
):
    """Plot one anode/cathode sheath metric versus time."""
    t = sheath_row_values(sheath_rows, "time_s")
    anode_values = sheath_metric(sheath_rows, "anode", metric)
    cathode_values = sheath_metric(sheath_rows, "cathode", metric)
    if tuple(sides) == ("cathode",):
        anode_values = np.full_like(cathode_values, np.nan)
    elif tuple(sides) == ("anode",):
        cathode_values = np.full_like(anode_values, np.nan)
    fig, ax = _plotting.plot_sheath_temporal_metric_from_arrays(
        t,
        anode_values,
        cathode_values,
        metric,
        t_start=t_start,
        t_end=t_end,
        t_unit=t_unit,
        y_scale_factor=y_scale_factor,
        ylabel=ylabel,
        title=title,
        xscale=xscale,
        yscale=yscale,
        figsize=figsize,
        linewidth=linewidth,
        grid=grid,
        legend=legend,
        legend_loc=legend_loc,
        ylim=ylim,
        xlim=xlim,
        show=False,
    )
    for line in ax.lines:
        line.set_linestyle(linestyle)
        if marker is not None:
            line.set_marker(marker)
    save_figure(ctx, fig, fig_name=fig_name or f"sheath_{metric}_temporal", save=save)
    show_figure()
    return fig, ax


def sheath_profile_data(
    ctx: RunContext,
    profile_quantity: str,
    k: int,
    *,
    sheath_density_floor_fraction=1.0e-8,
    sheath_density_floor_m3=None,
):
    """Return a spatial profile used for sheath-context plots."""
    if profile_quantity in ("eta", "quasineutrality_metric"):
        _, ne_arr = read_spatial(ctx, "ne")
        _, ni_arr = read_spatial(ctx, "ni")
        ne_row = np.asarray(ne_arr[k], dtype=np.float64)
        ni_row = np.asarray(ni_arr[k], dtype=np.float64)
        mean_density = 0.5 * (ne_row + ni_row)
        density_scale = float(np.nanmax(mean_density)) if mean_density.size else 0.0
        if sheath_density_floor_m3 is None:
            floor = max(sheath_density_floor_fraction * density_scale, 0.0)
        else:
            floor = float(sheath_density_floor_m3)
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.abs(ni_row - ne_row) / np.maximum(mean_density, floor)
        return y, r"$\eta=|n_i-n_e|/\max[0.5(n_i+n_e),n_\mathrm{floor}]$"
    if profile_quantity == "phi":
        _, arr = read_spatial(ctx, "phi")
        return np.asarray(arr[k], dtype=np.float64), r"$\phi$ [V]"
    if profile_quantity == "E":
        _, arr = read_spatial(ctx, "E")
        return np.asarray(arr[k], dtype=np.float64), r"$E$ [V/m]"
    if profile_quantity == "abs_E":
        _, arr = read_spatial(ctx, "E")
        return np.abs(np.asarray(arr[k], dtype=np.float64)), r"$|E|$ [V/m]"
    if profile_quantity == "rho":
        _, ne_arr = read_spatial(ctx, "ne")
        _, ni_arr = read_spatial(ctx, "ni")
        rho = e * (np.asarray(ni_arr[k], dtype=np.float64) - np.asarray(ne_arr[k], dtype=np.float64))
        return rho, r"$\rho=e(n_i-n_e)$ [C/m$^3$]"
    raise ValueError(f"Unknown sheath spatial profile quantity: {profile_quantity}")


def plot_sheath_spatial_context(
    ctx: RunContext,
    sheath_rows,
    *,
    profile_quantity="eta",
    t_samples=None,
    sides=("anode", "cathode"),
    x_unit="cm",
    y_scale_factor=1.0,
    ylabel=None,
    title=None,
    xscale="linear",
    yscale="linear",
    figsize=(5.2, 3.7),
    cmap="viridis",
    linewidth=2.0,
    linestyle="-",
    grid=True,
    colorbar=True,
    show_edge_markers=True,
    edge_marker_alpha=0.8,
    show_threshold=True,
    sheath_quasineutrality_tol=0.05,
    sheath_density_floor_fraction=1.0e-8,
    sheath_density_floor_m3=None,
    shade_latest_sheath=False,
    shade_alpha=0.12,
    time_unit_for_labels="us",
    xlim=None,
    ylim=None,
    save=False,
    fig_name=None,
):
    """Plot sheath-context spatial profiles with inferred sheath-edge markers."""
    idx = nearest_sheath_indices(sheath_rows, t_samples)
    if idx.size == 0:
        raise RuntimeError("No sheath snapshot times are available.")

    sample_times = sheath_row_values(sheath_rows, "time_s")[idx]
    profiles = []
    default_ylabel = "Value"
    for k in idx:
        y, default_ylabel = sheath_profile_data(
            ctx,
            profile_quantity,
            k,
            sheath_density_floor_fraction=sheath_density_floor_fraction,
            sheath_density_floor_m3=sheath_density_floor_m3,
        )
        profiles.append(y)

    anode_edges = sheath_metric(sheath_rows, "anode", "edge_x_m")[idx]
    cathode_edges = sheath_metric(sheath_rows, "cathode", "edge_x_m")[idx]
    if not show_edge_markers or "anode" not in sides:
        anode_edges = None
    if not show_edge_markers or "cathode" not in sides:
        cathode_edges = None

    fig, ax = _plotting.plot_sheath_spatial_context_from_arrays(
        ctx.x,
        np.vstack(profiles),
        sample_times,
        profile_quantity,
        anode_edges=anode_edges,
        cathode_edges=cathode_edges,
        x_unit=x_unit,
        time_unit_for_labels=time_unit_for_labels,
        value_mode="raw",
        y_scale_factor=y_scale_factor,
        ylabel=ylabel or default_ylabel,
        title=title,
        xscale=xscale,
        yscale=yscale,
        figsize=figsize,
        cmap=cmap,
        linewidth=linewidth,
        linestyle=linestyle,
        grid=grid,
        legend=True,
        colorbar=colorbar,
        ylim=ylim,
        xlim=xlim,
        show=False,
    )

    if show_edge_markers and edge_marker_alpha != 0.65:
        for line in ax.lines:
            if line.get_linestyle() in (":", "--") and len(line.get_xdata()) == 2:
                line.set_alpha(edge_marker_alpha)

    if profile_quantity in ("eta", "quasineutrality_metric") and show_threshold:
        ax.axhline(
            sheath_quasineutrality_tol,
            color="k",
            linestyle="--",
            linewidth=1.2,
            label=r"$\eta_\mathrm{tol}$",
        )

    if shade_latest_sheath:
        k = int(idx[-1])
        xp, _ = scale_x(ctx.x, x_unit)
        for side, color in (("anode", "tab:blue"), ("cathode", "tab:red")):
            edge = sheath_metric(sheath_rows, side, "edge_x_m")[k]
            if not np.isfinite(edge):
                continue
            edge_plot, _ = scale_x(np.array([edge]), x_unit)
            if side == "anode":
                ax.axvspan(xp[0], float(edge_plot[0]), color=color, alpha=shade_alpha)
            else:
                ax.axvspan(float(edge_plot[0]), xp[-1], color=color, alpha=shade_alpha)

    if (not colorbar or idx.size <= 1) and ax.get_legend() is None:
        ax.legend(frameon=True, loc="best")
    fig.tight_layout()
    save_figure(
        ctx,
        fig,
        fig_name=fig_name or f"sheath_{profile_quantity}_spatial_context",
        save=save,
    )
    show_figure()
    return fig, ax


def plot_temporal_group(
    ctx: RunContext,
    custom_group: Sequence[str],
    *,
    spatial_summary_by_quantity: dict[str, str] | None = None,
    value_mode_by_quantity: dict[str, str] | None = None,
    scale_factor_by_quantity: dict[str, float] | None = None,
    label_by_quantity: dict[str, str] | None = None,
    t_start=None,
    t_end=None,
    t_unit="us",
    ylabel="Value",
    title="Grouped temporal diagnostics",
    xscale="linear",
    yscale="linear",
    figsize=(5.4, 3.6),
    linewidth=2.0,
    grid=True,
    legend=True,
    legend_loc="best",
    max_points=30000,
    xlim=None,
    ylim=None,
    save=False,
    fig_name="temporal_custom_group",
):
    """Plot a custom group of scalar or spatial-summary temporal diagnostics."""
    spatial_summary_by_quantity = spatial_summary_by_quantity or {}
    value_mode_by_quantity = value_mode_by_quantity or {}
    scale_factor_by_quantity = scale_factor_by_quantity or {}
    label_by_quantity = label_by_quantity or {}

    series = []
    for q in custom_group:
        value_mode = value_mode_by_quantity.get(q, "raw")
        scale = scale_factor_by_quantity.get(q, 1.0)
        if q in available_temporal(ctx):
            t, y = read_temporal(ctx, q)
        elif q in available_spatial(ctx):
            ts, arr = read_spatial(ctx, q)
            summary = spatial_summary_by_quantity.get(q, "max")
            y, _ = summarize_spatial(ctx, arr, summary=summary, value_mode=value_mode)
            t = ts
            value_mode = "raw"
        else:
            print(f"Skipping unavailable diagnostic: {q}")
            continue
        series.append(
            {
                "time": t,
                "values": y,
                "quantity": q,
                "value_mode": value_mode,
                "scale": scale,
                "label": label_by_quantity.get(q, display_label(q)),
            }
        )

    fig, ax = _plotting.plot_temporal_group_from_series(
        series,
        t_start=t_start,
        t_end=t_end,
        t_unit=t_unit,
        ylabel=ylabel,
        title=title,
        xscale=xscale,
        yscale=yscale,
        figsize=figsize,
        linewidth=linewidth,
        grid=grid,
        legend=legend,
        legend_loc=legend_loc,
        max_points=max_points,
        xlim=xlim,
        ylim=ylim,
        show=False,
    )
    save_figure(ctx, fig, fig_name=fig_name, save=save)
    show_figure()
    return fig, ax


def plot_spatial_group_snapshot(
    ctx: RunContext,
    custom_group: Sequence[str],
    *,
    t_sample=None,
    t_samples=None,
    x_unit="cm",
    time_unit_for_labels="us",
    value_mode_by_quantity: dict[str, str] | None = None,
    scale_factor_by_quantity: dict[str, float] | None = None,
    label_by_quantity: dict[str, str] | None = None,
    ylabel="Value",
    title="Grouped spatial snapshot",
    xscale="linear",
    yscale="linear",
    figsize=(5.4, 3.6),
    linewidth=2.0,
    cmap="viridis",
    colorbar=False,
    grid=True,
    legend=True,
    legend_loc="best",
    show_time_annotation=True,
    xlim=None,
    ylim=None,
    save=False,
    fig_name="spatial_snapshot_custom_group",
):
    """Plot a custom group of spatial diagnostics at one or more snapshot times."""
    if t_sample is not None and t_samples is not None:
        raise ValueError("Use either t_sample for one snapshot or t_samples for multiple snapshots, not both.")

    if t_samples is None:
        requested_times = None if t_sample is None else (float(t_sample),)
    elif np.isscalar(t_samples):
        requested_times = (float(t_samples),)
    else:
        requested_times = tuple(float(t) for t in t_samples)

    value_mode_by_quantity = value_mode_by_quantity or {}
    scale_factor_by_quantity = scale_factor_by_quantity or {}
    label_by_quantity = label_by_quantity or {}

    actual_times = []
    profiles = []
    linestyle_cycle = ("-", "--", "-.", ":")
    linestyle_by_quantity = {
        q: linestyle_cycle[i % len(linestyle_cycle)]
        for i, q in enumerate(custom_group)
    }
    for q in custom_group:
        if q not in available_spatial(ctx):
            print(f"Skipping unavailable spatial diagnostic: {q}")
            continue
        ts, arr = read_spatial(ctx, q)
        value_mode = value_mode_by_quantity.get(q, "raw")
        scale = scale_factor_by_quantity.get(q, 1.0)
        base_label = label_by_quantity.get(q, display_label(q))
        idx = nearest_snapshot_indices(ts, requested_times)
        st_plot, _ = scale_time(ts[idx], time_unit_for_labels)
        include_time_in_label = (len(idx) > 1 or len(custom_group) > 1) and not colorbar
        for k, t_scaled in zip(idx, st_plot):
            actual_times.append(float(ts[k]))
            label = base_label
            if include_time_in_label:
                label = f"{base_label}, {t_scaled:.4g} {time_unit_for_labels}"
            profiles.append(
                {
                    "values": np.asarray(arr[k], dtype=np.float64),
                    "quantity": q,
                    "value_mode": value_mode,
                    "scale": scale,
                    "label": label,
                    "linestyle": linestyle_by_quantity.get(q, "-"),
                    "time_scaled": float(t_scaled),
                }
            )
    if actual_times:
        unique_times = np.unique(np.round(np.asarray(actual_times, dtype=np.float64), decimals=15))
        if len(unique_times) == 1:
            t_scaled, _ = scale_time(unique_times, time_unit_for_labels)
            t_label = f"t = {float(t_scaled[0]):.4g} {time_unit_for_labels}"
        else:
            t_label = f"{len(unique_times)} snapshots"
    else:
        t_label = "no available profiles"

    fig, ax = _plotting.plot_spatial_group_from_profiles(
        ctx.x,
        profiles,
        t_label=t_label,
        x_unit=x_unit,
        ylabel=ylabel,
        title=title if title is not None else t_label,
        xscale=xscale,
        yscale=yscale,
        figsize=figsize,
        linewidth=linewidth,
        cmap=cmap,
        colorbar=colorbar,
        colorbar_label=scale_time(np.array([0.0]), time_unit_for_labels)[1],
        grid=grid,
        legend=legend,
        legend_loc=legend_loc,
        show_time_annotation=show_time_annotation,
        xlim=xlim,
        ylim=ylim,
        show=False,
    )
    save_figure(ctx, fig, fig_name=fig_name, save=save)
    show_figure()
    return fig, ax


def plot_averaged_spatial_group(
    ctx: RunContext,
    custom_group: Sequence[str],
    *,
    averaged_mode="time_window",
    t_avg_start=None,
    t_avg_end=None,
    N_cycle_avg=1,
    statistic_by_quantity: dict[str, str] | None = None,
    value_mode_by_quantity: dict[str, str] | None = None,
    scale_factor_by_quantity: dict[str, float] | None = None,
    label_by_quantity: dict[str, str] | None = None,
    x_unit="cm",
    ylabel="Value",
    title="Grouped averaged spatial diagnostics",
    xscale="linear",
    yscale="linear",
    figsize=(5.4, 3.6),
    linewidth=2.0,
    grid=True,
    legend=True,
    legend_loc="best",
    xlim=None,
    ylim=None,
    save=False,
    fig_name="averaged_spatial_custom_group",
):
    """Plot a custom group of averaged spatial profiles."""
    statistic_by_quantity = statistic_by_quantity or {}
    value_mode_by_quantity = value_mode_by_quantity or {}
    scale_factor_by_quantity = scale_factor_by_quantity or {}
    label_by_quantity = label_by_quantity or {}

    profiles_to_plot = []
    averaging_label = ""
    for q in custom_group:
        if q not in available_spatial(ctx):
            print(f"Skipping unavailable spatial diagnostic: {q}")
            continue
        ts, arr = read_spatial(ctx, q)
        mask, averaging_label = averaging_indices(
            ctx,
            ts,
            averaged_mode=averaged_mode,
            t_avg_start=t_avg_start,
            t_avg_end=t_avg_end,
            N_cycle_avg=N_cycle_avg,
        )
        if not np.any(mask):
            print(f"Skipping {q}: averaging window contains no snapshots.")
            continue
        value_mode = value_mode_by_quantity.get(q, "raw")
        statistic = statistic_by_quantity.get(q, "mean")
        scale = scale_factor_by_quantity.get(q, 1.0)
        profiles = apply_value_mode(arr[mask], value_mode)
        profiles_to_plot.append(
            {
                "values": average_profiles(profiles, statistic=statistic),
                "quantity": q,
                "value_mode": "raw",
                "scale": scale,
                "label": label_by_quantity.get(q, display_label(q)),
            }
        )

    fig, ax = _plotting.plot_averaged_spatial_group_from_profiles(
        ctx.x,
        profiles_to_plot,
        averaging_label=averaging_label,
        x_unit=x_unit,
        ylabel=ylabel,
        title=title,
        xscale=xscale,
        yscale=yscale,
        figsize=figsize,
        linewidth=linewidth,
        grid=grid,
        legend=legend,
        legend_loc=legend_loc,
        xlim=xlim,
        ylim=ylim,
        show=False,
    )
    save_figure(
        ctx,
        fig,
        fig_name=fig_name,
        save=save,
    )
    show_figure()
    return fig, ax
