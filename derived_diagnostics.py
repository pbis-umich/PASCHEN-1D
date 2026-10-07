"""
derived_diagnostics.py

Derived diagnostics for saved PASCHEN-1D runs.

This module computes waveform metrics, current-decomposition metrics, sheath
diagnostics, and the retained saved-run replotting workflow without rerunning
the simulation.
"""

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from physical_constants import e
from physics import make_voltage_waveform


def _trapezoid_integral(y, *, x=None, axis=-1):
    """NumPy 1.x/2.x compatible trapezoidal integration."""
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:
        trapezoid = np.trapz
    return trapezoid(y, x=x, axis=axis)


TemporalQuantity = Literal[
    "V_app",
    "V_node",
    "V_source",
    "V_gap",
    "I_discharge",
    "I_transport_plasma",
    "I_transport_circuit",
    "I_emission_circuit",
    "I_emission_area",
    "I_displacement_gap",
    "cfl",
    "diffusion_cfl",
    "picard_iterations",
    "adaptive_substeps",
    "adaptive_dt_sub",
    "adaptive_cfl_est",
    "adaptive_diffusion_cfl_est",
    "particle_inventory",
]
SpatialQuantity = Literal[
    "ne",
    "ni",
    "phi",
    "E",
    "Gamma_i",
    "Gamma_e",
    "townsend_alpha",
    "nu_i",
    "S_ion",
    "S",
    "mu_e",
    "D_e",
]

AveragedSpatialMode = Literal["time_window", "last_n_cycles"]

CURRENT_TEMPORAL_QUANTITIES = {
    "I_discharge",
    "I_transport_plasma",
    "I_transport_circuit",
    "I_emission_circuit",
    "I_emission_area",
    "I_displacement_gap",
}


@dataclass
class TemporalReplotStyle:
    t_unit: Literal["s", "ms", "us", "ns"] = "ns"
    xscale: Literal["linear", "log"] = "linear"
    yscale: Literal["linear", "log"] = "linear"
    figsize: tuple[float, float] = (4.2, 3.0)


@dataclass
class SpatialReplotStyle:
    x_unit: Literal["m", "cm", "mm"] = "mm"
    xscale: Literal["linear", "log"] = "linear"
    yscale: Literal["linear", "log"] = "linear"
    figsize: tuple[float, float] = (4.4, 3.1)


def _time_scale(unit: str) -> tuple[float, str]:
    """Return the multiplicative scale and axis label for a requested time unit."""
    if unit == "s":
        return 1.0, "Time [s]"
    if unit == "ms":
        return 1e3, "Time [ms]"
    if unit == "us":
        return 1e6, "Time [us]"
    return 1e9, "Time [ns]"


def _x_scale(unit: str) -> tuple[float, str]:
    """Return the multiplicative scale and axis label for a requested position unit."""
    if unit == "m":
        return 1.0, "x [m]"
    if unit == "cm":
        return 1e2, "x [cm]"
    return 1e3, "x [mm]"


def load_run_metadata(run_name: str) -> dict:
    """Load and return the metadata dictionary for a saved run directory."""
    path = Path(run_name) / "run_metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _v_app_from_metadata(
    time: np.ndarray, meta: dict, *, project_dir: str | Path = "."
) -> np.ndarray:
    """Reconstruct the applied-voltage waveform from saved metadata on a given time array."""
    waveform_type = meta["waveform_type"]
    if waveform_type == "dc":
        return meta["V_peak"] * np.ones_like(time)
    if waveform_type == "step":
        return (
            meta["V_peak"] * ((time >= meta["tV_start"]) & (time <= meta["tV_end"]))
            + 1e-15 * ((time < meta["tV_start"]) | (time > meta["tV_end"]))
        ).astype(np.float64)
    if waveform_type == "gaussian":
        return meta["V_peak"] * np.exp(-((time - meta["t_peak"]) / meta["tau"]) ** 2)
    if waveform_type == "rf":
        omega = 2.0 * np.pi * meta["f_rf"]
        return meta["V_dc"] + meta["V_peak"] * np.sin(omega * time + meta["phi_rf"])
    if waveform_type in ("table", "tabulated", "measured_table"):
        waveform = SimpleNamespace(**meta)
        waveform.table_path = Path(project_dir) / meta["table_path"]
        return make_voltage_waveform(SimpleNamespace(waveform=waveform))(time)
    raise ValueError(f"Unknown waveform_type in metadata: {waveform_type}")


def _read_time_series(run_dir: Path, name: str, Nt: int) -> np.ndarray:
    """Return a memory-mapped scalar time-history array from a saved run folder."""
    path = run_dir / name
    itemsize = path.stat().st_size // int(Nt)
    if path.stat().st_size != int(Nt) * itemsize:
        raise ValueError(f"Unexpected scalar history file size for {path}")
    if itemsize == np.dtype(np.float64).itemsize:
        dtype = np.float64
    elif itemsize == np.dtype(np.float32).itemsize:
        dtype = np.float32
    else:
        raise ValueError(f"Cannot infer scalar history dtype for {path}")
    return np.memmap(path, mode="r", dtype=dtype, shape=(Nt,))


def _finite_or_none(value: float) -> float | None:
    """Return a JSON-friendly scalar, replacing NaN/Inf with None."""
    value = float(value)
    return value if np.isfinite(value) else None


def _crossing_time(
    time: np.ndarray,
    values: np.ndarray,
    threshold: float,
    *,
    start_idx: int,
    end_idx: int,
    direction: Literal["above", "below"],
) -> float:
    """Return first threshold-crossing time with linear interpolation."""
    if end_idx < start_idx:
        return float("nan")

    start_idx = max(0, int(start_idx))
    end_idx = min(len(values) - 1, int(end_idx))
    if direction == "above":
        crossed = values[start_idx : end_idx + 1] >= threshold
    else:
        crossed = values[start_idx : end_idx + 1] <= threshold

    hit = np.flatnonzero(crossed)
    if len(hit) == 0:
        return float("nan")

    i = start_idx + int(hit[0])
    if i == start_idx:
        return float(time[i])

    y0 = float(values[i - 1])
    y1 = float(values[i])
    t0 = float(time[i - 1])
    t1 = float(time[i])
    if y1 == y0:
        return t1
    frac = (float(threshold) - y0) / (y1 - y0)
    frac = min(1.0, max(0.0, frac))
    return t0 + frac * (t1 - t0)


def compute_discharge_waveform_metrics(
    run_name: str,
    *,
    laser_t0: float | None = None,
    pre_window: float | None = None,
    pre_guard: float = 100e-9,
    analysis_t_end: float | None = None,
    current_mode: Literal["absolute", "positive", "negative"] = "absolute",
    save: bool = True,
) -> dict[str, float | str | None]:
    """
    Compute compact transient-discharge waveform metrics from saved outputs.

    The exported quantities summarize the pre-trigger baseline, peak amplitude,
    finite rise delay, and discharge/recovery duration for a selected trigger
    time. The trigger is taken from saved laser timing metadata when available,
    or from the explicit ``laser_t0`` argument.
    """
    run_dir = Path(run_name)
    meta = load_run_metadata(run_name)

    Nt = int(meta["Nt"])
    T_total = float(meta["T_total"])
    time = np.linspace(0.0, T_total, Nt, dtype=np.float64)
    V_gap = np.asarray(_read_time_series(run_dir, "Vgap_mm.dat", Nt), dtype=np.float64)
    I_discharge = np.asarray(_read_time_series(run_dir, "Idischarge_mm.dat", Nt), dtype=np.float64)

    if laser_t0 is None:
        for key in ("cathode_laser_t0", "shared_laser_t0", "laser_t0"):
            if key in meta:
                laser_t0 = float(meta[key])
                break
    if laser_t0 is None:
        raise ValueError("laser_t0 is required because this run metadata does not contain laser timing.")

    laser_t0 = float(laser_t0)
    analysis_t_end = float(time[-1] if analysis_t_end is None else analysis_t_end)
    analysis_t_end = min(max(analysis_t_end, laser_t0), float(time[-1]))

    pre_end = max(float(time[0]), laser_t0 - float(pre_guard))
    if pre_window is None:
        pre_window = max(0.0, 0.20 * (pre_end - float(time[0])))
    pre_start = max(float(time[0]), pre_end - float(pre_window))
    pre_mask = (time >= pre_start) & (time <= pre_end)
    if not np.any(pre_mask):
        pre_mask = time < laser_t0
    if not np.any(pre_mask):
        raise ValueError("No pre-laser samples available for baseline calculation.")

    V_pre = V_gap[pre_mask]
    I_pre = I_discharge[pre_mask]
    V_pre_mean = float(np.mean(V_pre))
    V_pre_std = float(np.std(V_pre))
    I_pre_mean = float(np.mean(I_pre))
    I_pre_rms = float(np.sqrt(np.mean((I_pre - I_pre_mean) ** 2)))
    I_pre_abs_max = float(np.max(np.abs(I_pre)))

    analysis_mask = (time >= laser_t0) & (time <= analysis_t_end)
    analysis_indices = np.flatnonzero(analysis_mask)
    if len(analysis_indices) == 0:
        raise ValueError("No post-laser samples available in the selected analysis window.")
    post_start = int(analysis_indices[0])
    post_end = int(analysis_indices[-1])

    I_delta = I_discharge - I_pre_mean
    if current_mode == "positive":
        I_signal = I_delta
    elif current_mode == "negative":
        I_signal = -I_delta
    else:
        I_signal = np.abs(I_delta)

    local_peak_offset = int(np.argmax(I_signal[post_start : post_end + 1]))
    peak_idx = post_start + local_peak_offset
    I_peak_abs = float(I_signal[peak_idx])
    I_peak_signed = float(I_discharge[peak_idx])
    t_I_peak = float(time[peak_idx])

    local_vmin_offset = int(np.argmin(V_gap[post_start : post_end + 1]))
    vmin_idx = post_start + local_vmin_offset
    V_min = float(V_gap[vmin_idx])
    t_V_min = float(time[vmin_idx])
    V_drop = V_pre_mean - V_gap
    V_drop_peak = max(0.0, float(V_pre_mean - V_min))

    def rise_metrics(signal: np.ndarray, peak_value: float, peak_index: int, prefix: str) -> dict[str, float]:
        if peak_value <= 0.0:
            return {
                f"{prefix}_t_10_s": float("nan"),
                f"{prefix}_t_50_s": float("nan"),
                f"{prefix}_t_90_s": float("nan"),
                f"{prefix}_rise_10_90_s": float("nan"),
            }
        t10 = _crossing_time(time, signal, 0.10 * peak_value, start_idx=post_start, end_idx=peak_index, direction="above")
        t50 = _crossing_time(time, signal, 0.50 * peak_value, start_idx=post_start, end_idx=peak_index, direction="above")
        t90 = _crossing_time(time, signal, 0.90 * peak_value, start_idx=post_start, end_idx=peak_index, direction="above")
        return {
            f"{prefix}_t_10_s": t10,
            f"{prefix}_t_50_s": t50,
            f"{prefix}_t_90_s": t90,
            f"{prefix}_rise_10_90_s": t90 - t10 if np.isfinite(t10) and np.isfinite(t90) else float("nan"),
        }

    I_rise = rise_metrics(I_signal, I_peak_abs, peak_idx, "I")
    V_rise = rise_metrics(V_drop, V_drop_peak, vmin_idx, "Vdrop")

    I_decay_1e = (
        _crossing_time(time, I_signal, I_peak_abs / np.e, start_idx=peak_idx, end_idx=post_end, direction="below")
        if I_peak_abs > 0.0
        else float("nan")
    )
    I_decay_50 = (
        _crossing_time(time, I_signal, 0.50 * I_peak_abs, start_idx=peak_idx, end_idx=post_end, direction="below")
        if I_peak_abs > 0.0
        else float("nan")
    )
    I_decay_10 = (
        _crossing_time(time, I_signal, 0.10 * I_peak_abs, start_idx=peak_idx, end_idx=post_end, direction="below")
        if I_peak_abs > 0.0
        else float("nan")
    )
    I_decay_05 = (
        _crossing_time(time, I_signal, 0.05 * I_peak_abs, start_idx=peak_idx, end_idx=post_end, direction="below")
        if I_peak_abs > 0.0
        else float("nan")
    )

    V_recover_50 = (
        _crossing_time(time, V_drop, 0.50 * V_drop_peak, start_idx=vmin_idx, end_idx=post_end, direction="below")
        if V_drop_peak > 0.0
        else float("nan")
    )
    V_recover_90 = (
        _crossing_time(time, V_drop, 0.10 * V_drop_peak, start_idx=vmin_idx, end_idx=post_end, direction="below")
        if V_drop_peak > 0.0
        else float("nan")
    )
    V_recover_95 = (
        _crossing_time(time, V_drop, 0.05 * V_drop_peak, start_idx=vmin_idx, end_idx=post_end, direction="below")
        if V_drop_peak > 0.0
        else float("nan")
    )

    metrics: dict[str, float | str | None] = {
        "run_name": str(run_name),
        "laser_t0_s": laser_t0,
        "analysis_t_end_s": analysis_t_end,
        "pre_window_start_s": pre_start,
        "pre_window_end_s": pre_end,
        "current_mode": current_mode,
        "V_pre_mean_V": V_pre_mean,
        "V_pre_std_V": V_pre_std,
        "I_pre_mean_A": I_pre_mean,
        "I_pre_rms_A": I_pre_rms,
        "I_pre_abs_max_A": I_pre_abs_max,
        "I_peak_A": I_peak_signed,
        "I_peak_abs_A": I_peak_abs,
        "t_I_peak_s": t_I_peak,
        "delay_I_peak_s": t_I_peak - laser_t0,
        "V_min_V": V_min,
        "V_drop_peak_V": V_drop_peak,
        "t_V_min_s": t_V_min,
        "delay_V_min_s": t_V_min - laser_t0,
        "I_decay_to_1e_from_peak_s": I_decay_1e - t_I_peak if np.isfinite(I_decay_1e) else float("nan"),
        "I_decay_to_50pct_from_peak_s": I_decay_50 - t_I_peak if np.isfinite(I_decay_50) else float("nan"),
        "I_decay_to_10pct_from_peak_s": I_decay_10 - t_I_peak if np.isfinite(I_decay_10) else float("nan"),
        "I_decay_to_5pct_from_peak_s": I_decay_05 - t_I_peak if np.isfinite(I_decay_05) else float("nan"),
        "I_duration_above_10pct_s": I_decay_10 - I_rise["I_t_10_s"]
        if np.isfinite(I_decay_10) and np.isfinite(I_rise["I_t_10_s"])
        else float("nan"),
        "V_recover_50pct_from_min_s": V_recover_50 - t_V_min if np.isfinite(V_recover_50) else float("nan"),
        "V_recover_90pct_from_min_s": V_recover_90 - t_V_min if np.isfinite(V_recover_90) else float("nan"),
        "V_recover_95pct_from_min_s": V_recover_95 - t_V_min if np.isfinite(V_recover_95) else float("nan"),
        "V_duration_above_10pct_drop_s": V_recover_90 - V_rise["Vdrop_t_10_s"]
        if np.isfinite(V_recover_90) and np.isfinite(V_rise["Vdrop_t_10_s"])
        else float("nan"),
        "V_final_V": float(V_gap[post_end]),
        "I_final_A": float(I_discharge[post_end]),
    }
    metrics.update(I_rise)
    metrics.update(V_rise)

    metrics = {key: (_finite_or_none(value) if isinstance(value, (float, np.floating)) else value) for key, value in metrics.items()}

    if save:
        json_path = run_dir / "discharge_waveform_metrics.json"
        csv_path = run_dir / "discharge_waveform_metrics.csv"
        json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            writer.writeheader()
            writer.writerow(metrics)

    return metrics


def compute_current_decomposition_metrics(
    run_name: str,
    *,
    laser_t0: float | None = None,
    analysis_t_end: float | None = None,
    save: bool = True,
) -> list[dict[str, float | str | None]]:
    """Summarize saved current-decomposition histories for one run."""
    run_dir = Path(run_name)
    meta = load_run_metadata(run_name)
    Nt = int(meta["Nt"])
    T_total = float(meta["T_total"])
    time = np.linspace(0.0, T_total, Nt, dtype=np.float64)

    if laser_t0 is None:
        for key in ("cathode_laser_t0", "shared_laser_t0", "laser_t0"):
            if key in meta:
                laser_t0 = float(meta[key])
                break
    if laser_t0 is None:
        raise ValueError("laser_t0 is required because this run metadata does not contain laser timing.")

    laser_t0 = float(laser_t0)
    analysis_t_end = float(time[-1] if analysis_t_end is None else analysis_t_end)
    analysis_t_end = min(max(analysis_t_end, laser_t0), float(time[-1]))
    post_mask = (time >= laser_t0) & (time <= analysis_t_end)
    post_indices = np.flatnonzero(post_mask)
    if len(post_indices) == 0:
        raise ValueError("No post-laser samples available in the selected analysis window.")
    post_start = int(post_indices[0])
    post_end = int(post_indices[-1])

    component_files = (
        ("I_discharge", "Idischarge_mm.dat"),
        ("I_transport_plasma", "I_transport_plasma_mm.dat"),
        ("I_transport_circuit", "I_transport_circuit_mm.dat"),
        ("I_emission_circuit", "I_emission_circuit_mm.dat"),
        ("I_emission_area", "I_emission_area_mm.dat"),
        ("I_displacement_gap", "I_displacement_gap_mm.dat"),
    )

    rows: list[dict[str, float | str | None]] = []
    for quantity, filename in component_files:
        path = run_dir / filename
        if not path.exists():
            rows.append(
                {
                    "quantity": quantity,
                    "available": "no",
                    "peak_abs_A": None,
                    "peak_signed_A": None,
                    "delay_to_abs_peak_s": None,
                    "charge_integral_C": None,
                }
            )
            continue

        values = np.asarray(_read_time_series(run_dir, filename, Nt), dtype=np.float64)
        segment = values[post_start : post_end + 1]
        local_idx = int(np.argmax(np.abs(segment)))
        peak_idx = post_start + local_idx
        charge_integral = _trapezoid_integral(
            values[post_start : post_end + 1],
            x=time[post_start : post_end + 1],
        )
        rows.append(
            {
                "quantity": quantity,
                "available": "yes",
                "peak_abs_A": _finite_or_none(abs(values[peak_idx])),
                "peak_signed_A": _finite_or_none(values[peak_idx]),
                "delay_to_abs_peak_s": _finite_or_none(time[peak_idx] - laser_t0),
                "charge_integral_C": _finite_or_none(charge_integral),
            }
        )

    if save:
        json_path = run_dir / "current_decomposition_metrics.json"
        csv_path = run_dir / "current_decomposition_metrics.csv"
        json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "quantity",
                "available",
                "peak_abs_A",
                "peak_signed_A",
                "delay_to_abs_peak_s",
                "charge_integral_C",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field) for field in fieldnames})

    return rows


def run_current_decomposition_report(
    run_name: str,
    *,
    laser_t0: float | None = None,
    analysis_t_end: float | None = None,
    save: bool = True,
    print_table: bool = True,
) -> list[dict[str, float | str | None]]:
    """Compute, save, and optionally print current-decomposition metrics."""
    rows = compute_current_decomposition_metrics(
        run_name,
        laser_t0=laser_t0,
        analysis_t_end=analysis_t_end,
        save=save,
    )
    if print_table:
        print("\nCurrent decomposition metrics")
        print("-----------------------------")
        for row in rows:
            if row.get("available") != "yes":
                print(f"{row['quantity']:<26}: unavailable")
                continue
            peak_uA = row["peak_abs_A"] * 1.0e6 if row["peak_abs_A"] is not None else float("nan")
            delay_ns = (
                row["delay_to_abs_peak_s"] * 1.0e9
                if row["delay_to_abs_peak_s"] is not None
                else float("nan")
            )
            charge_pC = (
                row["charge_integral_C"] * 1.0e12
                if row["charge_integral_C"] is not None
                else float("nan")
            )
            print(
                f"{row['quantity']:<26}: "
                f"peak |I|={peak_uA: .6g} uA, "
                f"delay={delay_ns: .6g} ns, "
                f"integral={charge_pC: .6g} pC"
            )
    return rows


def _spatial_paths(run_dir: Path) -> dict[str, Path]:
    """Return the standard sampled-field file paths for a saved run directory."""
    return {
        "ne": run_dir / "ne_sampled_mm.dat",
        "ni": run_dir / "ni_sampled_mm.dat",
        "phi": run_dir / "phi_sampled_mm.dat",
        "E": run_dir / "E_sampled_mm.dat",
        "Gamma_i": run_dir / "Gamma_i_sampled_mm.dat",
        "Gamma_e": run_dir / "Gamma_e_sampled_mm.dat",
        "townsend_alpha": run_dir / "townsend_alpha_sampled_mm.dat",
        "nu_i": run_dir / "nu_i_sampled_mm.dat",
        "S_ion": run_dir / "S_ion_sampled_mm.dat",
        "S": run_dir / "S_sampled_mm.dat",
        "mu_e": run_dir / "mu_e_sampled_mm.dat",
        "D_e": run_dir / "D_e_sampled_mm.dat",
    }


def _sampled_count(path: Path, Nx: int, dtype=np.float32) -> int:
    """Return the number of saved spatial snapshots in a memmap file."""
    if not path.exists():
        raise FileNotFoundError(f"Required sampled field not found: {path}")
    itemsize = np.dtype(dtype).itemsize
    row_bytes = itemsize * int(Nx)
    size = path.stat().st_size
    if size % row_bytes != 0:
        raise ValueError(f"Unexpected sampled-field file size for {path}")
    return size // row_bytes


def _saved_snapshot_times(meta: dict, nsave: int) -> np.ndarray:
    """Return saved snapshot times from run metadata and save cadence."""
    Nt = int(meta["Nt"])
    T_total = float(meta["T_total"])
    save_every = int(meta["save_every"])
    dt = float(meta.get("dt", T_total / max(Nt - 1, 1)))
    saved_indices = np.minimum(np.arange(nsave, dtype=np.int64) * save_every, Nt - 1)
    return saved_indices.astype(np.float64) * dt


def _segment_trapezoid(values: np.ndarray, x: np.ndarray) -> float:
    """Integrate a possibly one-point segment; one-point segments have zero width."""
    if len(values) < 2:
        return 0.0
    return float(_trapezoid_integral(values, x=x))


def _sheath_segment_metrics(
    *,
    x: np.ndarray,
    phi: np.ndarray,
    E_field: np.ndarray,
    charge_density: np.ndarray,
    A: float,
    side: Literal["anode", "cathode"],
    edge_idx: int | None,
) -> dict[str, float | None]:
    """Return position, voltage, field, and charge metrics for one sheath."""
    if edge_idx is None:
        return {
            f"{side}_edge_x_m": None,
            f"{side}_width_m": None,
            f"{side}_delta_phi_edge_minus_electrode_V": None,
            f"{side}_voltage_drop_abs_V": None,
            f"{side}_peak_abs_E_V_m": None,
            f"{side}_space_charge_sigma_C_m2": None,
            f"{side}_space_charge_Q_C": None,
        }

    edge_idx = int(edge_idx)
    if side == "anode":
        region = slice(0, edge_idx + 1)
        electrode_idx = 0
        width = float(x[edge_idx] - x[0])
    else:
        region = slice(edge_idx, len(x))
        electrode_idx = len(x) - 1
        width = float(x[-1] - x[edge_idx])

    delta_phi = float(phi[edge_idx] - phi[electrode_idx])
    charge_sigma = _segment_trapezoid(charge_density[region], x[region])
    return {
        f"{side}_edge_x_m": float(x[edge_idx]),
        f"{side}_width_m": width,
        f"{side}_delta_phi_edge_minus_electrode_V": delta_phi,
        f"{side}_voltage_drop_abs_V": abs(delta_phi),
        f"{side}_peak_abs_E_V_m": float(np.max(np.abs(E_field[region]))),
        f"{side}_space_charge_sigma_C_m2": charge_sigma,
        f"{side}_space_charge_Q_C": float(A * charge_sigma),
    }


def compute_sheath_diagnostics(
    run_name: str,
    *,
    quasineutrality_tol: float = 0.05,
    density_floor_fraction: float = 1.0e-8,
    density_floor_m3: float | None = None,
    save: bool = True,
) -> list[dict[str, float | str | None]]:
    """
    Infer anode/cathode sheath metrics from saved spatial snapshots.

    The sheath edge is identified as the first quasineutral point reached from
    each electrode. Quasineutrality is defined by

        |n_i - n_e| / max(0.5*(n_i+n_e), density_floor) <= quasineutrality_tol

    where ``density_floor`` suppresses meaningless sheath edges in regions with
    negligible plasma density. The diagnostic is intentionally postprocessing
    only: it does not feed back into the plasma solve.

    Saved outputs
    -------------
    <run_name>/sheath_diagnostics.json
    <run_name>/sheath_diagnostics.csv
    """
    run_dir = Path(run_name)
    meta = load_run_metadata(run_name)
    Nx = int(meta["Nx"])
    L = float(meta["L"])
    A = float(meta["A"])
    x = np.linspace(0.0, L, Nx, dtype=np.float64)

    paths = _spatial_paths(run_dir)
    required = ("ne", "ni", "phi", "E")
    nsave = min(_sampled_count(paths[name], Nx) for name in required)
    times = _saved_snapshot_times(meta, nsave)

    ne = np.memmap(paths["ne"], mode="r", dtype=np.float32, shape=(nsave, Nx))
    ni = np.memmap(paths["ni"], mode="r", dtype=np.float32, shape=(nsave, Nx))
    phi = np.memmap(paths["phi"], mode="r", dtype=np.float32, shape=(nsave, Nx))
    E_field = np.memmap(paths["E"], mode="r", dtype=np.float32, shape=(nsave, Nx))

    rows: list[dict[str, float | str | None]] = []
    for k in range(nsave):
        ne_row = np.asarray(ne[k], dtype=np.float64)
        ni_row = np.asarray(ni[k], dtype=np.float64)
        phi_row = np.asarray(phi[k], dtype=np.float64)
        E_row = np.asarray(E_field[k], dtype=np.float64)

        mean_density = 0.5 * (ne_row + ni_row)
        density_scale = float(np.nanmax(mean_density)) if mean_density.size else 0.0
        floor = float(density_floor_m3) if density_floor_m3 is not None else (
            max(float(density_floor_fraction) * density_scale, 0.0)
        )
        denom = np.maximum(mean_density, floor)
        with np.errstate(divide="ignore", invalid="ignore"):
            imbalance = np.abs(ni_row - ne_row) / denom
        valid_density = mean_density > floor
        quasineutral = valid_density & np.isfinite(imbalance) & (imbalance <= quasineutrality_tol)

        anode_hits = np.flatnonzero(quasineutral)
        cathode_hits = anode_hits
        anode_edge_idx = int(anode_hits[0]) if len(anode_hits) else None
        cathode_edge_idx = int(cathode_hits[-1]) if len(cathode_hits) else None

        charge_density = e * (ni_row - ne_row)
        row: dict[str, float | str | None] = {
            "run_name": str(run_name),
            "snapshot_index": int(k),
            "time_s": _finite_or_none(times[k]),
            "quasineutrality_tol": float(quasineutrality_tol),
            "density_floor_m3": _finite_or_none(floor),
            "bulk_density_max_m3": _finite_or_none(density_scale),
            "anode_edge_found": "yes" if anode_edge_idx is not None else "no",
            "cathode_edge_found": "yes" if cathode_edge_idx is not None else "no",
        }
        row.update(
            _sheath_segment_metrics(
                x=x,
                phi=phi_row,
                E_field=E_row,
                charge_density=charge_density,
                A=A,
                side="anode",
                edge_idx=anode_edge_idx,
            )
        )
        row.update(
            _sheath_segment_metrics(
                x=x,
                phi=phi_row,
                E_field=E_row,
                charge_density=charge_density,
                A=A,
                side="cathode",
                edge_idx=cathode_edge_idx,
            )
        )
        row = {
            key: (_finite_or_none(value) if isinstance(value, (float, np.floating)) else value)
            for key, value in row.items()
        }
        rows.append(row)

    if save:
        json_path = run_dir / "sheath_diagnostics.json"
        csv_path = run_dir / "sheath_diagnostics.csv"
        json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        fieldnames = list(rows[0].keys()) if rows else []
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field) for field in fieldnames})

    return rows


def run_sheath_diagnostics_report(
    run_name: str,
    *,
    quasineutrality_tol: float = 0.05,
    density_floor_fraction: float = 1.0e-8,
    density_floor_m3: float | None = None,
    save: bool = True,
    print_table: bool = True,
) -> list[dict[str, float | str | None]]:
    """Compute, save, and optionally print compact sheath-diagnostic metrics."""
    rows = compute_sheath_diagnostics(
        run_name,
        quasineutrality_tol=quasineutrality_tol,
        density_floor_fraction=density_floor_fraction,
        density_floor_m3=density_floor_m3,
        save=save,
    )
    if print_table:
        print("\nSheath diagnostics")
        print("------------------")
        if not rows:
            print("No saved spatial snapshots were found.")
            return rows

        def values(name: str) -> np.ndarray:
            return np.array(
                [np.nan if row.get(name) is None else float(row[name]) for row in rows],
                dtype=np.float64,
            )

        time_us = values("time_s") * 1.0e6
        for side in ("anode", "cathode"):
            width_mm = values(f"{side}_width_m") * 1.0e3
            drop_V = values(f"{side}_voltage_drop_abs_V")
            peak_E_kV_cm = values(f"{side}_peak_abs_E_V_m") / 1.0e5
            if np.all(~np.isfinite(width_mm)):
                print(f"{side:<7}: no sheath edge found")
                continue
            last_idx = int(np.flatnonzero(np.isfinite(width_mm))[-1])
            max_idx = int(np.nanargmax(width_mm))
            print(
                f"{side:<7}: final width={width_mm[last_idx]:.6g} mm, "
                f"max width={width_mm[max_idx]:.6g} mm at t={time_us[max_idx]:.6g} us, "
                f"final |dphi|={drop_V[last_idx]:.6g} V, "
                f"final max |E|={peak_E_kV_cm[last_idx]:.6g} kV/cm"
            )
        if save:
            print(f"Saved sheath diagnostics for {run_name}")
            print(f"  {run_name}/sheath_diagnostics.json")
            print(f"  {run_name}/sheath_diagnostics.csv")
    return rows


def replot_from_saved(
    run_name: str,
    *,
    project_dir: str | Path = ".",
    temporal_groups: tuple[tuple[TemporalQuantity, ...], ...] | None = None,
    spatial_groups: tuple[tuple[SpatialQuantity, ...], ...] | None = None,
    averaged_spatial_groups: tuple[tuple[SpatialQuantity, ...], ...] | None = None,
    t_start: float | None = None,
    t_end: float | None = None,
    t_samples: tuple[float, ...] | None = None,
    averaged_mode: AveragedSpatialMode = "time_window",
    t_avg_start: float | None = None,
    t_avg_end: float | None = None,
    N_cycle_avg: int = 1,
    temporal_style: TemporalReplotStyle | None = None,
    spatial_style: SpatialReplotStyle | None = None,
    valid_end_idx: int | None = None,
    valid_nsave: int | None = None,
) -> None:
    """
    Regenerate diagnostics from saved files in ``run_name``.

    Relative run names and waveform table paths are resolved from
    ``project_dir``, which defaults to the current working directory. Absolute
    run paths are used directly; nested run names do not change the table base.

    This function can be called repeatedly with different style/units/scales
    without rerunning simulation.
    """
    run_dir = Path(project_dir) / run_name
    meta = load_run_metadata(str(run_dir))

    Nt = int(meta["Nt"])
    Nx = int(meta["Nx"])
    T_total = float(meta["T_total"])
    L = float(meta["L"])
    A = float(meta["A"])
    save_every = int(meta["save_every"])

    temporal_style = temporal_style or TemporalReplotStyle()
    spatial_style = spatial_style or SpatialReplotStyle()

    time_full = np.linspace(0.0, T_total, Nt, dtype=np.float64)
    if valid_end_idx is None:
        Nt_valid = Nt
    else:
        Nt_valid = max(1, min(int(valid_end_idx) + 1, Nt))
    time = time_full[:Nt_valid]
    x = np.linspace(0.0, L, Nx, dtype=np.float64)

    # ---------- Temporal ----------
    if temporal_groups is None:
        temporal_groups = (
            ("V_app", "V_gap"),
            (
                "I_discharge",
                "I_transport_plasma",
                "I_transport_circuit",
                "I_emission_circuit",
                "I_displacement_gap",
            ),
            ("I_emission_area",),
            ("cfl", "diffusion_cfl"),
            ("picard_iterations",),
            ("adaptive_substeps",),
            ("adaptive_cfl_est", "adaptive_diffusion_cfl_est"),
            ("particle_inventory",),
        )

    t_factor, t_label = _time_scale(temporal_style.t_unit)
    tw0 = float(time[0]) if t_start is None else float(t_start)
    tw1 = float(time[-1]) if t_end is None else float(t_end)
    if tw1 < tw0:
        tw0, tw1 = tw1, tw0
    mask_full = (time >= tw0) & (time <= tw1)

    temporal_values = {
        "V_app": _v_app_from_metadata(time, meta, project_dir=project_dir),
        "V_gap": np.asarray(_read_time_series(run_dir, "Vgap_mm.dat", Nt), dtype=np.float64)[:Nt_valid],
        "I_discharge": np.asarray(
            _read_time_series(run_dir, "Idischarge_mm.dat", Nt), dtype=np.float64
        )[:Nt_valid],
        "cfl": np.asarray(_read_time_series(run_dir, "c_cfl_mm.dat", Nt), dtype=np.float64)[:Nt_valid],
    }
    diffusion_cfl_path = run_dir / "diffusion_cfl_mm.dat"
    adaptive_substeps_path = run_dir / "adaptive_substeps_mm.dat"
    adaptive_dt_sub_path = run_dir / "adaptive_dt_sub_mm.dat"
    adaptive_cfl_est_path = run_dir / "adaptive_cfl_est_mm.dat"
    adaptive_diffusion_cfl_est_path = run_dir / "adaptive_diffusion_cfl_est_mm.dat"
    picard_iterations_path = run_dir / "picard_iterations_mm.dat"
    if diffusion_cfl_path.exists():
        temporal_values["diffusion_cfl"] = np.asarray(
            _read_time_series(run_dir, "diffusion_cfl_mm.dat", Nt), dtype=np.float64
        )[:Nt_valid]
    if adaptive_substeps_path.exists():
        temporal_values["adaptive_substeps"] = np.asarray(
            _read_time_series(run_dir, "adaptive_substeps_mm.dat", Nt), dtype=np.float64
        )[:Nt_valid]
    if adaptive_dt_sub_path.exists():
        temporal_values["adaptive_dt_sub"] = np.asarray(
            _read_time_series(run_dir, "adaptive_dt_sub_mm.dat", Nt), dtype=np.float64
        )[:Nt_valid]
    if adaptive_cfl_est_path.exists():
        temporal_values["adaptive_cfl_est"] = np.asarray(
            _read_time_series(run_dir, "adaptive_cfl_est_mm.dat", Nt), dtype=np.float64
        )[:Nt_valid]
    if adaptive_diffusion_cfl_est_path.exists():
        temporal_values["adaptive_diffusion_cfl_est"] = np.asarray(
            _read_time_series(run_dir, "adaptive_diffusion_cfl_est_mm.dat", Nt), dtype=np.float64
        )[:Nt_valid]
    if picard_iterations_path.exists():
        temporal_values["picard_iterations"] = np.asarray(
            _read_time_series(run_dir, "picard_iterations_mm.dat", Nt), dtype=np.float64
        )[:Nt_valid]
    for quantity, filename in (
        ("V_node", "Vnode_mm.dat"),
        ("V_source", "Vsource_mm.dat"),
    ):
        if (run_dir / filename).exists():
            temporal_values[quantity] = np.asarray(
                _read_time_series(run_dir, filename, Nt), dtype=np.float64
            )[:Nt_valid]
    for quantity, filename in (
        ("I_transport_plasma", "I_transport_plasma_mm.dat"),
        ("I_transport_circuit", "I_transport_circuit_mm.dat"),
        ("I_emission_circuit", "I_emission_circuit_mm.dat"),
        ("I_emission_area", "I_emission_area_mm.dat"),
        ("I_displacement_gap", "I_displacement_gap_mm.dat"),
    ):
        if (run_dir / filename).exists():
            temporal_values[quantity] = np.asarray(
                _read_time_series(run_dir, filename, Nt), dtype=np.float64
            )[:Nt_valid]

    # Particle inventory uses saved density snapshots.
    nsave_total = int((Nt - 1) // save_every + 1)
    nsave = nsave_total if valid_nsave is None else max(1, min(int(valid_nsave), nsave_total))
    saved_indices = np.arange(nsave, dtype=np.int64) * save_every
    saved_indices = np.minimum(saved_indices, Nt - 1)
    saved_times = time_full[saved_indices]
    ne_sampled = np.memmap(
        run_dir / "ne_sampled_mm.dat", mode="r", dtype=np.float32, shape=(nsave_total, Nx)
    )
    ni_sampled = np.memmap(
        run_dir / "ni_sampled_mm.dat", mode="r", dtype=np.float32, shape=(nsave_total, Nx)
    )
    N_e = A * _trapezoid_integral(np.asarray(ne_sampled[:nsave], dtype=np.float64), x=x, axis=1)
    N_i = A * _trapezoid_integral(np.asarray(ni_sampled[:nsave], dtype=np.float64), x=x, axis=1)

    for group in temporal_groups:
        if len(group) == 0:
            continue

        if "particle_inventory" in group:
            if len(group) > 1:
                print(f"Temporal group {group}: plotting 'particle_inventory' separately.")
            mask_inv = (saved_times >= tw0) & (saved_times <= tw1)
            if not np.any(mask_inv):
                print("Particle inventory skipped: empty selected time window.")
            else:
                Ne0 = N_e[0] if abs(N_e[0]) > 1e-30 else 1.0
                Ni0 = N_i[0] if abs(N_i[0]) > 1e-30 else 1.0
                fig, ax = plt.subplots(figsize=temporal_style.figsize)
                ax.plot(saved_times[mask_inv] * t_factor, N_e[mask_inv] / Ne0, label="N_e / N_e0")
                ax.plot(saved_times[mask_inv] * t_factor, N_i[mask_inv] / Ni0, label="N_i / N_i0")
                ax.set_xlabel(t_label)
                ax.set_ylabel("Normalized inventory")
                ax.set_xscale(temporal_style.xscale)
                ax.set_yscale(temporal_style.yscale)
                ax.set_title("Particle Inventory (saved snapshots)")
                ax.grid(True)
                ax.legend(frameon=False)
                fig.tight_layout()
                plt.show()

            # Continue with any non-inventory quantities in the same group.
            group = tuple(q for q in group if q != "particle_inventory")
            if len(group) == 0:
                continue

        valid = [q for q in group if q in temporal_values]
        if len(valid) == 0:
            print(f"Temporal group {group} has no available quantities.")
            continue

        fig, ax = plt.subplots(figsize=temporal_style.figsize)
        ylabel = None
        for q in valid:
            y = temporal_values[q][mask_full]
            if q in ("V_app", "V_node", "V_source", "V_gap"):
                y = y * 1e-3
                ylab = "Voltage [kV]"
            elif q in CURRENT_TEMPORAL_QUANTITIES:
                y = y * 1e3
                ylab = "Current [mA]"
            elif q == "adaptive_substeps":
                ylab = "Substeps per macro step"
            elif q == "adaptive_dt_sub":
                ylab = "Substep dt [s]"
            elif q == "adaptive_cfl_est":
                ylab = "Estimated macro drift CFL"
            elif q == "adaptive_diffusion_cfl_est":
                ylab = "Estimated macro diffusion stability"
            elif q == "cfl":
                ylab = "Drift CFL number"
            elif q == "diffusion_cfl":
                ylab = "Diffusion stability number"
            elif q == "picard_iterations":
                ylab = "Picard iterations per macro step"
            else:
                ylab = "Value"
            if ylabel is None:
                ylabel = ylab
            elif ylabel != ylab:
                ylabel = "Mixed units"
            ax.plot(time[mask_full] * t_factor, y, label=q)

        ax.set_xlabel(t_label)
        ax.set_ylabel(ylabel if ylabel is not None else "Value")
        ax.set_xscale(temporal_style.xscale)
        ax.set_yscale(temporal_style.yscale)
        ax.set_title(" + ".join(valid))
        ax.grid(True)
        ax.legend(frameon=False)
        fig.tight_layout()
        plt.show()

    # ---------- Spatial ----------
    if spatial_groups is None:
        spatial_groups = (("ne", "ni"), ("phi",), ("E",))

    x_factor, x_label = _x_scale(spatial_style.x_unit)
    x_plot = x * x_factor
    paths = _spatial_paths(run_dir)
    sampled_arrays: dict[str, np.ndarray] = {}
    for q, p in paths.items():
        if p.exists():
            sampled_arrays[q] = np.memmap(p, mode="r", dtype=np.float32, shape=(nsave_total, Nx))

    if t_samples is None:
        requested = np.array([saved_times[-1]], dtype=np.float64)
    else:
        requested = np.asarray(t_samples, dtype=np.float64)
        requested = np.clip(requested, saved_times[0], saved_times[-1])

    for group in spatial_groups:
        if len(group) == 0:
            continue
        fig, ax = plt.subplots(figsize=spatial_style.figsize)
        ylabel = None
        has_curve = False
        for q in group:
            arr = sampled_arrays.get(q)
            if arr is None:
                print(f"Spatial quantity '{q}' missing in saved files; skipping.")
                continue
            if q in ("ne", "ni"):
                ylab = "Density [m$^{-3}$]"
            elif q == "phi":
                ylab = "Potential [V]"
            elif q == "E":
                ylab = "Electric Field [V/m]"
            elif q in ("Gamma_i", "Gamma_e"):
                ylab = "Gamma [m$^{-2}$ s$^{-1}$]"
            elif q == "townsend_alpha":
                ylab = "Townsend alpha [m$^{-1}$]"
            elif q == "nu_i":
                ylab = "nu_i [s$^{-1}$]"
            elif q == "mu_e":
                ylab = "mu_e [m$^2$ V$^{-1}$ s$^{-1}$]"
            elif q == "D_e":
                ylab = "D_e [m$^2$ s$^{-1}$]"
            else:
                ylab = "Source [m$^{-3}$ s$^{-1}$]"
            if ylabel is None:
                ylabel = ylab
            elif ylabel != ylab:
                ylabel = "Mixed units"

            for t_req in requested:
                k = int(np.argmin(np.abs(saved_times - t_req)))
                lbl = q if requested.size == 1 else f"{q}, t={saved_times[k]*1e9:.1f} ns"
                ax.plot(x_plot, np.asarray(arr[k], dtype=np.float64), label=lbl)
                has_curve = True

        if not has_curve:
            plt.close(fig)
            print(f"Spatial group {group} has no available curves.")
            continue

        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel if ylabel is not None else "Value")
        ax.set_xscale(spatial_style.xscale)
        ax.set_yscale(spatial_style.yscale)
        ax.set_title(" + ".join(group))
        ax.grid(True)
        if len(ax.lines) <= 12:
            ax.legend(frameon=False)
        fig.tight_layout()
        plt.show()

    # ---------- Averaged Spatial ----------
    if averaged_spatial_groups is None:
        averaged_spatial_groups = (("ne", "ni"), ("phi",), ("E",))

    if averaged_mode == "last_n_cycles":
        if meta["waveform_type"] != "rf":
            print("Averaged spatial diagnostics skipped: 'last_n_cycles' requires waveform_type='rf'.")
            return
        f_rf = float(meta["f_rf"])
        if f_rf <= 0.0:
            print("Averaged spatial diagnostics skipped: f_rf must be positive for 'last_n_cycles'.")
            return
        n_cycles = max(int(N_cycle_avg), 1)
        t_window_end = float(saved_times[-1])
        t_window_start = t_window_end - n_cycles / f_rf
        averaging_label = f"Average over last {n_cycles} RF cycle(s)"
    else:
        t_window_start = float(saved_times[0]) if t_avg_start is None else float(t_avg_start)
        t_window_end = float(saved_times[-1]) if t_avg_end is None else float(t_avg_end)
        if t_window_end < t_window_start:
            t_window_start, t_window_end = t_window_end, t_window_start
        averaging_label = f"Average over [{t_window_start:.3e}, {t_window_end:.3e}] s"

    avg_mask = (saved_times >= t_window_start) & (saved_times <= t_window_end)
    if not np.any(avg_mask):
        print("Averaged spatial diagnostics skipped: empty averaging window in saved snapshots.")
        return

    for group in averaged_spatial_groups:
        if len(group) == 0:
            continue

        fig, ax = plt.subplots(figsize=spatial_style.figsize)
        ylabel = None
        has_curve = False

        for q in group:
            arr = sampled_arrays.get(q)
            if arr is None:
                print(f"Averaged spatial quantity '{q}' missing in saved files; skipping.")
                continue

            avg_profile = np.mean(np.asarray(arr[:nsave][avg_mask], dtype=np.float64), axis=0)
            if q in ("ne", "ni"):
                ylab = "Density [m$^{-3}$]"
            elif q == "phi":
                ylab = "Potential [V]"
            elif q == "E":
                ylab = "Electric Field [V/m]"
            elif q in ("Gamma_i", "Gamma_e"):
                ylab = "Gamma [m$^{-2}$ s$^{-1}$]"
            elif q == "townsend_alpha":
                ylab = "Townsend alpha [m$^{-1}$]"
            elif q == "nu_i":
                ylab = "nu_i [s$^{-1}$]"
            elif q == "mu_e":
                ylab = "mu_e [m$^2$ V$^{-1}$ s$^{-1}$]"
            elif q == "D_e":
                ylab = "D_e [m$^2$ s$^{-1}$]"
            else:
                ylab = "Source [m$^{-3}$ s$^{-1}$]"

            if ylabel is None:
                ylabel = ylab
            elif ylabel != ylab:
                ylabel = "Mixed units"

            ax.plot(x_plot, avg_profile, label=q)
            has_curve = True

        if not has_curve:
            plt.close(fig)
            print(f"Averaged spatial group {group} has no available curves.")
            continue

        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel if ylabel is not None else "Value")
        ax.set_xscale(spatial_style.xscale)
        ax.set_yscale(spatial_style.yscale)
        ax.set_title(" + ".join(group) + " (averaged)")
        ax.grid(True)
        ax.text(
            0.02,
            0.98,
            averaging_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
        if len(ax.lines) <= 12:
            ax.legend(frameon=False)
        fig.tight_layout()
        plt.show()
