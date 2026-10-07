from __future__ import annotations

import json

import numpy as np
import pytest

from derived_diagnostics import _v_app_from_metadata
from diagnostics_io import load_run_context, read_temporal


@pytest.mark.parametrize("waveform_type", ["table", "tabulated", "measured_table"])
@pytest.mark.parametrize("reader", ["notebook", "legacy_context", "legacy_cwd"])
@pytest.mark.parametrize("scaled_columns", [False, True])
def test_saved_table_waveform_reconstruction(
    tmp_path, monkeypatch, waveform_type, reader, scaled_columns
) -> None:
    project = tmp_path / "project"
    run_name = "runs/table_example"
    run_dir = project / run_name
    run_dir.mkdir(parents=True)
    metadata = {
        "Nt": 7, "Nx": 3, "T_total": 1.5, "L": 1.0,
        "A": 1.0, "save_every": 1, "waveform_type": waveform_type,
        "table_path": "voltage.csv",
    }
    if scaled_columns:
        table_contents = "voltage,time\n5,2\n1,0\n3,1\n"
        metadata.update(
            table_time_column=1, table_voltage_column=0,
            table_time_scale=0.5, table_time_offset=0.25,
            table_voltage_scale=2.0, table_voltage_offset=-1.0,
        )
    else:
        table_contents = "1.25,9\n0.25,1\n0.75,5\n"
    (project / "voltage.csv").write_text(table_contents)
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata))
    other_project = tmp_path / "other_project"
    other_project.mkdir()
    monkeypatch.chdir(project if reader == "legacy_cwd" else other_project)
    context = load_run_context(run_name, project_dir=project)

    if reader == "notebook":
        time, actual = read_temporal(context, "V_app")
        np.testing.assert_array_equal(time, context.time)
    elif reader == "legacy_context":
        actual = _v_app_from_metadata(context.time, context.meta, project_dir=project)
    else:
        actual = _v_app_from_metadata(context.time, context.meta)

    # Both tables describe (t, V) = (0.25, 1), (0.75, 5), (1.25, 9).
    # Include interpolation between rows and constant endpoint extrapolation.
    np.testing.assert_allclose(actual, [1, 1, 3, 5, 7, 9, 9], rtol=0, atol=0)
    assert context.meta == metadata
