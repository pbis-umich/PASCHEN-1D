from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from diagnostics_io import compute_sheath_rows, load_run_context


def write_sheath_run(run_dir: Path, *, has_sheaths: bool) -> None:
    run_dir.mkdir(parents=True)
    metadata = {
        "Nt": 2, "Nx": 5, "T_total": 1.0, "L": 4.0,
        "A": 2.0, "save_every": 1,
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata))
    profiles = {
        "ne_sampled_mm.dat": [0, 1, 1, 1, 0] if has_sheaths else [1] * 5,
        "ni_sampled_mm.dat": [1] * 5,
        "phi_sampled_mm.dat": [8, 6, 4, 2, 0],
        "E_sampled_mm.dat": [2] * 5,
    }
    for filename, profile in profiles.items():
        np.array([profile, profile], dtype=np.float32).tofile(run_dir / filename)


@pytest.mark.parametrize("same_named_run_in_cwd", [False, True])
def test_sheath_rows_use_context_project_directory(
    tmp_path, monkeypatch, same_named_run_in_cwd
) -> None:
    project = tmp_path / "project"
    run_name = "runs/example"
    run_dir = project / run_name
    write_sheath_run(run_dir, has_sheaths=True)
    cwd = tmp_path / "other_project"
    cwd.mkdir()
    if same_named_run_in_cwd:
        write_sheath_run(cwd / run_name, has_sheaths=False)
    monkeypatch.chdir(cwd)

    context = load_run_context(run_name, project_dir=project)
    rows = compute_sheath_rows(context, save=True)

    assert len(rows) == 2
    for row in rows:
        assert row["anode_width_m"] == 1.0
        assert row["cathode_width_m"] == 1.0
        assert row["anode_voltage_drop_abs_V"] == 2.0
        assert row["cathode_voltage_drop_abs_V"] == 2.0
    assert json.loads((run_dir / "sheath_diagnostics.json").read_text()) == rows
    assert (run_dir / "sheath_diagnostics.csv").exists()
    assert not (cwd / run_name / "sheath_diagnostics.json").exists()
    assert not (cwd / run_name / "sheath_diagnostics.csv").exists()
