from __future__ import annotations

import pytest

from config import SimulationConfig
from validation import validate_simulation_config


@pytest.mark.parametrize("prefix", ("shared", "anode", "cathode"))
def test_quantum_phase_quadrature_rejects_single_point(prefix: str) -> None:
    cfg = SimulationConfig()
    cfg.emission.electrode_material_mode = "shared" if prefix == "shared" else "separate"
    setattr(cfg.emission, f"{prefix}_emission_wt_points", 1)

    with pytest.raises(ValueError, match=rf"{prefix}_emission_wt_points must be an integer >= 2"):
        validate_simulation_config(cfg)


@pytest.mark.parametrize("prefix", ("shared", "anode", "cathode"))
def test_quantum_phase_quadrature_accepts_two_points(prefix: str) -> None:
    cfg = SimulationConfig()
    cfg.emission.electrode_material_mode = "shared" if prefix == "shared" else "separate"
    setattr(cfg.emission, f"{prefix}_emission_wt_points", 2)

    validate_simulation_config(cfg)
