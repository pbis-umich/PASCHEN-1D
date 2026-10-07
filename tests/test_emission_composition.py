from __future__ import annotations

import itertools

import numpy as np
import pytest

from config import SimulationConfig
from emission import (
    build_emission_model,
    fowler_nordheim_J,
    murphy_good_cold_J,
    richardson_dushman_J,
)
from validation import validate_simulation_config


MECHANISM_COMBINATIONS = [
    combination
    for count in (1, 2, 3)
    for combination in itertools.combinations(("fn", "mg", "rd"), count)
]


@pytest.mark.parametrize("material_mode", ("shared", "separate"))
@pytest.mark.parametrize("electrode", ("anode", "cathode"))
@pytest.mark.parametrize("mechanisms", MECHANISM_COMBINATIONS)
def test_composed_emitters_keep_each_mechanisms_material_parameters(
    material_mode: str, electrode: str, mechanisms: tuple[str, ...]
) -> None:
    cfg = SimulationConfig()
    cfg.emission.electrode_material_mode = material_mode
    for side in ("anode", "cathode"):
        setattr(cfg.emission, f"enable_{side}_external_emission", side == electrode)
        setattr(cfg.boundary, f"{side}_electron_boundary", "electron_emission")
        for mechanism in ("constant_J", "fn", "mg", "rd", "quantum_pulse"):
            setattr(
                cfg.emission,
                f"{side}_enable_{mechanism}_emission",
                side == electrode and mechanism in mechanisms,
            )

    # Distinct shared/anode/cathode values also check parameter resolution.
    for offset, prefix in enumerate(("shared", "anode", "cathode")):
        parameters = {
            "fn_work_function_eV": 5.0 + offset * 0.1,
            "fn_field_scale_factor": 1.0 + offset * 0.1,
            "mg_work_function_eV": 4.0 + offset * 0.1,
            "mg_field_scale_factor": 2.0 + offset * 0.1,
            "rd_work_function_eV": 3.0 + offset * 0.1,
            "rd_emitter_K": 1000.0 + offset * 100.0,
        }
        for name, value in parameters.items():
            setattr(cfg.emission, f"{prefix}_{name}", value)
    validate_simulation_config(cfg)

    prefix = "shared" if material_mode == "shared" else electrode

    def parameter(name: str) -> float:
        return getattr(cfg.emission, f"{prefix}_{name}")

    surface_field = 1.0e9
    expected_components = {
        "fn": fowler_nordheim_J(
            surface_field * parameter("fn_field_scale_factor"),
            parameter("fn_work_function_eV"),
        ),
        "mg": murphy_good_cold_J(
            surface_field * parameter("mg_field_scale_factor"),
            parameter("mg_work_function_eV"),
            parameter("mg_f_clip_min"),
            parameter("mg_f_clip_max"),
        ),
        "rd": richardson_dushman_J(
            parameter("rd_emitter_K"),
            parameter("rd_work_function_eV"),
            parameter("rd_A_R"),
        ),
    }
    expected = sum(expected_components[name] for name in mechanisms)
    model = build_emission_model(cfg)
    actual = model.current_density(
        0.0, 0.0, 1.0e-10, E_surface=surface_field, electrode=electrode
    )

    np.testing.assert_allclose(actual, expected, rtol=1.0e-13, atol=0.0)
