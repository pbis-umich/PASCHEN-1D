from __future__ import annotations

import importlib

import pytest

from config import SimulationConfig
from validation import validate_simulation_config


@pytest.mark.parametrize(
    "case_module",
    [
        "config_case_helium_photoemission_discharge",
        "config_case_deuterium_pulsed_discharge",
    ],
)
def test_lfa_rejects_unimplemented_empirical_electron_transport(case_module):
    cfg = importlib.import_module(case_module).SimulationConfig()
    # The shipped table-based configuration is valid for this gas.
    validate_simulation_config(cfg)
    cfg.local_field_approximation.electron_transport_source = "user_defined_equation"

    with pytest.raises(
        ValueError,
        match="user-defined electron transport is implemented only for argon and nitrogen",
    ):
        validate_simulation_config(cfg)


@pytest.mark.parametrize("gas", ["argon", "nitrogen"])
def test_lfa_accepts_supported_empirical_electron_transport(gas):
    cfg = SimulationConfig()
    cfg.plasma_state.gas = gas
    cfg.plasma.electron_kinetics_model = "local_field_approximation"
    cfg.local_field_approximation.electron_transport_source = "user_defined_equation"

    validate_simulation_config(cfg)
