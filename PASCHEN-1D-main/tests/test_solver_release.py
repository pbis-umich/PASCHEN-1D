from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from diagnostics_io import load_run_context, read_spatial, read_temporal
from numerics_jit import is_numba_available
from paschen_1d import run_simulation
from release_test_utils import make_miniature_config


CASES = {
    "argon_photoemission": "config_case_argon_photoemission_discharge",
    "nitrogen_pulsed": "config_case_nitrogen_pulsed_discharge",
    "argon_dc": "config_case_argon_dc_discharge",
    "deuterium_pulsed": "config_case_deuterium_pulsed_discharge",
}


def load_case(module_name: str):
    module = __import__(module_name, fromlist=["SimulationConfig"])
    return module.SimulationConfig()


def run_quietly(cfg):
    with contextlib.redirect_stdout(io.StringIO()), patch(
        "paschen_1d.tqdm", lambda iterable, **kwargs: iterable
    ):
        return run_simulation(cfg)


def state_metrics(state) -> dict[str, float]:
    return {
        "V_gap_final": float(state.V_gap[-1]),
        "I_discharge_final": float(state.I_discharge[-2]),
        "ne_sum_final": float(np.sum(state.ne_final, dtype=np.float64)),
        "ni_sum_final": float(np.sum(state.ni_final, dtype=np.float64)),
        "phi_l2_final": float(np.linalg.norm(state.phi_final.astype(np.float64))),
        "E_l2_final": float(np.linalg.norm(state.E_final.astype(np.float64))),
        "mu_e_mean_final": float(np.mean(state.mu_e_final, dtype=np.float64)),
        "D_e_mean_final": float(np.mean(state.D_e_final, dtype=np.float64)),
        "mu_i_mean_final": float(np.mean(state.mu_i_final, dtype=np.float64)),
        "D_i_mean_final": float(np.mean(state.D_i_final, dtype=np.float64)),
    }


class SolverReleaseTests(unittest.TestCase):
    def test_shipped_cases_match_miniature_golden_references(self) -> None:
        references = json.loads(
            (Path(__file__).parent / "fixtures" / "golden_mini_cases.json").read_text()
        )
        with tempfile.TemporaryDirectory() as directory, contextlib.chdir(directory):
            for case_name, module_name in CASES.items():
                with self.subTest(case=case_name):
                    cfg = make_miniature_config(
                        load_case(module_name), run_name=case_name
                    )
                    actual = state_metrics(run_quietly(cfg))
                    for metric, expected in references[case_name].items():
                        np.testing.assert_allclose(
                            actual[metric], expected, rtol=2.0e-6, atol=1.0e-14,
                            err_msg=f"{case_name}: {metric}",
                        )

    def test_saved_output_readers_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory, contextlib.chdir(directory):
            cfg = make_miniature_config(
                load_case("config_case_deuterium_pulsed_discharge"),
                run_name="reader_round_trip",
            )
            state = run_quietly(cfg)
            ctx = load_run_context("reader_round_trip", project_dir=directory)
            self.assertEqual(ctx.meta["software"]["version"], "1.0.1")
            electron_provenance = ctx.meta["transport_sources"][
                "electron_table_provenance"
            ]
            ion_provenance = ctx.meta["transport_sources"]["ion_table_provenance"]
            self.assertIn("tables", electron_provenance)
            self.assertIn("compatible_pair", ion_provenance)
            self.assertFalse(
                electron_provenance["tables"]["electron_transport"]["path"].startswith("/")
            )
            self.assertFalse(ion_provenance["mobility_table"]["path"].startswith("/"))
            time, voltage = read_temporal(ctx, "V_gap")
            saved_time, density = read_spatial(ctx, "ne")
            self.assertEqual(voltage.shape, (cfg.numerics.Nt,))
            self.assertEqual(density.shape, (cfg.numerics.Nt, cfg.numerics.Nx))
            np.testing.assert_allclose(time, state.time)
            np.testing.assert_allclose(saved_time, state.time)
            np.testing.assert_allclose(ctx.x, state.x)
            np.testing.assert_allclose(voltage, state.V_gap)
            np.testing.assert_allclose(density[-1], state.ne_final)

    @unittest.skipUnless(is_numba_available(), "Numba is not installed")
    def test_numpy_numba_backend_parity(self) -> None:
        with tempfile.TemporaryDirectory() as directory, contextlib.chdir(directory):
            numpy_cfg = make_miniature_config(
                load_case("config_case_argon_dc_discharge"),
                run_name="numpy_backend",
                backend="numpy",
            )
            numba_cfg = make_miniature_config(
                load_case("config_case_argon_dc_discharge"),
                run_name="numba_backend",
                backend="numba",
            )
            numpy_state = run_quietly(numpy_cfg)
            numba_state = run_quietly(numba_cfg)
            for name in (
                "V_gap", "I_discharge", "ne_final", "ni_final", "phi_final",
                "E_final", "mu_e_final", "D_e_final", "mu_i_final", "D_i_final",
            ):
                np.testing.assert_allclose(
                    getattr(numba_state, name), getattr(numpy_state, name),
                    rtol=2.0e-5, atol=1.0e-12, err_msg=name,
                )


if __name__ == "__main__":
    unittest.main()
