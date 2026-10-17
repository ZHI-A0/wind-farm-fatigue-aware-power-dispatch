"""Regression checks for invalid inputs and protection of experiment outputs."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from main import load_npz_data, load_yaml_like, main, parse_strategies
from optimizer import OptimizerConfig, build_feasible_bounds, project_bounded_simplex
sys.path.insert(0, str(ROOT / "tools"))
import prepare_data


class InputSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def arrays(self):
        return {key: np.ones((20, 2)) for key in ("v", "pref", "p_out", "tshaft", "ft")}

    def write_npz(self, arrays):
        path = self.root / "input.npz"
        np.savez(path, **arrays)
        return path

    def test_missing_load_is_rejected_instead_of_synthesized(self):
        arrays = self.arrays()
        del arrays["ft"]
        with self.assertRaisesRegex(ValueError, "missing measured channels"):
            load_npz_data(self.write_npz(arrays))

    def test_missing_measured_power_is_rejected(self):
        arrays = self.arrays()
        del arrays["p_out"]
        with self.assertRaisesRegex(ValueError, "measured power"):
            load_npz_data(self.write_npz(arrays))

    def test_alias_and_derived_grid_preserve_values(self):
        arrays = self.arrays()
        arrays["pout"] = arrays.pop("p_out")
        data = load_npz_data(self.write_npz(arrays))
        np.testing.assert_array_equal(data["p_out"], arrays["pout"])
        np.testing.assert_array_equal(data["p_grid"], arrays["pref"].sum(axis=1))

    def test_nonfinite_and_mismatched_channels_are_rejected(self):
        for key, value in (("v", np.full((20, 2), np.nan)),
                           ("ft", np.ones((20, 1))), ("p_grid", np.ones((20, 1)))):
            with self.subTest(key=key):
                arrays = self.arrays()
                arrays[key] = value
                with self.assertRaisesRegex(ValueError, "shape .* and finite"):
                    load_npz_data(self.write_npz(arrays))

    def test_yaml_errors_do_not_fall_back_to_partial_configuration(self):
        path = self.root / "config.yaml"
        for text in ("experiment: [", "!!python/object/apply:os.system ['false']"):
            path.write_text(text)
            with self.assertRaises(yaml.YAMLError):
                load_yaml_like(path)
        for text in ("- experiment", "experiment: []", ""):
            path.write_text(text)
            with self.assertRaises(ValueError):
                load_yaml_like(path)

    def test_unknown_and_duplicate_strategies_are_rejected(self):
        for text in ("mean,unknown", "mean,mean", "../output"):
            with self.assertRaises(ValueError):
                parse_strategies(text)

    def test_run_name_cannot_escape_output_directory(self):
        for name in ("../escape", "/absolute", "..", "sub/path", "sub\\path"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "single directory name"):
                main(["--out-root", str(self.root), "--run-name", name])
        self.assertEqual(list(self.root.iterdir()), [])

    def test_existing_results_are_preserved(self):
        run = self.root / "existing"
        run.mkdir()
        marker = run / "final_metrics.json"
        marker.write_text('{"preserve": true}')
        with self.assertRaises(FileExistsError):
            main(["--out-root", str(self.root), "--run-name", "existing"])
        self.assertEqual(marker.read_text(), '{"preserve": true}')

    def test_invalid_split_fails_before_creating_outputs(self):
        data_path = self.write_npz(self.arrays())
        cfg_path = self.root / "config.yaml"
        cfg_path.write_text("experiment:\n  train_steps: 12\n  eval_steps: 20\n")
        with self.assertRaisesRegex(ValueError, "Evaluation range"):
            main(["--config", str(cfg_path), "--data-npz", str(data_path),
                  "--out-root", str(self.root), "--run-name", "invalid"])
        self.assertFalse((self.root / "invalid").exists())

    def test_projection_rejects_nonfinite_values(self):
        for total, raw in ((np.nan, [1., 1.]), (2., [np.inf, 1.])):
            with self.assertRaisesRegex(ValueError, "finite"):
                project_bounded_simplex(raw, total=total, lower=[0., 0.], upper=[2., 2.])
        with self.assertRaisesRegex(ValueError, "finite"):
            build_feasible_bounds(p_grid=np.nan, p_prev=[1., 1.], cfg=OptimizerConfig())

    def test_projection_random_feasible_constraints(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            lower = rng.uniform(0., 2., 20)
            upper = lower + rng.uniform(.01, 3., 20)
            feasible = rng.uniform(lower, upper)
            projected = project_bounded_simplex(rng.normal(0., 10., 20), total=feasible.sum(),
                                                lower=lower, upper=upper)
            self.assertAlmostEqual(projected.sum(), feasible.sum(), places=9)
            self.assertTrue(np.all(projected >= lower - 1e-12))
            self.assertTrue(np.all(projected <= upper + 1e-12))

    def test_rebuild_does_not_require_existing_npz_archives(self):
        (self.root / "data/raw").mkdir(parents=True)
        (self.root / "data/raw/wind_farms.mat").write_bytes(b"raw")
        import hashlib
        import json
        manifest = {"files": {
            "raw/wind_farms.mat": {"sha256": hashlib.sha256(b"raw").hexdigest()},
            "data_wf1.npz": {"sha256": "missing"},
        }}
        (self.root / "data/manifest.json").write_text(json.dumps(manifest))
        arrays = {"wf1": {"v": np.ones((2000, 100)), "p_grid": np.ones(2000)}}
        with patch.object(prepare_data, "ROOT", self.root), patch.object(
                prepare_data, "arrays_from_mat", return_value=arrays):
            self.assertTrue(prepare_data.verify_dataset(check_npz=False)["passed"])
            with self.assertRaises(FileNotFoundError):
                prepare_data.verify_dataset()


if __name__ == "__main__":
    unittest.main()
