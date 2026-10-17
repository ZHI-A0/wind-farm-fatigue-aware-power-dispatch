"""Unit tests for fatigue estimation and constrained power allocation."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from damage_estimator import DamageEstimatorConfig, OnlineDamageEstimator
from optimizer import OptimizerConfig, optimize_power_step, project_bounded_simplex
from rainflow_eval import OfflineEvalConfig, damage_from_signal, strict_rainflow_cycles


class CoreTests(unittest.TestCase):
    def test_window_proxy_hand_calculated_triangle(self):
        cfg = DamageEstimatorConfig(window_size=10, sn_k=1, sn_m=3, use_goodman=False)
        estimator = OnlineDamageEstimator(1, cfg)
        increments = [estimator.update(np.array([x]), np.array([0.0]))[0] for x in [0., 2., 0.]]
        np.testing.assert_allclose(increments, [0., .1, .2], rtol=0, atol=1e-15)
        self.assertAlmostEqual(estimator.cumulative_damage[0], .3)

    def test_strict_rfc_triangle_counts_one_cycle_in_total(self):
        amp, mean, count = strict_rainflow_cycles([0., 2., 0.])
        np.testing.assert_array_equal(amp, [1., 1.])
        np.testing.assert_array_equal(mean, [1., 1.])
        self.assertEqual(count.sum(), 1.)
        cfg = OfflineEvalConfig(sn_k=1, sn_m=3, use_goodman=False)
        self.assertEqual(damage_from_signal([0., 2., 0.], cfg), 1.)
        self.assertEqual(damage_from_signal([2., 2., 2.], cfg), 0.)

    def test_projection_known_solution(self):
        actual = project_bounded_simplex([4., -1., 2.], total=3., lower=[0., 0., 0.], upper=[2., 2., 2.])
        np.testing.assert_allclose(actual, [2., 0., 1.], rtol=0, atol=1e-12)

    def test_projection_rejects_infeasible_total(self):
        with self.assertRaisesRegex(ValueError, "Infeasible projection target"):
            project_bounded_simplex([0., 0.], total=3., lower=[0., 0.], upper=[1., 1.])

    def test_damage_guidance_and_all_constraints(self):
        cfg = OptimizerConfig()
        previous = np.full(4, 3.)
        power, _ = optimize_power_step(p_prev=previous, p_grid=12., damage_hint=[0., 1., 2., 3.], cfg=cfg)
        self.assertAlmostEqual(power.sum(), 12., places=10)
        self.assertTrue(np.all(np.diff(power) < 0))
        self.assertTrue(np.all((power >= 0) & (power <= 5)))
        self.assertTrue(np.all(np.abs(power - 3) <= 1))
        self.assertTrue(np.all(np.abs(power - previous) <= .5))


if __name__ == "__main__":
    unittest.main(verbosity=2)
