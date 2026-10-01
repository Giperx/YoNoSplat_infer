"""Wide-K math for image-only nuScenes inference. No model and no GPU."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import inference_nuscenes_wide_pred as pred


class PredictedWideIntrinsicsTest(unittest.TestCase):
    def test_three_times_width_divides_only_fx(self):
        matrix, shape = pred.wide_intrinsics_from_predicted_focal(0.50, 0.90, (224, 392), 3.0)
        self.assertEqual(shape, (224, 1176))
        self.assertAlmostEqual(float(matrix[0, 0]), 0.50 / 3.0)
        self.assertAlmostEqual(float(matrix[1, 1]), 0.90)
        self.assertAlmostEqual(float(matrix[0, 2]), 0.5)
        self.assertAlmostEqual(float(matrix[1, 2]), 0.5)

    def test_placeholder_is_centered_and_overwritable(self):
        matrices = pred.placeholder_intrinsics(3)
        self.assertEqual(matrices.shape, (3, 3, 3))
        self.assertTrue(np.allclose(matrices[:, 0, 2], 0.5))
        self.assertTrue(np.allclose(matrices[:, 1, 2], 0.5))
        self.assertTrue(np.allclose(matrices[:, 0, 0], 1.0))

    def test_non_positive_focal_is_rejected(self):
        with self.assertRaises(ValueError):
            pred.wide_intrinsics_from_predicted_focal(0.0, 0.9, (224, 392), 3.0)


if __name__ == "__main__":
    unittest.main()
