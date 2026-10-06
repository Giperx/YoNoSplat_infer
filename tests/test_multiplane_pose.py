"""Predicted-pose multiplane yaw. No model and no GPU."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import inference_nuscenes_wide_pred as pred


def _axis(pose):
    return pose[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=np.float32)


class MultiplanePoseTest(unittest.TestCase):
    def test_left_looks_left_and_planes_stay_concentric(self):
        center = np.eye(4, dtype=np.float32)
        center[:3, 3] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        fx = 0.8
        planes = pred.multiplane_c2w(center, fx)
        self.assertEqual(planes.shape, (3, 4, 4))
        np.testing.assert_allclose(planes[1], center)
        for plane in planes:
            np.testing.assert_allclose(plane[:3, 3], center[:3, 3])
            self.assertAlmostEqual(float(np.linalg.det(plane[:3, :3])), 1.0, places=5)
        alpha = pred.plane_fov(fx)
        left = _axis(planes[0])
        right = _axis(planes[2])
        forward = _axis(planes[1])
        left_angle = math.acos(float(np.clip(np.dot(forward, left), -1.0, 1.0)))
        right_angle = math.acos(float(np.clip(np.dot(forward, right), -1.0, 1.0)))
        self.assertAlmostEqual(left_angle, alpha, places=5)
        self.assertAlmostEqual(right_angle, alpha, places=5)
        self.assertLess(float(left[0]), 0.0)
        self.assertGreater(float(right[0]), 0.0)

    def test_stitch_is_three_planes_wide(self):
        planes = np.zeros((3, 4, 5, 3), dtype=np.float32)
        planes[0, :, :, 0] = 1.0
        planes[2, :, :, 2] = 1.0
        image = pred.stitch_planes(planes)
        self.assertEqual(image.shape, (4, 15, 3))
        self.assertAlmostEqual(float(image[:, :5, 0].mean()), 1.0)
        self.assertAlmostEqual(float(image[:, 10:, 2].mean()), 1.0)

    def test_non_positive_focal_is_rejected(self):
        with self.assertRaises(ValueError):
            pred.multiplane_c2w(np.eye(4, dtype=np.float32), 0.0)


if __name__ == "__main__":
    unittest.main()
