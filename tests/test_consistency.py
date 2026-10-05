"""CBSR rises on a full-height exposure seam. PD falls when the image is blurred."""

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "metrics"))

import consistency


def _scene(height=48, width=1176, seed=0):
    rng = np.random.default_rng(seed)
    image = rng.integers(40, 180, size=(height, width, 3), dtype=np.uint8)
    return image


class ConsistencyTest(unittest.TestCase):
    def test_exposure_seam_raises_cbsr_and_blur_lowers_pd(self):
        image = _scene()
        width = image.shape[1]
        left, right = consistency.seam_columns(width)
        seamed = image.copy()
        seamed[:, :left] = (seamed[:, :left].astype(np.float32) * 0.55).astype(np.uint8)
        seamed[:, right:] = np.clip(seamed[:, right:].astype(np.float32) * 1.45, 0, 255).astype(np.uint8)
        blurred = gaussian_filter(image.astype(np.float32), sigma=(0, 3, 0)).astype(np.uint8)

        plain = consistency.score_image(image)
        exposed = consistency.score_image(seamed)
        soft = consistency.score_image(blurred)

        self.assertGreater(exposed["cbsr"], plain["cbsr"] * 3)
        self.assertLess(soft["pd"], plain["pd"])
        self.assertAlmostEqual(soft["cbsr"], plain["cbsr"], delta=plain["cbsr"] * 0.25 + 0.2)


if __name__ == "__main__":
    unittest.main()
