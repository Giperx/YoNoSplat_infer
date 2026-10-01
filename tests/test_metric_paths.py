"""Height-224 sparse GT paths. Does not read the missing GT folders."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "metrics"))

import common


class GtPathTest(unittest.TestCase):
    def test_every_sparse_gt_is_224_high_and_three_times_the_aspect_width(self):
        expected = {
            "nuscenes": (1176, 224),
            "lyft1920": (1176, 224),
            "lyft1224": (798, 224),
            "ddad": (1050, 224),
            "widedrive": (1176, 224),
        }
        for name, size in expected.items():
            preset = common.PRESETS[name]
            self.assertEqual(preset["expected_wh"], size)
            self.assertTrue(preset["gt_root"].endswith(f"sparseWideFOVImages3_{size[0]}x224"))
            self.assertEqual(preset["style"], "sparse")


if __name__ == "__main__":
    unittest.main()
