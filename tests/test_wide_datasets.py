"""Save-width and ego-mask paths for the driving datasets."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import wide_datasets as data


class SaveWidthTest(unittest.TestCase):
    def test_native_heights_land_on_the_nuscenes_rule(self):
        self.assertEqual(data.save_width((900, 1600)), 1176)
        self.assertEqual(data.save_width((1080, 1920)), 1176)
        self.assertEqual(data.save_width((1024, 1224)), 798)
        self.assertEqual(data.save_width((1216, 1936)), 1050)
        self.assertEqual(data.save_width((1080, 5760)), 3570)


class MaskPathTest(unittest.TestCase):
    def test_each_dataset_uses_its_own_mask_layout(self):
        nuscenes = data.DATASETS["nuscenes"]
        self.assertEqual(data.ego_mask_path(nuscenes, 5).name, "CAM_BACK_mask.png")
        self.assertEqual(data.ego_mask_path(data.DATASETS["lyft1920"], 4).name, "4.jpg")
        ddad = data.ego_mask_path(data.DATASETS["ddad"], 3, scene="000")
        self.assertEqual(ddad.parts[-3:], ("000", "ego_car_masks", "3.jpg"))
        self.assertIsNone(data.ego_mask_path(data.DATASETS["widedrive"], 5, scene="Town01_scene_0021"))


if __name__ == "__main__":
    unittest.main()
