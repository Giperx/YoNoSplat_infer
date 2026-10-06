"""Height-224 sparse GT paths. Does not read the dataset GT folders."""

import sys
import tempfile
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
        }
        for name, size in expected.items():
            preset = common.PRESETS[name]
            self.assertEqual(preset["expected_wh"], size)
            self.assertTrue(preset["gt_root"].endswith(f"sparseWideFOVImages3_{size[0]}x224"))
            self.assertEqual(preset["style"], "sparse")
        wide = common.PRESETS["widedrive"]
        self.assertEqual(wide["expected_wh"], (1176, 224))
        self.assertEqual(wide["style"], "dense")
        self.assertTrue(wide["gt_root"].endswith("WideDriveVal"))

    def test_sparse_gt_root_is_replaced_by_the_complete_widedrive_images(self):
        preset = common.PRESETS["widedrive"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sparse = root / "sparseWideFOVImages3_1176x224"
            full = root / "WideDriveVal"
            sparse.mkdir()
            full.mkdir()
            self.assertEqual(common._complete_gt_root(preset, sparse), full)
            self.assertEqual(common._complete_gt_root(preset, full), full)
            multiplane = root / "sparseMultiplaneImages3_672x224"
            multiplane.mkdir()
            self.assertEqual(common._complete_gt_root(preset, multiplane), multiplane)

    def test_dense_gt_uses_the_complete_camera2_image(self):
        preset = common.PRESETS["widedrive"]
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp) / "Town01_scene_0021" / "images"
            scene.mkdir(parents=True)
            (scene / "000_2.jpg").write_bytes(b"x")
            rgb, mask = common.find_gt(preset, Path(tmp), "Town01_scene_0021", "000")
            self.assertEqual(rgb.name, "000_2.jpg")
            self.assertIsNone(mask)
        self.assertEqual(common.dense_keys()[0], "Full_unmasked")
        self.assertEqual(common.dense_bounds(1176)["Full"], (0, 1176))

    def test_camera5_gt_wins_and_widedrive_camera2_name_is_accepted(self):
        preset = common.PRESETS["nuscenes"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = root / "Town01_scene_0021"
            for kind, name in (
                ("rgb", "000_5_sparse_wide.png"),
                ("rgb", "000_2_sparse_wide.png"),
                ("mask", "000_5_sparse_wide.png"),
                ("rgb", "001_2_sparse_wide.png"),
                ("mask", "001_2_sparse_wide.png"),
            ):
                directory = scene / kind
                directory.mkdir(parents=True, exist_ok=True)
                (directory / name).write_bytes(b"x")

            rgb, mask = common.find_gt(preset, root, scene.name, "000")
            self.assertEqual(rgb.name, "000_5_sparse_wide.png")
            self.assertEqual(mask.name, "000_5_sparse_wide.png")

            rgb, mask = common.find_gt(preset, root, scene.name, "001")
            self.assertEqual(rgb.name, "001_2_sparse_wide.png")
            self.assertEqual(mask.name, "001_2_sparse_wide.png")


if __name__ == "__main__":
    unittest.main()
