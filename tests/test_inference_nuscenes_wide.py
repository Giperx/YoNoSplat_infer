"""Geometry checks for single-frame nuScenes wide inference. No model and no GPU."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import inference_nuscenes_wide as wide


class InputSizeTest(unittest.TestCase):
    def test_nuscenes_short_side_matches_training_and_long_side_is_patch_aligned(self):
        # Native nuScenes rear cameras are 1600x900.
        height, width = wide.choose_input_hw((900, 1600))
        self.assertEqual((height, width), (224, 392))
        self.assertEqual(height % wide.PATCH_SIZE, 0)
        self.assertEqual(width % wide.PATCH_SIZE, 0)
        # 224 * 1600/900 = 398.22, nearest multiple of 14 is 392 (not 406).
        self.assertLess(abs(392 - 224 * 1600 / 900), abs(406 - 224 * 1600 / 900))

    def test_portrait_snaps_the_long_side(self):
        height, width = wide.choose_input_hw((1600, 900))
        self.assertEqual((height, width), (392, 224))


class ResizeIntrinsicsTest(unittest.TestCase):
    def test_cover_crop_preserves_aspect_and_shifts_principal_point(self):
        plan = wide.plan_resize_and_crop((900, 1600), (224, 392))
        self.assertGreaterEqual(plan.scaled_h, plan.out_h)
        self.assertGreaterEqual(plan.scaled_w, plan.out_w)
        self.assertEqual(plan.scaled_h, plan.out_h)  # height is the filled side
        self.assertGreater(plan.col, 0)
        self.assertEqual(plan.row, 0)
        self.assertLessEqual(plan.col + plan.out_w, plan.scaled_w)

        src = wide.PixelIntrinsics(fx=800.0, fy=800.0, cx=830.0, cy=480.0)
        out = wide.resize_and_crop_intrinsics(src, plan)
        scale_x = plan.scaled_w / plan.src_w
        scale_y = plan.scaled_h / plan.src_h
        self.assertAlmostEqual(out.fx, src.fx * scale_x)
        self.assertAlmostEqual(out.fy, src.fy * scale_y)
        self.assertAlmostEqual(out.cx, src.cx * scale_x - plan.col)
        self.assertAlmostEqual(out.cy, src.cy * scale_y - plan.row)

    def test_image_and_mask_share_the_crop(self):
        plan = wide.plan_resize_and_crop((8, 16), (4, 6))
        image = Image.new("RGB", (16, 8), (10, 20, 30))
        mask = Image.new("L", (16, 8), 255)
        cropped = wide.apply_plan_image(image, plan)
        keep = wide.apply_plan_mask(mask, plan)
        self.assertEqual(cropped.shape, (4, 6, 3))
        self.assertEqual(keep.shape, (4, 6))
        self.assertTrue(keep.all())


class WideIntrinsicsTest(unittest.TestCase):
    def test_three_times_width_keeps_pixel_focal_length_and_recenters(self):
        context = wide.PixelIntrinsics(fx=200.0, fy=180.0, cx=190.0, cy=100.0)
        wide_k, wide_hw = wide.make_wide_intrinsics(context, (224, 392), 3.0)
        self.assertEqual(wide_hw, (224, 1176))
        self.assertEqual(wide_k.fx, context.fx)
        self.assertEqual(wide_k.fy, context.fy)
        self.assertEqual(wide_k.cy, context.cy)
        self.assertEqual(wide_k.cx, 1176 / 2.0)

        context_n = wide.pixel_to_normalized(context, 224, 392)
        wide_n = wide.pixel_to_normalized(wide_k, *wide_hw)
        self.assertAlmostEqual(float(wide_n[0, 0]), float(context_n[0, 0]) / 3.0)
        self.assertAlmostEqual(float(wide_n[1, 1]), float(context_n[1, 1]))
        self.assertAlmostEqual(float(wide_n[0, 2]), 0.5)
        # Symmetric rasterizer FOV is 2*atan(0.5/fx_norm). Smaller fx_norm => wider FOV.
        context_half = 0.5 / float(context_n[0, 0])
        wide_half = 0.5 / float(wide_n[0, 0])
        self.assertAlmostEqual(wide_half, context_half * 3.0)


class PoseNormTest(unittest.TestCase):
    def test_render_camera_becomes_identity_and_baseline_becomes_one(self):
        c2w = np.eye(4, dtype=np.float64)[None].repeat(3, axis=0)
        c2w[0, :3, 3] = (0.0, 0.0, 0.0)
        c2w[1, :3, 3] = (1.5, 0.0, 0.2)
        c2w[2, :3, 3] = (-0.4, 0.2, 1.0)
        c2w[1, :3, :3] = np.array(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64
        )
        out, scale = wide.normalize_poses(c2w)
        self.assertAlmostEqual(scale, wide.max_pairwise_distance(c2w))
        self.assertGreater(scale, 1.0)
        self.assertTrue(np.allclose(out[0], np.eye(4), atol=1e-5))
        centers = out[:, :3, 3]
        pairwise = []
        for i in range(3):
            for j in range(i + 1, 3):
                pairwise.append(np.linalg.norm(centers[i] - centers[j]))
        self.assertAlmostEqual(max(pairwise), 1.0, places=5)


class EgoIndexTest(unittest.TestCase):
    def test_only_masked_views_and_black_pixels_are_indexed(self):
        keep = np.ones((3, 2, 4), dtype=bool)
        keep[1, 0, 1] = False
        keep[2, 1, 3] = False
        keep[0, 0, 0] = False  # render view, not listed
        indices = wide.ego_remove_indices(keep, [1, 2])
        self.assertEqual(set(indices.tolist()), {1 * 8 + 0 * 4 + 1, 2 * 8 + 1 * 4 + 3})

    def test_mask_names_match_rear_cameras(self):
        self.assertEqual(wide.camera_mask_path(Path("/m"), 5).name, "CAM_BACK_mask.png")
        self.assertEqual(wide.camera_mask_path(Path("/m"), 4).name, "CAM_BACK_RIGHT_mask.png")
        self.assertEqual(wide.camera_mask_path(Path("/m"), 3).name, "CAM_BACK_LEFT_mask.png")


class DatasetLayoutTest(unittest.TestCase):
    def test_scene_list_and_frame_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "list.txt"
            path.write_text("037\n\n# skip\n074\n")
            self.assertEqual(wide.read_scene_list(path), ["037", "074"])
        self.assertEqual(wide.normalize_frame_id("0"), "000")
        self.assertEqual(wide.normalize_frame_id("12"), "012")
        self.assertEqual(wide.normalize_frame_id("000"), "000")

    def test_real_val_list_and_one_frame_resolve(self):
        scenes = wide.read_scene_list(wide.DEFAULT_SCENE_LIST)
        self.assertEqual(scenes, ["037", "074", "076", "089"])
        scene = wide.DEFAULT_DATA_ROOT / "037"
        frames = wide.enumerate_frames(scene, [5, 4, 3])
        self.assertIn("000", frames)
        self.assertGreater(len(frames), 1)
        intrinsics = wide.load_pixel_intrinsics(scene / "intrinsics" / "5.txt")
        self.assertGreater(intrinsics.fx, 100.0)
        pose = wide.load_c2w(scene / "cam2ego_extrinsics" / "5.txt")
        self.assertEqual(pose.shape, (4, 4))
        self.assertTrue(np.allclose(pose[3], [0, 0, 0, 1]))


if __name__ == "__main__":
    unittest.main()
