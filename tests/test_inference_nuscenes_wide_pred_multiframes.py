"""Window and ego-mask indexing for image-only multi-frame inference."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import inference_nuscenes_wide_pred_multiframes as multi


class WindowTest(unittest.TestCase):
    def test_three_frame_windows_end_on_the_rendered_frame(self):
        windows = multi.enumerate_windows(["000", "001", "002", "003"], 3)
        self.assertEqual(windows, [("000", "001", "002"), ("001", "002", "003")])

    def test_requested_frame_selects_only_its_history(self):
        frames = ["000", "001", "002", "003"]
        self.assertEqual(multi.select_windows(frames, 3, "2"), [("000", "001", "002")])
        self.assertEqual(multi.select_windows(frames, 3, "000"), [])

    def test_short_scene_has_no_window(self):
        self.assertEqual(multi.enumerate_windows(["000", "001"], 3), [])


class MaskIndexTest(unittest.TestCase):
    def test_only_the_newest_camera_5_is_kept(self):
        cameras = [5, 4, 3]
        render_index = multi.newest_render_index(3, cameras, 5)
        self.assertEqual(render_index, 6)
        masked = multi.views_to_mask(3, cameras, render_index, mask_render_view=False)
        self.assertNotIn(6, masked)
        self.assertEqual(masked, [0, 1, 2, 3, 4, 5, 7, 8])

    def test_render_view_can_be_masked_too(self):
        cameras = [5, 4, 3]
        render_index = multi.newest_render_index(3, cameras, 5)
        masked = multi.views_to_mask(3, cameras, render_index, mask_render_view=True)
        self.assertEqual(masked, list(range(9)))


if __name__ == "__main__":
    unittest.main()
