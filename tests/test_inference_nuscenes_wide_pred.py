"""Wide-K math for image-only nuScenes inference. No model and no GPU."""

import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import inference_nuscenes_wide_pred as pred
import wide_datasets as datasets


class PredictedWideIntrinsicsTest(unittest.TestCase):
    def test_square_input_renders_three_times_then_saved_canvas_is_1176(self):
        matrix, shape = pred.wide_intrinsics_from_predicted_focal(
            0.96, 0.96, (pred.MODEL_SIZE, pred.MODEL_SIZE), 3.0
        )
        self.assertEqual(shape, (224, 672))
        self.assertEqual(pred.SAVE_WIDTH, 1176)
        self.assertAlmostEqual(float(matrix[0, 0]), 0.96 / 3.0)
        self.assertAlmostEqual(float(matrix[1, 1]), 0.96)
        self.assertAlmostEqual(float(matrix[0, 2]), 0.5)
        self.assertAlmostEqual(float(matrix[1, 2]), 0.5)

    def test_saved_image_is_stretched_to_1176_without_changing_height(self):
        source = np.zeros((224, 672, 3), dtype=np.float32)
        source[:, :, 0] = 1.0
        stretched = pred.stretch_rgb(source, pred.MODEL_SIZE, pred.SAVE_WIDTH)
        self.assertEqual(stretched.shape, (224, 1176, 3))
        self.assertGreater(float(stretched[:, :, 0].mean()), 0.99)

    def test_placeholder_is_centered_and_overwritable(self):
        matrices = pred.placeholder_intrinsics(3)
        self.assertEqual(matrices.shape, (3, 3, 3))
        self.assertTrue(np.allclose(matrices[:, 0, 2], 0.5))
        self.assertTrue(np.allclose(matrices[:, 1, 2], 0.5))
        self.assertTrue(np.allclose(matrices[:, 0, 0], 1.0))

    def test_non_positive_focal_is_rejected(self):
        with self.assertRaises(ValueError):
            pred.wide_intrinsics_from_predicted_focal(0.0, 0.9, (224, 392), 3.0)


class KeepAspectSizeTest(unittest.TestCase):
    def test_default_context_stays_square_and_aspect_render_matches_saved_width(self):
        cases = (
            ((900, 1600), 1176),
            ((1080, 1920), 1176),
            ((1024, 1224), 798),
            ((1216, 1936), 1050),
        )
        for src_hw, saved in cases:
            self.assertEqual(pred.context_hw(src_hw, False), (224, 224))
            context = pred.context_hw(src_hw, True)
            self.assertEqual(context[0], 224)
            self.assertEqual(context[1] % 14, 0)
            self.assertEqual(pred.raster_hw(context, 3.0), (224, saved))
            self.assertEqual(datasets.save_width(src_hw), saved)

    def test_aspect_fit_preserves_height_and_square_fit_does_not(self):
        image = Image.new("RGB", (1600, 900), (20, 40, 60))
        context = pred.context_hw((900, 1600), True)
        fitted = pred.fit_context_image(image, context, True)
        stretched = pred.fit_context_image(image, (224, 224), False)
        self.assertEqual(fitted.shape, (224, 392, 3))
        self.assertEqual(stretched.shape, (224, 224, 3))
        self.assertGreater(float(fitted.mean()), 0.0)

    def test_aspect_color_is_not_resized_again(self):
        color = np.zeros((224, 1176, 3), dtype=np.float32)
        color[:, :, 1] = 0.5
        finished = pred.finish_wide_color(color, 1176, True)
        self.assertIs(finished, color)
        with self.assertRaises(RuntimeError):
            pred.finish_wide_color(np.zeros((224, 672, 3), dtype=np.float32), 1176, True)

    def test_portrait_frame_is_rejected(self):
        with self.assertRaises(ValueError):
            pred.context_hw((1600, 900), True)

    def test_aspect_output_dir_is_separate_from_the_square_run(self):
        path = pred.aspect_output_dir(Path("outputs/nuscenes_wide_pred"))
        self.assertEqual(path, Path("outputs/nuscenes_wide_pred_aspect"))
        multi = pred.aspect_output_dir(Path("outputs/nuscenes_wide_pred_multiframes"))
        self.assertEqual(multi.name, "nuscenes_wide_pred_multiframes_aspect")


if __name__ == "__main__":
    unittest.main()
