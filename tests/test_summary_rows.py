"""Equal-weight region means and Lyft frame-count pooling."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "metrics"))

import summary_rows


SAMPLE = """\
================================================================================
Summary
================================================================================
Left                : n=2  PSNR=10.00  MAE=20.00  RMSE=30.00  SSIM=0.5000
Center              : n=2  PSNR=20.00  MAE=10.00  RMSE=15.00  SSIM=0.8000  LPIPS=0.2000
Center_masked       : n=2  PSNR=21.00  MAE=9.00  RMSE=14.00  SSIM=0.8100  LPIPS=0.1900
Right               : n=4  PSNR=30.00  MAE=40.00  RMSE=50.00  SSIM=0.7000
Overall             : n=4  PSNR=22.00  MAE=18.00  RMSE=24.00  SSIM=0.7200

================================================================================
Per-scene
================================================================================
"""


class SummaryRowsTest(unittest.TestCase):
    def test_means_are_equal_weight_and_skip_masked_and_lpips(self):
        updated, changed = summary_rows.insert_equal_means(SAMPLE)
        self.assertTrue(changed)
        rows = {row["key"]: row for row in summary_rows.summary_rows(updated)}
        self.assertAlmostEqual(rows["Mean_LR"]["values"]["PSNR"], 20.0)
        self.assertAlmostEqual(rows["Mean_LRC"]["values"]["PSNR"], 20.0)
        self.assertAlmostEqual(rows["Mean_LRC"]["values"]["MAE"], 23.33)
        self.assertNotIn("LPIPS", rows["Mean_LRC"]["values"])
        self.assertEqual(rows["Mean_LR"]["n"], 2)
        again, changed_again = summary_rows.insert_equal_means(updated)
        self.assertFalse(changed_again)
        self.assertEqual(again, updated)

    def test_dense_widedrive_is_left_unchanged(self):
        text = "Style: dense\nFull_unmasked       : n=1  PSNR=1.00\n"
        updated, changed = summary_rows.insert_equal_means(text)
        self.assertFalse(changed)
        self.assertEqual(updated, text)

    def test_lyft_pool_weights_by_frame_count(self):
        small = summary_rows.parse_metric_line("Left                : n=1  PSNR=10.00")
        large = summary_rows.parse_metric_line("Left                : n=3  PSNR=20.00")
        pooled = summary_rows.pool_by_count([small, large])
        self.assertEqual(pooled["n"], 4)
        self.assertAlmostEqual(pooled["values"]["PSNR"], 17.5)


if __name__ == "__main__":
    unittest.main()
