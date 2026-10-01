"""Timing-scope checks. Does not load the model."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import benchmark_wide as bench


class TimedViewsTest(unittest.TestCase):
    def test_single_frame_times_three_cameras(self):
        self.assertEqual(bench.timed_view_count(1, 3), 3)

    def test_multi_frame_times_the_whole_window(self):
        self.assertEqual(bench.timed_view_count(3, 3), 9)


if __name__ == "__main__":
    unittest.main()
