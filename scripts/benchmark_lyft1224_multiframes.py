#!/usr/bin/env python3
"""Multi-frame Lyft 1224 timing. All three frames are inside one timed forward."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_wide import main  # noqa: E402

if __name__ == "__main__":
    main(default_dataset="lyft1224", multi_frame=True)
