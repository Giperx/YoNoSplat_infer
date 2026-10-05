"""Pool Lyft 1920 and 1224 reports, and add equal-weight region means.

``Mean_LR`` and ``Mean_LRC`` are added to sparse photometric and histogram
reports. WideDrive is skipped because it already scores the full image.
The Lyft file is written under the 1920 render directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from summary_rows import insert_equal_means, write_lyft_report

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


def backfill_means(outputs=OUTPUTS):
    """Insert Mean_LR and Mean_LRC into existing photometric and HM reports."""
    changed = []
    for directory in sorted(path for path in outputs.iterdir() if path.is_dir()):
        if "widedrive" in directory.name:
            continue
        for path in sorted(directory.glob("yonosplat_*.txt")):
            if not (path.name.startswith("yonosplat_photometric_") or path.name.startswith("yonosplat_HM_")):
                continue
            original = path.read_text()
            updated, did_change = insert_equal_means(original)
            if did_change:
                path.write_text(updated)
                changed.append(path)
    return changed


def lyft_directories(outputs=OUTPUTS):
    return sorted(
        path for path in outputs.glob("lyft1920_wide_pred*")
        if path.is_dir() and "lyft1224" not in path.name
    )


def write_all_lyft(outputs=OUTPUTS):
    written = []
    for directory in lyft_directories(outputs):
        destination = write_lyft_report(directory)
        if destination is not None:
            written.append(destination)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-root", type=Path, default=None)
    parser.add_argument("--backfill-means", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if args.backfill_means:
        for path in backfill_means():
            print(f"means {path}")
    if args.all:
        for path in write_all_lyft():
            print(f"lyft {path}")
    elif args.render_root is not None:
        destination = write_lyft_report(args.render_root)
        if destination is None:
            print(f"Lyft sibling reports are not both ready for {args.render_root}", file=sys.stderr)
        else:
            print(f"lyft {destination}")
    elif not args.backfill_means:
        parser.error("pass --render-root, --all, or --backfill-means")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
