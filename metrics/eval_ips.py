"""Kept for comparison. ``metrics/run_wide.sh`` calls ``eval_consistency.py`` instead.

Seam low-frequency gradient on YoNoSplat wide renders. No GT image is required.

IPS is the mean horizontal gradient of a Gaussian low-pass, on the 0-255
scale, at the left and right third-boundaries. The primary score is unmasked.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    add_common_args,
    collect_renders,
    find_gt,
    load_binary_mask,
    load_scene_ids,
    load_uint8,
    open_report,
    resolve,
    timestamp,
    write_bucket,
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


REGIONS = ("L", "R", "MeanLR")
VARIANTS = ("unmasked", "gt_mask")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--seam-vertical-ratio", type=float, default=0.5)
    parser.add_argument("--band-ratio", type=float, default=0.05)
    parser.add_argument("--sigma", type=float, default=5.0)
    return parser.parse_args()


def low_frequency(image_uint8, mask, sigma):
    image = image_uint8.astype(np.float64) / 255.0
    weight = np.ones(image.shape[:2] + (1,), dtype=np.float64) if mask is None else mask.astype(np.float64)[..., None]
    blurred = gaussian_filter(image * weight, sigma=[sigma, sigma, 0], mode="nearest")
    blurred_weight = gaussian_filter(weight, sigma=[sigma, sigma, 0], mode="nearest")
    low = np.zeros_like(image)
    safe = blurred_weight[:, :, 0] > 1e-5
    for channel in range(3):
        low[:, :, channel][safe] = blurred[:, :, channel][safe] / blurred_weight[:, :, 0][safe]
    return low


def compute_ips(image_uint8, mask, vertical_ratio, band_ratio, sigma):
    height, width = image_uint8.shape[:2]
    low = low_frequency(image_uint8, mask, sigma)
    v_half = int(height * vertical_ratio / 2)
    v_center = height // 2
    v0, v1 = max(0, v_center - v_half), min(height, v_center + v_half)
    band_half = max(1, int(width * band_ratio))
    results = {}
    for name, center in (("L", width // 3), ("R", 2 * width // 3)):
        x0, x1 = max(0, center - band_half), min(width, center + band_half)
        grad = np.abs(low[v0:v1, x0:x1][:, 1:] - low[v0:v1, x0:x1][:, :-1]).mean(axis=2)
        unmasked = float(grad.mean() * 255.0) if grad.size else 0.0
        masked = None
        if mask is not None:
            pair = mask[v0:v1, x0:x1][:, 1:] & mask[v0:v1, x0:x1][:, :-1]
            if pair.any():
                masked = float(grad[pair].mean() * 255.0)
        results[name] = {"unmasked": unmasked, "masked": masked}
    return results


def _fmt(_name, value):
    return f"{float(value):.4f}"


def main():
    args = parse_args()
    preset, render_root, gt_root, val_list = resolve(args)
    if not render_root.is_dir():
        raise SystemExit(f"render root not found: {render_root}")
    scenes, missing_scenes = load_scene_ids(render_root, val_list)
    if not scenes:
        raise SystemExit(f"no rendered scenes under {render_root}")
    jobs = collect_renders(render_root, scenes, args.image_dir)
    if not jobs:
        raise SystemExit(f"no wide images under {render_root}")
    keys = tuple(f"{region}_{variant}" for region in REGIONS for variant in VARIANTS)
    global_buckets = {key: [] for key in keys}
    scene_buckets = {}
    bad_mask = 0
    gt_available = gt_root.is_dir()
    for scene, frame, path in tqdm(jobs, desc="IPS", unit="frame"):
        image = load_uint8(path)
        mask = None
        if gt_available:
            _, mask_path = find_gt(preset, gt_root, scene, frame)
            if mask_path is not None:
                mask = load_binary_mask(mask_path, (image.shape[1], image.shape[0]))
            else:
                bad_mask += 1
        else:
            bad_mask += 1
        plain = compute_ips(image, None, args.seam_vertical_ratio, args.band_ratio, args.sigma)
        masked_scores = compute_ips(image, mask, args.seam_vertical_ratio, args.band_ratio, args.sigma) if mask is not None else None
        plain["MeanLR"] = {"unmasked": 0.5 * (plain["L"]["unmasked"] + plain["R"]["unmasked"]), "masked": None}
        if masked_scores is not None:
            left, right = masked_scores["L"]["masked"], masked_scores["R"]["masked"]
            masked_scores["MeanLR"] = {
                "unmasked": None,
                "masked": None if left is None or right is None else 0.5 * (left + right),
            }
        buckets = scene_buckets.setdefault(scene, {key: [] for key in keys})
        for region in REGIONS:
            row = {"value": plain[region]["unmasked"]}
            buckets[f"{region}_unmasked"].append(row)
            global_buckets[f"{region}_unmasked"].append(row)
            if masked_scores is not None and masked_scores[region]["masked"] is not None:
                row = {"value": masked_scores[region]["masked"]}
                buckets[f"{region}_gt_mask"].append(row)
                global_buckets[f"{region}_gt_mask"].append(row)
    out_path = render_root / f"yonosplat_IPS_{timestamp()}.txt"
    meta = [
        f"Dataset: {args.dataset}",
        f"Mode: {args.mode}",
        f"Render root: {render_root}",
        f"GT root: {gt_root}",
        f"Sigma: {args.sigma}",
        "IPS is the mean horizontal gradient of a Gaussian low-pass, times 255.",
        "MeanLR is the mean of L and R. gt_mask uses the height-224 sparse mask when present.",
        f"Frames without a GT mask: {bad_mask}.",
        f"Rendered frames: {len(jobs)}.",
        f"Val-list scenes without a render directory: {len(missing_scenes)}.",
    ]
    handle = open_report(out_path, "YoNoSplat wide IPS", meta)
    with handle:
        handle.write("\n" + "=" * 80 + "\nSummary\n" + "=" * 80 + "\n")
        write_bucket(handle, keys, global_buckets, ("value",), _fmt)
        handle.write("\n" + "=" * 80 + "\nPer-scene\n" + "=" * 80 + "\n")
        for scene in sorted(scene_buckets):
            handle.write(f"\nScene {scene}:\n")
            write_bucket(handle, keys, scene_buckets[scene], ("value",), _fmt, indent="  ")
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
