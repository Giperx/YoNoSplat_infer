"""Photometric scores against the height-224 sparse wide GT.

The GT folders are not required to exist yet. When one is missing this script
exits before scoring. A render is bicubic-resized to the GT size only when the
two sizes differ. Histogram matching stays in memory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    PHOTOMETRIC_NAMES,
    SPARSE_REGIONS,
    collect_renders,
    find_gt,
    fmt_photometric,
    load_binary_mask,
    load_rgb,
    load_scene_ids,
    open_report,
    resolve,
    timestamp,
    write_bucket,
)
from photometric import load_lpips, score_sparse

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    from common import add_common_args
    add_common_args(parser)
    parser.add_argument("--histogram-match", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    preset, render_root, gt_root, val_list = resolve(args)
    if not render_root.is_dir():
        raise SystemExit(f"render root not found: {render_root}")
    if not gt_root.is_dir():
        raise SystemExit(
            f"GT root not found: {gt_root}\n"
            "Height-224 sparse GT is not in the checkout yet. "
            "CRCS and IPS do not need it."
        )
    scenes, missing_scenes = load_scene_ids(render_root, val_list)
    if not scenes:
        raise SystemExit(f"no rendered scenes under {render_root}")
    jobs = collect_renders(render_root, scenes, args.image_dir)
    if not jobs:
        raise SystemExit(f"no wide images under {render_root}")

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading LPIPS alex on {device}.", flush=True)
    lpips_fn = load_lpips(device)

    keys = SPARSE_REGIONS
    global_buckets = {key: [] for key in keys}
    scene_buckets = {}
    skipped = {"missing_gt": 0, "missing_mask": 0, "empty": 0, "resized": 0, "car_mask_missing": 0}
    render_size = None

    for scene, frame, path in tqdm(jobs, desc="photometric", unit="frame"):
        gt_path, mask_path = find_gt(preset, gt_root, scene, frame)
        if gt_path is None:
            skipped["missing_gt"] += 1
            continue
        if mask_path is None:
            skipped["missing_mask"] += 1
            continue
        gt, _ = load_rgb(gt_path)
        gt_mask = load_binary_mask(mask_path, (gt.shape[1], gt.shape[0]))
        render, resized = load_rgb(path, (gt.shape[1], gt.shape[0]))
        render_size = (render.shape[1], render.shape[0])
        if resized:
            skipped["resized"] += 1
        frame_metrics, car_missing = score_sparse(
            render, gt, gt_mask, preset, scene, device, lpips_fn, args.histogram_match,
        )
        if car_missing:
            skipped["car_mask_missing"] += 1
        if not frame_metrics:
            skipped["empty"] += 1
            continue
        buckets = scene_buckets.setdefault(scene, {key: [] for key in keys})
        for key, value in frame_metrics.items():
            if key not in buckets:
                continue
            buckets[key].append(value)
            global_buckets[key].append(value)

    scored = len(global_buckets[keys[0]])
    tag = "HM" if args.histogram_match else "photometric"
    out_path = render_root / f"yonosplat_{tag}_{timestamp()}.txt"
    expected = preset["expected_wh"]
    meta = [
        f"Dataset: {args.dataset}",
        f"Mode: {args.mode}",
        f"Render root: {render_root}",
        f"GT root: {gt_root}",
        f"Val list: {val_list}",
        f"Histogram match: {str(bool(args.histogram_match)).lower()}",
        f"Expected WxH: {expected[0]}x{expected[1]}",
        f"Observed WxH: {render_size[0]}x{render_size[1]}" if render_size else "Observed WxH: none",
        "GT is the forthcoming height-224 sparse wide set.",
        "JPEG quality 95. MAE and RMSE are on the 0-255 scale. PSNR uses [0, 1].",
        "Left/Right SSIM is sparse. Center SSIM is a dense window. LPIPS is Center and Center_masked.",
        "Center_masked is the GT mask AND the camera-5 ego-car mask. WideDrive has no ego-car mask.",
        "Histogram matching is in memory and does not write match/.",
        "Skipped missing GT: {missing_gt}. Missing mask: {missing_mask}. "
        "Empty: {empty}. Render resized to GT: {resized}. "
        "Frames without ego-car mask: {car_mask_missing}.".format(**skipped),
        f"Val-list scenes without a render directory: {len(missing_scenes)}.",
        f"Scored frames: {scored} / {len(jobs)}.",
    ]
    handle = open_report(out_path, f"YoNoSplat wide {tag}", meta)
    with handle:
        handle.write("\n" + "=" * 80 + "\nSummary\n" + "=" * 80 + "\n")
        write_bucket(handle, keys, global_buckets, PHOTOMETRIC_NAMES, fmt_photometric)
        handle.write("\n" + "=" * 80 + "\nPer-scene\n" + "=" * 80 + "\n")
        for scene in sorted(scene_buckets):
            handle.write(f"\nScene {scene}:\n")
            write_bucket(handle, keys, scene_buckets[scene], PHOTOMETRIC_NAMES, fmt_photometric, indent="  ")
    print(f"Wrote {out_path}", flush=True)
    if scored == 0:
        raise SystemExit("no frames were scored")


if __name__ == "__main__":
    main()
