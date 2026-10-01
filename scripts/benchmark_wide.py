#!/usr/bin/env python3
"""Wide-FOV inference timing for YoNoSplat.

Single-frame times one forward over cameras 5, 4 and 3, then the wide render.
Multi-frame times one forward over the whole window. YoNoSplat has no history
queue, so the two older frames are inside the timer. FPS is output images per
second, and each output image costs that full forward.

Image loading, checkpoint loading, and the CPU resize/JPEG save are not timed.
The timed raster is 672x224. The saved canvas (1176, 798, or 1050 wide) is
applied only when writing images.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import inference_nuscenes_wide as base
import inference_nuscenes_wide_pred as pred
import inference_nuscenes_wide_pred_multiframes as multi
import wide_datasets as datasets


def timed_view_count(num_frames: int, camera_count: int) -> int:
    """Views inside one timed forward. Multi-frame is the whole window."""
    if num_frames < 1 or camera_count < 1:
        raise ValueError("num_frames and camera_count must be positive.")
    return int(num_frames) * int(camera_count)


def summarize(times_ms: list[float]) -> dict[str, float]:
    mean_ms = statistics.mean(times_ms)
    return {
        "mean_ms": mean_ms,
        "median_ms": statistics.median(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "stdev_ms": statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        "fps": 1000.0 / mean_ms,
    }


def parse_args(argv=None, default_dataset="nuscenes", multi_frame=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=default_dataset, choices=sorted(datasets.DATASETS))
    parser.add_argument("--scene", default=None)
    parser.add_argument("--frame", default=None, help="Newest frame for multi; the only frame for single.")
    parser.add_argument("--num-frames", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=int(os.environ.get("WARMUP", "50")))
    parser.add_argument("--measure", type=int, default=int(os.environ.get("MEASURE", "50")))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--scene-list", type=Path, default=None)
    parser.add_argument("--cameras", default="5,4,3")
    parser.add_argument("--render-camera", type=int, default=5)
    parser.add_argument("--width-factor", type=float, default=3.0)
    parser.add_argument("--checkpoint", type=Path, default=base.DEFAULT_CHECKPOINT)
    parser.add_argument("--car-mask-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--disable-car-mask", action="store_true")
    parser.add_argument("--mask-render-view", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--multi", action="store_true", default=multi_frame)
    return parser.parse_args(argv)


def pick_case(args):
    scenes = [args.scene] if args.scene else base.read_scene_list(args.scene_list)
    for scene in scenes:
        frames = base.enumerate_frames(args.data_root / scene, args.cameras)
        if args.multi:
            windows = multi.select_windows(frames, args.num_frames, args.frame)
            if windows:
                return scene, windows[0]
        elif args.frame:
            wanted = base.normalize_frame_id(args.frame)
            if wanted in frames:
                return scene, (wanted,)
        elif frames:
            return scene, (frames[0],)
    kind = "window" if args.multi else "frame"
    raise SystemExit(f"No valid {kind} in {args.scene_list}.")


def main(argv=None, default_dataset="nuscenes", multi_frame=False):
    import torch

    args = parse_args(argv, default_dataset=default_dataset, multi_frame=multi_frame)
    args.multi = multi_frame or args.multi
    spec = datasets.apply_defaults(args, multi=args.multi)
    args.cameras = base.parse_cameras(args.cameras)
    if args.render_camera not in args.cameras:
        raise SystemExit(f"--render-camera {args.render_camera} is not in {args.cameras}.")
    if not args.multi:
        args.cameras = [args.render_camera] + [cam for cam in args.cameras if cam != args.render_camera]
        render_index = 0
        masked = []
        if not args.disable_car_mask:
            masked = [
                index
                for index, camera in enumerate(args.cameras)
                if args.mask_render_view or camera != args.render_camera
            ]
    else:
        render_index = multi.newest_render_index(args.num_frames, args.cameras, args.render_camera)
        masked = [] if args.disable_car_mask else multi.views_to_mask(
            args.num_frames, args.cameras, render_index, args.mask_render_view
        )
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise SystemExit("CUDA is required for this benchmark.")
    torch.set_float32_matmul_precision("high")
    scene, window = pick_case(args)
    scene_dir = args.data_root / scene
    if args.multi:
        images = multi.load_window_images(scene_dir, window, args.cameras, pred.MODEL_SIZE)
        keep = None
        if masked:
            camera_keeps = multi.load_camera_keep_masks(spec, scene, args.cameras, pred.MODEL_SIZE)
            keep = multi.build_window_keep(
                camera_keeps, args.cameras, args.num_frames, render_index, args.mask_render_view
            )
    else:
        images = pred.load_frame_images(scene_dir, window[0], args.cameras, pred.MODEL_SIZE)
        keep = None
        if masked:
            keep = pred.load_dataset_keep(
                spec, scene, args.cameras, pred.MODEL_SIZE, render_index, args.mask_render_view
            )
    remove_index = None
    if keep is not None and masked:
        indices = base.ego_remove_indices(keep, masked)
        if indices.size:
            remove_index = torch.as_tensor(indices, device=args.device, dtype=torch.long)
    images_t = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).contiguous()
    images_t = images_t[None].to(args.device)
    intrinsics = torch.from_numpy(pred.placeholder_intrinsics(images.shape[0]))[None].to(args.device)
    near = torch.full((1, 1), pred.PRED_NEAR, dtype=torch.float32, device=args.device)
    far = torch.full((1, 1), pred.PRED_FAR, dtype=torch.float32, device=args.device)
    print(f"Loading model from {args.checkpoint} ...", flush=True)
    encoder, decoder = pred.build_model(args.checkpoint, args.device)
    print("Model loaded.", flush=True)

    def once():
        with torch.no_grad():
            pred.forward_wide_gpu(
                encoder,
                decoder,
                images_t,
                intrinsics,
                remove_index,
                render_index,
                args.width_factor,
                near,
                far,
            )

    view_count = timed_view_count(len(window), len(args.cameras))
    scope = (
        f"one forward over {len(window)} frames x {len(args.cameras)} cameras "
        f"({view_count} views), then the newest camera wide render"
        if args.multi
        else f"one forward over {view_count} current cameras, then the wide render"
    )
    print(
        f"Scene {scene} frame {window[-1]} window={','.join(window)} "
        f"views={view_count} warmup={args.warmup} measure={args.measure}",
        flush=True,
    )
    print(f"[1/2] Warmup ({args.warmup})...", flush=True)
    for _ in range(args.warmup):
        once()
    torch.cuda.synchronize()
    print(f"[2/2] Measure ({args.measure})...", flush=True)
    times_ms = []
    for _ in range(args.measure):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        once()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
    stats = summarize(times_ms)
    mode = "multiframes" if args.multi else "single"
    raster_w = int(round(pred.MODEL_SIZE * args.width_factor))
    save_w = datasets.save_width(
        base.source_hw(scene_dir, window[-1], args.render_camera),
        short_side=pred.MODEL_SIZE,
        width_factor=args.width_factor,
    )
    print(f"\n{'=' * 64}")
    print(f" Benchmark — {args.dataset} {mode}")
    print(f"{'=' * 64}")
    print(f"  Scene:        {scene}")
    print(f"  Frame:        {window[-1]}")
    print(f"  Window:       {','.join(window)}")
    print(f"  Timed region: {scope}")
    print(f"  Input:        1 x {view_count} x 3 x {pred.MODEL_SIZE} x {pred.MODEL_SIZE}")
    print(f"  Raster:       1 x 3 x {pred.MODEL_SIZE} x {raster_w}")
    print(f"  Saved later:  {save_w} x {pred.MODEL_SIZE} (not timed)")
    print(f"  Warmup:       {args.warmup}")
    print(f"  Measure:      {args.measure}")
    print(
        f"  Latency (ms): mean={stats['mean_ms']:8.2f}  median={stats['median_ms']:8.2f}  "
        f"min={stats['min_ms']:8.2f}  max={stats['max_ms']:8.2f}  stdev={stats['stdev_ms']:6.2f}"
    )
    print(f"  Throughput:   {stats['fps']:8.2f} FPS")
    print(f"{'=' * 64}")
    payload = {
        "dataset": args.dataset,
        "mode": mode,
        "scene": scene,
        "frame": window[-1],
        "window": list(window),
        "views": view_count,
        "timed_region": scope,
        "input_hw": [pred.MODEL_SIZE, pred.MODEL_SIZE],
        "raster_hw": [pred.MODEL_SIZE, raster_w],
        "save_hw": [pred.MODEL_SIZE, save_w],
        "warmup": args.warmup,
        "measure": args.measure,
        **stats,
    }
    out_path = args.output_json or (ROOT / "outputs" / "benchmarks" / f"{args.dataset}_{mode}.json")
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {out_path}")
    return payload


if __name__ == "__main__":
    main()
