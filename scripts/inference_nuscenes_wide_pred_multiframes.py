#!/usr/bin/env python3
"""Multi-frame wide-FOV inference without ground-truth cameras.

``--dataset`` selects nuScenes, Lyft 1920, Lyft 1224, DDAD, or WideDrive.
A window is three consecutive frames by default, ordered oldest to newest.
Each frame contributes cameras 5, 4 and 3. By default they are stretched to
224x224. ``--keep-aspect`` scales them uniformly to height 224 instead. No
dataset intrinsics or extrinsics are read. Poses and focal lengths are predicted.

Only the newest frame's camera 5 is rendered. Ego-car Gaussians are removed
everywhere except that view: every historical image, plus the current frame's
cameras 4 and 3. The current camera 5 is kept whole. WideDrive has no ego-car
mask, so every Gaussian stays valid.

By default the rasterizer emits 672x224 and that image is stretched to three
times the 224-high aspect width (1176 for nuScenes and Lyft 1920).
``--keep-aspect`` emits that saved canvas directly. Results are saved as:

    <output>/<scene>/rgb/{frame}_{render_cam}_wide.jpg

The camera head rebases predicted poses onto view 0, which is the oldest
frame's first camera, not the render camera. The wide view therefore uses the
predicted pose and focal length of the newest camera 5.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import inference_nuscenes_wide as base
import inference_nuscenes_wide_pred as pred
import wide_datasets as datasets

DEFAULT_OUTPUT = ROOT / "outputs/nuscenes_wide_pred_multiframes"
DEFAULT_NUM_FRAMES = 3


def enumerate_windows(frames: list[str], num_frames: int) -> list[tuple[str, ...]]:
    """Causal windows, oldest to newest. The last id is the rendered frame."""
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}.")
    if len(frames) < num_frames:
        return []
    return [
        tuple(frames[start : start + num_frames])
        for start in range(len(frames) - num_frames + 1)
    ]


def select_windows(
    frames: list[str], num_frames: int, frame: str | None
) -> list[tuple[str, ...]]:
    windows = enumerate_windows(frames, num_frames)
    if frame is None:
        return windows
    wanted = base.normalize_frame_id(frame)
    return [window for window in windows if window[-1] == wanted]


def newest_render_index(num_frames: int, cameras: list[int], render_camera: int) -> int:
    if render_camera not in cameras:
        raise ValueError(f"render_camera {render_camera} is not in {cameras}.")
    return (int(num_frames) - 1) * len(cameras) + cameras.index(render_camera)


def views_to_mask(
    num_frames: int,
    cameras: list[int],
    render_index: int,
    mask_render_view: bool,
) -> list[int]:
    """Every flattened view except the newest render camera.

    With cameras [5, 4, 3] and 3 frames, index 6 is the newest camera 5 and is
    kept. Indices 7 and 8 are the current cameras 4 and 3. Indices 0..5 are
    the two historical frames.
    """
    view_count = int(num_frames) * len(cameras)
    if not 0 <= render_index < view_count:
        raise IndexError(f"render_index {render_index} is outside 0..{view_count - 1}.")
    if render_index // len(cameras) != num_frames - 1:
        raise ValueError("render_index must address the newest frame.")
    masked = list(range(view_count))
    if not mask_render_view:
        masked.remove(render_index)
    return masked


def load_camera_keep_masks(
    spec: datasets.DatasetSpec,
    scene: str,
    cameras: list[int],
    context: tuple[int, int],
    keep_aspect: bool,
) -> dict[int, np.ndarray]:
    keeps = {}
    for camera in cameras:
        path = datasets.ego_mask_path(spec, camera, scene)
        if path is None or not path.is_file():
            raise FileNotFoundError(
                f"Missing ego-car mask for camera {camera}: {path}. "
                "Pass --disable-car-mask to skip masking."
            )
        with Image.open(path) as mask:
            keeps[camera] = pred.fit_context_mask(mask, context, keep_aspect)
    return keeps


def build_window_keep(
    camera_keeps: dict[int, np.ndarray],
    cameras: list[int],
    num_frames: int,
    render_index: int,
    mask_render_view: bool,
) -> np.ndarray:
    sample = camera_keeps[cameras[0]]
    keep = np.ones((num_frames * len(cameras), sample.shape[0], sample.shape[1]), dtype=bool)
    for view_index in range(keep.shape[0]):
        if view_index == render_index and not mask_render_view:
            continue
        keep[view_index] = camera_keeps[cameras[view_index % len(cameras)]]
    return keep


def load_window_images(
    scene_dir: Path,
    window: tuple[str, ...],
    cameras: list[int],
    context: tuple[int, int],
    keep_aspect: bool,
) -> np.ndarray:
    images = []
    for frame in window:
        for camera in cameras:
            path = scene_dir / "images" / f"{frame}_{camera}.jpg"
            if not path.is_file():
                raise FileNotFoundError(path)
            with Image.open(path) as image:
                images.append(pred.fit_context_image(image, context, keep_aspect))
    return np.stack(images, axis=0)


def resolve_jobs(args) -> list[tuple[str, tuple[str, ...]]]:
    if args.scene:
        scenes = [args.scene]
    else:
        if not args.scene_list.is_file():
            raise FileNotFoundError(args.scene_list)
        scenes = base.read_scene_list(args.scene_list)
        if not scenes:
            raise RuntimeError(f"No scenes in {args.scene_list}.")
    jobs = []
    for scene in scenes:
        scene_dir = args.data_root / scene
        if not scene_dir.is_dir():
            message = f"Missing scene directory: {scene_dir}"
            if args.scene:
                raise FileNotFoundError(message)
            print(f"warning: {message}", file=sys.stderr)
            continue
        frames = base.enumerate_frames(scene_dir, args.cameras)
        if args.frame is None and args.max_frames >= 0:
            frames = frames[: args.max_frames]
        windows = select_windows(frames, args.num_frames, args.frame)
        if not windows:
            target = base.normalize_frame_id(args.frame) if args.frame is not None else "any"
            message = (
                f"No {args.num_frames}-frame window for scene {scene} ending at {target}."
            )
            if args.scene or args.frame is not None:
                raise RuntimeError(message)
            print(f"warning: {message}", file=sys.stderr)
            continue
        jobs.extend((scene, window) for window in windows)
    if not jobs:
        raise RuntimeError("No inference jobs.")
    return jobs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="nuscenes", choices=sorted(datasets.DATASETS))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--scene-list", type=Path, default=None)
    parser.add_argument("--scene", default=None)
    parser.add_argument(
        "--frame",
        default=None,
        help="Render the window whose newest frame is this id (0 or 000).",
    )
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--cameras", default="5,4,3")
    parser.add_argument("--render-camera", type=int, default=5)
    parser.add_argument("--width-factor", type=float, default=3.0)
    parser.add_argument(
        "--keep-aspect",
        action="store_true",
        help=(
            "Scale context views uniformly to height 224 instead of stretching "
            "them to 224x224. The 3x wide render is already the saved size. "
            "Default output is outputs/<dataset>_wide_pred_multiframes_aspect."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=base.DEFAULT_CHECKPOINT)
    parser.add_argument("--car-mask-root", type=Path, default=None)
    parser.add_argument("--mask-render-view", action="store_true")
    parser.add_argument("--disable-car-mask", action="store_true")
    parser.add_argument("--save-inputs", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    explicit_output = args.output_dir is not None
    spec = datasets.apply_defaults(args, multi=True)
    if args.keep_aspect and not explicit_output:
        args.output_dir = pred.aspect_output_dir(args.output_dir)
    args.cameras = base.parse_cameras(args.cameras)
    if args.render_camera not in args.cameras:
        raise SystemExit(
            f"--render-camera {args.render_camera} is not in --cameras {args.cameras}."
        )
    if args.num_frames < 1:
        raise SystemExit("--num-frames must be >= 1.")
    if abs(args.width_factor - 3.0) > 1e-6:
        raise SystemExit("This pipeline widens by 3 and requires --width-factor 3.")
    render_index = newest_render_index(args.num_frames, args.cameras, args.render_camera)
    masked = [] if args.disable_car_mask else views_to_mask(
        args.num_frames, args.cameras, render_index, args.mask_render_view
    )
    jobs = resolve_jobs(args)
    input_mode = (
        "aspect height 224" if args.keep_aspect else f"stretch {pred.MODEL_SIZE}x{pred.MODEL_SIZE}"
    )
    print(
        f"{spec.name}: windows {args.num_frames} x cameras {args.cameras}, "
        f"{input_mode}, render view {render_index}, "
        f"mask={spec.mask_kind} ({len(masked)} views), jobs {len(jobs)}, "
        f"output {args.output_dir}",
        file=sys.stderr,
    )
    keep_cache: dict[tuple, np.ndarray] = {}

    def keep_for(scene: str, context: tuple[int, int]) -> np.ndarray | None:
        if not masked:
            return None
        scene_key = scene if spec.mask_kind == "ddad" else ""
        key = (scene_key, context, args.keep_aspect)
        if key not in keep_cache:
            camera_keeps = load_camera_keep_masks(
                spec, scene, args.cameras, context, args.keep_aspect
            )
            keep_cache[key] = build_window_keep(
                camera_keeps,
                args.cameras,
                args.num_frames,
                render_index,
                args.mask_render_view,
            )
        return keep_cache[key]
    encoder = decoder = None
    if not args.dry_run:
        import torch

        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise SystemExit("CUDA is required for YoNoSplat inference.")
        torch.set_float32_matmul_precision("high")
        encoder, decoder = pred.build_model(args.checkpoint, args.device)

    for scene, window in jobs:
        newest = window[-1]
        scene_dir = args.data_root / scene
        src_hw = base.source_hw(scene_dir, newest, args.render_camera)
        context = pred.context_hw(src_hw, args.keep_aspect)
        save_w = datasets.save_width(
            src_hw,
            short_side=pred.MODEL_SIZE,
            width_factor=args.width_factor,
        )
        output_path = args.output_dir / scene / "rgb" / f"{newest}_{args.render_camera}_wide.jpg"
        if args.dry_run:
            print(
                f"dry-run {scene}/{'-'.join(window)}: context {context[1]}x{context[0]} -> "
                f"{save_w}x{pred.MODEL_SIZE} {output_path}",
                file=sys.stderr,
            )
            continue
        keep = keep_for(scene, context)
        images = load_window_images(
            scene_dir, window, args.cameras, context, args.keep_aspect
        )
        color, removed, focal, wide_k, identity_error = pred.render_wide(
            encoder,
            decoder,
            images,
            keep,
            masked,
            args.width_factor,
            args.device,
            render_index=render_index,
        )
        color = pred.finish_wide_color(color, save_w, args.keep_aspect)
        base.save_jpg(output_path, color)
        if args.save_inputs:
            view_index = 0
            for frame in window:
                for camera in args.cameras:
                    base.save_jpg(
                        args.output_dir / scene / "inputs" / f"{frame}_{camera}.jpg",
                        images[view_index],
                    )
                    view_index += 1
        print(
            f"{scene}/{'-'.join(window)}: pred fx={focal[0]:.4f} fy={focal[1]:.4f} "
            f"wide fx={wide_k[0, 0]:.4f} fy={wide_k[1, 1]:.4f} "
            f"pose_err={identity_error:.2e} removed {removed} "
            f"saved {color.shape[1]}x{color.shape[0]} mean={float(color.mean()):.4f} -> {output_path}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
