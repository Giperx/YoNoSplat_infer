#!/usr/bin/env python3
"""Three-plane wide inference from the predicted main camera.

The encoder is the same image-only path as the single-pinhole wide script.
No dataset intrinsics or extrinsics are read. The saved image is not one
canvas with ``fx`` divided by 3. It is three co-centric planes: the predicted
main-camera pose, plus that same pose yawed left and right by its predicted
horizontal field of view. Each plane keeps the predicted focal. The planes are
stitched Left | Center | Right.

A 224x224 context renders three 224-wide planes. That 672-wide panorama is
then stretched to the same canvas as the single-pinhole wide image: three
times the 224-high aspect width. nuScenes, Lyft 1920, and WideDrive save
1176x224, Lyft 1224 saves 798x224, and DDAD saves 1050x224. ``--keep-aspect``
already renders each plane at that aspect width, so no second stretch is
applied. Ground truth uses the same canvas,
``sparseMultiplaneImages3_{width}x224``.

Outputs:

    outputs/<dataset>_wide_pred_multiplane/<scene>/rgb/{frame}_{camera}_wide.jpg
    outputs/<dataset>_wide_pred_multiframes_multiplane/<scene>/rgb/{newest}_{camera}_wide.jpg
"""

from __future__ import annotations

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


def build_arg_parser():
    parser = pred.build_arg_parser()
    parser.description = __doc__
    parser.add_argument("--multiframes", action="store_true")
    parser.add_argument("--num-frames", type=int, default=multi.DEFAULT_NUM_FRAMES)
    return parser


def multiplane_dir(path: Path) -> Path:
    return path.parent / f"{path.name}_multiplane"


def prepare(args):
    explicit_output = args.output_dir is not None
    spec = datasets.apply_defaults(args, multi=args.multiframes)
    if not explicit_output:
        args.output_dir = multiplane_dir(args.output_dir)
        if args.keep_aspect:
            args.output_dir = pred.aspect_output_dir(args.output_dir)
    args.cameras = base.parse_cameras(args.cameras)
    if args.render_camera not in args.cameras:
        raise SystemExit(
            f"--render-camera {args.render_camera} is not in --cameras {args.cameras}."
        )
    if abs(args.width_factor - 3.0) > 1e-6:
        raise SystemExit("Three planes require --width-factor 3.")
    return spec


def load_model(args):
    if args.dry_run:
        return None, None
    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is required for YoNoSplat inference.")
    torch.set_float32_matmul_precision("high")
    return pred.build_model(args.checkpoint, args.device)


def saved_canvas(color, src_hw, keep_aspect):
    """Stretch the stitched planes onto the single-pinhole wide canvas."""
    save_w = datasets.save_width(src_hw, short_side=pred.MODEL_SIZE, width_factor=3.0)
    return pred.finish_wide_color(color, save_w, keep_aspect), save_w


def save_result(args, scene, frame, color, removed, focal, identity_error, fov, output_path):
    base.save_jpg(output_path, color)
    print(
        f"{scene}/{frame}: pred fx={focal[0]:.4f} fy={focal[1]:.4f} "
        f"plane_fov={np.degrees(fov):.2f}deg "
        f"pose_err={identity_error:.2e} removed {removed} "
        f"saved {color.shape[1]}x{color.shape[0]} mean={float(color.mean()):.4f} -> {output_path}",
        file=sys.stderr,
    )


def run_single(args, spec) -> int:
    args.cameras = [args.render_camera] + [
        camera for camera in args.cameras if camera != args.render_camera
    ]
    views_to_mask = []
    if not args.disable_car_mask:
        views_to_mask = [
            index
            for index, camera in enumerate(args.cameras)
            if args.mask_render_view or camera != args.render_camera
        ]
    jobs = base.resolve_jobs(args)
    input_mode = "aspect height 224" if args.keep_aspect else f"stretch {pred.MODEL_SIZE}x{pred.MODEL_SIZE}"
    print(
        f"{spec.name}: multiplane {input_mode}, planes Left|Center|Right, "
        f"mask={spec.mask_kind}, cameras {args.cameras}, jobs {len(jobs)}, "
        f"output {args.output_dir}",
        file=sys.stderr,
    )
    keep_cache: dict[tuple, np.ndarray] = {}

    def keep_for(scene: str, context: tuple[int, int]):
        if not views_to_mask:
            return None
        scene_key = scene if spec.mask_kind == "ddad" else ""
        key = (scene_key, context, args.keep_aspect)
        if key not in keep_cache:
            keep_cache[key] = pred.load_dataset_keep(
                spec,
                scene,
                args.cameras,
                context,
                render_index=0,
                mask_render_view=args.mask_render_view,
                keep_aspect=args.keep_aspect,
            )
        return keep_cache[key]

    encoder, decoder = load_model(args)
    for scene, frame in jobs:
        scene_dir = args.data_root / scene
        src_hw = base.source_hw(scene_dir, frame, args.render_camera)
        context = pred.context_hw(src_hw, args.keep_aspect)
        save_w = datasets.save_width(src_hw, short_side=pred.MODEL_SIZE, width_factor=3.0)
        output_path = args.output_dir / scene / "rgb" / f"{frame}_{args.render_camera}_wide.jpg"
        if args.dry_run:
            print(
                f"dry-run {scene}/{frame}: plane {context[1]}x{context[0]} -> "
                f"{save_w}x{pred.MODEL_SIZE} -> {output_path}",
                file=sys.stderr,
            )
            continue
        images = pred.load_frame_images(scene_dir, frame, args.cameras, context, args.keep_aspect)
        color, removed, focal, identity_error, fov = pred.render_multiplane(
            encoder, decoder, images, keep_for(scene, context), views_to_mask, args.device,
        )
        color, _ = saved_canvas(color, src_hw, args.keep_aspect)
        save_result(args, scene, frame, color, removed, focal, identity_error, fov, output_path)
    return 0


def run_multi(args, spec) -> int:
    if args.num_frames < 1:
        raise SystemExit("--num-frames must be >= 1.")
    render_index = multi.newest_render_index(args.num_frames, args.cameras, args.render_camera)
    masked = [] if args.disable_car_mask else multi.views_to_mask(
        args.num_frames, args.cameras, render_index, args.mask_render_view
    )
    jobs = multi.resolve_jobs(args)
    input_mode = "aspect height 224" if args.keep_aspect else f"stretch {pred.MODEL_SIZE}x{pred.MODEL_SIZE}"
    print(
        f"{spec.name}: multiplane windows {args.num_frames} x cameras {args.cameras}, "
        f"{input_mode}, render view {render_index}, "
        f"mask={spec.mask_kind} ({len(masked)} views), jobs {len(jobs)}, "
        f"output {args.output_dir}",
        file=sys.stderr,
    )
    keep_cache: dict[tuple, np.ndarray] = {}

    def keep_for(scene: str, context: tuple[int, int]):
        if not masked:
            return None
        scene_key = scene if spec.mask_kind == "ddad" else ""
        key = (scene_key, context, args.keep_aspect)
        if key not in keep_cache:
            camera_keeps = multi.load_camera_keep_masks(
                spec, scene, args.cameras, context, args.keep_aspect
            )
            keep_cache[key] = multi.build_window_keep(
                camera_keeps, args.cameras, args.num_frames, render_index, args.mask_render_view
            )
        return keep_cache[key]

    encoder, decoder = load_model(args)
    for scene, window in jobs:
        newest = window[-1]
        scene_dir = args.data_root / scene
        src_hw = base.source_hw(scene_dir, newest, args.render_camera)
        context = pred.context_hw(src_hw, args.keep_aspect)
        save_w = datasets.save_width(src_hw, short_side=pred.MODEL_SIZE, width_factor=3.0)
        output_path = args.output_dir / scene / "rgb" / f"{newest}_{args.render_camera}_wide.jpg"
        if args.dry_run:
            print(
                f"dry-run {scene}/{'-'.join(window)}: plane {context[1]}x{context[0]} -> "
                f"{save_w}x{pred.MODEL_SIZE} {output_path}",
                file=sys.stderr,
            )
            continue
        images = multi.load_window_images(scene_dir, window, args.cameras, context, args.keep_aspect)
        color, removed, focal, identity_error, fov = pred.render_multiplane(
            encoder,
            decoder,
            images,
            keep_for(scene, context),
            masked,
            args.device,
            render_index=render_index,
        )
        color, _ = saved_canvas(color, src_hw, args.keep_aspect)
        save_result(args, scene, "-".join(window), color, removed, focal, identity_error, fov, output_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    spec = prepare(args)
    if args.multiframes:
        return run_multi(args, spec)
    return run_single(args, spec)


if __name__ == "__main__":
    raise SystemExit(main())
