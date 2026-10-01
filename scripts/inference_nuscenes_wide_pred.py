#!/usr/bin/env python3
"""Single-frame nuScenes wide-FOV inference without ground-truth cameras.

Rear cameras 5, 4 and 3 are stretched to the square training resolution
(224x224) and passed to the encoder with no nuScenes intrinsics or cam2ego
poses. Camera poses come from the camera head (``pose_free=true``). Focal
lengths come from the intrinsic head and are written into the ray embedding
(``use_pred_intrinsics_for_embed=true``). The head predicts only normalized
``fx`` and ``fy``; the principal point is not predicted and stays at 0.5.

The wide intrinsics are built from the render camera's predicted focal:

    fx_wide = fx_pred / width_factor
    fy_wide = fy_pred
    cx = cy = 0.5

The rasterizer therefore emits a 672x224 image. That image is then stretched
horizontally to 1176x224, the same canvas as a 392x224 input widened by 3,
and saved under the original name.

Outputs:

    <output>/<scene>/rgb/{frame}_{render_cam}_wide.jpg
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

# Predicted poses already live in the training normalized frame, so these are
# the dataset near/far before a ground-truth baseline rescaling.
PRED_NEAR = 0.1
PRED_FAR = 100.0
DEFAULT_OUTPUT = ROOT / "outputs/nuscenes_wide_pred"
MODEL_SIZE = 224
# 224 * 1600/900 snapped to the patch size, then widened by 3.
SAVE_WIDTH = 1176


def placeholder_intrinsics(view_count: int) -> np.ndarray:
    """Centered K. Predicted fx/fy replace the diagonal before the ray embedding."""
    if view_count < 1:
        raise ValueError("view_count must be positive.")
    matrix = np.array(
        [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return np.repeat(matrix[None], view_count, axis=0)


def wide_intrinsics_from_predicted_focal(
    fx: float,
    fy: float,
    context_hw: tuple[int, int],
    width_factor: float,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Widen a predicted normalized focal.

    ``fx`` and ``fy`` are the render camera's predictions in the same units as
    training: ``fx_pixels / width`` and ``fy_pixels / height``. The wide canvas
    keeps that pixel focal length, so the normalized ``fx`` shrinks by
    ``width_factor`` and ``fy`` does not. The unknown principal point stays at
    the center of both the context image and the wide image.
    """
    if fx <= 0 or fy <= 0:
        raise ValueError(f"Predicted focal must be positive, got fx={fx}, fy={fy}.")
    if width_factor <= 0:
        raise ValueError("width_factor must be positive.")
    height, width = int(context_hw[0]), int(context_hw[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid context shape {context_hw}.")
    wide_w = int(round(width * float(width_factor)))
    if wide_w < 1:
        raise ValueError("Wide width must be positive.")
    matrix = np.array(
        [
            [float(fx) / float(width_factor), 0.0, 0.5],
            [0.0, float(fy), 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return matrix, (height, wide_w)


def stretch_image(image: Image.Image, height: int, width: int) -> np.ndarray:
    """Resize without preserving aspect ratio. PIL size is (width, height)."""
    if height < 1 or width < 1:
        raise ValueError(f"Invalid stretch size {width}x{height}.")
    resized = image.convert("RGB").resize((width, height), Image.LANCZOS)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    if array.shape != (height, width, 3):
        raise RuntimeError(f"Stretched image shape {array.shape} != {(height, width, 3)}.")
    return array


def stretch_rgb(image_hwc: np.ndarray, height: int, width: int) -> np.ndarray:
    """Stretch a rendered RGB image. ``image_hwc`` is float in [0, 1]."""
    if image_hwc.ndim != 3 or image_hwc.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB, got {image_hwc.shape}.")
    array = np.clip(image_hwc, 0.0, 1.0)
    array = (array * 255.0).round().astype(np.uint8)
    resized = Image.fromarray(array).resize((width, height), Image.LANCZOS)
    return np.asarray(resized, dtype=np.float32) / 255.0


def stretch_mask(mask: Image.Image, height: int, width: int) -> np.ndarray:
    resized = mask.convert("L").resize((width, height), Image.NEAREST)
    array = np.asarray(resized)
    if array.shape != (height, width):
        raise RuntimeError(f"Stretched mask shape {array.shape} != {(height, width)}.")
    return array >= base.CAR_MASK_KEEP_THRESHOLD


def load_frame_images(
    scene_dir: Path,
    frame: str,
    cameras: list[int],
    size: int = MODEL_SIZE,
) -> np.ndarray:
    images = []
    for camera in cameras:
        path = scene_dir / "images" / f"{frame}_{camera}.jpg"
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            images.append(stretch_image(image, size, size))
    return np.stack(images, axis=0)


def load_stretched_keep_masks(
    mask_root: Path,
    cameras: list[int],
    size: int,
    render_index: int,
    mask_render_view: bool,
) -> np.ndarray:
    keep = np.ones((len(cameras), size, size), dtype=bool)
    for index, camera in enumerate(cameras):
        if index == render_index and not mask_render_view:
            continue
        path = base.camera_mask_path(mask_root, camera)
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing ego-car mask for camera {camera}: {path}. "
                "Pass --disable-car-mask to skip masking."
            )
        with Image.open(path) as mask:
            keep[index] = stretch_mask(mask, size, size)
    return keep


def build_model(checkpoint: Path, device: str):
    import torch
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from src.config import load_typed_root_config
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "config")):
        cfg = compose(
            config_name="main",
            overrides=[
                "+experiment=yono_dl3dv",
                "mode=test",
                "wandb.mode=disabled",
                "model.encoder.pose_free=true",
                "model.encoder.backbone.use_pred_intrinsics_for_embed=true",
                "model.encoder.backbone.predict_intrinsics=true",
                "model.encoder.use_checkpoint=false",
                "model.decoder.prune_opacity_threshold=0.005",
            ],
        )
    typed = load_typed_root_config(cfg)
    if not typed.model.encoder.pose_free:
        raise RuntimeError("pose_free must stay true so GT extrinsics are not used.")
    if not typed.model.encoder.backbone.use_pred_intrinsics_for_embed:
        raise RuntimeError("The ray embedding must use the predicted focal.")
    if not typed.model.encoder.backbone.predict_intrinsics:
        raise RuntimeError("The intrinsic head must be enabled.")
    encoder, _ = get_encoder(typed.model.encoder)
    decoder = get_decoder(typed.model.decoder)
    checkpoint_blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint_blob["state_dict"]
    state = {key[len("encoder.") :] if key.startswith("encoder.") else key: value for key, value in state.items()}
    encoder.load_state_dict(state, strict=True)
    encoder = encoder.to(device).eval()
    decoder = decoder.to(device).eval()
    return encoder, decoder


def render_wide(
    encoder,
    decoder,
    images: np.ndarray,
    keep: np.ndarray | None,
    views_to_mask: list[int],
    width_factor: float,
    device: str,
    render_index: int = 0,
):
    import torch

    from src.model.types import Gaussians

    view_count, height, width = images.shape[0], images.shape[1], images.shape[2]
    images_t = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()
    context = {
        "image": images_t[None].to(device),
        "intrinsics": torch.from_numpy(placeholder_intrinsics(view_count))[None].to(device),
    }
    dump = {}
    with torch.no_grad():
        gaussians = encoder(context, global_step=0, visualization_dump=dump)
    gaussian_count = gaussians.opacities.shape[-1]
    expected = view_count * height * width
    if gaussian_count != expected:
        raise RuntimeError(
            f"Expected one Gaussian per context pixel ({expected}), got {gaussian_count}."
        )
    focal = dump.get("intrinsic_pred")
    poses = dump.get("c2w")
    if focal is None or poses is None:
        raise RuntimeError("Encoder did not return a predicted focal and pose.")
    focal = focal.detach().float().cpu().numpy().reshape(view_count, 2)
    poses = poses.detach().float().cpu().numpy()
    if poses.shape[0] != 1:
        raise RuntimeError(f"Expected one batch of predicted poses, got {poses.shape}.")
    if not 0 <= int(render_index) < view_count:
        raise IndexError(f"render_index {render_index} is outside 0..{view_count - 1}.")
    fx, fy = (float(value) for value in focal[int(render_index)])
    wide_k, wide_hw = wide_intrinsics_from_predicted_focal(fx, fy, (height, width), width_factor)
    render_pose = poses[0, int(render_index)]
    identity_error = float(np.max(np.abs(render_pose - np.eye(4))))

    removed = 0
    opacities = gaussians.opacities
    if keep is not None and views_to_mask:
        indices = base.ego_remove_indices(keep, views_to_mask)
        removed = int(indices.size)
        if removed:
            opacities = opacities.clone()
            flat = opacities.view(-1)
            flat[torch.as_tensor(indices, device=flat.device, dtype=torch.long)] = 0
            gaussians = Gaussians(
                gaussians.means,
                gaussians.covariances,
                gaussians.harmonics,
                opacities,
                gaussians.rotations,
                gaussians.scales,
            )
    near = torch.full((1, 1), PRED_NEAR, dtype=torch.float32, device=device)
    far = torch.full((1, 1), PRED_FAR, dtype=torch.float32, device=device)
    with torch.no_grad():
        output = decoder(
            gaussians,
            torch.from_numpy(render_pose)[None, None].to(device),
            torch.from_numpy(wide_k)[None, None].to(device),
            near,
            far,
            wide_hw,
        )
    color = output.color[0, 0].clamp(0, 1).permute(1, 2, 0).contiguous().cpu().numpy()
    return color, removed, (fx, fy), wide_k, identity_error


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=base.DEFAULT_DATA_ROOT)
    parser.add_argument("--scene-list", type=Path, default=base.DEFAULT_SCENE_LIST)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--frame", default=None)
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--cameras", default="5,4,3")
    parser.add_argument("--render-camera", type=int, default=5)
    parser.add_argument("--width-factor", type=float, default=3.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=base.DEFAULT_CHECKPOINT)
    parser.add_argument("--car-mask-root", type=Path, default=base.DEFAULT_MASK_ROOT)
    parser.add_argument("--mask-render-view", action="store_true")
    parser.add_argument("--disable-car-mask", action="store_true")
    parser.add_argument("--save-inputs", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    args.cameras = base.parse_cameras(args.cameras)
    if args.render_camera not in args.cameras:
        raise SystemExit(
            f"--render-camera {args.render_camera} is not in --cameras {args.cameras}."
        )
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
    if abs(args.width_factor - 3.0) > 1e-6:
        raise SystemExit("This pipeline saves a 1176-wide image and requires --width-factor 3.")
    jobs = base.resolve_jobs(args)
    print(
        f"stretch input to {MODEL_SIZE}x{MODEL_SIZE}, render {MODEL_SIZE * 3}x{MODEL_SIZE}, "
        f"save {SAVE_WIDTH}x{MODEL_SIZE}, cameras {args.cameras}, jobs {len(jobs)}",
        file=sys.stderr,
    )
    keep = None
    if views_to_mask:
        keep = load_stretched_keep_masks(
            args.car_mask_root,
            args.cameras,
            MODEL_SIZE,
            render_index=0,
            mask_render_view=args.mask_render_view,
        )
    encoder = decoder = None
    if not args.dry_run:
        import torch

        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise SystemExit("CUDA is required for YoNoSplat inference.")
        torch.set_float32_matmul_precision("high")
        encoder, decoder = build_model(args.checkpoint, args.device)

    for scene, frame in jobs:
        output_path = args.output_dir / scene / "rgb" / f"{frame}_{args.render_camera}_wide.jpg"
        if args.dry_run:
            print(
                f"dry-run {scene}/{frame}: {MODEL_SIZE}x{MODEL_SIZE} -> "
                f"{SAVE_WIDTH}x{MODEL_SIZE} -> {output_path}",
                file=sys.stderr,
            )
            continue
        images = load_frame_images(args.data_root / scene, frame, args.cameras, MODEL_SIZE)
        color, removed, focal, wide_k, identity_error = render_wide(
            encoder,
            decoder,
            images,
            keep,
            views_to_mask,
            args.width_factor,
            args.device,
        )
        if color.shape[0] != MODEL_SIZE:
            raise RuntimeError(f"Wide render height {color.shape[0]} != {MODEL_SIZE}.")
        color = stretch_rgb(color, MODEL_SIZE, SAVE_WIDTH)
        base.save_jpg(output_path, color)
        if args.save_inputs:
            for index, camera in enumerate(args.cameras):
                base.save_jpg(
                    args.output_dir / scene / "inputs" / f"{frame}_{camera}.jpg",
                    images[index],
                )
        print(
            f"{scene}/{frame}: pred fx={focal[0]:.4f} fy={focal[1]:.4f} "
            f"wide fx={wide_k[0, 0]:.4f} fy={wide_k[1, 1]:.4f} "
            f"identity_err={identity_error:.2e} removed {removed} "
            f"saved {color.shape[1]}x{color.shape[0]} mean={float(color.mean()):.4f} -> {output_path}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
