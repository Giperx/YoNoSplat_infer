#!/usr/bin/env python3
"""Single-frame wide-FOV inference without ground-truth cameras.

``--dataset`` selects nuScenes, Lyft 1920, Lyft 1224, DDAD, or WideDrive.
Rear cameras 5, 4 and 3 are stretched to 224x224 and passed to the encoder
with no dataset intrinsics or extrinsics. Camera poses come from the camera
head. Focal lengths come from the intrinsic head and condition the ray
embedding. The head predicts only normalized ``fx`` and ``fy``; the principal
point stays at 0.5.

The wide intrinsics divide the render camera's predicted ``fx`` by 3 and keep
``fy``. The rasterizer emits 672x224. That image is then stretched to three
times the width of a 224-high, aspect-aligned frame: 1176 for nuScenes, Lyft
1920, and WideDrive (rear cameras are 1920x1080), 798 for Lyft 1224, and 1050
for DDAD. WideDrive camera 2 is a 5760-wide image and is not used.

Ego-car Gaussians are removed on cameras 4 and 3. WideDrive has no ego-car
mask, so every Gaussian stays valid. The current camera 5 is kept whole.

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
import wide_datasets as datasets

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


def load_dataset_keep(
    spec: datasets.DatasetSpec,
    scene: str,
    cameras: list[int],
    size: int,
    render_index: int,
    mask_render_view: bool,
) -> np.ndarray:
    """Square keep mask. The render view stays fully valid unless requested."""
    keep = np.ones((len(cameras), size, size), dtype=bool)
    for index, camera in enumerate(cameras):
        if index == render_index and not mask_render_view:
            continue
        path = datasets.ego_mask_path(spec, camera, scene)
        if path is None or not path.is_file():
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


def forward_wide_gpu(
    encoder,
    decoder,
    images,
    intrinsics,
    remove_index,
    render_index: int,
    width_factor: float,
    near,
    far,
):
    """Encode every input view and render the chosen camera. Tensors stay on device.

    ``images`` is ``(1, V, 3, H, W)``. ``remove_index`` is a 1-D long tensor of
    Gaussian indices to make transparent, or ``None``. This is the timed region
    for both one frame and a multi-frame window: the window is one forward.
    """
    import torch

    from src.model.types import Gaussians

    view_count = int(images.shape[1])
    height, width = int(images.shape[-2]), int(images.shape[-1])
    if not 0 <= int(render_index) < view_count:
        raise IndexError(f"render_index {render_index} is outside 0..{view_count - 1}.")
    dump = {}
    gaussians = encoder(
        {"image": images, "intrinsics": intrinsics},
        global_step=0,
        visualization_dump=dump,
    )
    expected = view_count * height * width
    if int(gaussians.opacities.shape[-1]) != expected:
        raise RuntimeError(
            f"Expected one Gaussian per context pixel ({expected}), got {int(gaussians.opacities.shape[-1])}."
        )
    focal = dump.get("intrinsic_pred")
    poses = dump.get("c2w")
    if focal is None or poses is None:
        raise RuntimeError("Encoder did not return a predicted focal and pose.")
    focal = focal.detach().float().reshape(view_count, 2)
    poses = poses.detach().float()
    if poses.shape[0] != 1:
        raise RuntimeError(f"Expected one batch of predicted poses, got {tuple(poses.shape)}.")
    fx = focal[int(render_index), 0]
    fy = focal[int(render_index), 1]
    pose = poses[0, int(render_index)]
    wide_k = torch.zeros((1, 1, 3, 3), dtype=torch.float32, device=images.device)
    wide_k[0, 0, 0, 0] = fx / float(width_factor)
    wide_k[0, 0, 1, 1] = fy
    wide_k[0, 0, 0, 2] = 0.5
    wide_k[0, 0, 1, 2] = 0.5
    wide_k[0, 0, 2, 2] = 1.0
    removed = 0
    opacities = gaussians.opacities
    if remove_index is not None and int(remove_index.numel()) > 0:
        removed = int(remove_index.numel())
        opacities = opacities.clone()
        opacities.view(-1)[remove_index] = 0
        gaussians = Gaussians(
            gaussians.means,
            gaussians.covariances,
            gaussians.harmonics,
            opacities,
            gaussians.rotations,
            gaussians.scales,
        )
    output = decoder(
        gaussians,
        pose[None, None],
        wide_k,
        near,
        far,
        (height, int(round(width * float(width_factor)))),
    )
    return output, fx, fy, wide_k[0, 0], pose, removed


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

    view_count = images.shape[0]
    images_t = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).contiguous()
    intrinsics = torch.from_numpy(placeholder_intrinsics(view_count))[None].to(device)
    remove_index = None
    if keep is not None and views_to_mask:
        indices = base.ego_remove_indices(keep, views_to_mask)
        if indices.size:
            remove_index = torch.as_tensor(indices, device=device, dtype=torch.long)
    near = torch.full((1, 1), PRED_NEAR, dtype=torch.float32, device=device)
    far = torch.full((1, 1), PRED_FAR, dtype=torch.float32, device=device)
    with torch.no_grad():
        output, fx, fy, wide_k, pose, removed = forward_wide_gpu(
            encoder,
            decoder,
            images_t[None].to(device),
            intrinsics,
            remove_index,
            render_index,
            width_factor,
            near,
            far,
        )
    color = output.color[0, 0].clamp(0, 1).permute(1, 2, 0).contiguous().cpu().numpy()
    pose_np = pose.detach().float().cpu().numpy()
    identity_error = float(np.max(np.abs(pose_np - np.eye(4, dtype=np.float32))))
    return color, removed, (float(fx), float(fy)), wide_k.detach().float().cpu().numpy(), identity_error


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="nuscenes", choices=sorted(datasets.DATASETS))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--scene-list", type=Path, default=None)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--frame", default=None)
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--cameras", default="5,4,3")
    parser.add_argument("--render-camera", type=int, default=5)
    parser.add_argument("--width-factor", type=float, default=3.0)
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
    spec = datasets.apply_defaults(args, multi=False)
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
        raise SystemExit("This pipeline widens by 3 and requires --width-factor 3.")
    jobs = base.resolve_jobs(args)
    print(
        f"{spec.name}: stretch input to {MODEL_SIZE}x{MODEL_SIZE}, "
        f"render {MODEL_SIZE * 3}x{MODEL_SIZE}, save 3x the 224-high aspect width, "
        f"mask={spec.mask_kind}, cameras {args.cameras}, jobs {len(jobs)}",
        file=sys.stderr,
    )
    keep_cache: dict[str | None, np.ndarray] = {}

    def keep_for(scene: str) -> np.ndarray | None:
        if not views_to_mask:
            return None
        key = scene if spec.mask_kind == "ddad" else None
        if key not in keep_cache:
            keep_cache[key] = load_dataset_keep(
                spec,
                scene,
                args.cameras,
                MODEL_SIZE,
                render_index=0,
                mask_render_view=args.mask_render_view,
            )
        return keep_cache[key]
    encoder = decoder = None
    if not args.dry_run:
        import torch

        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise SystemExit("CUDA is required for YoNoSplat inference.")
        torch.set_float32_matmul_precision("high")
        encoder, decoder = build_model(args.checkpoint, args.device)

    for scene, frame in jobs:
        scene_dir = args.data_root / scene
        save_w = datasets.save_width(
            base.source_hw(scene_dir, frame, args.render_camera),
            short_side=MODEL_SIZE,
            width_factor=args.width_factor,
        )
        output_path = args.output_dir / scene / "rgb" / f"{frame}_{args.render_camera}_wide.jpg"
        if args.dry_run:
            print(
                f"dry-run {scene}/{frame}: {MODEL_SIZE}x{MODEL_SIZE} -> "
                f"{save_w}x{MODEL_SIZE} -> {output_path}",
                file=sys.stderr,
            )
            continue
        keep = keep_for(scene)
        images = load_frame_images(scene_dir, frame, args.cameras, MODEL_SIZE)
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
        color = stretch_rgb(color, MODEL_SIZE, save_w)
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
