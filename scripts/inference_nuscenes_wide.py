#!/usr/bin/env python3
"""Single-frame nuScenes wide-FOV inference for YoNoSplat.

Each call consumes one timestamp and the three rear cameras:

    5  CAM_BACK        main / render camera
    4  CAM_BACK_RIGHT  side
    3  CAM_BACK_LEFT   side

``cam2ego_extrinsics`` are OpenCV camera-to-ego matrices. For a single frame the
ego frame is the world, so they are used directly as camera-to-world, then
normalized the same way training does (max pairwise baseline, first camera
becomes the identity). Ground-truth intrinsics are injected as well
(``pose_free=false``, predicted intrinsics are not used as the condition).

After the scene Gaussians are predicted, ego-car Gaussians of cameras 4 and 3
are set fully transparent. The main camera is then re-rendered at the same
height and ``width_factor`` times the width (default 3), which widens the
horizontal field of view while keeping the pixel focal length.

The shipped checkpoints are 224x224 (patch size 14). The short side is kept at
that training length and the long side is scaled with the same factor, then
snapped to a multiple of 14. A uniform resize plus a centre crop absorbs the
snap so the aspect ratio is not stretched.

Outputs follow the DepthSplat nuScenes wide script:

    <output>/<scene>/rgb/{frame}_{render_cam}_wide.jpg
    <output>/<scene>/inputs/{frame}_{cam}.jpg     (only with --save-inputs)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PATCH_SIZE = 14
TRAIN_SHORT_SIDE = 224
NEAR = 0.1
FAR = 100.0
CAR_MASK_KEEP_THRESHOLD = 128
JPEG_QUALITY = 95

# nuScenes camera id -> ego-car mask filename. Black (<128) is the ego car.
NUSCENES_MASK_NAME = {
    0: "CAM_FRONT_mask.png",
    1: "CAM_FRONT_RIGHT_mask.png",
    2: "CAM_FRONT_LEFT_mask.png",
    3: "CAM_BACK_LEFT_mask.png",
    4: "CAM_BACK_RIGHT_mask.png",
    5: "CAM_BACK_mask.png",
}

DEFAULT_DATA_ROOT = ROOT / "datasets/nuscenes/processed_10Hz/trainval2"
DEFAULT_SCENE_LIST = DEFAULT_DATA_ROOT / "nuScenes_Val2.txt"
DEFAULT_MASK_ROOT = ROOT / "datasets/nuscenes/nuscenes_mask"
DEFAULT_CHECKPOINT = ROOT / "pretrained_weights/dl3dv_224x224_ctx2to32.ckpt"
DEFAULT_OUTPUT = ROOT / "outputs/nuscenes_wide"


@dataclass(frozen=True)
class PixelIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class ResizeCropPlan:
    src_h: int
    src_w: int
    scaled_h: int
    scaled_w: int
    out_h: int
    out_w: int
    row: int
    col: int


def snap_to_patch(value: float, patch: int = PATCH_SIZE) -> int:
    snapped = int(round(float(value) / patch)) * patch
    return max(patch, snapped)


def choose_input_hw(
    src_hw: tuple[int, int],
    short_side: int = TRAIN_SHORT_SIDE,
    patch: int = PATCH_SIZE,
) -> tuple[int, int]:
    """Match the training short side, scale the long side, snap both to ``patch``.

    Returns ``(height, width)``.
    """
    src_h, src_w = int(src_hw[0]), int(src_hw[1])
    if src_h <= 0 or src_w <= 0:
        raise ValueError(f"Invalid source shape {src_hw}.")
    short_side = snap_to_patch(short_side, patch)
    if src_w >= src_h:
        height = short_side
        width = snap_to_patch(short_side * src_w / src_h, patch)
    else:
        width = short_side
        height = snap_to_patch(short_side * src_h / src_w, patch)
    return height, width


def plan_resize_and_crop(
    src_hw: tuple[int, int], dst_hw: tuple[int, int]
) -> ResizeCropPlan:
    """Uniform scale that covers ``dst_hw``, then a centre crop."""
    src_h, src_w = int(src_hw[0]), int(src_hw[1])
    out_h, out_w = int(dst_hw[0]), int(dst_hw[1])
    if min(src_h, src_w, out_h, out_w) <= 0:
        raise ValueError(f"Invalid resize {src_hw} -> {dst_hw}.")
    scale = max(out_h / src_h, out_w / src_w)
    scaled_h = max(out_h, int(round(src_h * scale)))
    scaled_w = max(out_w, int(round(src_w * scale)))
    row = (scaled_h - out_h) // 2
    col = (scaled_w - out_w) // 2
    return ResizeCropPlan(src_h, src_w, scaled_h, scaled_w, out_h, out_w, row, col)


def resize_and_crop_intrinsics(
    intrinsics: PixelIntrinsics, plan: ResizeCropPlan
) -> PixelIntrinsics:
    scale_x = plan.scaled_w / plan.src_w
    scale_y = plan.scaled_h / plan.src_h
    return PixelIntrinsics(
        fx=intrinsics.fx * scale_x,
        fy=intrinsics.fy * scale_y,
        cx=intrinsics.cx * scale_x - plan.col,
        cy=intrinsics.cy * scale_y - plan.row,
    )


def make_wide_intrinsics(
    intrinsics: PixelIntrinsics,
    context_hw: tuple[int, int],
    width_factor: float,
) -> tuple[PixelIntrinsics, tuple[int, int]]:
    """Keep pixel fx/fy/cy and centre the principal point on the wider canvas.

    The CUDA rasterizer derives a symmetric FOV from the normalized intrinsics,
    so a 3x canvas with the same pixel focal length is a wider horizontal FOV
    at the same angular resolution. ``cy`` stays in pixels; height is unchanged.
    """
    if width_factor <= 0:
        raise ValueError("width_factor must be positive.")
    context_h, context_w = int(context_hw[0]), int(context_hw[1])
    wide_w = int(round(context_w * float(width_factor)))
    if wide_w < 1:
        raise ValueError("Wide width must be positive.")
    return (
        PixelIntrinsics(
            fx=intrinsics.fx,
            fy=intrinsics.fy,
            cx=wide_w / 2.0,
            cy=intrinsics.cy,
        ),
        (context_h, wide_w),
    )


def pixel_to_normalized(
    intrinsics: PixelIntrinsics, height: int, width: int
) -> np.ndarray:
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive.")
    return np.array(
        [
            [intrinsics.fx / width, 0.0, intrinsics.cx / width],
            [0.0, intrinsics.fy / height, intrinsics.cy / height],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def max_pairwise_distance(c2w: np.ndarray) -> float:
    """Max camera-center distance. Matches ``compute_pose_norm_scale(..., 'max_pairwise_d')``."""
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
        raise ValueError(f"Expected (V, 4, 4) poses, got {c2w.shape}.")
    scale = 0.0
    view_count = c2w.shape[0]
    for i in range(view_count):
        for j in range(i + 1, view_count):
            scale = max(scale, float(np.linalg.norm(c2w[i, :3, 3] - c2w[j, :3, 3])))
    return scale


def normalize_poses(c2w: np.ndarray) -> tuple[np.ndarray, float]:
    """Scale translations by the max baseline and make camera 0 the identity.

    This is the training ``relative_pose`` + ``max_pairwise_d`` normalization.
    Camera 0 must be the render camera so the wide view is the identity pose.
    """
    scale = max_pairwise_distance(c2w)
    if scale < 1e-8:
        raise ValueError("Camera baseline is zero; cannot normalize poses.")
    out = np.array(c2w, dtype=np.float64, copy=True)
    out[:, :3, 3] /= scale
    pivot_inv = np.linalg.inv(out[0])
    out = np.einsum("ij,vjk->vik", pivot_inv, out)
    return out.astype(np.float32), float(scale)


def ego_remove_indices(
    keep: np.ndarray, views_to_mask: list[int]
) -> np.ndarray:
    """Flat Gaussian indices to make transparent.

    ``keep`` is (V, H, W), True where the Gaussian stays. Gaussians are one per
    context pixel, stored view-major then row-major.
    """
    if keep.ndim != 3:
        raise ValueError(f"Expected keep mask (V, H, W), got {keep.shape}.")
    view_count, height, width = keep.shape
    indices = []
    for view in views_to_mask:
        if view < 0 or view >= view_count:
            raise IndexError(f"View {view} is outside 0..{view_count - 1}.")
        ys, xs = np.where(~keep[view])
        indices.append(view * height * width + ys * width + xs)
    if not indices:
        return np.zeros((0,), dtype=np.int64)
    return np.concatenate(indices).astype(np.int64, copy=False)


def read_scene_list(path: Path) -> list[str]:
    scenes = []
    for line in path.read_text().splitlines():
        scene = line.strip()
        if scene and not scene.startswith("#"):
            scenes.append(scene)
    return scenes


def normalize_frame_id(frame: str, width: int = 3) -> str:
    text = str(frame).strip()
    if text.isdigit():
        return f"{int(text):0{width}d}"
    return text


def parse_cameras(text: str) -> list[int]:
    cameras = [int(part) for part in text.split(",") if part.strip()]
    if not cameras:
        raise ValueError("At least one camera is required.")
    return cameras


def camera_mask_path(mask_root: Path, camera: int) -> Path:
    try:
        name = NUSCENES_MASK_NAME[camera]
    except KeyError as exc:
        known = ", ".join(str(cam) for cam in sorted(NUSCENES_MASK_NAME))
        raise KeyError(f"No ego-car mask for camera {camera}. Known ids: {known}.") from exc
    return mask_root / name


def load_floats(path: Path) -> np.ndarray:
    values = []
    for line in path.read_text().splitlines():
        for token in line.split():
            if token:
                values.append(float(token))
    if not values:
        raise ValueError(f"No numeric values in {path}.")
    return np.asarray(values, dtype=np.float64)


def load_pixel_intrinsics(path: Path) -> PixelIntrinsics:
    values = load_floats(path)
    if values.size < 4:
        raise ValueError(f"Intrinsics file {path} needs fx fy cx cy, got {values.size} values.")
    fx, fy, cx, cy = (float(v) for v in values[:4])
    return PixelIntrinsics(fx, fy, cx, cy)


def load_c2w(path: Path) -> np.ndarray:
    values = load_floats(path)
    if values.size != 16:
        raise ValueError(f"Extrinsics file {path} needs 16 values, got {values.size}.")
    return values.reshape(4, 4)


def enumerate_frames(scene_dir: Path, cameras: list[int]) -> list[str]:
    image_dir = scene_dir / "images"
    if not image_dir.is_dir():
        return []
    frame_sets = []
    for camera in cameras:
        stems = set()
        suffix = f"_{camera}.jpg"
        for path in image_dir.glob(f"*{suffix}"):
            stem = path.name[: -len(suffix)]
            if stem:
                stems.add(stem)
        frame_sets.append(stems)
    if not frame_sets:
        return []
    common = set.intersection(*frame_sets)
    return sorted(common)


def apply_plan_image(image: Image.Image, plan: ResizeCropPlan) -> np.ndarray:
    resized = image.convert("RGB").resize((plan.scaled_w, plan.scaled_h), Image.LANCZOS)
    cropped = resized.crop(
        (plan.col, plan.row, plan.col + plan.out_w, plan.row + plan.out_h)
    )
    array = np.asarray(cropped, dtype=np.float32) / 255.0
    if array.shape != (plan.out_h, plan.out_w, 3):
        raise RuntimeError(f"Cropped image shape {array.shape} != {(plan.out_h, plan.out_w, 3)}.")
    return array


def apply_plan_mask(mask: Image.Image, plan: ResizeCropPlan) -> np.ndarray:
    """Return a boolean keep-mask. True means the Gaussian stays."""
    resized = mask.convert("L").resize((plan.scaled_w, plan.scaled_h), Image.NEAREST)
    cropped = resized.crop(
        (plan.col, plan.row, plan.col + plan.out_w, plan.row + plan.out_h)
    )
    array = np.asarray(cropped)
    if array.shape != (plan.out_h, plan.out_w):
        raise RuntimeError(f"Cropped mask shape {array.shape} != {(plan.out_h, plan.out_w)}.")
    return array >= CAR_MASK_KEEP_THRESHOLD


def load_frame_inputs(
    scene_dir: Path,
    frame: str,
    cameras: list[int],
    plan: ResizeCropPlan,
):
    images = []
    intrinsics = []
    extrinsics = []
    for camera in cameras:
        image_path = scene_dir / "images" / f"{frame}_{camera}.jpg"
        intrinsic_path = scene_dir / "intrinsics" / f"{camera}.txt"
        extrinsic_path = scene_dir / "cam2ego_extrinsics" / f"{camera}.txt"
        for path in (image_path, intrinsic_path, extrinsic_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        with Image.open(image_path) as image:
            if (image.height, image.width) != (plan.src_h, plan.src_w):
                raise ValueError(
                    f"{image_path} is {image.width}x{image.height}, expected "
                    f"{plan.src_w}x{plan.src_h}."
                )
            images.append(apply_plan_image(image, plan))
        pixel_k = resize_and_crop_intrinsics(load_pixel_intrinsics(intrinsic_path), plan)
        intrinsics.append(pixel_to_normalized(pixel_k, plan.out_h, plan.out_w))
        extrinsics.append(load_c2w(extrinsic_path))
    return (
        np.stack(images, axis=0),
        np.stack(intrinsics, axis=0),
        np.stack(extrinsics, axis=0),
    )


def load_keep_masks(
    mask_root: Path,
    cameras: list[int],
    plan: ResizeCropPlan,
    render_index: int,
    mask_render_view: bool,
) -> np.ndarray:
    """(V, H, W) keep mask. Non-render views are masked unless disabled upstream."""
    keep = np.ones((len(cameras), plan.out_h, plan.out_w), dtype=bool)
    for index, camera in enumerate(cameras):
        if index == render_index and not mask_render_view:
            continue
        path = camera_mask_path(mask_root, camera)
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing ego-car mask for camera {camera}: {path}. "
                "Pass --disable-car-mask to skip masking."
            )
        with Image.open(path) as mask:
            keep[index] = apply_plan_mask(mask, plan)
    return keep


def save_jpg(path: Path, image_hwc: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.clip(image_hwc, 0.0, 1.0)
    array = (array * 255.0).round().astype(np.uint8)
    Image.fromarray(array).save(path, quality=JPEG_QUALITY)


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
                "model.encoder.pose_free=false",
                "model.encoder.backbone.use_pred_intrinsics_for_embed=false",
                "model.encoder.use_checkpoint=false",
                "model.decoder.prune_opacity_threshold=0.005",
            ],
        )
    typed = load_typed_root_config(cfg)
    if typed.model.encoder.pose_free:
        raise RuntimeError("pose_free must be false so GT extrinsics are injected.")
    if typed.model.encoder.backbone.use_pred_intrinsics_for_embed:
        raise RuntimeError("GT intrinsics must condition the backbone.")
    if not typed.model.decoder.make_scale_invariant:
        raise RuntimeError("DL3DV YoNoSplat was trained with make_scale_invariant=true.")

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
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    keep: np.ndarray | None,
    views_to_mask: list[int],
    wide_k: np.ndarray,
    wide_hw: tuple[int, int],
    pose_scale: float,
    device: str,
):
    import torch

    from src.model.types import Gaussians

    images_t = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()
    context = {
        "image": images_t[None].to(device),
        "intrinsics": torch.from_numpy(intrinsics)[None].to(device),
        "extrinsics": torch.from_numpy(extrinsics)[None].to(device),
    }
    with torch.no_grad():
        gaussians = encoder(context, global_step=0)
    view_count, height, width = images.shape[0], images.shape[1], images.shape[2]
    gaussian_count = gaussians.opacities.shape[-1]
    expected = view_count * height * width
    if gaussian_count != expected:
        raise RuntimeError(
            f"Expected one Gaussian per context pixel ({expected}), got {gaussian_count}."
        )
    removed = 0
    opacities = gaussians.opacities
    if keep is not None and views_to_mask:
        indices = ego_remove_indices(keep, views_to_mask)
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
    near = torch.full((1, 1), NEAR / pose_scale, dtype=torch.float32, device=device)
    far = torch.full((1, 1), FAR / pose_scale, dtype=torch.float32, device=device)
    render_ext = torch.from_numpy(extrinsics[0])[None, None].to(device)
    render_k = torch.from_numpy(wide_k)[None, None].to(device)
    with torch.no_grad():
        output = decoder(gaussians, render_ext, render_k, near, far, wide_hw)
    color = output.color[0, 0].clamp(0, 1).permute(1, 2, 0).contiguous().cpu().numpy()
    return color, removed


def resolve_jobs(args) -> list[tuple[str, str]]:
    if args.scene:
        scenes = [args.scene]
    else:
        if not args.scene_list.is_file():
            raise FileNotFoundError(args.scene_list)
        scenes = read_scene_list(args.scene_list)
        if not scenes:
            raise RuntimeError(f"No scenes in {args.scene_list}.")
    frame_filter = normalize_frame_id(args.frame) if args.frame is not None else None
    jobs = []
    for scene in scenes:
        scene_dir = args.data_root / scene
        if not scene_dir.is_dir():
            message = f"Missing scene directory: {scene_dir}"
            if args.scene:
                raise FileNotFoundError(message)
            print(f"warning: {message}", file=sys.stderr)
            continue
        frames = enumerate_frames(scene_dir, args.cameras)
        if frame_filter is not None:
            frames = [frame for frame in frames if frame == frame_filter]
        elif args.max_frames >= 0:
            frames = frames[: args.max_frames]
        if not frames:
            message = f"No frames for scene {scene} cameras {args.cameras}."
            if args.scene or frame_filter is not None:
                raise RuntimeError(message)
            print(f"warning: {message}", file=sys.stderr)
            continue
        jobs.extend((scene, frame) for frame in frames)
    if not jobs:
        raise RuntimeError("No inference jobs.")
    return jobs


def source_hw(scene_dir: Path, frame: str, camera: int) -> tuple[int, int]:
    path = scene_dir / "images" / f"{frame}_{camera}.jpg"
    with Image.open(path) as image:
        return image.height, image.width


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--scene-list", type=Path, default=DEFAULT_SCENE_LIST)
    parser.add_argument("--scene", default=None, help="Run one scene id instead of the list.")
    parser.add_argument("--frame", default=None, help="Run one frame id (for example 0 or 000).")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=-1,
        help="Max frames per scene. -1 means every frame. Ignored when --frame is set.",
    )
    parser.add_argument("--cameras", default="5,4,3")
    parser.add_argument("--render-camera", type=int, default=5)
    parser.add_argument("--width-factor", type=float, default=3.0)
    parser.add_argument("--short-side", type=int, default=TRAIN_SHORT_SIDE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--car-mask-root", type=Path, default=DEFAULT_MASK_ROOT)
    parser.add_argument("--mask-render-view", action="store_true")
    parser.add_argument("--disable-car-mask", action="store_true")
    parser.add_argument("--save-inputs", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    args.cameras = parse_cameras(args.cameras)
    if args.render_camera not in args.cameras:
        raise SystemExit(
            f"--render-camera {args.render_camera} is not in --cameras {args.cameras}."
        )
    # Camera 0 of the normalized rig is the render camera.
    args.cameras = [args.render_camera] + [
        camera for camera in args.cameras if camera != args.render_camera
    ]
    render_index = 0
    views_to_mask = []
    if not args.disable_car_mask:
        views_to_mask = [
            index
            for index, camera in enumerate(args.cameras)
            if args.mask_render_view or camera != args.render_camera
        ]

    jobs = resolve_jobs(args)
    src_hw = source_hw(args.data_root / jobs[0][0], jobs[0][1], args.cameras[0])
    dst_hw = choose_input_hw(src_hw, short_side=args.short_side)
    plan = plan_resize_and_crop(src_hw, dst_hw)
    print(
        f"input {src_hw[1]}x{src_hw[0]} -> {plan.out_w}x{plan.out_h} "
        f"(scaled {plan.scaled_w}x{plan.scaled_h}, crop col={plan.col} row={plan.row}), "
        f"wide x{args.width_factor:g}, cameras {args.cameras}, jobs {len(jobs)}",
        file=sys.stderr,
    )

    keep = None
    if views_to_mask:
        keep = load_keep_masks(
            args.car_mask_root,
            args.cameras,
            plan,
            render_index,
            args.mask_render_view,
        )
        for index in views_to_mask:
            removed_pixels = int((~keep[index]).sum())
            print(
                f"ego mask camera {args.cameras[index]}: remove {removed_pixels} / {keep.shape[1] * keep.shape[2]} pixels",
                file=sys.stderr,
            )

    encoder = decoder = None
    if not args.dry_run:
        import torch

        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise SystemExit("CUDA is required for YoNoSplat inference.")
        torch.set_float32_matmul_precision("high")
        encoder, decoder = build_model(args.checkpoint, args.device)

    for scene, frame in jobs:
        scene_dir = args.data_root / scene
        images, intrinsics, extrinsics = load_frame_inputs(scene_dir, frame, args.cameras, plan)
        normalized, pose_scale = normalize_poses(extrinsics)
        identity_error = float(np.max(np.abs(normalized[0] - np.eye(4))))
        if identity_error > 1e-4:
            raise RuntimeError(
                f"{scene}/{frame}: render camera was not normalized to identity "
                f"(max abs error {identity_error:.3e})."
            )
        pixel_k = resize_and_crop_intrinsics(
            load_pixel_intrinsics(scene_dir / "intrinsics" / f"{args.render_camera}.txt"),
            plan,
        )
        wide_pixel, wide_hw = make_wide_intrinsics(pixel_k, (plan.out_h, plan.out_w), args.width_factor)
        wide_k = pixel_to_normalized(wide_pixel, wide_hw[0], wide_hw[1])
        output_path = args.output_dir / scene / "rgb" / f"{frame}_{args.render_camera}_wide.jpg"
        if args.dry_run:
            identity_error = float(np.max(np.abs(normalized[0] - np.eye(4))))
            print(
                f"dry-run {scene}/{frame}: pose_scale={pose_scale:.4f} "
                f"identity_err={identity_error:.2e} wide={wide_hw[1]}x{wide_hw[0]} -> {output_path}",
                file=sys.stderr,
            )
            continue
        color, removed = render_wide(
            encoder,
            decoder,
            images,
            intrinsics,
            normalized,
            keep,
            views_to_mask,
            wide_k,
            wide_hw,
            pose_scale,
            args.device,
        )
        save_jpg(output_path, color)
        if args.save_inputs:
            for index, camera in enumerate(args.cameras):
                save_jpg(args.output_dir / scene / "inputs" / f"{frame}_{camera}.jpg", images[index])
        print(
            f"{scene}/{frame}: removed {removed} gaussians, "
            f"rgb mean={float(color.mean()):.4f} -> {output_path}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
