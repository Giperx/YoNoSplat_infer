"""Dataset presets for image-only wide-FOV inference.

By default the model sees 224x224 stretches. ``--keep-aspect`` instead feeds
the 224-high aspect canvas below, and the wide render is already the saved
size. The saved wide image keeps height 224 and uses the same width rule:
scale the native frame so its height is 224, snap the width to a multiple of
14, then multiply by 3. 1600x900 therefore stays 1176 wide.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import dataset_paths
import inference_nuscenes_wide as base

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    data_root: Path
    scene_list: Path
    mask_kind: str
    mask_root: Path | None
    output_single: Path
    output_multi: Path


def _spec(
    name: str,
    data_root: str,
    scene_list: str,
    mask_kind: str,
    mask_root: str | None,
) -> DatasetSpec:
    root = ROOT / data_root
    return DatasetSpec(
        name=name,
        data_root=root,
        scene_list=root / scene_list,
        mask_kind=mask_kind,
        mask_root=None if mask_root is None else ROOT / mask_root,
        output_single=ROOT / "outputs" / f"{name}_wide_pred",
        output_multi=ROOT / "outputs" / f"{name}_wide_pred_multiframes",
    )


DATASETS = {
    "nuscenes": _spec(
        "nuscenes",
        "datasets/nuscenes/processed_10Hz/trainval2",
        "nuScenes_Val.txt",
        "nuscenes",
        "datasets/nuscenes/nuscenes_mask",
    ),
    "lyft1920": _spec(
        "lyft1920",
        "datasets/lyft/lyft_val1920_3cams",
        "lyft_val1920.txt",
        "lyft",
        "datasets/lyft/lyft_val1920_3cams/ego_car_masks",
    ),
    "lyft1224": _spec(
        "lyft1224",
        "datasets/lyft/lyft_val1224_3cams",
        "lyft_val1224.txt",
        "lyft",
        "datasets/lyft/lyft_val1224_3cams/ego_car_masks",
    ),
    "ddad": _spec(
        "ddad",
        "datasets/ddad_process/valid",
        "valid.txt",
        "ddad",
        "datasets/ddad_process/valid",
    ),
    "widedrive": _spec(
        "widedrive",
        "datasets/WideDrive_processed/WideDriveVal",
        "val.txt",
        "none",
        None,
    ),
}


def save_width(src_hw: tuple[int, int], short_side: int = 224, width_factor: float = 3.0) -> int:
    """Width of the saved wide image for a native ``(height, width)`` frame."""
    _height, width = base.choose_input_hw(src_hw, short_side=int(short_side))
    saved = int(round(width * float(width_factor)))
    if saved < 1:
        raise ValueError(f"Invalid save width for {src_hw}.")
    return saved


def ego_mask_path(spec: DatasetSpec, camera: int, scene: str | None = None) -> Path | None:
    """Path of one camera's ego-car mask. ``None`` means every pixel stays valid."""
    if spec.mask_kind == "none":
        return None
    if spec.mask_kind == "nuscenes":
        return base.camera_mask_path(spec.mask_root, int(camera))
    if spec.mask_kind == "lyft":
        return Path(spec.mask_root) / f"{int(camera)}.jpg"
    if spec.mask_kind == "ddad":
        if not scene:
            raise ValueError("DDAD ego-car masks are stored per scene.")
        return Path(spec.mask_root) / str(scene) / "ego_car_masks" / f"{int(camera)}.jpg"
    raise ValueError(f"Unknown mask kind {spec.mask_kind!r}.")


def apply_defaults(args, multi: bool) -> DatasetSpec:
    """Fill unset paths from ``--dataset`` and turn masking off when there is no mask."""
    try:
        spec = DATASETS[args.dataset]
    except KeyError as exc:
        known = ", ".join(DATASETS)
        raise SystemExit(f"Unknown dataset {args.dataset!r}. Known: {known}.") from exc
    if args.data_root is None:
        args.data_root = spec.data_root
    if args.scene_list is None:
        args.scene_list = spec.scene_list
    if args.output_dir is None:
        args.output_dir = spec.output_multi if multi else spec.output_single
    if spec.mask_kind == "none":
        args.disable_car_mask = True
    elif args.car_mask_root is not None:
        spec = DatasetSpec(
            name=spec.name,
            data_root=args.data_root,
            scene_list=args.scene_list,
            mask_kind=spec.mask_kind,
            mask_root=args.car_mask_root,
            output_single=spec.output_single,
            output_multi=spec.output_multi,
        )
    elif spec.mask_kind == "ddad":
        spec = DatasetSpec(
            name=spec.name,
            data_root=args.data_root,
            scene_list=args.scene_list,
            mask_kind=spec.mask_kind,
            mask_root=args.data_root,
            output_single=spec.output_single,
            output_multi=spec.output_multi,
        )
    data_root, scene_list, mask_root = dataset_paths.relocate_layout(
        spec.data_root,
        spec.scene_list,
        spec.mask_root,
        spec.mask_kind,
        need_mask=spec.mask_kind != "none" and not args.disable_car_mask,
    )
    if (data_root, scene_list, mask_root) != (spec.data_root, spec.scene_list, spec.mask_root):
        spec = DatasetSpec(
            name=spec.name,
            data_root=data_root,
            scene_list=scene_list,
            mask_kind=spec.mask_kind,
            mask_root=mask_root,
            output_single=spec.output_single,
            output_multi=spec.output_multi,
        )
        args.data_root = data_root
        args.scene_list = scene_list
    return spec
