"""Resolve inference dataset folders when the checkout uses a different name.

Callers pass the path they expect. An existing path is kept. Otherwise each
missing folder is replaced by a known alias (``ddad`` <-> ``ddad_process``,
``valid`` <-> ``val``, ``trainval2`` <-> ``trainval``) and a missing scene list
is also looked up under the sibling split or an aliased filename
(``valid.txt`` <-> ``valid2.txt``, ``nuScenes_Val2.txt`` <-> ``nuScenes_Val.txt``).
"""

from __future__ import annotations

import sys
from pathlib import Path

DIR_ALIAS_GROUPS = (
    ("ddad", "ddad_process"),
    ("valid", "val", "validation"),
    ("trainval", "trainval2"),
    ("nuscenes_mask", "corrected_masks"),
)
LIST_ALIAS_GROUPS = (
    ("valid.txt", "valid2.txt", "val.txt", "validation.txt"),
    ("nuScenes_Val.txt", "nuScenes_Val2.txt"),
    ("nuScenes_Train.txt", "nuScenes_Train2.txt"),
    ("lyft_val1920.txt", "lyft_val19202.txt"),
    ("lyft_val1224.txt", "lyft_val12242.txt"),
    ("val.txt", "valid.txt", "valid2.txt"),
)


def _alias_names(name: str, groups) -> list[str]:
    ordered = [name]
    for group in groups:
        if name not in group:
            continue
        for item in group:
            if item not in ordered:
                ordered.append(item)
    return ordered


def locate_dir(path) -> Path | None:
    """Return ``path`` when it exists, else the same path with renamed folders."""
    path = Path(path).expanduser()
    if path.is_dir():
        return path
    if path.is_absolute():
        current = Path(path.anchor)
        parts = path.parts[1:]
    else:
        current = Path()
        parts = path.parts
    for part in parts:
        exact = current / part
        if exact.is_dir():
            current = exact
            continue
        found = None
        for alias in _alias_names(part, DIR_ALIAS_GROUPS):
            cand = current / alias
            if cand.is_dir():
                found = cand
                break
        if found is None:
            return None
        current = found
    return current if current.is_dir() else None


def _remember(dirs: list[Path], directory: Path) -> None:
    if directory.is_dir() and directory not in dirs:
        dirs.append(directory)


def locate_file(path, extra_dirs=()) -> Path | None:
    """Find a scene list after a split folder or list filename was renamed."""
    path = Path(path).expanduser()
    if path.is_file():
        return path
    names = _alias_names(path.name, LIST_ALIAS_GROUPS)
    dirs: list[Path] = []
    parent = locate_dir(path.parent)
    bases = []
    if parent is not None:
        _remember(dirs, parent)
        bases.append(parent)
    if path.parent != parent:
        bases.append(path.parent)
    for base in bases:
        grand = base.parent
        located_grand = grand if grand.is_dir() else locate_dir(grand)
        if located_grand is None:
            continue
        for alias in _alias_names(base.name, DIR_ALIAS_GROUPS):
            _remember(dirs, located_grand / alias)
        _remember(dirs, located_grand)
    for extra in extra_dirs:
        extra_path = Path(extra)
        _remember(dirs, extra_path)
        if extra_path.parent.is_dir():
            for alias in _alias_names(extra_path.name, DIR_ALIAS_GROUPS):
                _remember(dirs, extra_path.parent / alias)
    for directory in dirs:
        for name in names:
            cand = directory / name
            if cand.is_file():
                return cand
    return None


def locate_mask_root(path, data_root, mask_kind: str) -> Path | None:
    """Find an ego-car mask directory when its folder was renamed."""
    if not path:
        return None
    found = locate_dir(path)
    if found is not None:
        return found
    root = locate_dir(data_root) or Path(data_root)
    if mask_kind == "nuscenes":
        candidates = (
            root / "nuscenes_mask",
            root.parent / "nuscenes_mask",
            root.parent.parent / "nuscenes_mask",
            root.parent / "corrected_masks",
        )
    elif mask_kind == "lyft":
        candidates = (root / "ego_car_masks", root.parent / "ego_car_masks")
    elif mask_kind == "ddad":
        candidates = (root,)
    else:
        candidates = ()
    for cand in candidates:
        if cand.is_dir():
            return cand
    return None


def _note(kind: str, requested, found: Path) -> None:
    requested_path = Path(requested).expanduser()
    if requested_path.resolve() == found.resolve():
        return
    print(f"[data] {kind}: {requested_path} -> {found}", file=sys.stderr)


def relocate_layout(data_root, scene_list, mask_root=None, mask_kind: str = "", need_mask: bool = False):
    """Return ``(data_root, scene_list, mask_root)``, substituting aliases when needed."""
    root = locate_dir(data_root)
    if root is not None:
        _note("data_root", data_root, root)
        data_root = root
    listed = locate_file(scene_list, extra_dirs=(data_root,))
    if listed is not None:
        _note("scene_list", scene_list, listed)
        scene_list = listed
    if need_mask and mask_root:
        mask = locate_mask_root(mask_root, data_root, mask_kind)
        if mask is not None:
            _note("car_mask_root", mask_root, mask)
            mask_root = mask
    return data_root, scene_list, mask_root
