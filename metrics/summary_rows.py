"""Equal-weight region means and frame-count pooling for metric reports.

Mean_LR is (Left + Right) / 2. Mean_LRC is (Left + Right + Center) / 3.
Center_masked is not included. A score is averaged only when every region in
the mean reports it, so Left/Right LPIPS stays out.

Lyft 1920 and 1224 are pooled by frame count:
``(n_a * value_a + n_b * value_b) / (n_a + n_b)``. That is not the average of
the two subset scores.
"""

from __future__ import annotations

import re
from pathlib import Path

METRIC_LINE = re.compile(r"^(?P<indent> *)(?P<key>\S+)\s+:\s+(?P<body>.*)$")
REPORT_KIND = re.compile(
    r"^yonosplat_(photometric|HM|CRCS|IPS|consistency)_(\d{8}_\d{6})\.txt$"
)
REGION_MEANS = (
    ("Mean_LR", ("Left", "Right")),
    ("Mean_LRC", ("Left", "Right", "Center")),
)


def parse_metric_line(line):
    """Return a metric row, or None when the line is not one."""
    match = METRIC_LINE.match(line.rstrip("\n"))
    if match is None:
        return None
    body = match.group("body").strip()
    if not body or body == "no frames":
        return {
            "indent": match.group("indent"),
            "key": match.group("key"),
            "n": None,
            "values": {},
            "order": [],
            "empty": True,
        }
    count = None
    values = {}
    order = []
    for token in body.split():
        if "=" not in token:
            continue
        name, raw = token.split("=", 1)
        if name == "n":
            count = int(raw)
            continue
        if raw == "n/a":
            continue
        values[name] = float(raw)
        order.append(name)
    if count is None:
        return None
    return {
        "indent": match.group("indent"),
        "key": match.group("key"),
        "n": count,
        "values": values,
        "order": order,
        "empty": False,
    }


def _format_number(name, value):
    if name in {"PSNR", "MAE", "RMSE"}:
        return f"{value:.2f}"
    return f"{value:.4f}"


def format_metric_line(indent, key, row):
    if row is None or row.get("empty") or not row.get("order"):
        return f"{indent}{key:20s}: no frames\n"
    parts = [f"n={row['n']}"]
    for name in row["order"]:
        parts.append(f"{name}={_format_number(name, row['values'][name])}")
    return f"{indent}{key:20s}: {'  '.join(parts)}\n"


def unweighted_mean(rows):
    """Average region scores with equal weight. Shared names only."""
    if not rows or any(row is None or row.get("empty") for row in rows):
        return None
    names = set(rows[0]["values"])
    for row in rows[1:]:
        names &= set(row["values"])
    order = [name for name in rows[0]["order"] if name in names]
    if not order:
        return None
    return {
        "n": min(row["n"] for row in rows),
        "values": {
            name: sum(row["values"][name] for row in rows) / len(rows)
            for name in order
        },
        "order": order,
        "empty": False,
    }


def pool_by_count(rows):
    """Pool the same region across subsets. Each frame has equal weight."""
    if not rows or any(row is None or row.get("empty") or row["n"] <= 0 for row in rows):
        return None
    names = set(rows[0]["values"])
    for row in rows[1:]:
        names &= set(row["values"])
    order = [name for name in rows[0]["order"] if name in names]
    if not order:
        return None
    total = sum(row["n"] for row in rows)
    return {
        "n": total,
        "values": {
            name: sum(row["n"] * row["values"][name] for row in rows) / total
            for name in order
        },
        "order": order,
        "empty": False,
    }


def _mean_lines(group):
    by_key = {row["key"]: row for row in group}
    if "Mean_LR" in by_key or not {"Left", "Center", "Right"} <= set(by_key):
        return []
    indent = group[0]["indent"]
    lines = []
    for label, regions in REGION_MEANS:
        averaged = unweighted_mean([by_key[name] for name in regions])
        lines.append(format_metric_line(indent, label, averaged))
    return lines


def insert_equal_means(text):
    """Add Mean_LR and Mean_LRC after each Left/Center/Right block."""
    if "Full_unmasked" in text or "Style: dense" in text:
        return text, False
    lines = text.splitlines(keepends=True)
    output = []
    group = []
    changed = False

    def flush():
        nonlocal group, changed
        if not group:
            return
        parsed = [parse_metric_line(line) for line in group]
        if all(row is not None for row in parsed):
            extra = _mean_lines(parsed)
            output.extend(group)
            if extra:
                output.extend(extra)
                changed = True
        else:
            output.extend(group)
        group = []

    for line in lines:
        if parse_metric_line(line) is not None:
            group.append(line)
            continue
        flush()
        output.append(line)
    flush()
    return "".join(output), changed


def summary_rows(text):
    """Metric rows in the Summary block, in file order."""
    lines = text.splitlines()
    try:
        start = lines.index("Summary")
    except ValueError:
        return []
    rows = []
    begun = False
    for line in lines[start + 1:]:
        if line.startswith("=" * 10):
            if begun:
                break
            begun = True
            continue
        if not begun:
            continue
        parsed = parse_metric_line(line)
        if parsed is not None and not parsed["empty"]:
            rows.append(parsed)
    return rows


def latest_reports(directory):
    """Newest report of each kind in one render directory."""
    chosen = {}
    root = Path(directory)
    if not root.is_dir():
        return {}
    for path in root.glob("yonosplat_*.txt"):
        match = REPORT_KIND.match(path.name)
        if match is None:
            continue
        kind, stamp = match.group(1), match.group(2)
        current = chosen.get(kind)
        if current is None or stamp > current[0]:
            chosen[kind] = (stamp, path)
    return {kind: path for kind, (stamp, path) in chosen.items()}


def lyft_sibling(render_root):
    """Return ``(root_1920, root_1224)`` for one Lyft render directory."""
    path = Path(render_root)
    name = path.name
    if "lyft1920" in name:
        other = name.replace("lyft1920", "lyft1224", 1)
        return path, path.with_name(other)
    if "lyft1224" in name:
        other = name.replace("lyft1224", "lyft1920", 1)
        return path.with_name(other), path
    raise ValueError(f"{render_root} is not a Lyft 1920 or 1224 render directory.")


def _section(kind, left_path, right_path, left_rows, right_rows):
    by_right = {row["key"]: row for row in right_rows}
    lines = [
        f"=== {kind} ===",
        f"1920: {left_path.name}",
        f"1224: {right_path.name}",
        "",
    ]
    pooled = []
    for row in left_rows:
        if row["key"] in {"Mean_LR", "Mean_LRC"}:
            continue
        other = by_right.get(row["key"])
        if other is None:
            continue
        merged = pool_by_count([row, other])
        if merged is None:
            continue
        pooled.append((row["key"], merged))
        lines.append(format_metric_line("", row["key"], merged).rstrip("\n"))
    by_key = {key: row for key, row in pooled}
    if {"Left", "Center", "Right"} <= set(by_key):
        for label, regions in REGION_MEANS:
            averaged = unweighted_mean([by_key[name] for name in regions])
            lines.append(format_metric_line("", label, averaged).rstrip("\n"))
    return "\n".join(lines)


def render_lyft_report(root_1920, root_1224):
    """Pooled Lyft summary. None when either side has no reports."""
    left = latest_reports(root_1920)
    right = latest_reports(root_1224)
    kinds = [kind for kind in ("photometric", "HM", "CRCS", "IPS", "consistency") if kind in left and kind in right]
    if not kinds:
        return None
    sections = [
        "=== YoNoSplat Lyft combined ===",
        f"1920 root: {root_1920}",
        f"1224 root: {root_1224}",
        "1920 and 1224 are one dataset. Each region is pooled by frame count,",
        "(n_1920 * value_1920 + n_1224 * value_1224) / (n_1920 + n_1224).",
        "Mean_LR is (Left + Right) / 2 of those pooled scores.",
        "Mean_LRC is (Left + Right + Center) / 3. Center_masked is not included.",
        "WideDrive is not part of this file.",
        "",
    ]
    for kind in kinds:
        left_path = left[kind]
        right_path = right[kind]
        sections.append(
            _section(
                kind,
                left_path,
                right_path,
                summary_rows(left_path.read_text()),
                summary_rows(right_path.read_text()),
            )
        )
        sections.append("")
    return "\n".join(sections).rstrip() + "\n"


def write_lyft_report(render_root):
    """Write ``yonosplat_lyft.txt`` into the 1920 directory. Return its path or None."""
    root_1920, root_1224 = lyft_sibling(render_root)
    if not root_1920.is_dir() or not root_1224.is_dir():
        return None
    report = render_lyft_report(root_1920, root_1224)
    if report is None:
        return None
    destination = root_1920 / "yonosplat_lyft.txt"
    destination.write_text(report)
    return destination
