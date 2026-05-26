from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from math import atan2, degrees, hypot
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = BACKEND_DIR.parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from main import process_floor_image, UPLOADS_DIR  # noqa: E402


Point = Tuple[float, float]
Line = Tuple[float, float, float, float]


@dataclass
class ComparisonTolerances:
    line_endpoint_tol: float = 24.0
    line_angle_tol_deg: float = 8.0
    line_perp_tol: float = 16.0
    line_overlap_ratio_min: float = 0.45
    opening_center_tol: float = 26.0
    opening_width_tol: float = 26.0


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def line_angle(line: Line) -> float:
    x1, y1, x2, y2 = line
    angle = degrees(atan2(y2 - y1, x2 - x1))
    angle = abs(angle) % 180
    return angle


def line_length(line: Line) -> float:
    x1, y1, x2, y2 = line
    return hypot(x2 - x1, y2 - y1)


def endpoint_pair_distance(a: Line, b: Line) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    forward = hypot(ax1 - bx1, ay1 - by1) + hypot(ax2 - bx2, ay2 - by2)
    reverse = hypot(ax1 - bx2, ay1 - by2) + hypot(ax2 - bx1, ay2 - by1)
    return min(forward, reverse) / 2.0


def axis_signature(line: Line) -> Tuple[str, float, Tuple[float, float]]:
    x1, y1, x2, y2 = line
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    if dx >= dy:
        axis = "x"
        fixed = (y1 + y2) / 2.0
        span = (min(x1, x2), max(x1, x2))
    else:
        axis = "y"
        fixed = (x1 + x2) / 2.0
        span = (min(y1, y2), max(y1, y2))
    return axis, fixed, span


def line_overlap_ratio(a: Line, b: Line) -> float:
    axis_a, fixed_a, span_a = axis_signature(a)
    axis_b, fixed_b, span_b = axis_signature(b)
    if axis_a != axis_b:
        return 0.0
    overlap = max(0.0, min(span_a[1], span_b[1]) - max(span_a[0], span_b[0]))
    shorter = min(span_a[1] - span_a[0], span_b[1] - span_b[0])
    if shorter <= 0:
        return 0.0
    return overlap / shorter


def line_perpendicular_distance(a: Line, b: Line) -> float:
    axis_a, fixed_a, _ = axis_signature(a)
    axis_b, fixed_b, _ = axis_signature(b)
    if axis_a != axis_b:
        return float("inf")
    return abs(fixed_a - fixed_b)


def bbox_from_polygon(polygon: Sequence[Sequence[float]]) -> Optional[Dict[str, float]]:
    if not polygon:
        return None
    xs = [float(pt[0]) for pt in polygon]
    ys = [float(pt[1]) for pt in polygon]
    return {
        "min_x": min(xs),
        "max_x": max(xs),
        "min_y": min(ys),
        "max_y": max(ys),
        "width": max(xs) - min(xs),
        "height": max(ys) - min(ys),
    }


def normalize_point_to_bbox(
    x: float,
    y: float,
    source_bbox: Optional[Dict[str, float]],
    target_bbox: Optional[Dict[str, float]],
) -> Point:
    if (
        not source_bbox
        or not target_bbox
        or source_bbox["width"] <= 0
        or source_bbox["height"] <= 0
    ):
        return (x, y)
    scale_x = target_bbox["width"] / source_bbox["width"]
    scale_y = target_bbox["height"] / source_bbox["height"]
    return (
        target_bbox["min_x"] + ((x - source_bbox["min_x"]) * scale_x),
        target_bbox["min_y"] + ((y - source_bbox["min_y"]) * scale_y),
    )


def normalize_lines_to_bbox(
    lines: Sequence[Sequence[float]],
    source_bbox: Optional[Dict[str, float]],
    target_bbox: Optional[Dict[str, float]],
) -> List[Line]:
    normalized: List[Line] = []
    for line in lines:
        p1 = normalize_point_to_bbox(float(line[0]), float(line[1]), source_bbox, target_bbox)
        p2 = normalize_point_to_bbox(float(line[2]), float(line[3]), source_bbox, target_bbox)
        normalized.append((p1[0], p1[1], p2[0], p2[1]))
    return normalized


def normalize_openings_to_bbox(
    openings: Sequence[Dict[str, Any]],
    source_bbox: Optional[Dict[str, float]],
    target_bbox: Optional[Dict[str, float]],
) -> List[Dict[str, float]]:
    if (
        not source_bbox
        or not target_bbox
        or source_bbox["width"] <= 0
        or source_bbox["height"] <= 0
    ):
        return [
            {"x": float(item["x"]), "y": float(item["y"]), "width": float(item["width"])}
            for item in openings
        ]
    scale_x = target_bbox["width"] / source_bbox["width"]
    scale_y = target_bbox["height"] / source_bbox["height"]
    width_scale = (scale_x + scale_y) / 2.0
    normalized = []
    for item in openings:
        px, py = normalize_point_to_bbox(float(item["x"]), float(item["y"]), source_bbox, target_bbox)
        normalized.append(
            {
                "x": round(px, 3),
                "y": round(py, 3),
                "width": round(float(item["width"]) * width_scale, 3),
            }
        )
    return normalized


def compare_polygon(
    expected_polygon: Sequence[Sequence[float]],
    actual_polygon: Sequence[Sequence[float]],
) -> Dict[str, Any]:
    expected_bbox = bbox_from_polygon(expected_polygon)
    actual_bbox = bbox_from_polygon(actual_polygon)
    if not expected_bbox or not actual_bbox:
        return {
            "expected_exists": bool(expected_bbox),
            "actual_exists": bool(actual_bbox),
            "bbox_diff": None,
        }
    return {
        "expected_exists": True,
        "actual_exists": True,
        "expected_point_count": len(expected_polygon),
        "actual_point_count": len(actual_polygon),
        "expected_bbox": expected_bbox,
        "actual_bbox": actual_bbox,
        "bbox_diff": {
            key: round(actual_bbox[key] - expected_bbox[key], 3)
            for key in ("min_x", "max_x", "min_y", "max_y", "width", "height")
        },
    }


def line_to_dict(line: Line) -> Dict[str, float]:
    x1, y1, x2, y2 = line
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def compare_lines(
    expected_lines: Sequence[Sequence[float]],
    actual_lines: Sequence[Sequence[float]],
    tolerances: ComparisonTolerances,
) -> Dict[str, Any]:
    expected = [tuple(map(float, line)) for line in expected_lines]
    actual = [tuple(map(float, line)) for line in actual_lines]
    unmatched_actual = set(range(len(actual)))
    matched_pairs: List[Dict[str, Any]] = []
    missing: List[Dict[str, float]] = []

    for expected_line in expected:
        best_idx = None
        best_score = None
        best_meta = None
        expected_angle = line_angle(expected_line)
        for idx in list(unmatched_actual):
            actual_line = actual[idx]
            angle_diff = min(
                abs(expected_angle - line_angle(actual_line)),
                180 - abs(expected_angle - line_angle(actual_line)),
            )
            overlap_ratio = line_overlap_ratio(expected_line, actual_line)
            perp_distance = line_perpendicular_distance(expected_line, actual_line)
            endpoint_distance = endpoint_pair_distance(expected_line, actual_line)
            if angle_diff > tolerances.line_angle_tol_deg:
                continue
            if perp_distance > tolerances.line_perp_tol:
                continue
            if overlap_ratio < tolerances.line_overlap_ratio_min:
                continue
            if endpoint_distance > tolerances.line_endpoint_tol:
                continue
            score = endpoint_distance + perp_distance + (1.0 - overlap_ratio) * 10.0
            if best_score is None or score < best_score:
                best_idx = idx
                best_score = score
                best_meta = {
                    "endpoint_distance": round(endpoint_distance, 3),
                    "perpendicular_distance": round(perp_distance, 3),
                    "overlap_ratio": round(overlap_ratio, 3),
                    "angle_diff_deg": round(angle_diff, 3),
                }

        if best_idx is None:
            missing.append(line_to_dict(expected_line))
            continue

        unmatched_actual.remove(best_idx)
        matched_pairs.append(
            {
                "expected": line_to_dict(expected_line),
                "actual": line_to_dict(actual[best_idx]),
                "match_meta": best_meta,
            }
        )

    extra = [line_to_dict(actual[idx]) for idx in sorted(unmatched_actual)]
    return {
        "matched_count": len(matched_pairs),
        "missing": missing,
        "extra": extra,
        "matched_pairs": matched_pairs,
    }


def compare_openings(
    expected_openings: Sequence[Dict[str, Any]],
    actual_openings: Sequence[Dict[str, Any]],
    tolerances: ComparisonTolerances,
) -> Dict[str, Any]:
    unmatched_actual = set(range(len(actual_openings)))
    matched_pairs = []
    missing = []

    for expected in expected_openings:
        best_idx = None
        best_score = None
        best_meta = None
        for idx in list(unmatched_actual):
            actual = actual_openings[idx]
            center_distance = hypot(
                float(expected["x"]) - float(actual["x"]),
                float(expected["y"]) - float(actual["y"]),
            )
            width_distance = abs(float(expected["width"]) - float(actual["width"]))
            if center_distance > tolerances.opening_center_tol:
                continue
            if width_distance > tolerances.opening_width_tol:
                continue
            score = center_distance + width_distance * 0.3
            if best_score is None or score < best_score:
                best_idx = idx
                best_score = score
                best_meta = {
                    "center_distance": round(center_distance, 3),
                    "width_distance": round(width_distance, 3),
                }
        if best_idx is None:
            missing.append(expected)
            continue

        unmatched_actual.remove(best_idx)
        matched_pairs.append(
            {
                "expected": expected,
                "actual": actual_openings[best_idx],
                "match_meta": best_meta,
            }
        )

    extra = [actual_openings[idx] for idx in sorted(unmatched_actual)]
    return {
        "matched_count": len(matched_pairs),
        "missing": missing,
        "extra": extra,
        "matched_pairs": matched_pairs,
    }


def prepare_project_image(source_image: Path, project_id: str) -> Path:
    project_dir = UPLOADS_DIR / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    target_image = project_dir / "floor_1.png"
    shutil.copyfile(source_image, target_image)
    return target_image


def build_report(ground_truth: Dict[str, Any], actual: Dict[str, Any]) -> Dict[str, Any]:
    tolerances = ComparisonTolerances()
    polygon_report = compare_polygon(ground_truth["polygon"], actual.get("polygon", []))
    expected_bbox = bbox_from_polygon(ground_truth["polygon"])
    actual_bbox = bbox_from_polygon(actual.get("polygon", []))
    normalized_actual_lines = normalize_lines_to_bbox(
        actual.get("inner_walls", []),
        actual_bbox,
        expected_bbox,
    )
    normalized_actual_doors = normalize_openings_to_bbox(
        actual.get("doors", []),
        actual_bbox,
        expected_bbox,
    )
    normalized_actual_windows = normalize_openings_to_bbox(
        actual.get("windows", []),
        actual_bbox,
        expected_bbox,
    )
    wall_report = compare_lines(
        ground_truth["inner_walls"],
        normalized_actual_lines,
        tolerances,
    )
    door_report = compare_openings(
        ground_truth["doors"],
        normalized_actual_doors,
        tolerances,
    )
    window_report = compare_openings(
        ground_truth["windows"],
        normalized_actual_windows,
        tolerances,
    )
    actual_room_count = len(actual.get("rooms", []))
    expected_room_count = int(ground_truth["room_count"])
    return {
        "floor_name": ground_truth["floor_name"],
        "polygon": polygon_report,
        "inner_walls": {
            "expected_count": len(ground_truth["inner_walls"]),
            "actual_count": len(actual.get("inner_walls", [])),
            "missing_inner_walls": wall_report["missing"],
            "extra_inner_walls": wall_report["extra"],
            "matched_count": wall_report["matched_count"],
        },
        "doors": {
            "expected_count": len(ground_truth["doors"]),
            "actual_count": len(actual.get("doors", [])),
            "missing_doors": door_report["missing"],
            "extra_doors": door_report["extra"],
            "matched_count": door_report["matched_count"],
        },
        "windows": {
            "expected_count": len(ground_truth["windows"]),
            "actual_count": len(actual.get("windows", [])),
            "missing_windows": window_report["missing"],
            "extra_windows": window_report["extra"],
            "matched_count": window_report["matched_count"],
        },
        "rooms": {
            "expected_room_count": expected_room_count,
            "actual_room_count": actual_room_count,
            "room_count_diff": actual_room_count - expected_room_count,
        },
    }


def print_human_summary(report: Dict[str, Any]) -> None:
    print(f"Comparison floor: {report['floor_name']}")
    polygon = report["polygon"]
    print(
        "Polygon: "
        f"expected_exists={polygon['expected_exists']} "
        f"actual_exists={polygon['actual_exists']} "
        f"bbox_diff={polygon['bbox_diff']}"
    )
    for key, label in (
        ("inner_walls", "Inner walls"),
        ("doors", "Doors"),
        ("windows", "Windows"),
    ):
        section = report[key]
        missing_key = next(k for k in section.keys() if k.startswith("missing_"))
        extra_key = next(k for k in section.keys() if k.startswith("extra_"))
        print(
            f"{label}: expected={section['expected_count']} actual={section['actual_count']} "
            f"missing={len(section[missing_key])} extra={len(section[extra_key])} "
            f"matched={section['matched_count']}"
        )
    rooms = report["rooms"]
    print(
        "Rooms: "
        f"expected={rooms['expected_room_count']} actual={rooms['actual_room_count']} "
        f"diff={rooms['room_count_diff']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare backend extraction output against clean floor ground truth.")
    parser.add_argument(
        "--ground-truth",
        default=str(BACKEND_DIR / "ground_truth" / "floor_1_clean.json"),
        help="Path to a ground truth JSON file.",
    )
    parser.add_argument(
        "--project-id",
        default="compare-floor-1-clean",
        help="Temporary project id used to save debug artifacts.",
    )
    parser.add_argument(
        "--floor-height",
        type=float,
        default=3.2,
        help="Floor height passed into process_floor_image.",
    )
    args = parser.parse_args()

    ground_truth_path = Path(args.ground_truth)
    ground_truth = load_json(ground_truth_path)
    source_image = REPO_DIR / ground_truth["source_image"]
    if not source_image.exists():
        raise FileNotFoundError(f"Source image not found: {source_image}")

    target_image = prepare_project_image(source_image, args.project_id)
    actual = process_floor_image(
        image_path=target_image,
        project_id=args.project_id,
        floor_index=1,
        floor_height=args.floor_height,
    )
    report = build_report(ground_truth, actual)
    print_human_summary(report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
