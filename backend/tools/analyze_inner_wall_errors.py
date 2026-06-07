from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

import cv2
import numpy as np


BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = BACKEND_DIR.parent
DEBUG_COMPARE_DIR = BACKEND_DIR / "debug_upload_compare"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(BACKEND_DIR / "tools") not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR / "tools"))

from compare_clean_floor import (  # noqa: E402
    ComparisonTolerances,
    axis_signature,
    bbox_from_polygon,
    endpoint_pair_distance,
    line_angle,
    line_length,
    line_overlap_ratio,
    line_perpendicular_distance,
    normalize_lines_to_bbox,
)
from main import process_floor_image  # noqa: E402


Line = Tuple[float, float, float, float]


def load_json_any_bom(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


@dataclass
class ExpectedAnalysis:
    gt_line: Line
    closest_actual: Optional[Line]
    orientation_match: bool
    fixed_axis_offset: Optional[float]
    span_overlap_ratio: float
    actual_vs_gt_length: str
    fragment_count_near_gt: int
    nearest_candidates_count: int
    reason: str


def line_to_dict(line: Optional[Line]) -> Any:
    if line is None:
        return None
    x1, y1, x2, y2 = line
    return {"x1": round(x1, 3), "y1": round(y1, 3), "x2": round(x2, 3), "y2": round(y2, 3)}


def reason_priority(reason: str) -> int:
    order = {
        "orientation_mismatch": 0,
        "missing_candidate": 1,
        "fixed_axis_offset_too_high": 2,
        "fragmented_wall": 3,
        "span_overlap_too_low": 4,
        "too_short": 5,
        "too_long": 6,
        "unknown": 7,
    }
    return order.get(reason, 99)


def classify_expected_vs_actual(gt_line: Line, actual_lines: Sequence[Line]) -> ExpectedAnalysis:
    gt_axis, _, gt_span = axis_signature(gt_line)
    gt_len = max(1.0, line_length(gt_line))
    scored: List[Dict[str, Any]] = []

    for actual in actual_lines:
        actual_axis, _, _ = axis_signature(actual)
        orientation_match = actual_axis == gt_axis
        fixed_axis_offset = line_perpendicular_distance(gt_line, actual) if orientation_match else None
        overlap_ratio = line_overlap_ratio(gt_line, actual) if orientation_match else 0.0
        endpoint_distance = endpoint_pair_distance(gt_line, actual)
        angle_diff = min(abs(line_angle(gt_line) - line_angle(actual)), 180 - abs(line_angle(gt_line) - line_angle(actual)))
        actual_len = max(1.0, line_length(actual))
        length_ratio = actual_len / gt_len
        score = endpoint_distance + (fixed_axis_offset or 120.0) + ((1.0 - overlap_ratio) * 30.0) + (0 if orientation_match else 80.0)
        scored.append(
            {
                "line": actual,
                "orientation_match": orientation_match,
                "fixed_axis_offset": fixed_axis_offset,
                "span_overlap_ratio": overlap_ratio,
                "endpoint_distance": endpoint_distance,
                "angle_diff": angle_diff,
                "length_ratio": length_ratio,
                "score": score,
            }
        )

    scored.sort(key=lambda item: item["score"])
    closest = scored[0] if scored else None

    near_fragments = [
        item for item in scored
        if item["orientation_match"]
        and item["fixed_axis_offset"] is not None
        and item["fixed_axis_offset"] <= 28.0
        and item["span_overlap_ratio"] >= 0.18
    ]
    fragment_count = len(near_fragments)

    reason = "unknown"
    actual_vs_gt_length = "not_available"
    closest_line = None
    orientation_match = False
    fixed_axis_offset = None
    overlap_ratio = 0.0

    if closest is None:
        reason = "missing_candidate"
    else:
        closest_line = closest["line"]
        orientation_match = bool(closest["orientation_match"])
        fixed_axis_offset = closest["fixed_axis_offset"]
        overlap_ratio = float(closest["span_overlap_ratio"])
        length_ratio = float(closest["length_ratio"])

        if length_ratio < 0.85:
            actual_vs_gt_length = "shorter"
        elif length_ratio > 1.15:
            actual_vs_gt_length = "longer"
        else:
            actual_vs_gt_length = "similar"

        if not orientation_match:
            reason = "orientation_mismatch"
        elif fragment_count >= 2 and overlap_ratio < 0.45:
            reason = "fragmented_wall"
        elif fixed_axis_offset is not None and fixed_axis_offset > 16.0:
            reason = "fixed_axis_offset_too_high"
        elif overlap_ratio < 0.45:
            reason = "span_overlap_too_low"
        elif actual_vs_gt_length == "shorter" and overlap_ratio >= 0.45:
            reason = "too_short"
        elif actual_vs_gt_length == "longer" and overlap_ratio >= 0.45:
            reason = "too_long"
        elif fragment_count >= 2:
            reason = "fragmented_wall"
        else:
            reason = "unknown"

    if not scored or min((item["endpoint_distance"] for item in scored), default=999.0) > 120.0:
        reason = "missing_candidate"

    return ExpectedAnalysis(
        gt_line=gt_line,
        closest_actual=closest_line,
        orientation_match=orientation_match,
        fixed_axis_offset=None if fixed_axis_offset is None else round(float(fixed_axis_offset), 3),
        span_overlap_ratio=round(overlap_ratio, 3),
        actual_vs_gt_length=actual_vs_gt_length,
        fragment_count_near_gt=fragment_count,
        nearest_candidates_count=len(scored),
        reason=reason,
    )


def classify_actual_line(actual_line: Line, gt_lines: Sequence[Line]) -> Tuple[str, Dict[str, Any]]:
    actual_axis, actual_fixed, actual_span = axis_signature(actual_line)
    actual_len = max(1.0, line_length(actual_line))
    candidates: List[Dict[str, Any]] = []
    for gt_line in gt_lines:
        gt_axis, gt_fixed, gt_span = axis_signature(gt_line)
        orientation_match = gt_axis == actual_axis
        fixed_axis_offset = line_perpendicular_distance(gt_line, actual_line) if orientation_match else None
        overlap_ratio = line_overlap_ratio(gt_line, actual_line) if orientation_match else 0.0
        endpoint_distance = endpoint_pair_distance(gt_line, actual_line)
        gt_len = max(1.0, line_length(gt_line))
        score = endpoint_distance + (fixed_axis_offset or 120.0) + ((1.0 - overlap_ratio) * 30.0) + (0 if orientation_match else 80.0)
        candidates.append(
            {
                "gt_line": gt_line,
                "orientation_match": orientation_match,
                "fixed_axis_offset": fixed_axis_offset,
                "overlap_ratio": overlap_ratio,
                "endpoint_distance": endpoint_distance,
                "length_ratio": actual_len / gt_len,
                "score": score,
            }
        )

    candidates.sort(key=lambda item: item["score"])
    best = candidates[0] if candidates else None
    classification = "extra_false_positive"
    meta = {
        "closest_gt": line_to_dict(best["gt_line"]) if best else None,
        "orientation_match": bool(best["orientation_match"]) if best else False,
        "fixed_axis_offset": round(float(best["fixed_axis_offset"]), 3) if best and best["fixed_axis_offset"] is not None else None,
        "span_overlap_ratio": round(float(best["overlap_ratio"]), 3) if best else 0.0,
    }

    if best is None:
        return classification, meta

    offset = best["fixed_axis_offset"] if best["fixed_axis_offset"] is not None else 999.0
    overlap = best["overlap_ratio"]
    length_ratio = best["length_ratio"]

    if best["orientation_match"] and offset <= 16.0 and overlap >= 0.45:
        classification = "true_candidate_near_gt"
    elif best["orientation_match"] and offset <= 28.0 and overlap >= 0.18 and length_ratio < 0.85:
        classification = "fragmented_near_gt"
    elif best["orientation_match"] and offset <= 40.0 and overlap >= 0.18:
        classification = "shifted_from_gt"
    elif best["orientation_match"] and offset <= 24.0 and overlap < 0.18:
        classification = "possible_outer_or_symbolic_artifact"
    elif not best["orientation_match"] or offset > 40.0:
        classification = "possible_outer_or_symbolic_artifact"

    return classification, meta


def render_overlay(
    floor_name: str,
    gt_lines: Sequence[Line],
    actual_lines: Sequence[Line],
    expected_reports: Sequence[Dict[str, Any]],
    output_path: Path,
    canvas_size: Tuple[int, int] = (620, 380),
) -> None:
    width, height = canvas_size
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    def draw_line(line: Line, color: Tuple[int, int, int], thickness: int) -> None:
        x1, y1, x2, y2 = line
        cv2.line(canvas, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), color, thickness)

    for line in gt_lines:
        draw_line(line, (40, 170, 40), 3)

    for line in actual_lines:
        draw_line(line, (30, 90, 220), 2)

    for idx, report in enumerate(expected_reports, start=1):
        gt = report["gt_segment"]
        actual = report.get("closest_actual_segment")
        if actual:
            gx = int(round((gt["x1"] + gt["x2"]) / 2.0))
            gy = int(round((gt["y1"] + gt["y2"]) / 2.0))
            ax = int(round((actual["x1"] + actual["x2"]) / 2.0))
            ay = int(round((actual["y1"] + actual["y2"]) / 2.0))
            cv2.line(canvas, (gx, gy), (ax, ay), (160, 160, 160), 1)
            cv2.putText(canvas, str(idx), (gx + 4, gy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 0), 1, cv2.LINE_AA)
            cv2.putText(canvas, str(idx), (ax + 4, ay + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 30, 180), 1, cv2.LINE_AA)

    cv2.putText(canvas, f"{floor_name}: GT green, actual blue", (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 50), 1, cv2.LINE_AA)
    cv2.imwrite(str(output_path), canvas)


def analyze_floor(floor_name: str, gt_path: Path) -> Dict[str, Any]:
    gt = load_json_any_bom(gt_path)
    source_image = REPO_DIR / gt["source_image"]
    project_id = f"iwall-analysis-{floor_name}-{uuid4().hex[:8]}"
    result = process_floor_image(source_image, project_id=project_id, floor_index=1)

    expected_bbox = bbox_from_polygon(gt["polygon"])
    actual_bbox = bbox_from_polygon(result["polygon"])
    actual_final_lines = normalize_lines_to_bbox(result["inner_walls"], actual_bbox, expected_bbox)
    gt_lines: List[Line] = [tuple(map(float, line)) for line in gt["inner_walls"]]
    actual_lines: List[Line] = [tuple(map(float, line)) for line in actual_final_lines]

    expected_reports: List[Dict[str, Any]] = []
    summary_counts = {
        "matched": 0,
        "shifted": 0,
        "fragmented": 0,
        "missing": 0,
    }

    for gt_line in gt_lines:
        analysis = classify_expected_vs_actual(gt_line, actual_lines)
        if analysis.reason == "missing_candidate":
            summary_counts["missing"] += 1
        elif analysis.reason == "fragmented_wall":
            summary_counts["fragmented"] += 1
        elif analysis.reason in {"fixed_axis_offset_too_high", "span_overlap_too_low", "too_short", "too_long", "orientation_mismatch"}:
            summary_counts["shifted"] += 1
        else:
            if analysis.orientation_match and analysis.fixed_axis_offset is not None and analysis.fixed_axis_offset <= 16.0 and analysis.span_overlap_ratio >= 0.45:
                summary_counts["matched"] += 1
            else:
                summary_counts["shifted"] += 1

        same_gt_candidates = [
            line_to_dict(actual)
            for actual in actual_lines
            if axis_signature(actual)[0] == axis_signature(gt_line)[0]
            and line_perpendicular_distance(gt_line, actual) <= 28.0
            and line_overlap_ratio(gt_line, actual) >= 0.18
        ]
        expected_reports.append(
            {
                "gt_segment": line_to_dict(analysis.gt_line),
                "closest_actual_segment": line_to_dict(analysis.closest_actual),
                "orientation_match": analysis.orientation_match,
                "fixed_axis_offset": analysis.fixed_axis_offset,
                "span_overlap_ratio": analysis.span_overlap_ratio,
                "actual_vs_gt_length": analysis.actual_vs_gt_length,
                "actual_fragmented": analysis.fragment_count_near_gt >= 2,
                "nearby_actual_segments_for_same_gt": same_gt_candidates,
                "match_failure_reason": analysis.reason,
            }
        )

    actual_reports: List[Dict[str, Any]] = []
    extra_false_positive = 0
    for actual_line in actual_lines:
        classification, meta = classify_actual_line(actual_line, gt_lines)
        if classification == "extra_false_positive":
            extra_false_positive += 1
        actual_reports.append(
            {
                "actual_segment": line_to_dict(actual_line),
                "classification": classification,
                **meta,
            }
        )

    out_dir = DEBUG_COMPARE_DIR / floor_name
    out_dir.mkdir(parents=True, exist_ok=True)
    render_overlay(
        floor_name=floor_name,
        gt_lines=gt_lines,
        actual_lines=actual_lines,
        expected_reports=expected_reports,
        output_path=out_dir / "inner_wall_gt_vs_actual.png",
    )

    report = {
        "floor_name": floor_name,
        "expected_wall_count": len(gt_lines),
        "actual_wall_count": len(actual_lines),
        "expected_wall_analysis": expected_reports,
        "actual_wall_analysis": actual_reports,
        "summary": {
            "matched": summary_counts["matched"],
            "shifted": summary_counts["shifted"],
            "fragmented": summary_counts["fragmented"],
            "missing": summary_counts["missing"],
            "extra_false_positive": extra_false_positive,
        },
    }
    (out_dir / "inner_wall_error_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    debug_dir = BACKEND_DIR / "debug" / project_id
    if debug_dir.exists():
        shutil.rmtree(debug_dir, ignore_errors=True)

    return report


def main() -> None:
    gt_map = {
        "floor_1_clean": BACKEND_DIR / "ground_truth" / "floor_1_clean.json",
        "floor_2_clean": BACKEND_DIR / "debug" / "compare_tmp" / "floor_2_clean.json",
        "floor_3_clean": BACKEND_DIR / "debug" / "compare_tmp" / "floor_3_clean.json",
    }

    reports = [analyze_floor(floor_name, gt_path) for floor_name, gt_path in gt_map.items()]
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
