from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _axis_aligned_segment_count(lines: List[List[int]], tol: int = 6) -> int:
    count = 0
    for line in lines:
        if len(line) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in line]
        if abs(y1 - y2) <= tol or abs(x1 - x2) <= tol:
            count += 1
    return count


def _polygon_axis_aligned_ratio(polygon: List[List[int]], tol: int = 8) -> float:
    if len(polygon) < 3:
        return 0.0
    axis_aligned = 0
    total = 0
    for index in range(len(polygon)):
        x1, y1 = polygon[index]
        x2, y2 = polygon[(index + 1) % len(polygon)]
        total += 1
        if abs(int(y1) - int(y2)) <= tol or abs(int(x1) - int(x2)) <= tol:
            axis_aligned += 1
    return axis_aligned / max(1, total)


def _polygon_bbox(polygon: List[List[int]]) -> Tuple[int, int, int, int]:
    xs = [int(point[0]) for point in polygon]
    ys = [int(point[1]) for point in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _normalize_line(line: List[int], tol: int = 6) -> Dict[str, int] | None:
    if len(line) != 4:
        return None
    x1, y1, x2, y2 = [int(v) for v in line]
    if abs(y1 - y2) <= tol:
        fixed = int(round((y1 + y2) / 2))
        return {
            "orientation": "h",
            "fixed": fixed,
            "start": int(min(x1, x2)),
            "end": int(max(x1, x2)),
        }
    if abs(x1 - x2) <= tol:
        fixed = int(round((x1 + x2) / 2))
        return {
            "orientation": "v",
            "fixed": fixed,
            "start": int(min(y1, y2)),
            "end": int(max(y1, y2)),
        }
    return None


def _segment_midpoint_inside_bbox(
    info: Dict[str, int],
    bbox: Tuple[int, int, int, int],
    margin: int = 10,
) -> bool:
    min_x, min_y, max_x, max_y = bbox
    min_x -= margin
    min_y -= margin
    max_x += margin
    max_y += margin
    if info["orientation"] == "h":
        mid_x = int(round((info["start"] + info["end"]) / 2))
        mid_y = int(info["fixed"])
    else:
        mid_x = int(info["fixed"])
        mid_y = int(round((info["start"] + info["end"]) / 2))
    return min_x <= mid_x <= max_x and min_y <= mid_y <= max_y


def _segment_overlaps_bbox(
    info: Dict[str, int],
    bbox: Tuple[int, int, int, int],
    tol: int = 10,
) -> bool:
    min_x, min_y, max_x, max_y = bbox
    if info["orientation"] == "h":
        if not (min_y - tol <= info["fixed"] <= max_y + tol):
            return False
        overlap_start = max(info["start"], min_x)
        overlap_end = min(info["end"], max_x)
        return overlap_end - overlap_start >= max(12, (info["end"] - info["start"]) * 0.5)
    if not (min_x - tol <= info["fixed"] <= max_x + tol):
        return False
    overlap_start = max(info["start"], min_y)
    overlap_end = min(info["end"], max_y)
    return overlap_end - overlap_start >= max(12, (info["end"] - info["start"]) * 0.5)


def _is_stair_like_segment(
    info: Dict[str, int],
    stair_regions: List[Tuple[int, int, int, int]],
) -> bool:
    if not stair_regions:
        return False
    length = info["end"] - info["start"]
    for bbox in stair_regions:
        if not _segment_midpoint_inside_bbox(info, bbox, margin=10):
            continue
        if not _segment_overlaps_bbox(info, bbox, tol=10):
            continue
        min_x, min_y, max_x, max_y = bbox
        bbox_span = max(max_x - min_x, max_y - min_y)
        if length <= bbox_span + 32:
            return True
    return False


def _filter_isolated_top_spurs(
    lines: List[List[int]],
    polygon_bbox: Tuple[int, int, int, int],
) -> Tuple[List[List[int]], int]:
    if not lines:
        return lines, 0
    min_x, min_y, max_x, max_y = polygon_bbox
    width = max_x - min_x
    height = max_y - min_y
    if width <= 0 or height <= 0:
        return lines, 0

    top_band = max(28, int(round(height * 0.11)))
    max_spur_len = max(96, int(round(height * 0.22)))
    x_margin = int(round(width * 0.22))
    tol = 10
    norm = [_normalize_line(line, tol=tol) for line in lines]

    def bottom_has_horizontal_connection(item: Dict[str, int]) -> bool:
        end_y = item["end"]
        x = item["fixed"]
        for other in norm:
            if other is None or other["orientation"] != "h":
                continue
            if abs(other["fixed"] - end_y) > tol:
                continue
            if other["start"] - tol <= x <= other["end"] + tol:
                return True
        return False

    def has_longer_vertical_neighbors(item: Dict[str, int]) -> bool:
        x = item["fixed"]
        length = item["end"] - item["start"]
        left_ok = False
        right_ok = False
        for other in norm:
            if other is None or other["orientation"] != "v":
                continue
            if other["fixed"] == x:
                continue
            other_len = other["end"] - other["start"]
            if other["start"] > min_y + top_band:
                continue
            if other_len < max(length * 1.9, int(round(height * 0.45))):
                continue
            if other["fixed"] < x:
                left_ok = True
            if other["fixed"] > x:
                right_ok = True
        return left_ok and right_ok

    kept: List[List[int]] = []
    removed = 0
    for line, item in zip(lines, norm):
        if item is None or item["orientation"] != "v":
            kept.append(line)
            continue
        length = item["end"] - item["start"]
        if (
            item["start"] <= min_y + top_band
            and length <= max_spur_len
            and (min_x + x_margin) <= item["fixed"] <= (max_x - x_margin)
            and not bottom_has_horizontal_connection(item)
            and has_longer_vertical_neighbors(item)
        ):
            removed += 1
            continue
        kept.append(line)
    return kept, removed


def filter_supported_clean_plan_isolated_top_spurs(
    lines: List[List[int]],
    polygon: List[List[int]],
) -> Tuple[List[List[int]], int]:
    if not lines or len(polygon) < 4:
        return [list(map(int, line)) for line in lines], 0
    return _filter_isolated_top_spurs(
        [list(map(int, line)) for line in lines],
        _polygon_bbox(polygon),
    )


def close_supported_clean_plan_micro_gaps(
    lines: List[List[int]],
    max_gap: int = 15,
    tol: int = 6,
) -> Tuple[List[List[int]], int]:
    if not lines:
        return [], 0

    infos: List[Dict[str, int] | None] = [_normalize_line(line, tol=tol) for line in lines]
    closed = 0

    for idx, info in enumerate(infos):
        if info is None or info["orientation"] != "h":
            continue
        best_left_gap = max_gap + 1
        best_left_x = None
        best_right_gap = max_gap + 1
        best_right_x = None
        for other in infos:
            if other is None or other["orientation"] != "v":
                continue
            if not (other["start"] - tol <= info["fixed"] <= other["end"] + tol):
                continue
            left_gap = abs(info["start"] - other["fixed"])
            right_gap = abs(info["end"] - other["fixed"])
            if 1 <= left_gap <= max_gap and left_gap < best_left_gap:
                best_left_gap = left_gap
                best_left_x = other["fixed"]
            if 1 <= right_gap <= max_gap and right_gap < best_right_gap:
                best_right_gap = right_gap
                best_right_x = other["fixed"]
        if best_left_x is not None and best_left_x != info["start"]:
            info["start"] = int(best_left_x)
            closed += 1
        if best_right_x is not None and best_right_x != info["end"]:
            info["end"] = int(best_right_x)
            closed += 1

    for idx, info in enumerate(infos):
        if info is None or info["orientation"] != "v":
            continue
        best_top_gap = max_gap + 1
        best_top_y = None
        best_bottom_gap = max_gap + 1
        best_bottom_y = None
        for other in infos:
            if other is None or other["orientation"] != "h":
                continue
            if not (other["start"] - tol <= info["fixed"] <= other["end"] + tol):
                continue
            top_gap = abs(info["start"] - other["fixed"])
            bottom_gap = abs(info["end"] - other["fixed"])
            if 1 <= top_gap <= max_gap and top_gap < best_top_gap:
                best_top_gap = top_gap
                best_top_y = other["fixed"]
            if 1 <= bottom_gap <= max_gap and bottom_gap < best_bottom_gap:
                best_bottom_gap = bottom_gap
                best_bottom_y = other["fixed"]
        if best_top_y is not None and best_top_y != info["start"]:
            info["start"] = int(best_top_y)
            closed += 1
        if best_bottom_y is not None and best_bottom_y != info["end"]:
            info["end"] = int(best_bottom_y)
            closed += 1

    normalized_lines: List[List[int]] = []
    for line, info in zip(lines, infos):
        if info is None:
            normalized_lines.append([int(v) for v in line])
            continue
        if info["orientation"] == "h":
            normalized_lines.append([int(info["start"]), int(info["fixed"]), int(info["end"]), int(info["fixed"])])
        else:
            normalized_lines.append([int(info["fixed"]), int(info["start"]), int(info["fixed"]), int(info["end"])])
    return normalized_lines, int(closed)


def normalize_supported_clean_plan_polygon(polygon: List[List[int]]) -> List[List[int]]:
    if len(polygon) < 3:
        return []
    xs = [int(point[0]) for point in polygon]
    ys = [int(point[1]) for point in polygon]
    min_x = int(min(xs))
    max_x = int(max(xs))
    min_y = int(min(ys))
    max_y = int(max(ys))
    if max_x - min_x < 10 or max_y - min_y < 10:
        return [list(map(int, point)) for point in polygon]
    return [
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y],
    ]


def normalize_supported_clean_plan_inner_walls(
    inner_walls: List[List[int]],
    polygon: List[List[int]],
    stair_regions: List[Tuple[int, int, int, int]] | None = None,
    axis_tol: int = 8,
    snap_tol: int = 16,
    outer_snap_tol: int = 28,
    min_length: int = 24,
    merge_gap: int = 14,
) -> Dict[str, Any]:
    if not inner_walls:
        return {
            "lines": [],
            "endpoint_snap_count": 0,
            "stair_filtered_count": 0,
        }
    if len(polygon) >= 4:
        xs = [int(point[0]) for point in polygon]
        ys = [int(point[1]) for point in polygon]
        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)
    else:
        min_x = min(min(int(line[0]), int(line[2])) for line in inner_walls)
        max_x = max(max(int(line[0]), int(line[2])) for line in inner_walls)
        min_y = min(min(int(line[1]), int(line[3])) for line in inner_walls)
        max_y = max(max(int(line[1]), int(line[3])) for line in inner_walls)

    normalized = []
    h_groups: List[List[Dict[str, int]]] = []
    v_groups: List[List[Dict[str, int]]] = []
    candidates: List[Dict[str, int]] = []
    stair_filtered_count = 0
    endpoint_snap_count = 0
    stair_regions = stair_regions or []

    for line in inner_walls:
        info = _normalize_line(line, tol=axis_tol)
        if info is None:
            continue
        if info["end"] - info["start"] < min_length:
            continue
        if _is_stair_like_segment(info, stair_regions):
            stair_filtered_count += 1
            continue
        candidates.append(info)

    for orientation in ("h", "v"):
        group_store = h_groups if orientation == "h" else v_groups
        values = sorted([item for item in candidates if item["orientation"] == orientation], key=lambda item: item["fixed"])
        for item in values:
            if not group_store:
                group_store.append([item])
                continue
            last_group = group_store[-1]
            if abs(last_group[-1]["fixed"] - item["fixed"]) <= snap_tol:
                last_group.append(item)
            else:
                group_store.append([item])

    def _merge_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if not spans:
            return []
        spans = sorted(spans)
        merged: List[List[int]] = [[spans[0][0], spans[0][1]]]
        for start, end in spans[1:]:
            current = merged[-1]
            if start <= current[1] + merge_gap:
                current[1] = max(current[1], end)
            else:
                merged.append([start, end])
        return [(int(start), int(end)) for start, end in merged]

    for group in h_groups:
        fixed = int(round(sum(item["fixed"] for item in group) / max(1, len(group))))
        spans = []
        for item in group:
            start = max(min_x, item["start"])
            end = min(max_x, item["end"])
            if end - start >= min_length:
                spans.append((int(start), int(end)))
        for start, end in _merge_spans(spans):
            if end - start >= min_length:
                if abs(start - min_x) <= outer_snap_tol:
                    start = min_x
                    endpoint_snap_count += 1
                if abs(end - max_x) <= outer_snap_tol:
                    end = max_x
                    endpoint_snap_count += 1
                normalized.append([int(start), int(fixed), int(end), int(fixed)])

    for group in v_groups:
        fixed = int(round(sum(item["fixed"] for item in group) / max(1, len(group))))
        spans = []
        for item in group:
            start = max(min_y, item["start"])
            end = min(max_y, item["end"])
            if end - start >= min_length:
                spans.append((int(start), int(end)))
        for start, end in _merge_spans(spans):
            if end - start >= min_length:
                if abs(start - min_y) <= outer_snap_tol:
                    start = min_y
                    endpoint_snap_count += 1
                if abs(end - max_y) <= outer_snap_tol:
                    end = max_y
                    endpoint_snap_count += 1
                normalized.append([int(fixed), int(start), int(fixed), int(end)])

    deduped: List[List[int]] = []
    seen = set()
    for line in normalized:
        key = tuple(int(v) for v in line)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(list(key))
    deduped, isolated_top_spur_filtered_count = _filter_isolated_top_spurs(
        deduped,
        (min_x, min_y, max_x, max_y),
    )
    return {
        "lines": deduped,
        "endpoint_snap_count": endpoint_snap_count,
        "stair_filtered_count": stair_filtered_count,
        "isolated_top_spur_filtered_count": isolated_top_spur_filtered_count,
    }


def analyze_supported_clean_plan_input(
    polygon: List[List[int]],
    inner_walls: List[List[int]],
    doors: List[Dict[str, int]],
    windows: List[Dict[str, int]],
    rooms: List[Dict[str, int]],
    symbolic_cluster_debug: List[Dict[str, Any]],
) -> Dict[str, Any]:
    normalized_polygon = normalize_supported_clean_plan_polygon(polygon)
    stair_regions = []
    for cluster in symbolic_cluster_debug:
        if not (cluster.get("accepted") and cluster.get("kind") == "stair_like_parallel_cluster"):
            continue
        bbox = cluster.get("bbox") or []
        if len(bbox) != 4:
            continue
        x, y, w, h = [int(v) for v in bbox]
        stair_regions.append((x, y, x + w, y + h))
    normalized_wall_result = normalize_supported_clean_plan_inner_walls(
        inner_walls,
        normalized_polygon or polygon,
        stair_regions=stair_regions,
    )
    normalized_inner_walls = normalized_wall_result["lines"]
    polygon_ratio = _polygon_axis_aligned_ratio(normalized_polygon or polygon)
    wall_ratio = _axis_aligned_segment_count(normalized_inner_walls or inner_walls) / max(1, len(normalized_inner_walls or inner_walls))
    accepted_symbolic_clusters = [
        cluster
        for cluster in symbolic_cluster_debug
        if cluster.get("accepted") and cluster.get("kind") == "stair_like_parallel_cluster"
    ]

    clean_like = (
        len(polygon) >= 4
        and 4 <= len(inner_walls) <= 16
        and polygon_ratio >= 0.95
        and wall_ratio >= 0.9
    )

    reasons: List[str] = []
    if polygon_ratio < 0.95:
        reasons.append("polygon_not_orthogonal")
    if wall_ratio < 0.9:
        reasons.append("inner_walls_not_orthogonal_enough")
    if len(inner_walls) < 4:
        reasons.append("too_few_inner_walls")
    if len(inner_walls) > 16:
        reasons.append("too_many_inner_walls")

    score = 0.0
    score += polygon_ratio * 0.35
    score += wall_ratio * 0.35
    score += min(1.0, len(rooms) / 4.0) * 0.1
    score += min(1.0, len(windows) / 4.0) * 0.1
    score += min(1.0, len(doors) / 3.0) * 0.05
    score += min(1.0, len(accepted_symbolic_clusters) / 1.0) * 0.05

    return {
        "enabled": True,
        "eligible": clean_like,
        "score": round(float(score), 4),
        "raw_polygon_axis_ratio": round(float(_polygon_axis_aligned_ratio(polygon)), 4),
        "polygon_axis_ratio": round(float(polygon_ratio), 4),
        "normalized_polygon_point_count": len(normalized_polygon or polygon),
        "normalized_inner_wall_count": len(normalized_inner_walls or inner_walls),
        "inner_wall_axis_ratio": round(float(wall_ratio), 4),
        "endpoint_snap_count": int(normalized_wall_result["endpoint_snap_count"]),
        "stair_filtered_segment_count": int(normalized_wall_result["stair_filtered_count"]),
        "isolated_top_spur_filtered_count": int(normalized_wall_result["isolated_top_spur_filtered_count"]),
        "symbolic_stair_cluster_count": len(accepted_symbolic_clusters),
        "reasons": reasons,
    }


def build_supported_clean_plan_stairs(
    symbolic_cluster_debug: List[Dict[str, Any]],
    max_count: int = 2,
) -> List[Dict[str, Any]]:
    stairs: List[Dict[str, Any]] = []
    for cluster in symbolic_cluster_debug:
        if not cluster.get("accepted"):
            continue
        if cluster.get("kind") != "stair_like_parallel_cluster":
            continue
        bbox = cluster.get("bbox") or []
        if len(bbox) != 4:
            continue
        x, y, w, h = [int(v) for v in bbox]
        if w <= 0 or h <= 0:
            continue
        orientation = str(cluster.get("orientation", "h"))
        stairs.append(
            {
                "id": f"upload-stair-{len(stairs) + 1}",
                "bounds": [x, y, x + w, y + h],
                "direction": "down",
                "steps": int(max(3, min(8, cluster.get("line_count", 5) or 5))),
                "orientation_hint": orientation,
            }
        )
        if len(stairs) >= max_count:
            break
    return stairs


def build_supported_clean_plan_candidate(
    *,
    floor_index: int,
    polygon: List[List[int]],
    inner_walls: List[List[int]],
    doors: List[Dict[str, int]],
    windows: List[Dict[str, int]],
    rooms: List[Dict[str, int]],
    stairs: List[Dict[str, Any]],
    contract_meta: Dict[str, Any],
) -> Dict[str, Any]:
    normalized_polygon = normalize_supported_clean_plan_polygon(polygon)
    stair_regions = []
    for stair in stairs:
        bounds = stair.get("bounds") or []
        if len(bounds) != 4:
            continue
        stair_regions.append(tuple(int(v) for v in bounds))
    normalized_wall_result = normalize_supported_clean_plan_inner_walls(
        inner_walls,
        normalized_polygon or polygon,
        stair_regions=stair_regions,
    )
    normalized_inner_walls = normalized_wall_result["lines"]
    candidate_contract_meta = dict(contract_meta)
    candidate_contract_meta["endpoint_snap_count"] = int(normalized_wall_result["endpoint_snap_count"])
    candidate_contract_meta["stair_filtered_segment_count"] = int(normalized_wall_result["stair_filtered_count"])
    candidate_contract_meta["isolated_top_spur_filtered_count"] = int(normalized_wall_result["isolated_top_spur_filtered_count"])
    return {
        "floor_index": floor_index,
        "polygon": [list(map(int, point)) for point in (normalized_polygon or polygon)],
        "inner_walls": [[int(v) for v in line] for line in (normalized_inner_walls or inner_walls)],
        "doors": [{"x": int(item["x"]), "y": int(item["y"]), "width": int(item.get("width", 0))} for item in doors],
        "windows": [{"x": int(item["x"]), "y": int(item["y"]), "width": int(item.get("width", 0))} for item in windows],
        "rooms": [
            {
                "id": int(room.get("id", index + 1)),
                "x": int(room["x"]),
                "y": int(room["y"]),
            }
            for index, room in enumerate(rooms)
            if "x" in room and "y" in room
        ],
        "stairs": stairs,
        "contract_meta": candidate_contract_meta,
    }


def evaluate_supported_clean_plan_candidate(
    *,
    current_polygon: List[List[int]],
    current_inner_walls: List[List[int]],
    current_doors: List[Dict[str, int]],
    current_windows: List[Dict[str, int]],
    current_rooms: List[Dict[str, int]],
    candidate: Dict[str, Any],
    contract_meta: Dict[str, Any],
) -> Dict[str, Any]:
    if not contract_meta.get("eligible"):
        return {"select": False, "reason": "candidate_not_eligible"}

    candidate_polygon = candidate.get("polygon", []) or []
    candidate_inner_walls = candidate.get("inner_walls", []) or []
    candidate_doors = candidate.get("doors", []) or []
    candidate_windows = candidate.get("windows", []) or []
    candidate_rooms = candidate.get("rooms", []) or []

    if len(candidate_polygon) < 4:
        return {"select": False, "reason": "candidate_polygon_too_small"}
    if len(candidate_inner_walls) < 4:
        return {"select": False, "reason": "candidate_too_few_inner_walls"}
    if len(candidate_doors) < len(current_doors):
        return {"select": False, "reason": "candidate_doors_worse"}
    if len(candidate_windows) < len(current_windows):
        return {"select": False, "reason": "candidate_windows_worse"}
    if len(candidate_rooms) < len(current_rooms):
        return {"select": False, "reason": "candidate_rooms_worse"}
    if len(candidate_inner_walls) > len(current_inner_walls) + 1:
        return {"select": False, "reason": "candidate_too_many_inner_walls"}

    current_bbox = _polygon_bbox(current_polygon)
    candidate_bbox = _polygon_bbox(candidate_polygon)
    current_area = max(1, (current_bbox[2] - current_bbox[0]) * (current_bbox[3] - current_bbox[1]))
    candidate_area = max(1, (candidate_bbox[2] - candidate_bbox[0]) * (candidate_bbox[3] - candidate_bbox[1]))
    bbox_area_ratio = candidate_area / current_area
    if bbox_area_ratio < 0.88 or bbox_area_ratio > 1.12:
        return {"select": False, "reason": "candidate_bbox_changed_too_much", "bbox_area_ratio": round(float(bbox_area_ratio), 4)}

    current_polygon_ratio = _polygon_axis_aligned_ratio(current_polygon)
    candidate_polygon_ratio = _polygon_axis_aligned_ratio(candidate_polygon)
    polygon_improved = candidate_polygon_ratio >= current_polygon_ratio + 0.2
    polygon_simplified = len(candidate_polygon) <= max(4, len(current_polygon) - 2)
    wall_stable = abs(len(candidate_inner_walls) - len(current_inner_walls)) <= 1

    if polygon_improved and polygon_simplified and wall_stable:
        return {
            "select": True,
            "reason": "polygon_normalized_and_walls_stable",
            "current_polygon_axis_ratio": round(float(current_polygon_ratio), 4),
            "candidate_polygon_axis_ratio": round(float(candidate_polygon_ratio), 4),
            "bbox_area_ratio": round(float(bbox_area_ratio), 4),
        }

    if (
        candidate_polygon_ratio >= 0.95
        and len(candidate_polygon) == 4
        and len(current_polygon) >= 6
        and wall_stable
        and bbox_area_ratio >= 0.9
        and bbox_area_ratio <= 1.1
    ):
        return {
            "select": True,
            "reason": "polygon_simplified_to_clean_envelope",
            "current_polygon_axis_ratio": round(float(current_polygon_ratio), 4),
            "candidate_polygon_axis_ratio": round(float(candidate_polygon_ratio), 4),
            "bbox_area_ratio": round(float(bbox_area_ratio), 4),
        }

    return {
        "select": False,
        "reason": "candidate_not_significantly_better",
        "current_polygon_axis_ratio": round(float(current_polygon_ratio), 4),
        "candidate_polygon_axis_ratio": round(float(candidate_polygon_ratio), 4),
        "bbox_area_ratio": round(float(bbox_area_ratio), 4),
    }
