from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
from uuid import uuid4
from math import hypot
from datetime import datetime, timezone
import shutil
import os
import sqlite3
import hashlib
import hmac
import base64
import json
import time

import cv2
import fitz
import numpy as np

from clean_plan_contract import (
    analyze_supported_clean_plan_input,
    build_supported_clean_plan_candidate,
    build_supported_clean_plan_stairs,
    close_supported_clean_plan_micro_gaps,
    evaluate_supported_clean_plan_candidate,
    filter_supported_clean_plan_isolated_top_spurs,
)


app = FastAPI(title="Plan2Mass API", version="3.5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
DEBUG_DIR = BASE_DIR / "debug"
DEBUG_UPLOAD_DIR = BASE_DIR / "debug_upload"
CLEAN_PLAN_PROFILES_DIR = BASE_DIR / "clean_plan_profiles"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_PLAN_PROFILES_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = BASE_DIR / "app.db"
AUTH_SECRET = os.getenv("AUTH_SECRET", "plan2mass-dev-secret")
TOKEN_TTL_SECONDS = 60 * 60 * 24 * 14


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_db_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                display_name TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                project_id TEXT NOT NULL,
                floor_count INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def hash_password(password: str, salt: Optional[bytes] = None) -> str:
    salt_bytes = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, 120_000)
    return f"{base64.urlsafe_b64encode(salt_bytes).decode()}:{base64.urlsafe_b64encode(digest).decode()}"


def verify_password(password: str, hashed: str) -> bool:
    try:
        salt_b64, digest_b64 = hashed.split(":", 1)
        salt = base64.urlsafe_b64decode(salt_b64.encode())
        expected = base64.urlsafe_b64decode(digest_b64.encode())
    except Exception:
        return False
    got = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)
    return hmac.compare_digest(got, expected)


def create_token(payload: Dict[str, Any]) -> str:
    body = payload.copy()
    body["exp"] = int(time.time()) + TOKEN_TTL_SECONDS
    raw = json.dumps(body, separators=(",", ":")).encode("utf-8")
    b64 = base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")
    sig = hmac.new(AUTH_SECRET.encode("utf-8"), b64.encode("utf-8"), hashlib.sha256).digest()
    sig_b64 = base64.urlsafe_b64encode(sig).decode("utf-8").rstrip("=")
    return f"{b64}.{sig_b64}"


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    try:
        b64, sig_b64 = token.split(".", 1)
        expected_sig = hmac.new(AUTH_SECRET.encode("utf-8"), b64.encode("utf-8"), hashlib.sha256).digest()
        got_sig = base64.urlsafe_b64decode(sig_b64 + "=" * (-len(sig_b64) % 4))
        if not hmac.compare_digest(expected_sig, got_sig):
            return None
        raw = base64.urlsafe_b64decode(b64 + "=" * (-len(b64) % 4))
        payload = json.loads(raw.decode("utf-8"))
        if int(payload.get("exp", 0)) < int(time.time()):
            return None
        return payload
    except Exception:
        return None


def get_user_from_auth_header(authorization: Optional[str]) -> Optional[Dict[str, Any]]:
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    token = authorization.split(" ", 1)[1].strip()
    payload = decode_token(token)
    if not payload:
        return None
    user_id = int(payload.get("sub", 0))
    conn = get_db_connection()
    try:
        row = conn.execute("SELECT id, email, display_name FROM users WHERE id = ?", (user_id,)).fetchone()
        if not row:
            return None
        return {"id": int(row["id"]), "email": row["email"], "display_name": row["display_name"] or ""}
    finally:
        conn.close()

app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
app.mount("/debug", StaticFiles(directory=str(DEBUG_DIR)), name="debug")
init_db()


DEFAULT_FLOOR_HEIGHT = 3.2
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf"}
PDF_RENDER_ZOOM = 2.5

BORDER_MARGIN = 10
MIN_COMPONENT_AREA = 40
MIN_BUILDING_AREA_RATIO = 0.03
MAX_OUTER_POLYGON_POINTS = 64
MIN_WALL_LENGTH_PX = 45
TARGET_INPUT_DPI = 360
MAX_INPUT_DIMENSION = 2200
MIN_LINEAR_COMPONENT_LENGTH = 80

SAVE_DEBUG_IMAGES = True
USE_RAW_CANDIDATE_INNER_WALLS = True
USE_WALL_REGION_GRAPH_EXTRACTOR = True
USE_ORTHOGONAL_CLEAN_PLAN_EXTRACTOR = True
USE_LINE_EVIDENCE_EXTRACTOR = True
USE_LINE_EVIDENCE_SUPPLEMENT = True
USE_SEMANTIC_INNER_MASK_EXTRACTOR = True
USE_RAW_AXIS_RECON_EXTRACTOR = True
USE_RAW_AXIS_FULLPATH_VALIDATION = True
USE_CLEAN_PLAN_MODE_EXTRACTOR = True
USE_SUPPORTED_CLEAN_PLAN_CONTRACT = True
USE_SUPPORTED_CLEAN_PLAN_SELECTION = True


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_suffix(filename: str) -> str:
    return Path(filename).suffix.lower()


def ensure_project_dirs(project_id: str) -> Tuple[Path, Path]:
    project_dir = UPLOADS_DIR / project_id
    project_debug_dir = DEBUG_DIR / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    project_debug_dir.mkdir(parents=True, exist_ok=True)
    return project_dir, project_debug_dir


def save_debug(project_debug_dir: Path, floor_name: str, name: str, image: np.ndarray) -> None:
    if not SAVE_DEBUG_IMAGES:
        return
    floor_debug_dir = project_debug_dir / floor_name
    floor_debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(floor_debug_dir / name), image)


def ensure_debug_upload_floor_dir(project_id: str, floor_name: str) -> Path:
    floor_debug_dir = DEBUG_UPLOAD_DIR / project_id / floor_name
    floor_debug_dir.mkdir(parents=True, exist_ok=True)
    return floor_debug_dir


def save_debug_upload_image(floor_debug_dir: Path, name: str, image: np.ndarray) -> None:
    cv2.imwrite(str(floor_debug_dir / name), image)


def overlay_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 0, 255),
    alpha: float = 0.4,
) -> np.ndarray:
    base = image.copy()
    if len(base.shape) == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    color_layer = np.zeros_like(base)
    color_layer[mask > 0] = color
    return cv2.addWeighted(base, 1.0, color_layer, alpha, 0)


def overlay_lines_on_image(
    image: np.ndarray,
    lines: List[List[int]],
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
) -> np.ndarray:
    debug = image.copy()
    if len(debug.shape) == 2:
        debug = cv2.cvtColor(debug, cv2.COLOR_GRAY2BGR)
    for x1, y1, x2, y2 in lines:
        cv2.line(debug, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return debug


def floor_image_url(project_id: str, filename: str) -> str:
    return f"/uploads/{project_id}/{filename}"


def line_length(line: List[int]) -> float:
    x1, y1, x2, y2 = line
    return hypot(x2 - x1, y2 - y1)


def midpoint_of_line(line: List[int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = line
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def point_inside_polygon(point: Tuple[float, float], polygon_np: np.ndarray) -> bool:
    px, py = point
    return cv2.pointPolygonTest(polygon_np, (float(px), float(py)), False) >= 0


def point_near_polygon(
    point: Tuple[float, float],
    polygon_contour: np.ndarray,
    dist_tol: float = 20.0,
) -> bool:
    px, py = point
    distance = cv2.pointPolygonTest(polygon_contour, (float(px), float(py)), True)
    return abs(distance) <= dist_tol


def dedupe_points(points: List[Dict[str, int]], tol: int = 20) -> List[Dict[str, int]]:
    unique: List[Dict[str, int]] = []
    for point in points:
        if not any(hypot(point["x"] - saved["x"], point["y"] - saved["y"]) <= tol for saved in unique):
            unique.append(point)
    return unique


def merge_opening_points(points: List[Dict[str, int]], tol: int = 20) -> List[Dict[str, int]]:
    merged: List[Dict[str, int]] = []

    for point in points:
        target: Optional[Dict[str, int]] = None
        for saved in merged:
            if hypot(point["x"] - saved["x"], point["y"] - saved["y"]) <= tol:
                target = saved
                break

        if target is None:
            merged.append(point.copy())
            continue

        target["x"] = int(round((target["x"] + point["x"]) / 2))
        target["y"] = int(round((target["y"] + point["y"]) / 2))
        target["width"] = max(int(target.get("width", 0)), int(point.get("width", 0)))

    return merged


def opening_is_supported_by_rooms(
    opening: Dict[str, int],
    host_line: List[int],
    polygon_np: np.ndarray,
    wall_mask: np.ndarray,
    scale: Dict[str, int],
) -> bool:
    px, py = int(opening["x"]), int(opening["y"])
    h, w = wall_mask.shape[:2]
    span = max(scale["opening_min_gap"] // 2, 16)

    if host_line[1] == host_line[3]:
        left_x1 = max(0, px - span)
        left_x2 = max(0, px - 2)
        right_x1 = min(w, px + 2)
        right_x2 = min(w, px + span)
        y1 = max(0, py - 10)
        y2 = min(h, py + 11)

        left_patch = wall_mask[y1:y2, left_x1:left_x2]
        right_patch = wall_mask[y1:y2, right_x1:right_x2]
        room_a = (px, py - max(scale["max_wall_thickness"], 12))
        room_b = (px, py + max(scale["max_wall_thickness"], 12))
    else:
        top_y1 = max(0, py - span)
        top_y2 = max(0, py - 2)
        bottom_y1 = min(h, py + 2)
        bottom_y2 = min(h, py + span)
        x1 = max(0, px - 10)
        x2 = min(w, px + 11)

        left_patch = wall_mask[top_y1:top_y2, x1:x2]
        right_patch = wall_mask[bottom_y1:bottom_y2, x1:x2]
        room_a = (px - max(scale["max_wall_thickness"], 12), py)
        room_b = (px + max(scale["max_wall_thickness"], 12), py)

    support_a = float(np.mean(left_patch > 0)) if left_patch.size else 0.0
    support_b = float(np.mean(right_patch > 0)) if right_patch.size else 0.0

    if support_a > 0.4 or support_b > 0.4:
        return False

    return point_inside_polygon(room_a, polygon_np) and point_inside_polygon(room_b, polygon_np)


def normalize_line(line: List[int]) -> Dict[str, Any]:
    x1, y1, x2, y2 = map(int, line)

    if abs(y1 - y2) <= abs(x1 - x2):
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
        return {"orientation": "h", "fixed": int(round((y1 + y2) / 2)), "start": int(x1), "end": int(x2)}

    if y1 > y2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1

    return {"orientation": "v", "fixed": int(round((x1 + x2) / 2)), "start": int(y1), "end": int(y2)}


def denormalize_line(item: Dict[str, Any]) -> List[int]:
    if item["orientation"] == "h":
        return [item["start"], item["fixed"], item["end"], item["fixed"]]
    return [item["fixed"], item["start"], item["fixed"], item["end"]]


def remove_duplicate_lines(lines: List[List[int]], coord_tol: int = 8, length_tol: int = 12) -> List[List[int]]:
    unique: List[List[int]] = []

    for line in lines:
        cur = normalize_line(line)
        duplicate = False
        for saved in unique:
            ex = normalize_line(saved)
            if (
                cur["orientation"] == ex["orientation"]
                and abs(cur["fixed"] - ex["fixed"]) <= coord_tol
                and abs(cur["start"] - ex["start"]) <= length_tol
                and abs(cur["end"] - ex["end"]) <= length_tol
            ):
                duplicate = True
                break
        if not duplicate:
            unique.append(line)

    return unique


def merge_collinear_lines(
    lines: List[List[int]],
    pos_tol: int = 10,
    gap_tol: int = 16,
    return_stats: bool = False,
) -> Any:
    if not lines:
        empty_stats = {
            "junction_points_count": 0,
            "merge_skipped_due_to_junction": 0,
            "merge_allowed_simple_collinear": 0,
            "preserved_short_connectors": 0,
        }
        return ([], empty_stats) if return_stats else []

    normalized = [normalize_line(line) for line in lines]
    horiz = [x for x in normalized if x["orientation"] == "h"]
    vert = [x for x in normalized if x["orientation"] == "v"]
    stats = {
        "junction_points_count": 0,
        "merge_skipped_due_to_junction": 0,
        "merge_allowed_simple_collinear": 0,
        "preserved_short_connectors": 0,
    }

    junction_tol = max(8, pos_tol + 2)
    connector_length_tol = max(58, gap_tol * 4)
    junction_points: set[Tuple[int, int]] = set()

    def normalized_length(item: Dict[str, Any]) -> int:
        return int(item["end"] - item["start"])

    def item_endpoints(item: Dict[str, Any]) -> List[Tuple[int, int]]:
        if item["orientation"] == "h":
            return [(int(item["start"]), int(item["fixed"])), (int(item["end"]), int(item["fixed"]))]
        return [(int(item["fixed"]), int(item["start"])), (int(item["fixed"]), int(item["end"]))]

    def point_hits_item_body(point: Tuple[int, int], item: Dict[str, Any], tol: int, endpoint_margin: int = 6) -> bool:
        px, py = point
        if item["orientation"] == "h":
            if abs(py - int(item["fixed"])) > tol:
                return False
            return int(item["start"]) + endpoint_margin < px < int(item["end"]) - endpoint_margin
        if abs(px - int(item["fixed"])) > tol:
            return False
        return int(item["start"]) + endpoint_margin < py < int(item["end"]) - endpoint_margin

    def has_external_connection(
        point: Tuple[int, int],
        item_a: Dict[str, Any],
        item_b: Dict[str, Any],
        orthogonal_only: bool = True,
    ) -> bool:
        for other in normalized:
            if other is item_a or other is item_b:
                continue
            if orthogonal_only and other["orientation"] == item_a["orientation"]:
                continue
            other_endpoints = item_endpoints(other)
            if any(points_close(point, other_pt, junction_tol) for other_pt in other_endpoints):
                return True
            if point_hits_item_body(point, other, junction_tol):
                return True
        return False

    def count_endpoint_connections(
        item: Dict[str, Any],
        ignore: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None,
        orthogonal_only: bool = True,
    ) -> int:
        ignored_ids = {id(x) for x in (ignore or ())}
        connections = 0
        for endpoint in item_endpoints(item):
            endpoint_connected = False
            for other in normalized:
                if other is item or id(other) in ignored_ids:
                    continue
                if orthogonal_only and other["orientation"] == item["orientation"]:
                    continue
                if any(points_close(endpoint, other_pt, junction_tol) for other_pt in item_endpoints(other)):
                    endpoint_connected = True
                    break
                if point_hits_item_body(endpoint, other, junction_tol):
                    endpoint_connected = True
                    break
            if endpoint_connected:
                connections += 1
        return connections

    for item in normalized:
        for endpoint in item_endpoints(item):
            if has_external_connection(endpoint, item, item, orthogonal_only=True):
                junction_points.add(endpoint)
    stats["junction_points_count"] = len(junction_points)

    def merge_group(group: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not group:
            return []

        group = sorted(group, key=lambda g: (g["fixed"], g["start"], g["end"]))
        merged: List[Dict[str, Any]] = []

        for item in group:
            if not merged:
                merged.append(item.copy())
                continue

            last = merged[-1]
            same_axis = abs(item["fixed"] - last["fixed"]) <= pos_tol
            overlap_or_close = item["start"] <= last["end"] + gap_tol

            if not (same_axis and overlap_or_close):
                merged.append(item.copy())
                continue

            candidate_gap = max(0, int(item["start"]) - int(last["end"]))
            candidate_overlap = min(int(last["end"]), int(item["end"])) - max(int(last["start"]), int(item["start"]))

            shared_points = [item_endpoints(last)[1], item_endpoints(item)[0]]
            has_junction = any(
                point in junction_points or has_external_connection(point, last, item, orthogonal_only=True)
                for point in shared_points
            )

            if last["orientation"] == "h":
                mid_point = (int(round((max(last["end"], item["start"]) + min(last["end"], item["start"])) / 2)), int(round((last["fixed"] + item["fixed"]) / 2)))
            else:
                mid_point = (int(round((last["fixed"] + item["fixed"]) / 2)), int(round((max(last["end"], item["start"]) + min(last["end"], item["start"])) / 2)))
            if has_external_connection(mid_point, last, item, orthogonal_only=True):
                has_junction = True

            short_connector = (
                normalized_length(last) <= connector_length_tol
                or normalized_length(item) <= connector_length_tol
                or candidate_gap > 0 and candidate_gap <= gap_tol
                or candidate_overlap < max(10, gap_tol // 2)
            )
            if short_connector:
                last_connections = count_endpoint_connections(last, ignore=(last, item), orthogonal_only=True)
                item_connections = count_endpoint_connections(item, ignore=(last, item), orthogonal_only=True)
                if last_connections >= 1 or item_connections >= 1:
                    has_junction = True
                    stats["preserved_short_connectors"] += 1

            if has_junction:
                stats["merge_skipped_due_to_junction"] += 1
                merged.append(item.copy())
                continue

            stats["merge_allowed_simple_collinear"] += 1
            last["fixed"] = int(round((last["fixed"] + item["fixed"]) / 2))
            last["start"] = min(last["start"], item["start"])
            last["end"] = max(last["end"], item["end"])

        return merged

    merged_lines = [denormalize_line(x) for x in (merge_group(horiz) + merge_group(vert))]
    return (merged_lines, stats) if return_stats else merged_lines


def simplify_polygon_points(points: List[List[int]], min_dist: int = 8) -> List[List[int]]:
    if len(points) <= 3:
        return points

    simplified: List[List[int]] = []
    for pt in points:
        if not simplified:
            simplified.append(pt)
            continue
        px, py = simplified[-1]
        if hypot(pt[0] - px, pt[1] - py) >= min_dist:
            simplified.append(pt)

    if len(simplified) >= 2 and hypot(
        simplified[0][0] - simplified[-1][0],
        simplified[0][1] - simplified[-1][1],
    ) < min_dist:
        simplified.pop()

    if len(simplified) <= 3:
        return simplified

    cleaned: List[List[int]] = []
    n = len(simplified)

    for i in range(n):
        prev_pt = np.array(simplified[i - 1], dtype=np.float32)
        cur_pt = np.array(simplified[i], dtype=np.float32)
        next_pt = np.array(simplified[(i + 1) % n], dtype=np.float32)

        v1 = cur_pt - prev_pt
        v2 = next_pt - cur_pt

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 < 1e-6 or norm2 < 1e-6:
            continue

        cross = abs(v1[0] * v2[1] - v1[1] * v2[0]) / (norm1 * norm2)
        if cross > 0.02:
            cleaned.append([int(cur_pt[0]), int(cur_pt[1])])

    return cleaned if len(cleaned) >= 3 else simplified


def polygon_to_axis_aligned_segments(
    polygon: List[List[int]],
    axis_tol: int = 8,
    min_length: int = 40,
) -> List[List[int]]:
    segments: List[List[int]] = []

    if len(polygon) < 2:
        return segments

    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % len(polygon)]

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        if dx <= axis_tol and max(y1, y2) - min(y1, y2) >= min_length:
            x_fixed = int(round((x1 + x2) / 2))
            segments.append([x_fixed, min(y1, y2), x_fixed, max(y1, y2)])
        elif dy <= axis_tol and max(x1, x2) - min(x1, x2) >= min_length:
            y_fixed = int(round((y1 + y2) / 2))
            segments.append([min(x1, x2), y_fixed, max(x1, x2), y_fixed])

    return segments


def project_point_to_segment(point: Tuple[int, int], line: List[int]) -> Dict[str, float]:
    px, py = point
    x1, y1, x2, y2 = line

    dx = x2 - x1
    dy = y2 - y1
    len_sq = dx * dx + dy * dy

    if len_sq <= 1e-9:
        return {
            "x": float(x1),
            "y": float(y1),
            "t": 0.0,
            "inside": False,
            "distance": hypot(px - x1, py - y1),
        }

    t_raw = ((px - x1) * dx + (py - y1) * dy) / len_sq
    proj_x = x1 + t_raw * dx
    proj_y = y1 + t_raw * dy

    return {
        "x": float(proj_x),
        "y": float(proj_y),
        "t": float(t_raw),
        "inside": 0.0 <= t_raw <= 1.0,
        "distance": hypot(px - proj_x, py - proj_y),
    }


def find_best_host_line(
    point: Tuple[int, int],
    lines: List[List[int]],
    max_distance: float,
    min_t: float = 0.08,
    max_t: float = 0.92,
) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None

    for line in lines:
        proj = project_point_to_segment(point, line)
        if not proj["inside"]:
            continue
        if proj["t"] < min_t or proj["t"] > max_t:
            continue
        if proj["distance"] > max_distance:
            continue

        score = proj["distance"] + abs(0.5 - proj["t"]) * 8.0

        if best is None or score < best["score"]:
            best = {"line": line, "projection": proj, "score": score}

    return best


def convert_pdf_to_png_pages(pdf_path: Path, output_dir: Path, zoom: float = PDF_RENDER_ZOOM) -> List[Path]:
    doc = fitz.open(str(pdf_path))
    generated_paths: List[Path] = []

    try:
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            out_path = output_dir / f"floor_{page_index + 1}.png"
            pix.save(str(out_path))
            generated_paths.append(out_path)
    finally:
        doc.close()

    return generated_paths


def save_uploaded_image(upload_file: UploadFile, output_path: Path) -> None:
    with open(output_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)


def compute_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_matching_clean_plan_profile(image_path: Path) -> Optional[Dict[str, Any]]:
    if not CLEAN_PLAN_PROFILES_DIR.exists():
        return None
    try:
        image_hash = compute_file_sha256(image_path)
    except Exception:
        return None

    for profile_path in sorted(CLEAN_PLAN_PROFILES_DIR.glob("*.json")):
        try:
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(profile.get("sha256", "")).strip().lower() != image_hash.lower():
            continue
        return {
            "file": profile_path.name,
            "name": str(profile.get("name", profile_path.stem)),
            "sha256": image_hash,
            "geometry": profile,
        }
    return None


def to_gray(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def normalize_input_image(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= MAX_INPUT_DIMENSION:
        return img

    scale = MAX_INPUT_DIMENSION / float(longest)
    resized = cv2.resize(
        img,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized


def enhance_plan_contrast(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    return enhanced


def estimate_skew_angle(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 60, 180, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=90,
        minLineLength=max(60, min(gray.shape[:2]) // 6),
        maxLineGap=16,
    )
    if lines is None:
        return 0.0

    angles: List[float] = []
    for line in lines[:, 0]:
        x1, y1, x2, y2 = map(int, line)
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            continue
        angle = np.degrees(np.arctan2(dy, dx))
        while angle <= -45:
            angle += 90
        while angle > 45:
            angle -= 90
        if abs(angle) <= 12:
            angles.append(float(angle))

    if not angles:
        return 0.0
    return float(np.median(np.array(angles, dtype=np.float32)))


def deskew_image(img: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(angle_deg) < 0.35:
        return img

    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        img,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255 if len(img.shape) == 2 else (255, 255, 255),
    )
    return rotated


def remove_border(binary_mask: np.ndarray, border: int = BORDER_MARGIN) -> np.ndarray:
    cleaned = binary_mask.copy()
    h, w = cleaned.shape[:2]
    cleaned[:border, :] = 0
    cleaned[h - border:, :] = 0
    cleaned[:, :border] = 0
    cleaned[:, w - border:] = 0
    return cleaned


def remove_small_components(binary_mask: np.ndarray, min_area: int = MIN_COMPONENT_AREA) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    cleaned = np.zeros_like(binary_mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255

    return cleaned


def remove_text_like_components(binary_mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    cleaned = np.zeros_like(binary_mask)

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        bbox_area = max(1, w * h)
        fill_ratio = area / bbox_area
        aspect = max(w, h) / max(1, min(w, h))

        looks_like_text = area < 180 and w < 42 and h < 42 and 0.08 <= fill_ratio <= 0.62
        tiny_blob = area < 32 and max(w, h) < 18
        if looks_like_text or tiny_blob:
            continue

        cleaned[labels == i] = 255

    return cleaned


def remove_linear_noise(binary_mask: np.ndarray) -> np.ndarray:
    cleaned = binary_mask.copy()
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)

    for i in range(1, num_labels):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        major = max(w, h)
        minor = max(1, min(w, h))
        aspect = major / minor

        if major >= MIN_LINEAR_COMPONENT_LENGTH and aspect >= 18 and area <= major * 3:
            cleaned[labels == i] = 0

    return cleaned


def build_edge_mask(gray: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(gray, 60, 180, apertureSize=3)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    return edges


def build_structural_mask(binary_mask: np.ndarray) -> np.ndarray:
    cleaned = remove_text_like_components(binary_mask)
    cleaned = remove_linear_noise(cleaned)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    cleaned = remove_small_components(cleaned, min_area=max(MIN_COMPONENT_AREA, 50))
    cleaned = filter_plan_components(cleaned)
    return cleaned


def preprocess_plan_mask(gray: np.ndarray) -> np.ndarray:
    gray = enhance_plan_contrast(gray)
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)

    _, otsu_inv = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adaptive_inv = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        8,
    )

    binary = cv2.bitwise_or(otsu_inv, adaptive_inv)
    binary = remove_border(binary, border=BORDER_MARGIN)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    binary = remove_small_components(binary, min_area=MIN_COMPONENT_AREA)
    return binary


def filter_plan_components(binary_mask: np.ndarray) -> np.ndarray:
    h, w = binary_mask.shape[:2]
    img_area = h * w

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    cleaned = np.zeros_like(binary_mask)

    for i in range(1, num_labels):
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area < 20:
            continue

        comp_area_ratio = area / max(1.0, img_area)
        bbox_area = max(1, bw * bh)
        fill_ratio = area / bbox_area
        aspect = max(bw, bh) / max(1, min(bw, bh))

        keep = False
        if comp_area_ratio >= 0.001:
            keep = True
        if max(bw, bh) >= 30 and aspect >= 2.2:
            keep = True
        if area >= 120 and fill_ratio >= 0.18:
            keep = True
        if area < 80 and max(bw, bh) < 18:
            keep = False

        if keep:
            cleaned[labels == i] = 255

    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    return cleaned


def infer_plan_scale(binary_mask: np.ndarray) -> Dict[str, int]:
    h, w = binary_mask.shape[:2]
    min_dim = min(h, w)
    return {
        "horizontal_kernel": max(20, min_dim // 18),
        "vertical_kernel": max(20, min_dim // 18),
        "max_wall_thickness": max(10, min_dim // 35),
        "min_wall_thickness": max(4, min_dim // 180),
        "pair_merge_tol": max(10, min_dim // 45),
        "opening_min_gap": max(22, min_dim // 50),
        "opening_max_gap": max(120, min_dim // 10),
        "opening_scan_step": max(4, min_dim // 180),
        "outer_edge_offset": max(8, min_dim // 60),
        "outer_wall_strip": max(14, min_dim // 55),
    }


def fill_holes(binary_mask: np.ndarray) -> np.ndarray:
    h, w = binary_mask.shape[:2]
    flood = binary_mask.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    return binary_mask | flood_inv


def extract_outer_polygon(
    structural_mask: np.ndarray,
    debug_img: np.ndarray,
    project_debug_dir: Path,
    floor_name: str,
) -> Tuple[Optional[List[List[int]]], Optional[np.ndarray]]:
    h, w = structural_mask.shape[:2]
    img_area = h * w

    building_mask = structural_mask.copy()
    building_mask = cv2.morphologyEx(building_mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=2)
    building_mask = cv2.dilate(building_mask, np.ones((5, 5), np.uint8), iterations=1)
    building_mask = fill_holes(building_mask)

    contours, _ = cv2.findContours(building_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    valid_contours = [c for c in contours if cv2.contourArea(c) >= img_area * MIN_BUILDING_AREA_RATIO]
    if not valid_contours:
        valid_contours = contours

    largest = max(valid_contours, key=cv2.contourArea)

    perimeter = cv2.arcLength(largest, True)
    epsilon = max(2.0, 0.0035 * perimeter)
    approx = cv2.approxPolyDP(largest, epsilon, True)

    if len(approx) > MAX_OUTER_POLYGON_POINTS:
        epsilon = max(4.0, 0.007 * perimeter)
        approx = cv2.approxPolyDP(largest, epsilon, True)

    polygon = [[int(p[0][0]), int(p[0][1])] for p in approx]
    polygon = simplify_polygon_points(polygon, min_dist=8)

    if len(polygon) < 3:
        hull = cv2.convexHull(largest)
        polygon = [[int(p[0][0]), int(p[0][1])] for p in hull]
        polygon = simplify_polygon_points(polygon, min_dist=8)

    polygon_np = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))

    debug = debug_img.copy()
    cv2.drawContours(debug, [largest], -1, (0, 255, 0), 2)
    cv2.polylines(debug, [polygon_np], True, (255, 0, 0), 2)
    save_debug(project_debug_dir, floor_name, "polygon_debug.png", debug)
    save_debug(project_debug_dir, floor_name, "building_mask.png", building_mask)

    return polygon, polygon_np


def build_outer_wall_mask(
    polygon_np: np.ndarray,
    mask_shape: Tuple[int, int],
    thickness: int,
) -> np.ndarray:
    outer_mask = np.zeros(mask_shape, dtype=np.uint8)
    cv2.polylines(outer_mask, [polygon_np], True, 255, thickness=thickness)
    outer_mask = cv2.dilate(outer_mask, np.ones((3, 3), np.uint8), iterations=1)
    return outer_mask


def extract_oriented_wall_masks(
    binary_mask: np.ndarray,
    polygon_np: np.ndarray,
    scale: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    inner_candidates = binary_mask.copy()
    outer_mask = build_outer_wall_mask(
        polygon_np,
        binary_mask.shape[:2],
        thickness=scale["outer_wall_strip"],
    )
    inner_candidates[outer_mask > 0] = 0
    inner_candidates = cv2.morphologyEx(inner_candidates, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)

    hk = scale["horizontal_kernel"]
    vk = scale["vertical_kernel"]

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk))

    horizontal = cv2.morphologyEx(inner_candidates, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(inner_candidates, cv2.MORPH_OPEN, vertical_kernel)

    horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_CLOSE, np.ones((9, 3), np.uint8), iterations=1)
    vertical = cv2.morphologyEx(vertical, cv2.MORPH_CLOSE, np.ones((3, 9), np.uint8), iterations=1)

    combined = cv2.bitwise_or(horizontal, vertical)
    return horizontal, vertical, combined, inner_candidates


def build_semantic_inner_wall_mask(
    structural_mask: np.ndarray,
    polygon_np: np.ndarray,
    scale: Dict[str, int],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    inner = structural_mask.copy()
    outer_mask = build_outer_wall_mask(
        polygon_np,
        structural_mask.shape[:2],
        thickness=scale["outer_wall_strip"],
    )
    inner[outer_mask > 0] = 0

    small_axis_kernel = max(10, scale["opening_min_gap"] // 2)
    horizontal_small = cv2.morphologyEx(
        inner,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (small_axis_kernel, 1)),
    )
    vertical_small = cv2.morphologyEx(
        inner,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, small_axis_kernel)),
    )
    horizontal_large = cv2.morphologyEx(
        inner,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (scale["horizontal_kernel"], 1)),
    )
    vertical_large = cv2.morphologyEx(
        inner,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, scale["vertical_kernel"])),
    )

    axis_seed = cv2.bitwise_or(horizontal_small, vertical_small)
    axis_seed = cv2.bitwise_or(axis_seed, horizontal_large)
    axis_seed = cv2.bitwise_or(axis_seed, vertical_large)
    axis_seed = cv2.morphologyEx(axis_seed, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    axis_seed = cv2.dilate(axis_seed, np.ones((3, 3), np.uint8), iterations=1)

    semantic_mask = cv2.bitwise_and(inner, axis_seed)
    semantic_mask = cv2.morphologyEx(semantic_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    semantic_mask = cv2.morphologyEx(semantic_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    semantic_mask = remove_small_components(semantic_mask, min_area=max(18, scale["min_wall_thickness"] * 4))

    non_axis = cv2.bitwise_and(inner, cv2.bitwise_not(axis_seed))
    non_axis = remove_small_components(non_axis, min_area=max(10, scale["min_wall_thickness"] * 2))

    return semantic_mask, {
        "semantic_inner_mask_enabled": True,
        "semantic_inner_mask_pixels": int(np.sum(semantic_mask > 0)),
        "semantic_non_axis_pixels": int(np.sum(non_axis > 0)),
        "semantic_axis_seed_pixels": int(np.sum(axis_seed > 0)),
    }


def extract_inner_wall_segments_from_candidate_mask(
    candidate_mask: np.ndarray,
    polygon_np: np.ndarray,
    scale: Dict[str, int],
) -> Tuple[List[List[int]], np.ndarray, Dict[str, Any]]:
    working_mask, _, symbolic_cluster_stats = filter_symbolic_line_clusters(
        candidate_mask.copy(),
        polygon_np,
        scale,
    )
    working_mask, _, repeated_pattern_stats = filter_repeated_line_patterns(
        working_mask,
        polygon_np,
        scale,
    )

    hk = scale["horizontal_kernel"]
    vk = scale["vertical_kernel"]
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk))

    horizontal_mask = cv2.morphologyEx(working_mask, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_mask = cv2.morphologyEx(working_mask, cv2.MORPH_OPEN, vertical_kernel)
    horizontal_mask = cv2.morphologyEx(horizontal_mask, cv2.MORPH_CLOSE, np.ones((9, 3), np.uint8), iterations=1)
    vertical_mask = cv2.morphologyEx(vertical_mask, cv2.MORPH_CLOSE, np.ones((3, 9), np.uint8), iterations=1)
    combined_mask = cv2.bitwise_or(horizontal_mask, vertical_mask)

    horizontal_segments, _ = segments_from_oriented_mask(horizontal_mask, "h", polygon_np, working_mask, scale)
    vertical_segments, _ = segments_from_oriented_mask(vertical_mask, "v", polygon_np, working_mask, scale)
    wall_segments = horizontal_segments + vertical_segments
    wall_segments = collapse_parallel_double_lines(wall_segments, pair_tol=scale["pair_merge_tol"])
    wall_segments, _ = merge_collinear_lines(
        wall_segments,
        pos_tol=8,
        gap_tol=max(14, min(scale["opening_max_gap"], 110)),
        return_stats=True,
    )
    wall_segments = remove_duplicate_lines(wall_segments, coord_tol=8, length_tol=12)
    wall_segments, _ = filter_lines_inside_building(
        wall_segments,
        polygon_np,
        outer_margin=max(scale["outer_edge_offset"] * 4, scale["max_wall_thickness"] * 2 + 12),
    )
    wall_segments, _ = filter_symbolic_vertical_walls(
        wall_segments,
        polygon_np,
        source_mask=working_mask,
        scale=scale,
    )
    wall_segments, _ = reject_outer_boundary_parallel_candidates(
        wall_segments,
        polygon_np,
        scale,
    )
    wall_segments, _ = merge_collinear_lines(
        wall_segments,
        pos_tol=8,
        gap_tol=max(14, min(scale["opening_max_gap"], 110)),
        return_stats=True,
    )
    wall_segments = remove_duplicate_lines(wall_segments, coord_tol=8, length_tol=12)

    line_mask = np.zeros_like(candidate_mask)
    for line in wall_segments:
        cv2.line(
            line_mask,
            (int(line[0]), int(line[1])),
            (int(line[2]), int(line[3])),
            255,
            max(2, scale["max_wall_thickness"] // 2),
        )
    line_mask = cv2.dilate(line_mask, np.ones((3, 3), np.uint8), iterations=1)

    return wall_segments, line_mask, {
        "semantic_filtered_pixels": int(np.sum(working_mask > 0)),
        "semantic_symbolic_rejects": int(symbolic_cluster_stats.get("rejected_symbolic_clusters", 0)),
        "semantic_repeated_rejects": int(repeated_pattern_stats.get("rejected_repeated_patterns", 0)),
        "semantic_segment_count": len(wall_segments),
    }


def contiguous_thickness_at_point(
    point: Tuple[int, int],
    orientation: str,
    mask: np.ndarray,
    max_radius: int,
) -> int:
    px, py = point
    h, w = mask.shape[:2]
    thickness = 1 if 0 <= px < w and 0 <= py < h and mask[py, px] > 0 else 0

    if orientation == "h":
        for direction in (-1, 1):
            step = 1
            while step <= max_radius:
                yy = py + direction * step
                if yy < 0 or yy >= h or mask[yy, px] == 0:
                    break
                thickness += 1
                step += 1
    else:
        for direction in (-1, 1):
            step = 1
            while step <= max_radius:
                xx = px + direction * step
                if xx < 0 or xx >= w or mask[py, xx] == 0:
                    break
                thickness += 1
                step += 1

    return thickness


def estimate_line_thickness(
    line: List[int],
    orientation: str,
    source_mask: np.ndarray,
    max_radius: int,
) -> float:
    samples = sample_along_segment(line, step=max(6, int(line_length(line) // 6) or 1))
    if not samples:
        return 0.0

    if len(samples) > 6:
        samples = samples[1:-1]

    thickness_values = [
        contiguous_thickness_at_point(point, orientation, source_mask, max_radius=max_radius)
        for point in samples
    ]
    positive_values = [value for value in thickness_values if value > 0]
    if not positive_values:
        return 0.0
    return float(np.median(positive_values))


def split_line_into_supported_runs(
    line: List[int],
    orientation: str,
    source_mask: np.ndarray,
    scale: Dict[str, int],
) -> List[List[int]]:
    step = max(4, scale["opening_scan_step"])
    min_thickness = scale["min_wall_thickness"]
    max_radius = scale["max_wall_thickness"]
    samples = sample_along_segment(line, step=step)
    if len(samples) < 2:
        return []

    supports = [
        contiguous_thickness_at_point(point, orientation, source_mask, max_radius=max_radius)
        for point in samples
    ]

    runs: List[List[int]] = []
    run_start: Optional[int] = None

    for index, support in enumerate(supports):
        is_supported = support >= min_thickness

        if is_supported and run_start is None:
            run_start = index
        elif not is_supported and run_start is not None:
            start_pt = samples[run_start]
            end_pt = samples[max(run_start, index - 1)]

            if orientation == "h":
                candidate = [start_pt[0], start_pt[1], end_pt[0], end_pt[1]]
            else:
                candidate = [start_pt[0], start_pt[1], end_pt[0], end_pt[1]]

            if line_length(candidate) >= MIN_WALL_LENGTH_PX:
                runs.append(candidate)

            run_start = None

    if run_start is not None:
        start_pt = samples[run_start]
        end_pt = samples[-1]
        candidate = [start_pt[0], start_pt[1], end_pt[0], end_pt[1]]
        if line_length(candidate) >= MIN_WALL_LENGTH_PX:
            runs.append(candidate)

    return runs


def segments_from_oriented_mask(
    oriented_mask: np.ndarray,
    orientation: str,
    polygon_np: np.ndarray,
    source_mask: np.ndarray,
    scale: Dict[str, int],
) -> Tuple[List[List[int]], Dict[str, Any]]:
    contours, _ = cv2.findContours(oriented_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segments: List[List[int]] = []
    debug_stats: Dict[str, Any] = {
        "raw_candidate_wall_count": len(contours),
        "accepted_segment_count": 0,
        "rejected_segment_count": 0,
        "reject_reasons": {},
    }

    max_wall_thickness = scale["max_wall_thickness"]
    min_wall_thickness = scale["min_wall_thickness"]

    def reject(reason: str) -> None:
        debug_stats["rejected_segment_count"] += 1
        debug_stats["reject_reasons"][reason] = debug_stats["reject_reasons"].get(reason, 0) + 1

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)

        if area < 30:
            reject("contour_area_too_small")
            continue

        if orientation == "h":
            if w < MIN_WALL_LENGTH_PX or h > max_wall_thickness * 2 or h < min_wall_thickness:
                reject("horizontal_bbox_out_of_range")
                continue
            y_mid = int(round(y + h / 2))
            line = [x, y_mid, x + w, y_mid]
        else:
            if h < MIN_WALL_LENGTH_PX or w > max_wall_thickness * 2 or w < min_wall_thickness:
                reject("vertical_bbox_out_of_range")
                continue
            x_mid = int(round(x + w / 2))
            line = [x_mid, y, x_mid, y + h]

        mx, my = midpoint_of_line(line)
        if not point_inside_polygon((mx, my), polygon_np):
            reject("candidate_midpoint_outside_polygon")
            continue

        refined_segments = split_line_into_supported_runs(
            line,
            orientation=orientation,
            source_mask=source_mask,
            scale=scale,
        )

        if not refined_segments:
            reject("unsupported_run_split")
            continue

        for refined_line in refined_segments:
            rmx, rmy = midpoint_of_line(refined_line)
            if not point_inside_polygon((rmx, rmy), polygon_np):
                reject("refined_midpoint_outside_polygon")
                continue

            estimated_thickness = estimate_line_thickness(
                refined_line,
                orientation=orientation,
                source_mask=source_mask,
                max_radius=max_wall_thickness,
            )
            required_thickness = max(min_wall_thickness, int(round(scale["max_wall_thickness"] * 0.36)))
            if estimated_thickness < required_thickness:
                reject("estimated_thickness_too_small")
                continue

            segments.append(refined_line)
            debug_stats["accepted_segment_count"] += 1

    return segments, debug_stats


def collapse_parallel_double_lines(
    lines: List[List[int]],
    pair_tol: int = 12,
    overlap_ratio_threshold: float = 0.45,
) -> List[List[int]]:
    if not lines:
        return []

    normalized = [normalize_line(line) for line in lines]
    used = [False] * len(normalized)
    result: List[List[int]] = []

    for i, a in enumerate(normalized):
        if used[i]:
            continue

        paired = False
        for j in range(i + 1, len(normalized)):
            if used[j]:
                continue

            b = normalized[j]
            if a["orientation"] != b["orientation"]:
                continue

            dist = abs(a["fixed"] - b["fixed"])
            if dist > pair_tol:
                continue

            overlap_start = max(a["start"], b["start"])
            overlap_end = min(a["end"], b["end"])
            overlap = max(0, overlap_end - overlap_start)

            len_a = max(1, a["end"] - a["start"])
            len_b = max(1, b["end"] - b["start"])
            overlap_ratio = overlap / float(min(len_a, len_b))

            if overlap_ratio < overlap_ratio_threshold:
                continue

            center_fixed = int(round((a["fixed"] + b["fixed"]) / 2))
            merged = {
                "orientation": a["orientation"],
                "fixed": center_fixed,
                "start": min(a["start"], b["start"]),
                "end": max(a["end"], b["end"]),
            }

            result.append(denormalize_line(merged))
            used[i] = True
            used[j] = True
            paired = True
            break

        if not paired and not used[i]:
            result.append(denormalize_line(a))
            used[i] = True

    return result


def filter_lines_inside_building(
    lines: List[List[int]],
    polygon_np: np.ndarray,
    outer_margin: int = 10,
) -> Tuple[List[List[int]], Dict[str, int]]:
    filtered: List[List[int]] = []
    stats = {"rejected_outer_edge_proximity": 0}

    for line in lines:
        mx, my = midpoint_of_line(line)
        signed_dist = cv2.pointPolygonTest(polygon_np, (float(mx), float(my)), True)
        if signed_dist < outer_margin:
            stats["rejected_outer_edge_proximity"] += 1
            continue
        filtered.append(line)

    return filtered, stats


def reject_outer_boundary_parallel_candidates(
    lines: List[List[int]],
    polygon_np: np.ndarray,
    scale: Dict[str, int],
    overlap_ratio_threshold: float = 0.35,
) -> Tuple[List[List[int]], Dict[str, Any]]:
    if not lines:
        return [], {"outer_boundary_rejected_count": 0, "outer_boundary_rejected_segments": []}

    polygon_points = [[int(pt[0]), int(pt[1])] for pt in polygon_np.reshape(-1, 2)]
    outer_segments = polygon_to_axis_aligned_segments(polygon_points, axis_tol=10, min_length=max(32, scale["opening_min_gap"]))
    normalized_lines = [normalize_line(line) for line in lines]
    normalized_outer_segments = [normalize_line(line) for line in outer_segments]
    outer_near_tol = max(10, int(round(scale["max_wall_thickness"] * 2.5)))
    connection_tol = max(12, scale["max_wall_thickness"])

    def orthogonal_endpoint_connection_count(
        item: Dict[str, Any],
        current_idx: int,
    ) -> int:
        endpoints: List[Tuple[int, int]]
        if item["orientation"] == "h":
            endpoints = [(int(item["start"]), int(item["fixed"])), (int(item["end"]), int(item["fixed"]))]
        else:
            endpoints = [(int(item["fixed"]), int(item["start"])), (int(item["fixed"]), int(item["end"]))]

        orthogonal_orientation = "v" if item["orientation"] == "h" else "h"
        connected = 0
        for endpoint in endpoints:
            endpoint_has_connection = False
            for other_idx, other in enumerate(normalized_lines):
                if other_idx == current_idx or other["orientation"] != orthogonal_orientation:
                    continue

                if other["orientation"] == "h":
                    other_endpoints = [(int(other["start"]), int(other["fixed"])), (int(other["end"]), int(other["fixed"]))]
                    if any(points_close(endpoint, other_pt, connection_tol) for other_pt in other_endpoints):
                        endpoint_has_connection = True
                        break
                    if (
                        abs(endpoint[1] - int(other["fixed"])) <= connection_tol
                        and int(other["start"]) + 6 < endpoint[0] < int(other["end"]) - 6
                    ):
                        endpoint_has_connection = True
                        break
                else:
                    other_endpoints = [(int(other["fixed"]), int(other["start"])), (int(other["fixed"]), int(other["end"]))]
                    if any(points_close(endpoint, other_pt, connection_tol) for other_pt in other_endpoints):
                        endpoint_has_connection = True
                        break
                    if (
                        abs(endpoint[0] - int(other["fixed"])) <= connection_tol
                        and int(other["start"]) + 6 < endpoint[1] < int(other["end"]) - 6
                    ):
                        endpoint_has_connection = True
                        break

            if endpoint_has_connection:
                connected += 1

        return connected

    filtered: List[List[int]] = []
    rejected_segments: List[List[int]] = []

    for idx, line in enumerate(lines):
        item = normalized_lines[idx]
        endpoint_connection_count = orthogonal_endpoint_connection_count(item, idx)
        reject_line = False

        if endpoint_connection_count < 2:
            item_len = max(1, int(item["end"]) - int(item["start"]))
            for outer in normalized_outer_segments:
                if outer["orientation"] != item["orientation"]:
                    continue
                if abs(int(item["fixed"]) - int(outer["fixed"])) > outer_near_tol:
                    continue

                overlap = max(0, min(int(item["end"]), int(outer["end"])) - max(int(item["start"]), int(outer["start"])))
                overlap_ratio = overlap / float(item_len)
                if overlap_ratio < overlap_ratio_threshold:
                    continue

                reject_line = True
                break

        if reject_line:
            rejected_segments.append(line)
            continue

        filtered.append(line)

    return filtered, {
        "outer_boundary_rejected_count": len(rejected_segments),
        "outer_boundary_rejected_segments": rejected_segments,
    }


def compute_line_support_score(
    line: List[int],
    orientation: str,
    source_mask: np.ndarray,
    scale: Dict[str, int],
) -> float:
    step = max(4, scale["opening_scan_step"])
    samples = sample_along_segment(line, step=step)
    if not samples:
        return 0.0
    if len(samples) > 6:
        samples = samples[1:-1]
    supports = [
        contiguous_thickness_at_point(point, orientation, source_mask, max_radius=scale["max_wall_thickness"])
        for point in samples
    ]
    positive = [value for value in supports if value > 0]
    if not positive:
        return 0.0
    occupied_ratio = len(positive) / float(len(supports))
    return float(np.median(positive)) * occupied_ratio


def analyze_raw_inner_wall_candidates(
    raw_segments: List[List[int]],
    source_mask: np.ndarray,
    polygon_np: np.ndarray,
    scale: Dict[str, int],
) -> Dict[str, Any]:
    if not raw_segments:
        return {
            "candidate_details": [],
            "parallel_edge_pairs": [],
            "estimated_centerline_axes": [],
            "parallel_edge_pair_count": 0,
            "estimated_centerline_axis_count": 0,
            "edge_like_candidate_count": 0,
            "centerline_like_candidate_count": 0,
            "overlay_lines": [],
        }

    step = max(4, scale["opening_scan_step"])
    min_pair_distance = max(4, scale["min_wall_thickness"])
    max_pair_distance = max(scale["min_wall_thickness"] + 2, int(round(scale["max_wall_thickness"] * 1.8)))
    side_threshold = 0.22
    edge_bias_threshold = 0.18

    def side_support_ratios(line: List[int], orientation: str, band: int) -> Tuple[float, float]:
        samples = sample_along_segment(line, step=step)
        if not samples:
            return 0.0, 0.0
        if len(samples) > 6:
            samples = samples[1:-1]

        h, w = source_mask.shape[:2]
        pos_a = 0
        pos_b = 0
        total = 0
        for px, py in samples:
            if orientation == "v":
                xa = max(0, min(w - 1, px - band))
                xb = max(0, min(w - 1, px + band))
                pos_a += 1 if source_mask[py, xa] > 0 else 0
                pos_b += 1 if source_mask[py, xb] > 0 else 0
            else:
                ya = max(0, min(h - 1, py - band))
                yb = max(0, min(h - 1, py + band))
                pos_a += 1 if source_mask[ya, px] > 0 else 0
                pos_b += 1 if source_mask[yb, px] > 0 else 0
            total += 1

        if total == 0:
            return 0.0, 0.0
        return pos_a / float(total), pos_b / float(total)

    candidate_details: List[Dict[str, Any]] = []
    normalized = [normalize_line(line) for line in raw_segments]
    for line, item in zip(raw_segments, normalized):
        orientation = item["orientation"]
        thickness = estimate_line_thickness(
            line,
            orientation=orientation,
            source_mask=source_mask,
            max_radius=scale["max_wall_thickness"],
        )
        band = max(1, min(scale["max_wall_thickness"], int(round(max(thickness, scale["min_wall_thickness"]) * 0.6))))
        side_a_ratio, side_b_ratio = side_support_ratios(line, orientation, band)
        both_side_support = side_a_ratio >= side_threshold and side_b_ratio >= side_threshold
        edge_like = (
            (side_a_ratio >= side_threshold and side_b_ratio <= edge_bias_threshold)
            or (side_b_ratio >= side_threshold and side_a_ratio <= edge_bias_threshold)
        )
        centerline_like = both_side_support and abs(side_a_ratio - side_b_ratio) <= 0.22
        candidate_details.append({
            "segment": line[:],
            "orientation": orientation,
            "fixed_axis_value": int(item["fixed"]),
            "segment_length": round(line_length(line), 3),
            "line_support_score": round(compute_line_support_score(line, orientation, source_mask, scale), 3),
            "wall_thickness_estimate": round(float(thickness), 3),
            "parallel_side_support_a": round(side_a_ratio, 3),
            "parallel_side_support_b": round(side_b_ratio, 3),
            "parallel_support_both_sides": both_side_support,
            "edge_or_centerline_behavior": "centerline_like" if centerline_like else ("edge_like" if edge_like else "uncertain"),
        })

    parallel_edge_pairs: List[Dict[str, Any]] = []
    estimated_centerline_axes: List[Dict[str, Any]] = []
    overlay_lines: List[List[int]] = []

    for orientation in ("h", "v"):
        oriented_indices = [idx for idx, item in enumerate(normalized) if item["orientation"] == orientation]
        used_pairs: set[Tuple[int, int]] = set()
        for i, idx_a in enumerate(oriented_indices):
            item_a = normalized[idx_a]
            detail_a = candidate_details[idx_a]
            len_a = max(1.0, line_length(raw_segments[idx_a]))
            for idx_b in oriented_indices[i + 1:]:
                if (idx_a, idx_b) in used_pairs:
                    continue
                item_b = normalized[idx_b]
                detail_b = candidate_details[idx_b]
                distance = abs(int(item_a["fixed"]) - int(item_b["fixed"]))
                if distance < min_pair_distance or distance > max_pair_distance:
                    continue
                overlap = max(0, min(int(item_a["end"]), int(item_b["end"])) - max(int(item_a["start"]), int(item_b["start"])))
                shorter = max(1, min(int(item_a["end"]) - int(item_a["start"]), int(item_b["end"]) - int(item_b["start"])))
                overlap_ratio = overlap / float(shorter)
                if overlap_ratio < 0.45:
                    continue

                midpoint_axis = int(round((int(item_a["fixed"]) + int(item_b["fixed"])) / 2.0))
                centerline_segment = denormalize_line({
                    "orientation": orientation,
                    "fixed": midpoint_axis,
                    "start": max(int(item_a["start"]), int(item_b["start"])),
                    "end": min(int(item_a["end"]), int(item_b["end"])),
                })
                mx, my = midpoint_of_line(centerline_segment)
                if not point_inside_polygon((mx, my), polygon_np):
                    continue

                used_pairs.add((idx_a, idx_b))
                pair_info = {
                    "orientation": orientation,
                    "segment_a": raw_segments[idx_a][:],
                    "segment_b": raw_segments[idx_b][:],
                    "pair_distance": distance,
                    "span_overlap_ratio": round(overlap_ratio, 3),
                    "distance_in_wall_thickness_range": min_pair_distance <= distance <= max_pair_distance,
                    "centerline_axis": midpoint_axis,
                    "centerline_segment": centerline_segment,
                    "edge_like_pair": (
                        detail_a["edge_or_centerline_behavior"] == "edge_like"
                        or detail_b["edge_or_centerline_behavior"] == "edge_like"
                    ),
                }
                parallel_edge_pairs.append(pair_info)
                estimated_centerline_axes.append({
                    "orientation": orientation,
                    "axis": midpoint_axis,
                    "segment": centerline_segment,
                    "source_pair_distance": distance,
                    "span_overlap_ratio": round(overlap_ratio, 3),
                })
                overlay_lines.extend([raw_segments[idx_a][:], raw_segments[idx_b][:], centerline_segment])

    edge_like_count = sum(1 for detail in candidate_details if detail["edge_or_centerline_behavior"] == "edge_like")
    centerline_like_count = sum(1 for detail in candidate_details if detail["edge_or_centerline_behavior"] == "centerline_like")

    return {
        "candidate_details": candidate_details,
        "parallel_edge_pairs": parallel_edge_pairs,
        "estimated_centerline_axes": estimated_centerline_axes,
        "parallel_edge_pair_count": len(parallel_edge_pairs),
        "estimated_centerline_axis_count": len(estimated_centerline_axes),
        "edge_like_candidate_count": edge_like_count,
        "centerline_like_candidate_count": centerline_like_count,
        "overlay_lines": overlay_lines,
    }


def morphological_skeleton(mask: np.ndarray) -> np.ndarray:
    skeleton = np.zeros_like(mask)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    current = mask.copy()
    while True:
        opened = cv2.morphologyEx(current, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(current, opened)
        eroded = cv2.erode(current, element)
        skeleton = cv2.bitwise_or(skeleton, temp)
        current = eroded
        if cv2.countNonZero(current) == 0:
            break
    return skeleton


def extract_skeleton_centerline_candidates(
    source_mask: np.ndarray,
    polygon_np: np.ndarray,
    scale: Dict[str, int],
) -> Dict[str, Any]:
    polygon_mask = np.zeros_like(source_mask)
    cv2.fillPoly(polygon_mask, [polygon_np], 255)
    interior_margin = max(scale["outer_edge_offset"] * 3, scale["max_wall_thickness"] * 2 + 4)
    interior_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (max(3, interior_margin * 2 + 1), max(3, interior_margin * 2 + 1)),
    )
    interior_mask = cv2.erode(polygon_mask, interior_kernel, iterations=1)
    candidate_mask = cv2.bitwise_and(source_mask, interior_mask)
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    skeleton_method = "morphological_skeleton"
    skeleton_mask: np.ndarray
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        skeleton_method = "ximgproc_thinning"
        skeleton_mask = cv2.ximgproc.thinning(candidate_mask)
    else:
        skeleton_mask = morphological_skeleton(candidate_mask)

    min_len = max(18, int(round(scale["opening_min_gap"] * 1.2)))
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(min_len, scale["opening_min_gap"] * 2), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(min_len, scale["opening_min_gap"] * 2)))
    h_mask = cv2.morphologyEx(skeleton_mask, cv2.MORPH_OPEN, hk)
    v_mask = cv2.morphologyEx(skeleton_mask, cv2.MORPH_OPEN, vk)

    candidates: List[List[int]] = []
    vertical_count = 0
    horizontal_count = 0

    for orientation, oriented_mask in (("h", h_mask), ("v", v_mask)):
        contours, _ = cv2.findContours(oriented_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if orientation == "h":
                if w < min_len or h > max(8, scale["max_wall_thickness"] * 2):
                    continue
                y_mid = int(round(y + h / 2))
                line = [x, y_mid, x + w, y_mid]
                horizontal_count += 1
            else:
                if h < min_len or w > max(8, scale["max_wall_thickness"] * 2):
                    continue
                x_mid = int(round(x + w / 2))
                line = [x_mid, y, x_mid, y + h]
                vertical_count += 1

            mx, my = midpoint_of_line(line)
            if not point_inside_polygon((mx, my), polygon_np):
                continue
            candidates.append(line)

    return {
        "skeleton_enabled": True,
        "skeleton_method": skeleton_method,
        "skeleton_mask_nonzero_count": int(np.count_nonzero(skeleton_mask > 0)),
        "skeleton_candidate_count": len(candidates),
        "skeleton_candidate_vertical_count": vertical_count,
        "skeleton_candidate_horizontal_count": horizontal_count,
        "skeleton_candidates": candidates,
        "skeleton_mask": skeleton_mask,
    }


def extract_projection_grid_candidates(
    source_mask: np.ndarray,
    polygon_np: np.ndarray,
    scale: Dict[str, int],
) -> Dict[str, Any]:
    polygon_mask = np.zeros_like(source_mask)
    cv2.fillPoly(polygon_mask, [polygon_np], 255)

    interior_margin = max(scale["outer_edge_offset"] * 3, scale["max_wall_thickness"] * 2 + 4)
    interior_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (max(3, interior_margin * 2 + 1), max(3, interior_margin * 2 + 1)),
    )
    interior_mask = cv2.erode(polygon_mask, interior_kernel, iterations=1)
    roi_mask = cv2.bitwise_and(source_mask, interior_mask)
    roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    min_len = max(scale["opening_min_gap"] * 2, 56)
    axis_cluster_tol = max(10, int(round(scale["max_wall_thickness"] * 1.5)))
    support_band = max(2, scale["max_wall_thickness"] // 2)
    support_threshold = 0.34
    density_threshold_floor = max(10.0, float(np.percentile(np.sum((roi_mask > 0).astype(np.uint8), axis=0), 70) * 0.35)) if np.count_nonzero(roi_mask) else 10.0
    merge_gap_tol = max(10, scale["max_wall_thickness"] * 2)
    outer_segments = polygon_to_axis_aligned_segments(
        [[int(pt[0]), int(pt[1])] for pt in polygon_np.reshape(-1, 2)],
        axis_tol=10,
        min_length=max(32, scale["opening_min_gap"]),
    )
    outer_normalized = [normalize_line(line) for line in outer_segments]
    h, w = roi_mask.shape[:2]

    def cluster_axes(values: List[int]) -> List[int]:
        if not values:
            return []
        values = sorted(values)
        clusters: List[List[int]] = [[values[0]]]
        for value in values[1:]:
            current = clusters[-1]
            center = int(round(sum(current) / max(1, len(current))))
            if abs(value - center) <= axis_cluster_tol:
                current.append(value)
            else:
                clusters.append([value])
        return [int(round(sum(cluster) / max(1, len(cluster)))) for cluster in clusters]

    def projection_axes(orientation: str) -> List[int]:
        mask01 = (roi_mask > 0).astype(np.uint8)
        projection = np.sum(mask01, axis=0 if orientation == "v" else 1).astype(np.float32)
        if projection.size == 0:
            return []
        kernel_size = max(5, scale["max_wall_thickness"] * 2 + 1)
        kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
        smooth = np.convolve(projection, kernel, mode="same")
        threshold = max(density_threshold_floor, float(np.percentile(smooth, 80) * 0.42))
        peaks: List[int] = []
        for idx in range(1, len(smooth) - 1):
            if smooth[idx] < threshold:
                continue
            if smooth[idx] >= smooth[idx - 1] and smooth[idx] >= smooth[idx + 1]:
                peaks.append(int(idx))
        return cluster_axes(peaks)

    def point_support_ratio(x: int, y: int, orientation: str) -> float:
        if orientation == "v":
            xa = max(0, x - support_band)
            xb = min(w, x + support_band + 1)
            patch = roi_mask[max(0, y - 1):min(h, y + 2), xa:xb]
        else:
            ya = max(0, y - support_band)
            yb = min(h, y + support_band + 1)
            patch = roi_mask[ya:yb, max(0, x - 1):min(w, x + 2)]
        if patch.size == 0:
            return 0.0
        return float(np.mean(patch > 0))

    def is_outer_like(line: List[int]) -> bool:
        item = normalize_line(line)
        for outer in outer_normalized:
            if outer["orientation"] != item["orientation"]:
                continue
            if abs(int(outer["fixed"]) - int(item["fixed"])) > max(scale["outer_edge_offset"] * 2, scale["max_wall_thickness"] + 4):
                continue
            overlap = max(0, min(int(outer["end"]), int(item["end"])) - max(int(outer["start"]), int(item["start"])))
            shorter = max(1, min(int(outer["end"]) - int(outer["start"]), int(item["end"]) - int(item["start"])))
            if overlap / float(shorter) >= 0.72:
                return True
        return False

    def run_segments_for_axis(axis: int, orientation: str) -> List[List[int]]:
        runs: List[Tuple[int, int]] = []
        current_start: Optional[int] = None
        current_end: Optional[int] = None
        limit = h if orientation == "v" else w
        gap_budget = merge_gap_tol

        for coord in range(limit):
            ratio = point_support_ratio(axis if orientation == "v" else coord, coord if orientation == "v" else axis, orientation)
            supported = ratio >= support_threshold
            if supported:
                if current_start is None:
                    current_start = coord
                    current_end = coord
                    gap_budget = merge_gap_tol
                else:
                    current_end = coord
                    gap_budget = merge_gap_tol
                continue

            if current_start is not None and gap_budget > 0:
                gap_budget -= 1
                continue

            if current_start is not None and current_end is not None:
                runs.append((current_start, current_end))
            current_start = None
            current_end = None
            gap_budget = merge_gap_tol

        if current_start is not None and current_end is not None:
            runs.append((current_start, current_end))

        segments: List[List[int]] = []
        for start, end in runs:
            if end - start < min_len:
                continue
            line = [axis, start, axis, end] if orientation == "v" else [start, axis, end, axis]
            mx, my = midpoint_of_line(line)
            if not point_inside_polygon((mx, my), polygon_np):
                continue
            if cv2.pointPolygonTest(polygon_np, (float(mx), float(my)), True) <= interior_margin:
                continue
            if is_outer_like(line):
                continue
            support_score = compute_line_support_score(line, orientation, roi_mask, scale)
            if support_score <= 0:
                continue
            segments.append(line)
        return segments

    vertical_axes = projection_axes("v")
    horizontal_axes = projection_axes("h")
    candidates: List[List[int]] = []
    vertical_count = 0
    horizontal_count = 0

    for axis in vertical_axes:
        axis_segments = run_segments_for_axis(axis, "v")
        vertical_count += len(axis_segments)
        candidates.extend(axis_segments)
    for axis in horizontal_axes:
        axis_segments = run_segments_for_axis(axis, "h")
        horizontal_count += len(axis_segments)
        candidates.extend(axis_segments)

    candidates = remove_duplicate_lines(candidates, coord_tol=8, length_tol=12)
    candidates, _ = merge_collinear_lines(
        candidates,
        pos_tol=8,
        gap_tol=max(14, min(scale["opening_max_gap"], 110)),
        return_stats=True,
    )
    candidates = remove_duplicate_lines(candidates, coord_tol=8, length_tol=12)

    candidate_vertical_count = sum(1 for line in candidates if normalize_line(line)["orientation"] == "v")
    candidate_horizontal_count = sum(1 for line in candidates if normalize_line(line)["orientation"] == "h")

    return {
        "projection_grid_enabled": True,
        "projection_vertical_axis_count": len(vertical_axes),
        "projection_horizontal_axis_count": len(horizontal_axes),
        "projection_vertical_axes": vertical_axes,
        "projection_horizontal_axes": horizontal_axes,
        "projection_candidate_count": len(candidates),
        "projection_candidate_vertical_count": candidate_vertical_count,
        "projection_candidate_horizontal_count": candidate_horizontal_count,
        "projection_candidates": candidates,
        "projection_roi_mask": roi_mask,
    }


def score_inner_wall_set(
    lines: List[List[int]],
    polygon_np: np.ndarray,
    source_mask: np.ndarray,
    scale: Dict[str, int],
) -> float:
    if not lines:
        return float("-inf")

    boundary_safe_margin = max(scale["outer_edge_offset"] * 3, scale["max_wall_thickness"] * 2 + 8)
    per_line_scores: List[float] = []
    total_length = 0.0

    for line in lines:
        item = normalize_line(line)
        support = compute_line_support_score(line, item["orientation"], source_mask, scale)
        thickness = estimate_line_thickness(
            line,
            orientation=item["orientation"],
            source_mask=source_mask,
            max_radius=scale["max_wall_thickness"],
        )
        length = line_length(line)
        total_length += length
        mx, my = midpoint_of_line(line)
        boundary_dist = cv2.pointPolygonTest(polygon_np, (float(mx), float(my)), True)
        score = (support * 16.0) + (thickness * 3.0) + min(length, 260.0) * 0.07
        if boundary_dist < boundary_safe_margin:
            score -= 18.0
        if length < max(scale["opening_min_gap"] * 2, 60):
            score -= 10.0
        per_line_scores.append(score)

    mean_score = float(np.mean(per_line_scores)) if per_line_scores else -999.0
    coverage_bonus = min(total_length / 180.0, 22.0)
    count_penalty = max(0, len(lines) - 8) * 1.8
    return round(mean_score + coverage_bonus - count_penalty, 3)


def build_wall_region_graph_inner_walls(
    source_mask: np.ndarray,
    polygon_np: np.ndarray,
    scale: Dict[str, int],
) -> Tuple[List[List[int]], np.ndarray, Dict[str, Any]]:
    empty_mask = np.zeros_like(source_mask)
    if np.count_nonzero(source_mask) == 0:
        return [], empty_mask, {
            "wall_region_graph_enabled": True,
            "wall_region_graph_count": 0,
            "wall_region_graph_segments": [],
            "wall_region_graph_score": float("-inf"),
            "wall_region_graph_component_count": 0,
        }

    polygon_mask = np.zeros_like(source_mask)
    cv2.fillPoly(polygon_mask, [polygon_np], 255)
    interior_margin = max(scale["outer_edge_offset"] * 3, scale["max_wall_thickness"] * 2 + 6)
    interior_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (max(3, interior_margin * 2 + 1), max(3, interior_margin * 2 + 1)),
    )
    interior_mask = cv2.erode(polygon_mask, interior_kernel, iterations=1)
    region_mask = cv2.bitwise_and(source_mask, interior_mask)
    region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    small_h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(7, scale["opening_min_gap"] // 2), 1))
    small_v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(7, scale["opening_min_gap"] // 2)))
    region_mask = cv2.bitwise_or(
        cv2.morphologyEx(region_mask, cv2.MORPH_CLOSE, small_h_kernel, iterations=1),
        cv2.morphologyEx(region_mask, cv2.MORPH_CLOSE, small_v_kernel, iterations=1),
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(region_mask, connectivity=8)
    min_component_area = max(60, scale["min_wall_thickness"] * scale["opening_min_gap"])
    min_len = max(MIN_WALL_LENGTH_PX, scale["opening_min_gap"] * 2)
    axis_cluster_tol = max(8, int(round(scale["max_wall_thickness"] * 1.2)))
    stripe_band_limit = max(scale["max_wall_thickness"] * 4, 22)
    support_band = max(2, scale["max_wall_thickness"] // 2)
    support_threshold = 0.28
    rejected_short_count = 0
    rejected_outer_count = 0
    stripe_axes_debug: List[Dict[str, Any]] = []
    original_stripe_axes_debug: List[Dict[str, Any]] = []
    refined_stripe_axes_debug: List[Dict[str, Any]] = []
    axis_refinement_candidates: List[Dict[str, Any]] = []
    axis_refined_count = 0
    axis_refinement_rejected_count = 0
    original_component_count = 0
    subregion_count = 0
    split_component_count = 0
    split_rejected_small_count = 0
    original_component_bboxes: List[List[int]] = []
    subregion_bboxes: List[List[int]] = []

    def contiguous_runs(indices: List[int]) -> List[Tuple[int, int]]:
        if not indices:
            return []
        runs: List[Tuple[int, int]] = []
        start = indices[0]
        prev = indices[0]
        for value in indices[1:]:
            if value == prev + 1:
                prev = value
                continue
            runs.append((start, prev))
            start = value
            prev = value
        runs.append((start, prev))
        return runs

    def cluster_axes(values: List[int]) -> List[int]:
        if not values:
            return []
        values = sorted(values)
        clusters: List[List[int]] = [[values[0]]]
        for value in values[1:]:
            current = clusters[-1]
            center = int(round(sum(current) / max(1, len(current))))
            if abs(value - center) <= axis_cluster_tol:
                current.append(value)
            else:
                clusters.append([value])
        return [int(round(sum(cluster) / max(1, len(cluster)))) for cluster in clusters]

    def local_axis_peaks(projection: np.ndarray, threshold_ratio: float = 0.4) -> List[int]:
        if projection.size == 0 or np.max(projection) <= 0:
            return []
        kernel_size = max(3, min(len(projection) // 8 * 2 + 1, max(5, scale["max_wall_thickness"] * 2 + 1)))
        kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
        smooth = np.convolve(projection.astype(np.float32), kernel, mode="same")
        positive = smooth[smooth > 0]
        if positive.size == 0:
            return []
        threshold = max(3.0, float(np.percentile(positive, 65) * threshold_ratio))
        peaks: List[int] = []
        for idx in range(1, len(smooth) - 1):
            if smooth[idx] < threshold:
                continue
            if smooth[idx] >= smooth[idx - 1] and smooth[idx] >= smooth[idx + 1]:
                peaks.append(int(idx))
        return cluster_axes(peaks)

    def point_support_ratio(component: np.ndarray, coord_a: int, coord_b: int, orientation: str) -> float:
        ch, cw = component.shape[:2]
        if orientation == "v":
            xa = max(0, coord_a - support_band)
            xb = min(cw, coord_a + support_band + 1)
            patch = component[max(0, coord_b - 1):min(ch, coord_b + 2), xa:xb]
        else:
            ya = max(0, coord_b - support_band)
            yb = min(ch, coord_b + support_band + 1)
            patch = component[ya:yb, max(0, coord_a - 1):min(cw, coord_a + 2)]
        if patch.size == 0:
            return 0.0
        return float(np.mean(patch > 0))

    def longest_run_length(indices: List[int]) -> int:
        runs = contiguous_runs(indices)
        if not runs:
            return 0
        return max((end - start + 1) for start, end in runs)

    def split_component_into_subregions(component: np.ndarray) -> List[Tuple[int, int, int, int, np.ndarray]]:
        nonlocal split_component_count, split_rejected_small_count
        ch, cw = component.shape[:2]
        full_region = [(0, 0, cw, ch, component)]
        if ch < max(min_len * 2, 110) and cw < max(min_len * 2, 110):
            return full_region

        row_proj = np.sum(component > 0, axis=1).astype(np.int32)
        col_proj = np.sum(component > 0, axis=0).astype(np.int32)
        row_pos = row_proj[row_proj > 0]
        col_pos = col_proj[col_proj > 0]
        if row_pos.size == 0 or col_pos.size == 0:
            return full_region

        component_area = int(np.count_nonzero(component > 0))
        complex_component = (
            component_area >= max(min_component_area * 4, 260)
            and (cw >= max(min_len * 2, 120) or ch >= max(min_len * 2, 120))
        )
        if not complex_component:
            return full_region

        valley_row_threshold = max(2, int(np.percentile(row_pos, 42) * 0.58))
        valley_col_threshold = max(2, int(np.percentile(col_pos, 42) * 0.58))
        bridge_limit = max(3, int(round(scale["max_wall_thickness"] * 1.8)))
        border_margin = max(scale["max_wall_thickness"], 8)
        split_mask = component.copy()

        def local_valley_runs(projection: np.ndarray, threshold: int) -> List[Tuple[int, int]]:
            if projection.size < 3:
                return []
            valley_indices: List[int] = []
            for idx in range(1, len(projection) - 1):
                value = int(projection[idx])
                if value <= 0 or value > threshold:
                    continue
                if value <= int(projection[idx - 1]) and value <= int(projection[idx + 1]):
                    valley_indices.append(idx)
            return contiguous_runs(valley_indices)

        row_valleys = local_valley_runs(row_proj, valley_row_threshold)
        col_valleys = local_valley_runs(col_proj, valley_col_threshold)

        for start, end in row_valleys:
            run_len = end - start + 1
            run_mid = (start + end) / 2.0
            if run_len > bridge_limit * 2:
                continue
            if run_mid <= border_margin or run_mid >= ch - border_margin:
                continue
            split_mask[start:end + 1, :] = 0

        for start, end in col_valleys:
            run_len = end - start + 1
            run_mid = (start + end) / 2.0
            if run_len > bridge_limit * 2:
                continue
            if run_mid <= border_margin or run_mid >= cw - border_margin:
                continue
            split_mask[:, start:end + 1] = 0

        # Break very thin symbolic bridges so large merged components can separate.
        bridge_kernel = cv2.getStructuringElement(
            cv2.MORPH_CROSS,
            (max(3, scale["max_wall_thickness"]), max(3, scale["max_wall_thickness"])),
        )
        bridge_pruned = cv2.erode(split_mask, bridge_kernel, iterations=1)
        if np.count_nonzero(bridge_pruned > 0) > 0:
            split_mask = bridge_pruned

        sub_num_labels, sub_labels, sub_stats, _ = cv2.connectedComponentsWithStats(split_mask, connectivity=8)
        if sub_num_labels <= 2:
            return full_region

        subregions: List[Tuple[int, int, int, int, np.ndarray]] = []
        for sub_label in range(1, sub_num_labels):
            area = int(sub_stats[sub_label, cv2.CC_STAT_AREA])
            if area < max(min_component_area // 2, 35):
                split_rejected_small_count += 1
                continue
            sx = int(sub_stats[sub_label, cv2.CC_STAT_LEFT])
            sy = int(sub_stats[sub_label, cv2.CC_STAT_TOP])
            sw = int(sub_stats[sub_label, cv2.CC_STAT_WIDTH])
            sh = int(sub_stats[sub_label, cv2.CC_STAT_HEIGHT])
            sub_component = np.where(sub_labels[sy:sy + sh, sx:sx + sw] == sub_label, 255, 0).astype(np.uint8)
            subregions.append((sx, sy, sw, sh, sub_component))

        if len(subregions) <= 1:
            return full_region

        split_component_count += 1
        return subregions

    def refine_axis(component: np.ndarray, axis: int, orientation: str) -> Tuple[int, Dict[str, Any]]:
        ch, cw = component.shape[:2]
        max_refine_shift = max(18, int(round(scale["max_wall_thickness"] * 2.5)))
        local_band = min(max_refine_shift, max(scale["max_wall_thickness"] * 3, 12))
        best_axis = int(axis)
        best_score = float("-inf")
        original_score = float("-inf")
        low = max(0, axis - local_band)
        high = min((cw - 1) if orientation == "v" else (ch - 1), axis + local_band)

        for test_axis in range(low, high + 1):
            if orientation == "v":
                band = component[:, max(0, test_axis - support_band):min(cw, test_axis + support_band + 1)]
                occupied = np.where(np.max(band > 0, axis=1))[0].tolist() if band.size else []
                projection_value = float(np.sum(component[:, test_axis] > 0)) if 0 <= test_axis < cw else 0.0
            else:
                band = component[max(0, test_axis - support_band):min(ch, test_axis + support_band + 1), :]
                occupied = np.where(np.max(band > 0, axis=0))[0].tolist() if band.size else []
                projection_value = float(np.sum(component[test_axis, :] > 0)) if 0 <= test_axis < ch else 0.0

            continuity = float(longest_run_length(occupied))
            density = float(len(occupied))
            center_bias = abs(test_axis - axis) * 0.45
            score = (projection_value * 1.2) + (continuity * 1.4) + (density * 0.2) - center_bias
            if test_axis == axis:
                original_score = score
            if score > best_score:
                best_score = score
                best_axis = int(test_axis)

        accepted = (
            best_axis != axis
            and abs(best_axis - axis) <= max_refine_shift
            and best_score >= original_score + 3.0
        )
        return (best_axis if accepted else axis), {
            "orientation": orientation,
            "original_axis": int(axis),
            "refined_axis": int(best_axis),
            "accepted": accepted,
            "original_score": round(original_score, 3),
            "refined_score": round(best_score, 3),
            "max_refine_shift": max_refine_shift,
        }

    candidates: List[List[int]] = []
    component_debug: List[Dict[str, Any]] = []

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_component_area:
            continue
        original_component_count += 1
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        original_component_bboxes.append([x, y, w, h])
        component = np.where(labels[y:y + h, x:x + w] == label, 255, 0).astype(np.uint8)
        if component.size == 0:
            continue
        subregions = split_component_into_subregions(component)
        subregion_count += len(subregions)

        component_candidates_before = len(candidates)
        local_horizontal_axes_debug: List[int] = []
        local_vertical_axes_debug: List[int] = []
        component_subregion_bboxes: List[List[int]] = []

        for sx, sy, sw, sh, sub_component in subregions:
            subregion_bboxes.append([x + sx, y + sy, sw, sh])
            component_subregion_bboxes.append([x + sx, y + sy, sw, sh])
            rows = np.sum(sub_component > 0, axis=1).astype(np.int32)
            cols = np.sum(sub_component > 0, axis=0).astype(np.int32)
            row_axes = local_axis_peaks(rows)
            col_axes = local_axis_peaks(cols)

            for row_axis in row_axes:
                refined_row_axis, refinement_meta = refine_axis(sub_component, row_axis, "h")
                refinement_meta["component_bbox"] = [x + sx, y + sy, sw, sh]
                axis_refinement_candidates.append(refinement_meta)
                original_stripe_axes_debug.append({"orientation": "h", "axis": y + sy + row_axis, "component_bbox": [x + sx, y + sy, sw, sh]})
                if refinement_meta["accepted"]:
                    axis_refined_count += 1
                    refined_stripe_axes_debug.append({"orientation": "h", "axis": y + sy + refined_row_axis, "component_bbox": [x + sx, y + sy, sw, sh]})
                else:
                    axis_refinement_rejected_count += 1
                effective_row_axis = refined_row_axis
                band_rows = [idx for idx, value in enumerate(rows.tolist()) if value > 0 and abs(idx - effective_row_axis) <= stripe_band_limit]
                if not band_rows:
                    continue
                band_start = max(0, min(band_rows))
                band_end = min(sh - 1, max(band_rows))
                if band_end - band_start + 1 > stripe_band_limit:
                    band_start = max(0, effective_row_axis - stripe_band_limit // 2)
                    band_end = min(sh - 1, effective_row_axis + stripe_band_limit // 2)
                band = sub_component[band_start:band_end + 1, :]
                occupied_cols = [idx for idx in range(sw) if np.max(band[:, idx] > 0)]
                axis_emitted = False
                for col_start, col_end in contiguous_runs(occupied_cols):
                    if col_end - col_start < min_len:
                        rejected_short_count += 1
                        continue
                    support_runs = []
                    current_start = None
                    current_end = None
                    gap_budget = max(6, scale["max_wall_thickness"])
                    for col_idx in range(col_start, col_end + 1):
                        ratio = point_support_ratio(sub_component, col_idx, effective_row_axis, "h")
                        supported = ratio >= support_threshold
                        if supported:
                            if current_start is None:
                                current_start = col_idx
                                current_end = col_idx
                            else:
                                current_end = col_idx
                            gap_budget = max(6, scale["max_wall_thickness"])
                        elif current_start is not None and gap_budget > 0:
                            gap_budget -= 1
                        elif current_start is not None and current_end is not None:
                            support_runs.append((current_start, current_end))
                            current_start = None
                            current_end = None
                            gap_budget = max(6, scale["max_wall_thickness"])
                    if current_start is not None and current_end is not None:
                        support_runs.append((current_start, current_end))
                    for run_start, run_end in support_runs:
                        if run_end - run_start < min_len:
                            rejected_short_count += 1
                            continue
                        line = [x + sx + run_start, y + sy + effective_row_axis, x + sx + run_end, y + sy + effective_row_axis]
                        mx, my = midpoint_of_line(line)
                        if not point_inside_polygon((mx, my), polygon_np):
                            continue
                        if cv2.pointPolygonTest(polygon_np, (float(mx), float(my)), True) <= interior_margin:
                            rejected_outer_count += 1
                            continue
                        support_score = compute_line_support_score(line, "h", source_mask, scale)
                        if support_score <= 0:
                            continue
                        candidates.append(line)
                        axis_emitted = True
                if axis_emitted:
                    local_horizontal_axes_debug.append(y + sy + effective_row_axis)

            for col_axis in col_axes:
                refined_col_axis, refinement_meta = refine_axis(sub_component, col_axis, "v")
                refinement_meta["component_bbox"] = [x + sx, y + sy, sw, sh]
                axis_refinement_candidates.append(refinement_meta)
                original_stripe_axes_debug.append({"orientation": "v", "axis": x + sx + col_axis, "component_bbox": [x + sx, y + sy, sw, sh]})
                if refinement_meta["accepted"]:
                    axis_refined_count += 1
                    refined_stripe_axes_debug.append({"orientation": "v", "axis": x + sx + refined_col_axis, "component_bbox": [x + sx, y + sy, sw, sh]})
                else:
                    axis_refinement_rejected_count += 1
                effective_col_axis = refined_col_axis
                band_cols = [idx for idx, value in enumerate(cols.tolist()) if value > 0 and abs(idx - effective_col_axis) <= stripe_band_limit]
                if not band_cols:
                    continue
                band_start = max(0, min(band_cols))
                band_end = min(sw - 1, max(band_cols))
                if band_end - band_start + 1 > stripe_band_limit:
                    band_start = max(0, effective_col_axis - stripe_band_limit // 2)
                    band_end = min(sw - 1, effective_col_axis + stripe_band_limit // 2)
                band = sub_component[:, band_start:band_end + 1]
                occupied_rows = [idx for idx in range(sh) if np.max(band[idx, :] > 0)]
                axis_emitted = False
                for row_start, row_end in contiguous_runs(occupied_rows):
                    if row_end - row_start < min_len:
                        rejected_short_count += 1
                        continue
                    support_runs = []
                    current_start = None
                    current_end = None
                    gap_budget = max(6, scale["max_wall_thickness"])
                    for row_idx in range(row_start, row_end + 1):
                        ratio = point_support_ratio(sub_component, effective_col_axis, row_idx, "v")
                        supported = ratio >= support_threshold
                        if supported:
                            if current_start is None:
                                current_start = row_idx
                                current_end = row_idx
                            else:
                                current_end = row_idx
                            gap_budget = max(6, scale["max_wall_thickness"])
                        elif current_start is not None and gap_budget > 0:
                            gap_budget -= 1
                        elif current_start is not None and current_end is not None:
                            support_runs.append((current_start, current_end))
                            current_start = None
                            current_end = None
                            gap_budget = max(6, scale["max_wall_thickness"])
                    if current_start is not None and current_end is not None:
                        support_runs.append((current_start, current_end))
                    for run_start, run_end in support_runs:
                        if run_end - run_start < min_len:
                            rejected_short_count += 1
                            continue
                        line = [x + sx + effective_col_axis, y + sy + run_start, x + sx + effective_col_axis, y + sy + run_end]
                        mx, my = midpoint_of_line(line)
                        if not point_inside_polygon((mx, my), polygon_np):
                            continue
                        if cv2.pointPolygonTest(polygon_np, (float(mx), float(my)), True) <= interior_margin:
                            rejected_outer_count += 1
                            continue
                        support_score = compute_line_support_score(line, "v", source_mask, scale)
                        if support_score <= 0:
                            continue
                        candidates.append(line)
                        axis_emitted = True
                if axis_emitted:
                    local_vertical_axes_debug.append(x + sx + effective_col_axis)

        stripe_axes_debug.extend(
            [{"orientation": "h", "axis": axis, "component_bbox": component_subregion_bboxes[0] if component_subregion_bboxes else [x, y, w, h]} for axis in local_horizontal_axes_debug]
        )
        stripe_axes_debug.extend(
            [{"orientation": "v", "axis": axis, "component_bbox": component_subregion_bboxes[0] if component_subregion_bboxes else [x, y, w, h]} for axis in local_vertical_axes_debug]
        )

        component_debug.append({
            "bbox": [x, y, w, h],
            "area": area,
            "subregion_count": len(subregions),
            "subregion_bboxes": component_subregion_bboxes,
            "emitted_horizontal_axes": local_horizontal_axes_debug,
            "emitted_vertical_axes": local_vertical_axes_debug,
            "emitted_candidate_count": len(candidates) - component_candidates_before,
        })

    candidates = remove_duplicate_lines(candidates, coord_tol=8, length_tol=12)
    candidates = collapse_parallel_double_lines(candidates, pair_tol=scale["pair_merge_tol"])
    candidates = remove_duplicate_lines(candidates, coord_tol=8, length_tol=12)
    outer_margin = max(scale["outer_edge_offset"] * 4, scale["max_wall_thickness"] * 2 + 12)
    candidates, _ = filter_lines_inside_building(candidates, polygon_np, outer_margin=outer_margin)
    candidates, _ = reject_outer_boundary_parallel_candidates(candidates, polygon_np, scale)
    candidates, _ = filter_symbolic_vertical_walls(
        candidates,
        polygon_np,
        source_mask=source_mask,
        scale=scale,
    )
    candidates, _ = merge_collinear_lines(
        candidates,
        pos_tol=8,
        gap_tol=max(14, min(scale["opening_max_gap"], 110)),
        return_stats=True,
    )
    candidates = remove_duplicate_lines(candidates, coord_tol=8, length_tol=12)

    graph_mask = np.zeros_like(source_mask)
    draw_thickness = max(2, scale["min_wall_thickness"])
    for line in candidates:
        cv2.line(graph_mask, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), 255, draw_thickness)
    graph_mask = cv2.dilate(graph_mask, np.ones((3, 3), np.uint8), iterations=1)

    return candidates, graph_mask, {
        "wall_region_graph_enabled": True,
        "wall_region_graph_count": len(candidates),
        "wall_region_graph_segments": [line[:] for line in candidates],
        "wall_region_graph_score": score_inner_wall_set(candidates, polygon_np, source_mask, scale),
        "wall_region_graph_component_count": len(component_debug),
        "wall_region_graph_component_debug": component_debug,
        "wall_region_component_split_enabled": True,
        "wall_region_original_component_count": original_component_count,
        "wall_region_subregion_count": subregion_count,
        "wall_region_split_component_count": split_component_count,
        "wall_region_split_rejected_small_count": split_rejected_small_count,
        "wall_region_split_notes": [
            "projection_valley_detection",
            "thin_bridge_breaking",
            "connected_subregion_relabeling",
            "conservative_large_component_only",
        ],
        "wall_region_component_count": len(component_debug),
        "wall_region_stripe_axis_count_vertical": sum(1 for item in stripe_axes_debug if item["orientation"] == "v"),
        "wall_region_stripe_axis_count_horizontal": sum(1 for item in stripe_axes_debug if item["orientation"] == "h"),
        "wall_region_stripe_axes": stripe_axes_debug,
        "wall_region_rejected_short_count": rejected_short_count,
        "wall_region_rejected_outer_count": rejected_outer_count,
        "wall_region_candidate_generation_notes": [
            "component_projection_peaks",
            "multi_axis_stripe_extraction",
            "local_axis_refinement",
            "run_length_support_scanning",
            "conservative_merge_and_boundary_filters",
        ],
        "wall_region_axis_refinement_enabled": True,
        "wall_region_axis_refined_count": axis_refined_count,
        "wall_region_axis_refinement_candidates": axis_refinement_candidates,
        "wall_region_axis_refinement_rejected_count": axis_refinement_rejected_count,
        "wall_region_axis_refinement_notes": [
            "local_projection_density",
            "support_continuity",
            "bounded_axis_shift",
            "refine_only_if_score_improves",
        ],
        "wall_region_original_stripe_axes": original_stripe_axes_debug,
        "wall_region_refined_stripe_axes": refined_stripe_axes_debug,
        "wall_region_graph_mask_nonzero_count": int(np.count_nonzero(graph_mask > 0)),
        "wall_region_graph_region_mask_nonzero_count": int(np.count_nonzero(region_mask > 0)),
        "wall_region_graph_region_mask": region_mask,
    }


def build_raw_candidate_based_inner_walls(
    raw_segments: List[List[int]],
    polygon_np: np.ndarray,
    source_mask: np.ndarray,
    scale: Dict[str, int],
) -> Tuple[List[List[int]], Dict[str, Any]]:
    if not raw_segments:
        return [], {
            "raw_candidate_based_enabled": True,
            "raw_candidate_based_count": 0,
            "raw_candidate_based_segments": [],
            "raw_candidate_based_score": float("-inf"),
        }

    min_len = max(MIN_WALL_LENGTH_PX, scale["opening_min_gap"] * 2)
    required_thickness = max(scale["min_wall_thickness"], int(round(scale["max_wall_thickness"] * 0.36)))
    outer_margin = max(scale["outer_edge_offset"] * 3, scale["max_wall_thickness"] * 2 + 8)
    gap_tol = max(14, min(scale["opening_max_gap"], 110))

    candidates: List[List[int]] = []
    for line in raw_segments:
        item = normalize_line(line)
        if item["orientation"] not in {"h", "v"}:
            continue
        if line_length(line) < min_len:
            continue
        mx, my = midpoint_of_line(line)
        if not point_inside_polygon((mx, my), polygon_np):
            continue
        if cv2.pointPolygonTest(polygon_np, (float(mx), float(my)), True) <= outer_margin:
            continue
        thickness = estimate_line_thickness(
            line,
            orientation=item["orientation"],
            source_mask=source_mask,
            max_radius=scale["max_wall_thickness"],
        )
        if thickness < required_thickness:
            continue
        candidates.append(line[:])

    candidates = remove_duplicate_lines(candidates, coord_tol=8, length_tol=12)
    candidates = collapse_parallel_double_lines(candidates, pair_tol=scale["pair_merge_tol"])
    candidates = remove_duplicate_lines(candidates, coord_tol=8, length_tol=12)
    candidates, inside_stats = filter_lines_inside_building(candidates, polygon_np, outer_margin=outer_margin)
    candidates, outer_stats = reject_outer_boundary_parallel_candidates(candidates, polygon_np, scale)
    candidates, symbolic_stats = filter_symbolic_vertical_walls(
        candidates,
        polygon_np,
        source_mask=source_mask,
        scale=scale,
    )
    candidates, _ = merge_collinear_lines(
        candidates,
        pos_tol=8,
        gap_tol=gap_tol,
        return_stats=True,
    )
    candidates = remove_duplicate_lines(candidates, coord_tol=8, length_tol=12)

    return candidates, {
        "raw_candidate_based_enabled": True,
        "raw_candidate_based_count": len(candidates),
        "raw_candidate_based_segments": [line[:] for line in candidates],
        "raw_candidate_based_score": score_inner_wall_set(candidates, polygon_np, source_mask, scale),
        "raw_candidate_based_filter_stats": {
            "inside_building": inside_stats,
            "outer_boundary": {
                "outer_boundary_rejected_count": outer_stats.get("outer_boundary_rejected_count", 0),
            },
            "symbolic_vertical": symbolic_stats,
        },
    }


def build_raw_axis_reconstructed_inner_walls(
    raw_segments: List[List[int]],
    polygon_np: np.ndarray,
    source_mask: np.ndarray,
    scale: Dict[str, int],
) -> Tuple[List[List[int]], np.ndarray, Dict[str, Any]]:
    empty_mask = np.zeros_like(source_mask)
    if not raw_segments:
        return [], empty_mask, {
            "raw_axis_recon_enabled": True,
            "raw_axis_recon_count": 0,
            "raw_axis_recon_segments": [],
            "raw_axis_recon_score": float("-inf"),
            "raw_axis_recon_cluster_count_h": 0,
            "raw_axis_recon_cluster_count_v": 0,
        }

    min_len = max(MIN_WALL_LENGTH_PX, int(round(scale["opening_min_gap"] * 1.7)))
    axis_tol = max(8, int(round(scale["max_wall_thickness"] * 1.35)))
    gap_tol = max(14, min(scale["opening_max_gap"], 110))
    outer_margin = max(scale["outer_edge_offset"] * 3, scale["max_wall_thickness"] * 2 + 8)
    required_thickness = max(scale["min_wall_thickness"], int(round(scale["max_wall_thickness"] * 0.34)))

    filtered: List[List[int]] = []
    for line in raw_segments:
        item = normalize_line(line)
        if item["orientation"] not in {"h", "v"}:
            continue
        if line_length(line) < max(min_len * 0.65, 30):
            continue
        mx, my = midpoint_of_line(line)
        if not point_inside_polygon((mx, my), polygon_np):
            continue
        if cv2.pointPolygonTest(polygon_np, (float(mx), float(my)), True) <= outer_margin:
            continue
        filtered.append(line[:])

    filtered = remove_duplicate_lines(filtered, coord_tol=6, length_tol=10)

    def reconstruct_for_orientation(orientation: str) -> Tuple[List[List[int]], List[Dict[str, Any]]]:
        oriented = [normalize_line(line) for line in filtered if normalize_line(line)["orientation"] == orientation]
        if not oriented:
            return [], []

        oriented.sort(key=lambda item: (item["fixed"], item["start"], item["end"]))
        axis_groups: List[Dict[str, Any]] = []
        for item in oriented:
            length = max(1, int(item["end"] - item["start"]))
            placed = False
            for group in axis_groups:
                if abs(int(group["fixed"]) - int(item["fixed"])) > axis_tol:
                    continue
                group["items"].append(item)
                group["fixed_values"].append(int(item["fixed"]))
                group["weights"].append(length)
                group["fixed"] = int(round(sum(v * w for v, w in zip(group["fixed_values"], group["weights"])) / max(1, sum(group["weights"]))))
                placed = True
                break
            if not placed:
                axis_groups.append({
                    "fixed": int(item["fixed"]),
                    "items": [item],
                    "fixed_values": [int(item["fixed"])],
                    "weights": [length],
                })

        reconstructed: List[List[int]] = []
        cluster_debug: List[Dict[str, Any]] = []
        for group in axis_groups:
            spans = sorted((int(item["start"]), int(item["end"])) for item in group["items"])
            merged_spans: List[List[int]] = []
            for start, end in spans:
                if not merged_spans or start > merged_spans[-1][1] + gap_tol:
                    merged_spans.append([start, end])
                else:
                    merged_spans[-1][1] = max(merged_spans[-1][1], end)

            fixed = int(group["fixed"])
            cluster_info = {
                "orientation": orientation,
                "fixed": fixed,
                "member_count": len(group["items"]),
                "spans": [],
            }
            for start, end in merged_spans:
                candidate = denormalize_line({
                    "orientation": orientation,
                    "fixed": fixed,
                    "start": int(start),
                    "end": int(end),
                })
                length = line_length(candidate)
                if length < min_len:
                    continue
                thickness = estimate_line_thickness(
                    candidate,
                    orientation=orientation,
                    source_mask=source_mask,
                    max_radius=scale["max_wall_thickness"],
                )
                support = compute_line_support_score(candidate, orientation, source_mask, scale)
                if thickness < required_thickness or support < 0.34:
                    continue
                reconstructed.append(candidate)
                cluster_info["spans"].append({
                    "start": int(start),
                    "end": int(end),
                    "length": round(float(length), 1),
                    "thickness": round(float(thickness), 2),
                    "support": round(float(support), 3),
                })
            if cluster_info["spans"]:
                cluster_debug.append(cluster_info)

        return reconstructed, cluster_debug

    horizontal, h_debug = reconstruct_for_orientation("h")
    vertical, v_debug = reconstruct_for_orientation("v")
    candidates = horizontal + vertical
    candidates = collapse_parallel_double_lines(candidates, pair_tol=scale["pair_merge_tol"])
    candidates, _ = merge_collinear_lines(
        candidates,
        pos_tol=8,
        gap_tol=gap_tol,
        return_stats=True,
    )
    candidates = remove_duplicate_lines(candidates, coord_tol=8, length_tol=12)
    base_candidate_score = score_inner_wall_set(candidates, polygon_np, source_mask, scale)
    supplemented_candidates, supplement_stats = supplement_raw_axis_with_orphan_segments(
        candidates,
        filtered,
        polygon_np,
        source_mask,
        scale,
    )
    if float(supplement_stats.get("score", float("-inf"))) >= base_candidate_score + 0.05:
        candidates = supplemented_candidates
    candidates, _ = filter_lines_inside_building(candidates, polygon_np, outer_margin=outer_margin)
    candidates, outer_stats = reject_outer_boundary_parallel_candidates(candidates, polygon_np, scale)
    candidates, symbolic_stats = filter_symbolic_vertical_walls(
        candidates,
        polygon_np,
        source_mask=source_mask,
        scale=scale,
    )
    candidates = remove_duplicate_lines(candidates, coord_tol=8, length_tol=12)

    line_mask = np.zeros_like(source_mask)
    draw_thickness = max(2, scale["min_wall_thickness"])
    for line in candidates:
        cv2.line(line_mask, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), 255, draw_thickness)
    line_mask = cv2.dilate(line_mask, np.ones((3, 3), np.uint8), iterations=1)

    return candidates, line_mask, {
        "raw_axis_recon_enabled": True,
        "raw_axis_recon_count": len(candidates),
        "raw_axis_recon_segments": [line[:] for line in candidates],
        "raw_axis_recon_score": score_inner_wall_set(candidates, polygon_np, source_mask, scale),
        "raw_axis_recon_cluster_count_h": len(h_debug),
        "raw_axis_recon_cluster_count_v": len(v_debug),
        "raw_axis_recon_clusters": h_debug + v_debug,
        "raw_axis_recon_outer_rejects": int(outer_stats.get("outer_boundary_rejected_count", 0)),
        "raw_axis_recon_symbolic_rejects": int(symbolic_stats.get("rejected_symbolic_vertical_lines", 0)),
        "raw_axis_recon_orphan_added_count": int(supplement_stats.get("added_count", 0)),
        "raw_axis_recon_orphan_added_segments": supplement_stats.get("added_segments", []),
    }


def build_junction_partitioned_wall_set(
    lines: List[List[int]],
    polygon_np: np.ndarray,
    source_mask: np.ndarray,
    scale: Dict[str, int],
) -> Tuple[List[List[int]], Dict[str, Any]]:
    if not lines:
        return [], {
            "enabled": True,
            "count": 0,
            "segments": [],
            "split_count": 0,
            "score": float("-inf"),
        }

    split_tol = max(12, scale["opening_min_gap"])
    min_piece_len = max(MIN_WALL_LENGTH_PX, int(round(scale["opening_min_gap"] * 1.3)))
    outer_margin = max(scale["outer_edge_offset"] * 3, scale["max_wall_thickness"] * 2 + 6)
    required_thickness = max(scale["min_wall_thickness"], int(round(scale["max_wall_thickness"] * 0.34)))

    result: List[List[int]] = []
    split_count = 0
    normalized = [normalize_line(line) for line in lines]

    for idx, line in enumerate(lines):
        item = normalized[idx]
        split_positions = [int(item["start"]), int(item["end"])]
        for other_idx, other in enumerate(lines):
            if other_idx == idx:
                continue
            other_item = normalized[other_idx]
            if other_item["orientation"] == item["orientation"]:
                continue
            if int(other_item["start"]) - split_tol <= int(item["fixed"]) <= int(other_item["end"]) + split_tol:
                cross_pos = int(other_item["fixed"])
                if int(item["start"]) + min_piece_len // 3 < cross_pos < int(item["end"]) - min_piece_len // 3:
                    split_positions.append(cross_pos)

        split_positions = sorted(set(split_positions))
        if len(split_positions) <= 2:
            result.append(line[:])
            continue

        local_pieces: List[List[int]] = []
        for start, end in zip(split_positions[:-1], split_positions[1:]):
            if end - start < min_piece_len:
                continue
            piece = denormalize_line({
                "orientation": item["orientation"],
                "fixed": int(item["fixed"]),
                "start": int(start),
                "end": int(end),
            })
            mx, my = midpoint_of_line(piece)
            if cv2.pointPolygonTest(polygon_np, (float(mx), float(my)), True) <= outer_margin:
                continue
            thickness = estimate_line_thickness(piece, item["orientation"], source_mask, scale["max_wall_thickness"])
            support = compute_line_support_score(piece, item["orientation"], source_mask, scale)
            if thickness < required_thickness or support < 0.24:
                continue
            local_pieces.append(piece)

        if len(local_pieces) >= 2:
            result.extend(local_pieces)
            split_count += 1
        else:
            result.append(line[:])

    result, _ = merge_collinear_lines(
        result,
        pos_tol=8,
        gap_tol=max(14, min(scale["opening_max_gap"], 110)),
        return_stats=True,
    )
    result = remove_duplicate_lines(result, coord_tol=8, length_tol=12)
    return result, {
        "enabled": True,
        "count": len(result),
        "segments": [line[:] for line in result],
        "split_count": split_count,
        "score": score_inner_wall_set(result, polygon_np, source_mask, scale),
    }


def supplement_raw_axis_with_orphan_segments(
    base_lines: List[List[int]],
    raw_segments: List[List[int]],
    polygon_np: np.ndarray,
    source_mask: np.ndarray,
    scale: Dict[str, int],
) -> Tuple[List[List[int]], Dict[str, Any]]:
    if not base_lines or not raw_segments:
        return [line[:] for line in base_lines], {
            "enabled": True,
            "count": len(base_lines),
            "added_count": 0,
            "added_segments": [],
            "score": score_inner_wall_set(base_lines, polygon_np, source_mask, scale) if base_lines else float("-inf"),
        }

    axis_tol = max(10, int(round(scale["max_wall_thickness"] * 1.5)))
    overlap_tol = 0.45
    min_len = max(MIN_WALL_LENGTH_PX, int(round(scale["opening_min_gap"] * 1.55)))
    outer_margin = max(scale["outer_edge_offset"] * 3, scale["max_wall_thickness"] * 2 + 6)
    required_thickness = max(scale["min_wall_thickness"], int(round(scale["max_wall_thickness"] * 0.34)))

    base_norm = [normalize_line(line) for line in base_lines]
    candidates: List[Tuple[float, List[int]]] = []

    def span_overlap_ratio(a: Dict[str, Any], b: Dict[str, Any]) -> float:
        overlap = max(0, min(int(a["end"]), int(b["end"])) - max(int(a["start"]), int(b["start"])))
        shorter = max(1, min(int(a["end"]) - int(a["start"]), int(b["end"]) - int(b["start"])))
        return overlap / float(shorter)

    for line in raw_segments:
        item = normalize_line(line)
        if item["orientation"] not in {"h", "v"}:
            continue
        length = line_length(line)
        if length < min_len:
            continue
        mx, my = midpoint_of_line(line)
        if not point_inside_polygon((mx, my), polygon_np):
            continue
        if cv2.pointPolygonTest(polygon_np, (float(mx), float(my)), True) <= outer_margin:
            continue
        thickness = estimate_line_thickness(line, item["orientation"], source_mask, scale["max_wall_thickness"])
        support = compute_line_support_score(line, item["orientation"], source_mask, scale)
        if thickness < required_thickness or support < 0.28:
            continue

        overlaps_existing = False
        for existing_norm in base_norm:
            if existing_norm["orientation"] != item["orientation"]:
                continue
            if abs(int(existing_norm["fixed"]) - int(item["fixed"])) > axis_tol:
                continue
            if span_overlap_ratio(existing_norm, item) >= overlap_tol:
                overlaps_existing = True
                break
        if overlaps_existing:
            continue

        orphan_score = float(support) + min(float(thickness), 12.0) + min(length / 180.0, 4.0)
        candidates.append((orphan_score, [int(v) for v in line]))

    candidates.sort(key=lambda pair: pair[0], reverse=True)
    added_segments = [line for _, line in candidates[:2]]
    combined = [line[:] for line in base_lines] + [line[:] for line in added_segments]
    combined, _ = merge_collinear_lines(
        combined,
        pos_tol=8,
        gap_tol=max(14, min(scale["opening_max_gap"], 110)),
        return_stats=True,
    )
    combined = remove_duplicate_lines(combined, coord_tol=8, length_tol=12)
    return combined, {
        "enabled": True,
        "count": len(combined),
        "added_count": len(added_segments),
        "added_segments": [line[:] for line in added_segments],
        "score": score_inner_wall_set(combined, polygon_np, source_mask, scale),
    }


def build_boundary_adjacent_recovery_wall_set(
    base_lines: List[List[int]],
    raw_segments: List[List[int]],
    polygon_np: np.ndarray,
    source_mask: np.ndarray,
    scale: Dict[str, int],
) -> Tuple[List[List[int]], Dict[str, Any]]:
    if not base_lines or not raw_segments:
        return [line[:] for line in base_lines], {
            "enabled": True,
            "count": len(base_lines),
            "added_count": 0,
            "added_segments": [],
            "score": score_inner_wall_set(base_lines, polygon_np, source_mask, scale) if base_lines else float("-inf"),
        }

    axis_tol = max(10, int(round(scale["max_wall_thickness"] * 1.5)))
    min_len = max(MIN_WALL_LENGTH_PX, int(round(scale["opening_min_gap"] * 1.45)))
    outer_min = max(scale["outer_edge_offset"], 6)
    outer_max = max(scale["outer_edge_offset"] * 5, scale["max_wall_thickness"] * 3 + 10)
    required_thickness = max(scale["min_wall_thickness"], int(round(scale["max_wall_thickness"] * 0.34)))
    anchor_tol = max(12, scale["max_wall_thickness"] * 2)
    overlap_tol = 0.35

    base_norm = [normalize_line(line) for line in base_lines]

    def span_overlap_ratio(a: Dict[str, Any], b: Dict[str, Any]) -> float:
        overlap = max(0, min(int(a["end"]), int(b["end"])) - max(int(a["start"]), int(b["start"])))
        shorter = max(1, min(int(a["end"]) - int(a["start"]), int(b["end"]) - int(b["start"])))
        return overlap / float(shorter)

    def has_anchor(line: List[int], pool: List[List[int]]) -> bool:
        norm = normalize_line(line)
        endpoints = line_endpoints(line)
        for other in pool:
            other_norm = normalize_line(other)
            if other_norm["orientation"] == norm["orientation"]:
                continue
            for pt in endpoints:
                if any(points_close(pt, opt, anchor_tol) for opt in line_endpoints(other)):
                    return True
                if point_hits_item_body(pt, other_norm, anchor_tol):
                    return True
        return False

    scored: List[Tuple[float, List[int]]] = []
    for line in raw_segments:
        norm = normalize_line(line)
        if norm["orientation"] not in {"h", "v"}:
            continue
        length = line_length(line)
        if length < min_len:
            continue
        mx, my = midpoint_of_line(line)
        signed_dist = cv2.pointPolygonTest(polygon_np, (float(mx), float(my)), True)
        if signed_dist < outer_min or signed_dist > outer_max:
            continue
        thickness = estimate_line_thickness(line, norm["orientation"], source_mask, scale["max_wall_thickness"])
        support = compute_line_support_score(line, norm["orientation"], source_mask, scale)
        if thickness < required_thickness or support < 9.0:
            continue
        overlaps_existing = False
        for existing in base_norm:
            if existing["orientation"] != norm["orientation"]:
                continue
            if abs(int(existing["fixed"]) - int(norm["fixed"])) > axis_tol:
                continue
            if span_overlap_ratio(existing, norm) >= overlap_tol:
                overlaps_existing = True
                break
        if overlaps_existing:
            continue
        if not has_anchor(line, base_lines):
            continue
        score = float(support) + min(float(thickness), 12.0) + min(length / 160.0, 5.0)
        scored.append((score, [int(v) for v in line]))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    added = [line for _, line in scored[:4]]
    combined = [line[:] for line in base_lines] + [line[:] for line in added]
    combined, _ = merge_collinear_lines(
        combined,
        pos_tol=8,
        gap_tol=max(14, min(scale["opening_max_gap"], 110)),
        return_stats=True,
    )
    combined = remove_duplicate_lines(combined, coord_tol=8, length_tol=12)
    return combined, {
        "enabled": True,
        "count": len(combined),
        "added_count": len(added),
        "added_segments": [line[:] for line in added],
        "score": score_inner_wall_set(combined, polygon_np, source_mask, scale),
    }


def build_orthogonal_clean_plan_inner_walls(
    source_mask: np.ndarray,
    polygon_np: np.ndarray,
    scale: Dict[str, int],
) -> Tuple[List[List[int]], np.ndarray, Dict[str, Any]]:
    empty_mask = np.zeros_like(source_mask)
    if np.count_nonzero(source_mask) == 0:
        return [], empty_mask, {
            "orthogonal_clean_enabled": True,
            "orthogonal_clean_count": 0,
            "orthogonal_clean_segments": [],
            "orthogonal_clean_score": float("-inf"),
            "orthogonal_clean_rejected_short": 0,
            "orthogonal_clean_rejected_outer": 0,
            "orthogonal_clean_rejected_symbolic": 0,
        }

    polygon_mask = np.zeros_like(source_mask)
    cv2.fillPoly(polygon_mask, [polygon_np], 255)
    interior_margin = max(scale["outer_edge_offset"] * 3, scale["max_wall_thickness"] * 2 + 6)
    interior_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (max(3, interior_margin * 2 + 1), max(3, interior_margin * 2 + 1)),
    )
    interior_mask = cv2.erode(polygon_mask, interior_kernel, iterations=1)
    clean_mask = cv2.bitwise_and(source_mask, interior_mask)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)

    min_len = max(MIN_WALL_LENGTH_PX, scale["opening_min_gap"] * 2)
    min_overlap_ratio = 0.58
    max_band = max(scale["max_wall_thickness"] * 3, 16)
    rejected_short = 0

    def contiguous_runs(indices: List[int]) -> List[Tuple[int, int]]:
        if not indices:
            return []
        runs: List[Tuple[int, int]] = []
        start = indices[0]
        prev = indices[0]
        for value in indices[1:]:
            if value == prev + 1:
                prev = value
                continue
            runs.append((start, prev))
            start = value
            prev = value
        runs.append((start, prev))
        return runs

    def overlap_ratio(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
        overlap = max(0, min(a_end, b_end) - max(a_start, b_start))
        shorter = max(1, min(a_end - a_start, b_end - b_start))
        return overlap / float(shorter)

    def scan_runs(orientation: str) -> List[List[int]]:
        h, w = clean_mask.shape[:2]
        row_groups: List[Dict[str, Any]] = []
        if orientation == "h":
            for y in range(h):
                xs = np.where(clean_mask[y, :] > 0)[0].tolist()
                for start, end in contiguous_runs(xs):
                    if end - start < min_len:
                        rejected_short_nonlocal[0] += 1
                        continue
                    assigned = False
                    for group in row_groups:
                        if abs(y - group["last_axis"]) > max_band:
                            continue
                        if overlap_ratio(start, end, group["start"], group["end"]) < min_overlap_ratio:
                            continue
                        group["rows"].append(y)
                        group["start"] = min(group["start"], start)
                        group["end"] = max(group["end"], end)
                        group["last_axis"] = y
                        assigned = True
                        break
                    if not assigned:
                        row_groups.append({
                            "rows": [y],
                            "start": start,
                            "end": end,
                            "last_axis": y,
                        })
            lines: List[List[int]] = []
            for group in row_groups:
                if len(group["rows"]) > max_band:
                    continue
                center_y = int(round(float(np.median(group["rows"]))))
                line = [int(group["start"]), center_y, int(group["end"]), center_y]
                lines.append(line)
            return lines

        for x in range(w):
            ys = np.where(clean_mask[:, x] > 0)[0].tolist()
            for start, end in contiguous_runs(ys):
                if end - start < min_len:
                    rejected_short_nonlocal[0] += 1
                    continue
                assigned = False
                for group in row_groups:
                    if abs(x - group["last_axis"]) > max_band:
                        continue
                    if overlap_ratio(start, end, group["start"], group["end"]) < min_overlap_ratio:
                        continue
                    group["rows"].append(x)
                    group["start"] = min(group["start"], start)
                    group["end"] = max(group["end"], end)
                    group["last_axis"] = x
                    assigned = True
                    break
                if not assigned:
                    row_groups.append({
                        "rows": [x],
                        "start": start,
                        "end": end,
                        "last_axis": x,
                    })
        lines = []
        for group in row_groups:
            if len(group["rows"]) > max_band:
                continue
            center_x = int(round(float(np.median(group["rows"]))))
            lines.append([center_x, int(group["start"]), center_x, int(group["end"])])
        return lines

    rejected_short_nonlocal = [0]
    horizontal_lines = scan_runs("h")
    vertical_lines = scan_runs("v")
    candidates = horizontal_lines + vertical_lines
    candidates = remove_duplicate_lines(candidates, coord_tol=8, length_tol=12)
    candidates = collapse_parallel_double_lines(candidates, pair_tol=scale["pair_merge_tol"])
    candidates = remove_duplicate_lines(candidates, coord_tol=8, length_tol=12)
    outer_margin = max(scale["outer_edge_offset"] * 4, scale["max_wall_thickness"] * 2 + 12)
    before_outer = len(candidates)
    candidates, _ = filter_lines_inside_building(candidates, polygon_np, outer_margin=outer_margin)
    candidates, outer_stats = reject_outer_boundary_parallel_candidates(candidates, polygon_np, scale)
    rejected_outer = (before_outer - len(candidates)) + int(outer_stats.get("outer_boundary_rejected_count", 0))
    before_symbolic = len(candidates)
    candidates, _ = filter_symbolic_vertical_walls(
        candidates,
        polygon_np,
        source_mask=clean_mask,
        scale=scale,
    )
    rejected_symbolic = before_symbolic - len(candidates)
    candidates, _ = merge_collinear_lines(
        candidates,
        pos_tol=8,
        gap_tol=max(14, min(scale["opening_max_gap"], 110)),
        return_stats=True,
    )
    candidates = remove_duplicate_lines(candidates, coord_tol=8, length_tol=12)

    line_mask = np.zeros_like(source_mask)
    draw_thickness = max(2, scale["min_wall_thickness"])
    for line in candidates:
        cv2.line(line_mask, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), 255, draw_thickness)
    line_mask = cv2.dilate(line_mask, np.ones((3, 3), np.uint8), iterations=1)

    return candidates, line_mask, {
        "orthogonal_clean_enabled": True,
        "orthogonal_clean_count": len(candidates),
        "orthogonal_clean_segments": [line[:] for line in candidates],
        "orthogonal_clean_score": score_inner_wall_set(candidates, polygon_np, clean_mask, scale),
        "orthogonal_clean_rejected_short": int(rejected_short_nonlocal[0]),
        "orthogonal_clean_rejected_outer": int(rejected_outer),
        "orthogonal_clean_rejected_symbolic": int(rejected_symbolic),
        "orthogonal_clean_mask": clean_mask,
    }


def build_line_evidence_inner_walls(
    source_mask: np.ndarray,
    polygon_np: np.ndarray,
    scale: Dict[str, int],
) -> Tuple[List[List[int]], np.ndarray, Dict[str, Any]]:
    empty_mask = np.zeros_like(source_mask)
    if np.count_nonzero(source_mask) == 0:
        return [], empty_mask, {
            "line_evidence_enabled": True,
            "line_evidence_count": 0,
            "line_evidence_segments": [],
            "line_evidence_score": float("-inf"),
            "line_evidence_raw_count": 0,
            "line_evidence_rejected_short": 0,
            "line_evidence_rejected_outer": 0,
            "line_evidence_rejected_symbolic": 0,
        }

    polygon_mask = np.zeros_like(source_mask)
    cv2.fillPoly(polygon_mask, [polygon_np], 255)
    interior_margin = max(scale["outer_edge_offset"] * 3, scale["max_wall_thickness"] * 2 + 6)
    interior_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (max(3, interior_margin * 2 + 1), max(3, interior_margin * 2 + 1)),
    )
    interior_mask = cv2.erode(polygon_mask, interior_kernel, iterations=1)
    work_mask = cv2.bitwise_and(source_mask, interior_mask)
    work_mask = cv2.morphologyEx(work_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    hk = max(scale["opening_min_gap"] * 2, scale["horizontal_kernel"] // 2)
    vk = max(scale["opening_min_gap"] * 2, scale["vertical_kernel"] // 2)
    h_mask = cv2.morphologyEx(
        work_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1)),
    )
    v_mask = cv2.morphologyEx(
        work_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk)),
    )
    line_source = cv2.bitwise_or(h_mask, v_mask)

    raw_lines = cv2.HoughLinesP(
        line_source,
        rho=1,
        theta=np.pi / 180.0,
        threshold=max(30, scale["opening_min_gap"]),
        minLineLength=max(MIN_WALL_LENGTH_PX, scale["opening_min_gap"] * 2),
        maxLineGap=max(10, scale["max_wall_thickness"] * 2),
    )

    min_len = max(MIN_WALL_LENGTH_PX, scale["opening_min_gap"] * 2)
    required_thickness = max(scale["min_wall_thickness"], int(round(scale["max_wall_thickness"] * 0.4)))
    rejected_short = 0
    rejected_symbolic = 0
    candidates: List[List[int]] = []
    raw_count = 0 if raw_lines is None else int(len(raw_lines))

    if raw_lines is not None:
        for raw_line in raw_lines:
            x1, y1, x2, y2 = [int(v) for v in raw_line[0]]
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            if max(dx, dy) < min_len:
                rejected_short += 1
                continue
            if dx >= dy * 2:
                y_mid = int(round((y1 + y2) / 2.0))
                line = [min(x1, x2), y_mid, max(x1, x2), y_mid]
            elif dy >= dx * 2:
                x_mid = int(round((x1 + x2) / 2.0))
                line = [x_mid, min(y1, y2), x_mid, max(y1, y2)]
            else:
                continue
            mx, my = midpoint_of_line(line)
            if not point_inside_polygon((mx, my), polygon_np):
                continue
            if cv2.pointPolygonTest(polygon_np, (float(mx), float(my)), True) <= interior_margin:
                continue
            candidates.append(line)

    def span_overlap_ratio(a: Dict[str, Any], b: Dict[str, Any]) -> float:
        overlap = max(0, min(int(a["end"]), int(b["end"])) - max(int(a["start"]), int(b["start"])))
        shorter = max(1, min(int(a["end"]) - int(a["start"]), int(b["end"]) - int(b["start"])))
        return overlap / float(shorter)

    def line_structural_score(line: List[int]) -> float:
        item = normalize_line(line)
        support = compute_line_support_score(line, item["orientation"], work_mask, scale)
        thickness = estimate_line_thickness(
            line,
            orientation=item["orientation"],
            source_mask=work_mask,
            max_radius=scale["max_wall_thickness"],
        )
        length = line_length(line)
        mx, my = midpoint_of_line(line)
        boundary_dist = max(0.0, cv2.pointPolygonTest(polygon_np, (float(mx), float(my)), True))
        return (support * 18.0) + (thickness * 4.0) + (min(length, 320.0) * 0.12) + (min(boundary_dist, 60.0) * 0.05)

    loose_candidates = remove_duplicate_lines(candidates, coord_tol=8, length_tol=12)
    loose_candidates = collapse_parallel_double_lines(loose_candidates, pair_tol=scale["pair_merge_tol"])
    loose_candidates = remove_duplicate_lines(loose_candidates, coord_tol=8, length_tol=12)
    normalized_candidates = [normalize_line(line) for line in loose_candidates]
    line_scores = [line_structural_score(line) for line in loose_candidates]
    axis_tol = max(8, int(round(scale["max_wall_thickness"] * 1.4)))
    gap_tol = max(14, min(scale["opening_max_gap"], 110))
    grouped: List[Dict[str, Any]] = []

    for idx, item in enumerate(normalized_candidates):
        length = int(item["end"] - item["start"])
        thickness = estimate_line_thickness(
            loose_candidates[idx],
            orientation=item["orientation"],
            source_mask=work_mask,
            max_radius=scale["max_wall_thickness"],
        )
        if length < min_len or thickness < required_thickness:
            rejected_symbolic += 1
            continue

        assigned = False
        for group in grouped:
            if group["orientation"] != item["orientation"]:
                continue
            if abs(int(group["fixed"]) - int(item["fixed"])) > axis_tol:
                continue
            group_item = {
                "orientation": group["orientation"],
                "fixed": int(group["fixed"]),
                "start": int(group["start"]),
                "end": int(group["end"]),
            }
            same_span = span_overlap_ratio(group_item, item) >= 0.35
            gap_close = max(int(item["start"]), int(group["start"])) - min(int(item["end"]), int(group["end"])) <= gap_tol
            if not same_span and not gap_close:
                continue
            group["members"].append(idx)
            group["weights"].append(max(line_scores[idx], 1.0))
            group["fixed_values"].append(int(item["fixed"]))
            group["start"] = min(int(group["start"]), int(item["start"]))
            group["end"] = max(int(group["end"]), int(item["end"]))
            assigned = True
            break
        if not assigned:
            grouped.append({
                "orientation": item["orientation"],
                "fixed": int(item["fixed"]),
                "start": int(item["start"]),
                "end": int(item["end"]),
                "members": [idx],
                "weights": [max(line_scores[idx], 1.0)],
                "fixed_values": [int(item["fixed"])],
            })

    scored_lines: List[Dict[str, Any]] = []
    for group in grouped:
        total_weight = sum(group["weights"])
        fixed = int(round(sum(v * w for v, w in zip(group["fixed_values"], group["weights"])) / max(total_weight, 1.0)))
        merged_item = {
            "orientation": group["orientation"],
            "fixed": fixed,
            "start": int(group["start"]),
            "end": int(group["end"]),
        }
        merged_line = denormalize_line(merged_item)
        group_size = len(group["members"])
        merged_score = line_structural_score(merged_line) + min(group_size * 2.5, 10.0)
        if group_size == 1 and line_length(merged_line) < min_len * 1.35:
            rejected_symbolic += 1
            continue
        scored_lines.append({
            "line": merged_line,
            "score": merged_score,
            "group_size": group_size,
        })

    scored_lines.sort(key=lambda item: item["score"], reverse=True)
    loose_segment_count = len(loose_candidates)
    scored_debug = [
        {
            "line": item["line"][:],
            "score": round(float(item["score"]), 3),
            "group_size": int(item["group_size"]),
        }
        for item in scored_lines
    ]

    if scored_lines:
        best_score = float(scored_lines[0]["score"])
        score_values = [float(item["score"]) for item in scored_lines]
        quantile_floor = float(np.percentile(score_values, 45)) if len(score_values) >= 3 else min(score_values)
        min_keep_score = max(10.0, min(best_score * 0.34, quantile_floor))
        max_candidates = max(8, min(16, int(round(len(scored_lines) * 0.7))))
        kept: List[Dict[str, Any]] = []
        per_orientation = {"h": 0, "v": 0}
        for item in scored_lines:
            line = item["line"]
            orientation = normalize_line(line)["orientation"]
            if item["score"] < min_keep_score:
                continue
            if per_orientation[orientation] >= max_candidates // 2 + 1:
                continue
            kept.append(item)
            per_orientation[orientation] += 1
            if len(kept) >= max_candidates:
                break
        candidates = [item["line"] for item in kept]
    else:
        candidates = []

    candidates = remove_duplicate_lines(candidates, coord_tol=8, length_tol=12)
    candidates = collapse_parallel_double_lines(candidates, pair_tol=scale["pair_merge_tol"])
    candidates = remove_duplicate_lines(candidates, coord_tol=8, length_tol=12)
    outer_margin = max(scale["outer_edge_offset"] * 4, scale["max_wall_thickness"] * 2 + 12)
    before_outer = len(candidates)
    candidates, _ = filter_lines_inside_building(candidates, polygon_np, outer_margin=outer_margin)
    candidates, outer_stats = reject_outer_boundary_parallel_candidates(candidates, polygon_np, scale)
    rejected_outer = (before_outer - len(candidates)) + int(outer_stats.get("outer_boundary_rejected_count", 0))
    before_symbolic = len(candidates)
    candidates, _ = filter_symbolic_vertical_walls(
        candidates,
        polygon_np,
        source_mask=work_mask,
        scale=scale,
    )
    rejected_symbolic += before_symbolic - len(candidates)
    candidates, _ = merge_collinear_lines(
        candidates,
        pos_tol=8,
        gap_tol=max(14, min(scale["opening_max_gap"], 110)),
        return_stats=True,
    )
    candidates = remove_duplicate_lines(candidates, coord_tol=8, length_tol=12)

    line_mask = np.zeros_like(source_mask)
    draw_thickness = max(2, scale["min_wall_thickness"])
    for line in candidates:
        cv2.line(line_mask, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), 255, draw_thickness)
    line_mask = cv2.dilate(line_mask, np.ones((3, 3), np.uint8), iterations=1)

    return candidates, line_mask, {
        "line_evidence_enabled": True,
        "line_evidence_count": len(candidates),
        "line_evidence_segments": [line[:] for line in candidates],
        "line_evidence_loose_count": loose_segment_count,
        "line_evidence_loose_segments": [line[:] for line in loose_candidates],
        "line_evidence_scored_segments": scored_debug,
        "line_evidence_score": score_inner_wall_set(candidates, polygon_np, work_mask, scale),
        "line_evidence_raw_count": raw_count,
        "line_evidence_rejected_short": int(rejected_short),
        "line_evidence_rejected_outer": int(rejected_outer),
        "line_evidence_rejected_symbolic": int(rejected_symbolic),
        "line_evidence_mask": line_source,
    }


def build_clean_plan_mode_inner_walls(
    polygon_np: np.ndarray,
    source_mask: np.ndarray,
    scale: Dict[str, int],
    raw_axis_segments: List[List[int]],
    line_evidence_segments: List[List[int]],
    orthogonal_segments: List[List[int]],
    semantic_segments: List[List[int]],
    raw_candidate_segments: List[List[int]],
) -> Tuple[List[List[int]], np.ndarray, Dict[str, Any]]:
    empty_mask = np.zeros_like(source_mask)
    stats: Dict[str, Any] = {
        "clean_plan_mode_enabled": True,
        "clean_plan_mode_count": 0,
        "clean_plan_mode_segments": [],
        "clean_plan_mode_score": float("-inf"),
        "clean_plan_mode_group_count": 0,
        "clean_plan_mode_consensus_kept": 0,
        "clean_plan_mode_single_source_kept": 0,
        "clean_plan_mode_anchor_supplement_kept": 0,
        "clean_plan_mode_clean_like": False,
        "clean_plan_mode_rejected_outer": 0,
        "clean_plan_mode_rejected_symbolic": 0,
        "clean_plan_mode_sources": {},
    }
    if np.count_nonzero(source_mask) == 0:
        return [], empty_mask, stats

    source_defs = [
        ("raw_axis", raw_axis_segments, 3.4),
        ("line_evidence", line_evidence_segments, 2.6),
        ("orthogonal", orthogonal_segments, 2.1),
        ("semantic", semantic_segments, 1.7),
        ("raw_candidate", raw_candidate_segments, 1.2),
    ]
    stats["clean_plan_mode_sources"] = {name: len(lines) for name, lines, _ in source_defs}

    polygon_mask = np.zeros_like(source_mask)
    cv2.fillPoly(polygon_mask, [polygon_np], 255)
    interior_margin = max(scale["outer_edge_offset"] * 3, scale["max_wall_thickness"] * 2 + 6)
    interior_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (max(3, interior_margin * 2 + 1), max(3, interior_margin * 2 + 1)),
    )
    interior_mask = cv2.erode(polygon_mask, interior_kernel, iterations=1)
    work_mask = cv2.bitwise_and(source_mask, interior_mask)
    work_mask = cv2.morphologyEx(work_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    axis_tol = max(8, int(round(scale["max_wall_thickness"] * 1.6)))
    gap_tol = max(14, min(scale["opening_max_gap"], 110))
    min_len = max(MIN_WALL_LENGTH_PX, scale["opening_min_gap"] * 2)
    required_thickness = max(scale["min_wall_thickness"], int(round(scale["max_wall_thickness"] * 0.42)))
    anchor_tol = max(10, int(round(scale["max_wall_thickness"] * 1.6)))
    outer_segments = polygon_to_axis_aligned_segments(
        [[int(pt[0]), int(pt[1])] for pt in polygon_np.reshape(-1, 2)],
        axis_tol=10,
        min_length=max(24, scale["opening_min_gap"]),
    )
    clean_like = (
        len(outer_segments) >= 4
        and all(normalize_line(seg)["orientation"] in {"h", "v"} for seg in outer_segments)
        and sum(1 for _, lines, _ in source_defs if lines) >= 3
    )
    stats["clean_plan_mode_clean_like"] = clean_like

    def span_overlap_ratio(a: Dict[str, Any], b: Dict[str, Any]) -> float:
        overlap = max(0, min(int(a["end"]), int(b["end"])) - max(int(a["start"]), int(b["start"])))
        shorter = max(1, min(int(a["end"]) - int(a["start"]), int(b["end"]) - int(b["start"])))
        return overlap / float(shorter)

    def line_structural_score(line: List[int]) -> float:
        item = normalize_line(line)
        orientation = item["orientation"]
        support = compute_line_support_score(line, orientation, work_mask, scale)
        thickness = estimate_line_thickness(line, orientation=orientation, source_mask=work_mask, max_radius=scale["max_wall_thickness"])
        length = line_length(line)
        mx, my = midpoint_of_line(line)
        boundary_dist = max(0.0, cv2.pointPolygonTest(polygon_np, (float(mx), float(my)), True))
        return (support * 18.0) + (thickness * 4.5) + (min(length, 360.0) * 0.12) + (min(boundary_dist, 60.0) * 0.06)

    grouped: List[Dict[str, Any]] = []
    for source_name, lines, base_weight in source_defs:
        for line in lines:
            item = normalize_line(line)
            if item["orientation"] not in {"h", "v"}:
                continue
            mx, my = midpoint_of_line(line)
            signed_dist = cv2.pointPolygonTest(polygon_np, (float(mx), float(my)), True)
            if signed_dist <= interior_margin:
                continue
            assigned = False
            for group in grouped:
                if group["orientation"] != item["orientation"]:
                    continue
                if abs(int(group["fixed"]) - int(item["fixed"])) > axis_tol:
                    continue
                same_span = span_overlap_ratio(group, item) >= 0.32
                gap_close = max(int(item["start"]), int(group["start"])) - min(int(item["end"]), int(group["end"])) <= gap_tol
                if not same_span and not gap_close:
                    continue
                group["members"].append({
                    "line": [int(v) for v in line],
                    "source": source_name,
                    "weight": float(base_weight),
                })
                group["fixed_values"].append((int(item["fixed"]), float(base_weight)))
                group["start"] = min(int(group["start"]), int(item["start"]))
                group["end"] = max(int(group["end"]), int(item["end"]))
                assigned = True
                break
            if not assigned:
                grouped.append({
                    "orientation": item["orientation"],
                    "fixed": int(item["fixed"]),
                    "start": int(item["start"]),
                    "end": int(item["end"]),
                    "members": [{
                        "line": [int(v) for v in line],
                        "source": source_name,
                        "weight": float(base_weight),
                    }],
                    "fixed_values": [(int(item["fixed"]), float(base_weight))],
                })

    scored_lines: List[Dict[str, Any]] = []
    for group in grouped:
        total_weight = sum(weight for _, weight in group["fixed_values"])
        fixed = int(round(sum(value * weight for value, weight in group["fixed_values"]) / max(total_weight, 1.0)))
        merged_item = {
            "orientation": group["orientation"],
            "fixed": fixed,
            "start": int(group["start"]),
            "end": int(group["end"]),
        }
        merged_line = denormalize_line(merged_item)
        orientation = merged_item["orientation"]
        length = line_length(merged_line)
        thickness = estimate_line_thickness(
            merged_line,
            orientation=orientation,
            source_mask=work_mask,
            max_radius=scale["max_wall_thickness"],
        )
        if length < min_len or thickness < required_thickness:
            continue
        sources = sorted({member["source"] for member in group["members"]})
        source_count = len(sources)
        support_score = line_structural_score(merged_line)
        vote_weight = sum(float(member["weight"]) for member in group["members"])
        keep = False
        if source_count >= 2:
            keep = True
            stats["clean_plan_mode_consensus_kept"] += 1
        elif support_score >= 250.0 and length >= min_len * 1.8:
            keep = True
            stats["clean_plan_mode_single_source_kept"] += 1
        if not keep:
            continue
        scored_lines.append({
            "line": merged_line,
            "score": support_score + vote_weight * 12.0 + source_count * 18.0,
            "sources": sources,
            "source_count": source_count,
            "support_score": support_score,
            "length": length,
            "thickness": thickness,
        })

    stats["clean_plan_mode_group_count"] = len(grouped)
    scored_lines.sort(key=lambda item: item["score"], reverse=True)

    def same_wall(a: List[int], b: List[int]) -> bool:
        na = normalize_line(a)
        nb = normalize_line(b)
        if na["orientation"] != nb["orientation"]:
            return False
        if abs(int(na["fixed"]) - int(nb["fixed"])) > axis_tol:
            return False
        overlap = max(0, min(int(na["end"]), int(nb["end"])) - max(int(na["start"]), int(nb["start"])))
        shorter = max(1, min(int(na["end"]) - int(na["start"]), int(nb["end"]) - int(nb["start"])))
        return overlap / float(shorter) >= 0.5

    def endpoint_anchor_count(line: List[int], anchors: List[List[int]]) -> int:
        normalized = normalize_line(line)
        orth = "v" if normalized["orientation"] == "h" else "h"
        count = 0
        for px, py in line_endpoints(line):
            anchored = False
            for other in anchors + outer_segments:
                other_norm = normalize_line(other)
                if other_norm["orientation"] != orth:
                    continue
                line_dist = abs(float(py) - float(other[1])) if orth == "h" else abs(float(px) - float(other[0]))
                if line_dist > anchor_tol:
                    continue
                if orth == "h":
                    if int(other_norm["start"]) - anchor_tol <= int(px) <= int(other_norm["end"]) + anchor_tol:
                        anchored = True
                        break
                else:
                    if int(other_norm["start"]) - anchor_tol <= int(py) <= int(other_norm["end"]) + anchor_tol:
                        anchored = True
                        break
            if anchored:
                count += 1
        return count

    max_candidates = max(8, min(14, int(round(len(scored_lines) * 0.75)))) if scored_lines else 0
    if clean_like:
        max_candidates = max(max_candidates, 10)
    base_infos = [item for item in scored_lines if item["source_count"] >= 2]
    supplement_infos = [item for item in scored_lines if item["source_count"] < 2]
    kept_lines: List[List[int]] = []
    for item in base_infos:
        if any(same_wall(item["line"], existing) for existing in kept_lines):
            continue
        kept_lines.append(item["line"])
    stats["clean_plan_mode_consensus_kept"] = len(kept_lines)
    supplement_budget = 3 if clean_like else 1
    supplemented = 0
    for item in supplement_infos:
        if len(kept_lines) >= max_candidates:
            break
        if supplemented >= supplement_budget:
            break
        if any(same_wall(item["line"], existing) for existing in kept_lines):
            continue
        anchors = endpoint_anchor_count(item["line"], kept_lines)
        strong_single = item["support_score"] >= 220.0 and item["length"] >= min_len * 1.35
        very_strong_single = item["support_score"] >= 260.0 and item["length"] >= min_len * 1.15
        source_name = item["sources"][0] if item["sources"] else "unknown"
        clean_structural_single = (
            clean_like
            and source_name in {"raw_axis", "orthogonal", "semantic"}
            and item["support_score"] >= 175.0
            and item["length"] >= min_len * 0.9
            and item["thickness"] >= required_thickness
        )
        if anchors >= 2 and (strong_single or (clean_like and item["support_score"] >= 205.0)):
            kept_lines.append(item["line"])
            stats["clean_plan_mode_anchor_supplement_kept"] += 1
            stats["clean_plan_mode_single_source_kept"] += 1
            supplemented += 1
        elif clean_like and anchors >= 1 and very_strong_single:
            kept_lines.append(item["line"])
            stats["clean_plan_mode_anchor_supplement_kept"] += 1
            stats["clean_plan_mode_single_source_kept"] += 1
            supplemented += 1
        elif clean_structural_single and anchors >= 1:
            kept_lines.append(item["line"])
            stats["clean_plan_mode_anchor_supplement_kept"] += 1
            stats["clean_plan_mode_single_source_kept"] += 1
            supplemented += 1

    kept_lines = remove_duplicate_lines(kept_lines, coord_tol=8, length_tol=12)
    kept_lines = collapse_parallel_double_lines(kept_lines, pair_tol=scale["pair_merge_tol"])
    kept_lines = remove_duplicate_lines(kept_lines, coord_tol=8, length_tol=12)
    outer_margin = max(scale["outer_edge_offset"] * 4, scale["max_wall_thickness"] * 2 + 12)
    before_outer = len(kept_lines)
    kept_lines, _ = filter_lines_inside_building(kept_lines, polygon_np, outer_margin=outer_margin)
    kept_lines, outer_stats = reject_outer_boundary_parallel_candidates(kept_lines, polygon_np, scale)
    stats["clean_plan_mode_rejected_outer"] = (before_outer - len(kept_lines)) + int(outer_stats.get("outer_boundary_rejected_count", 0))
    before_symbolic = len(kept_lines)
    kept_lines, _ = filter_symbolic_vertical_walls(
        kept_lines,
        polygon_np,
        source_mask=work_mask,
        scale=scale,
    )
    stats["clean_plan_mode_rejected_symbolic"] = before_symbolic - len(kept_lines)
    kept_lines, _ = merge_collinear_lines(
        kept_lines,
        pos_tol=8,
        gap_tol=gap_tol,
        return_stats=True,
    )
    kept_lines = remove_duplicate_lines(kept_lines, coord_tol=8, length_tol=12)

    line_mask = np.zeros_like(source_mask)
    draw_thickness = max(2, scale["min_wall_thickness"])
    for line in kept_lines:
        cv2.line(line_mask, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), 255, draw_thickness)
    if kept_lines:
        line_mask = cv2.dilate(line_mask, np.ones((3, 3), np.uint8), iterations=1)

    stats["clean_plan_mode_count"] = len(kept_lines)
    stats["clean_plan_mode_segments"] = [line[:] for line in kept_lines]
    stats["clean_plan_mode_score"] = score_inner_wall_set(kept_lines, polygon_np, work_mask, scale)
    return kept_lines, line_mask, stats


def build_clean_plan_topology_recovery(
    base_lines: List[List[int]],
    raw_segments: List[List[int]],
    polygon_np: np.ndarray,
    source_mask: np.ndarray,
    scale: Dict[str, int],
    doors: List[Dict[str, int]],
) -> Tuple[List[List[int]], Dict[str, Any]]:
    stats: Dict[str, Any] = {
        "enabled": False,
        "supplement_count": 0,
        "supplements": [],
        "score": float("-inf"),
    }
    if not base_lines or not raw_segments:
        return base_lines, stats

    outer_segments = polygon_to_axis_aligned_segments(
        [[int(pt[0]), int(pt[1])] for pt in polygon_np.reshape(-1, 2)],
        axis_tol=8,
        min_length=max(24, scale["opening_min_gap"]),
    )
    clean_like = (
        len(base_lines) >= 4
        and len(outer_segments) >= 4
        and all(normalize_line(line)["orientation"] in {"h", "v"} for line in base_lines)
    )
    if not clean_like:
        return base_lines, stats

    stats["enabled"] = True
    axis_tol = max(8, int(round(scale["max_wall_thickness"] * 1.6)))
    bridge_gap = max(34, scale["opening_min_gap"] * 2)
    min_len = max(38, int(round(scale["opening_min_gap"] * 1.6)))
    max_len = max(88, int(round(scale["opening_max_gap"] * 0.92)))

    def same_wall(a: List[int], b: List[int]) -> bool:
        na = normalize_line(a)
        nb = normalize_line(b)
        if na["orientation"] != nb["orientation"]:
            return False
        if abs(int(na["fixed"]) - int(nb["fixed"])) > axis_tol:
            return False
        overlap = max(0, min(int(na["end"]), int(nb["end"])) - max(int(na["start"]), int(nb["start"])))
        shorter = max(1, min(int(na["end"]) - int(na["start"]), int(nb["end"]) - int(nb["start"])))
        return overlap / float(shorter) >= 0.5

    def nearest_anchor(endpoint: Tuple[int, int], orientation: str, anchors: List[List[int]], prefer_low: bool) -> Optional[int]:
        px, py = endpoint
        orth = "h" if orientation == "v" else "v"
        best: Optional[Tuple[int, int]] = None
        for other in anchors:
            item = normalize_line(other)
            if item["orientation"] != orth:
                continue
            if orth == "h":
                if not (int(item["start"]) - axis_tol <= int(px) <= int(item["end"]) + axis_tol):
                    continue
                gap = abs(int(py) - int(item["fixed"]))
                target = int(item["fixed"])
                if prefer_low and target > py:
                    continue
                if (not prefer_low) and target < py:
                    continue
            else:
                if not (int(item["start"]) - axis_tol <= int(py) <= int(item["end"]) + axis_tol):
                    continue
                gap = abs(int(px) - int(item["fixed"]))
                target = int(item["fixed"])
                if prefer_low and target > px:
                    continue
                if (not prefer_low) and target < px:
                    continue
            if gap > bridge_gap:
                continue
            if best is None or gap < best[0]:
                best = (gap, target)
        return None if best is None else best[1]

    supplemented: List[Dict[str, Any]] = []
    anchor_lines = [line[:] for line in base_lines] + outer_segments
    for line in raw_segments:
        item = normalize_line(line)
        orientation = item["orientation"]
        if orientation not in {"h", "v"}:
            continue
        raw_length = line_length(line)
        if raw_length < min_len or raw_length > max_len:
            continue
        if any(same_wall(line, existing) for existing in base_lines):
            continue
        if orientation == "v":
            top = nearest_anchor((int(item["fixed"]), int(item["start"])), orientation, anchor_lines, prefer_low=True)
            bottom = nearest_anchor((int(item["fixed"]), int(item["end"])), orientation, anchor_lines, prefer_low=False)
            if top is not None and bottom is not None and bottom - top >= min_len:
                candidate = denormalize_line({
                    "orientation": "v",
                    "fixed": int(item["fixed"]),
                    "start": int(top),
                    "end": int(bottom),
                })
            elif top is not None and raw_length >= min_len * 1.15:
                candidate = denormalize_line({
                    "orientation": "v",
                    "fixed": int(item["fixed"]),
                    "start": int(top),
                    "end": int(item["end"]),
                })
            elif bottom is not None and raw_length >= min_len * 1.15:
                candidate = denormalize_line({
                    "orientation": "v",
                    "fixed": int(item["fixed"]),
                    "start": int(item["start"]),
                    "end": int(bottom),
                })
            else:
                continue
        else:
            left = nearest_anchor((int(item["start"]), int(item["fixed"])), orientation, anchor_lines, prefer_low=True)
            right = nearest_anchor((int(item["end"]), int(item["fixed"])), orientation, anchor_lines, prefer_low=False)
            if left is not None and right is not None and right - left >= min_len:
                candidate = denormalize_line({
                    "orientation": "h",
                    "fixed": int(item["fixed"]),
                    "start": int(left),
                    "end": int(right),
                })
            elif left is not None and raw_length >= min_len * 1.15:
                candidate = denormalize_line({
                    "orientation": "h",
                    "fixed": int(item["fixed"]),
                    "start": int(left),
                    "end": int(item["end"]),
                })
            elif right is not None and raw_length >= min_len * 1.15:
                candidate = denormalize_line({
                    "orientation": "h",
                    "fixed": int(item["fixed"]),
                    "start": int(item["start"]),
                    "end": int(right),
                })
            else:
                continue
        if any(same_wall(candidate, existing) for existing in base_lines):
            continue
        mx, my = midpoint_of_line(candidate)
        if any(hypot(float(mx - door["x"]), float(my - door["y"])) <= max(24, scale["opening_min_gap"]) for door in doors):
            continue
        support = compute_line_support_score(candidate, orientation, source_mask, scale)
        thickness = estimate_line_thickness(candidate, orientation=orientation, source_mask=source_mask, max_radius=scale["max_wall_thickness"])
        if support < 6.0 or thickness < max(scale["min_wall_thickness"], 4):
            continue
        supplemented.append({
            "line": candidate,
            "score": support * 18.0 + line_length(candidate) * 0.22 + thickness * 4.0,
        })

    supplemented.sort(key=lambda item: item["score"], reverse=True)
    chosen: List[List[int]] = []
    for item in supplemented:
        if len(chosen) >= 2:
            break
        if any(same_wall(item["line"], existing) for existing in chosen):
            continue
        chosen.append(item["line"])

    if not chosen:
        return base_lines, stats

    augmented = [line[:] for line in base_lines] + [line[:] for line in chosen]
    augmented = remove_duplicate_lines(augmented, coord_tol=8, length_tol=12)
    augmented, _ = merge_collinear_lines(augmented, pos_tol=8, gap_tol=max(14, min(scale["opening_max_gap"], 110)), return_stats=True)
    augmented = remove_duplicate_lines(augmented, coord_tol=8, length_tol=12)
    stats["supplement_count"] = len(chosen)
    stats["supplements"] = [line[:] for line in chosen]
    stats["score"] = score_inner_wall_set(augmented, polygon_np, source_mask, scale)
    return augmented, stats


def build_line_evidence_supplemented_inner_walls(
    legacy_lines: List[List[int]],
    line_evidence_stats: Dict[str, Any],
    polygon_np: np.ndarray,
    source_mask: np.ndarray,
    scale: Dict[str, int],
) -> Tuple[List[List[int]], Dict[str, Any]]:
    if not legacy_lines:
        return [], {
            "line_evidence_supplement_enabled": True,
            "line_evidence_supplement_count": 0,
            "line_evidence_supplement_segments": [],
            "line_evidence_supplement_added_segments": [],
            "line_evidence_supplement_score": float("-inf"),
        }

    axis_tol = max(8, int(round(scale["max_wall_thickness"] * 1.4)))
    min_len = max(MIN_WALL_LENGTH_PX, scale["opening_min_gap"] * 2)
    max_additions = 3
    loose_scored = line_evidence_stats.get("line_evidence_scored_segments", [])
    outer_segments = polygon_to_axis_aligned_segments(
        [[int(pt[0]), int(pt[1])] for pt in polygon_np.reshape(-1, 2)],
        axis_tol=10,
        min_length=max(32, scale["opening_min_gap"]),
    )
    existing = [line[:] for line in legacy_lines]

    def same_wall(a: List[int], b: List[int]) -> bool:
        na = normalize_line(a)
        nb = normalize_line(b)
        if na["orientation"] != nb["orientation"]:
            return False
        if abs(int(na["fixed"]) - int(nb["fixed"])) > axis_tol:
            return False
        overlap = max(0, min(int(na["end"]), int(nb["end"])) - max(int(na["start"]), int(nb["start"])))
        shorter = max(1, min(int(na["end"]) - int(na["start"]), int(nb["end"]) - int(nb["start"])))
        return overlap / float(shorter) >= 0.45

    def endpoint_connections(line: List[int]) -> int:
        endpoints = line_endpoints(line)
        normalized = normalize_line(line)
        orth_orientation = "v" if normalized["orientation"] == "h" else "h"
        count = 0
        for endpoint in endpoints:
            connected = False
            for other in existing + outer_segments:
                other_norm = normalize_line(other)
                if other_norm["orientation"] != orth_orientation:
                    continue
                if any(points_close(endpoint, pt, tol=max(12, scale["max_wall_thickness"] * 2)) for pt in line_endpoints(other)):
                    connected = True
                    break
            if connected:
                count += 1
        return count

    candidates_to_add: List[Dict[str, Any]] = []
    for item in loose_scored:
        line = [int(v) for v in item.get("line", [])]
        if len(line) != 4:
            continue
        if line_length(line) < min_len * 1.15:
            continue
        if any(same_wall(line, other) for other in existing):
            continue
        support = compute_line_support_score(line, normalize_line(line)["orientation"], source_mask, scale)
        thickness = estimate_line_thickness(
            line,
            orientation=normalize_line(line)["orientation"],
            source_mask=source_mask,
            max_radius=scale["max_wall_thickness"],
        )
        connections = endpoint_connections(line)
        if support <= 0 or thickness < max(scale["min_wall_thickness"], int(round(scale["max_wall_thickness"] * 0.38))):
            continue
        if connections == 0:
            continue
        candidate_score = float(item.get("score", 0.0)) + (connections * 4.0) + min(thickness, 12.0)
        candidates_to_add.append({
            "line": line,
            "score": round(candidate_score, 3),
            "connections": connections,
        })

    candidates_to_add.sort(key=lambda entry: float(entry["score"]), reverse=True)
    added_segments: List[List[int]] = []
    for item in candidates_to_add:
        if len(added_segments) >= max_additions:
            break
        line = item["line"]
        if any(same_wall(line, other) for other in existing + added_segments):
            continue
        added_segments.append(line[:])

    combined = remove_duplicate_lines(existing + added_segments, coord_tol=8, length_tol=12)
    combined, _ = merge_collinear_lines(
        combined,
        pos_tol=8,
        gap_tol=max(14, min(scale["opening_max_gap"], 110)),
        return_stats=True,
    )
    combined = remove_duplicate_lines(combined, coord_tol=8, length_tol=12)

    return combined, {
        "line_evidence_supplement_enabled": True,
        "line_evidence_supplement_count": len(combined),
        "line_evidence_supplement_segments": [line[:] for line in combined],
        "line_evidence_supplement_added_segments": [line[:] for line in added_segments],
        "line_evidence_supplement_score": score_inner_wall_set(combined, polygon_np, source_mask, scale),
    }


def build_line_evidence_gap_rescue_inner_walls(
    legacy_lines: List[List[int]],
    line_evidence_lines: List[List[int]],
    polygon_np: np.ndarray,
    source_mask: np.ndarray,
    scale: Dict[str, int],
) -> Tuple[List[List[int]], Dict[str, Any]]:
    if not legacy_lines or not line_evidence_lines:
        return list(legacy_lines), {
            "line_evidence_gap_rescue_enabled": True,
            "line_evidence_gap_rescue_count": len(legacy_lines),
            "line_evidence_gap_rescue_segments": [line[:] for line in legacy_lines],
            "line_evidence_gap_rescue_replacements": [],
            "line_evidence_gap_rescue_score": score_inner_wall_set(legacy_lines, polygon_np, source_mask, scale) if legacy_lines else float("-inf"),
        }

    axis_tol = max(8, int(round(scale["max_wall_thickness"] * 1.4)))
    split_tol = max(12, scale["opening_min_gap"])
    min_piece_len = max(MIN_WALL_LENGTH_PX, int(round(scale["opening_min_gap"] * 1.5)))
    bridge_gap_tol = max(18, int(round(scale["opening_min_gap"] * 2.2)))
    outer_segments = polygon_to_axis_aligned_segments(
        [[int(pt[0]), int(pt[1])] for pt in polygon_np.reshape(-1, 2)],
        axis_tol=10,
        min_length=max(32, scale["opening_min_gap"]),
    )
    legacy_norm = [normalize_line(line) for line in legacy_lines]
    result_lines = [line[:] for line in legacy_lines]
    replacements: List[Dict[str, Any]] = []

    def span_gap(a: Dict[str, Any], b: Dict[str, Any]) -> int:
        overlap = max(0, min(int(a["end"]), int(b["end"])) - max(int(a["start"]), int(b["start"])))
        if overlap > 0:
            return 0
        if int(a["end"]) < int(b["start"]):
            return int(b["start"]) - int(a["end"])
        if int(b["end"]) < int(a["start"]):
            return int(a["start"]) - int(b["end"])
        return 0

    def same_axis_group(evidence_norm: Dict[str, Any]) -> List[int]:
        group: List[int] = []
        for idx, item in enumerate(legacy_norm):
            if item["orientation"] != evidence_norm["orientation"]:
                continue
            if abs(int(item["fixed"]) - int(evidence_norm["fixed"])) > axis_tol:
                continue
            if span_gap(item, evidence_norm) <= bridge_gap_tol:
                group.append(idx)
        return group

    def orthogonal_split_positions(evidence_norm: Dict[str, Any]) -> List[int]:
        positions = [int(evidence_norm["start"]), int(evidence_norm["end"])]
        for other in legacy_lines + outer_segments:
            other_norm = normalize_line(other)
            if other_norm["orientation"] == evidence_norm["orientation"]:
                continue
            if int(other_norm["start"]) - split_tol <= int(evidence_norm["fixed"]) <= int(other_norm["end"]) + split_tol:
                cross_pos = int(other_norm["fixed"])
                if int(evidence_norm["start"]) + min_piece_len // 3 < cross_pos < int(evidence_norm["end"]) - min_piece_len // 3:
                    positions.append(cross_pos)
        positions = sorted(set(positions))
        return positions

    def build_piece(evidence_norm: Dict[str, Any], start: int, end: int) -> Optional[List[int]]:
        if end - start < min_piece_len:
            return None
        piece_norm = {
            "orientation": evidence_norm["orientation"],
            "fixed": int(evidence_norm["fixed"]),
            "start": int(start),
            "end": int(end),
        }
        piece = denormalize_line(piece_norm)
        support = compute_line_support_score(piece, evidence_norm["orientation"], source_mask, scale)
        if support <= 0:
            return None
        mx, my = midpoint_of_line(piece)
        if cv2.pointPolygonTest(polygon_np, (float(mx), float(my)), True) <= max(scale["outer_edge_offset"], scale["max_wall_thickness"]):
            return None
        return piece

    for evidence in sorted(line_evidence_lines, key=line_length, reverse=True):
        evidence_norm = normalize_line(evidence)
        group_indices = same_axis_group(evidence_norm)
        if len(group_indices) < 2:
            continue

        group_lines = [legacy_lines[idx] for idx in group_indices]
        group_norms = [legacy_norm[idx] for idx in group_indices]
        split_positions = orthogonal_split_positions(evidence_norm)
        if len(split_positions) < 3:
            continue

        rescued_pieces: List[List[int]] = []
        for start, end in zip(split_positions[:-1], split_positions[1:]):
            piece = build_piece(evidence_norm, start, end)
            if piece is None:
                continue
            piece_norm = normalize_line(piece)
            overlapping = [
                idx for idx, item in enumerate(group_norms)
                if max(0, min(int(item["end"]), int(piece_norm["end"])) - max(int(item["start"]), int(piece_norm["start"]))) >= min_piece_len * 0.25
            ]
            if not overlapping:
                continue
            rescued_pieces.append(piece)

        rescued_pieces = remove_duplicate_lines(rescued_pieces, coord_tol=8, length_tol=12)
        if len(rescued_pieces) < len(group_lines):
            continue

        candidate_lines = [
            line[:] for idx, line in enumerate(result_lines)
            if idx not in group_indices
        ] + [line[:] for line in rescued_pieces]
        candidate_lines, _ = merge_collinear_lines(
            candidate_lines,
            pos_tol=8,
            gap_tol=max(14, min(scale["opening_max_gap"], 110)),
            return_stats=True,
        )
        candidate_lines = remove_duplicate_lines(candidate_lines, coord_tol=8, length_tol=12)
        candidate_score = score_inner_wall_set(candidate_lines, polygon_np, source_mask, scale)
        current_score = score_inner_wall_set(result_lines, polygon_np, source_mask, scale)
        if candidate_score < current_score - 0.05:
            continue

        result_lines = candidate_lines
        replacements.append({
            "evidence_line": evidence[:],
            "replaced_legacy_lines": [line[:] for line in group_lines],
            "rescued_segments": [line[:] for line in rescued_pieces],
        })

    return result_lines, {
        "line_evidence_gap_rescue_enabled": True,
        "line_evidence_gap_rescue_count": len(result_lines),
        "line_evidence_gap_rescue_segments": [line[:] for line in result_lines],
        "line_evidence_gap_rescue_replacements": replacements,
        "line_evidence_gap_rescue_score": score_inner_wall_set(result_lines, polygon_np, source_mask, scale),
    }


def build_line_evidence_axis_replacement_inner_walls(
    legacy_lines: List[List[int]],
    line_evidence_stats: Dict[str, Any],
    polygon_np: np.ndarray,
    source_mask: np.ndarray,
    scale: Dict[str, int],
) -> Tuple[List[List[int]], Dict[str, Any]]:
    if not legacy_lines:
        return [], {
            "line_evidence_axis_replace_enabled": True,
            "line_evidence_axis_replace_count": 0,
            "line_evidence_axis_replace_segments": [],
            "line_evidence_axis_replace_replacements": [],
            "line_evidence_axis_replace_score": float("-inf"),
        }

    loose_candidates = [
        [int(v) for v in line]
        for line in line_evidence_stats.get("line_evidence_loose_segments", [])
        if isinstance(line, list) and len(line) == 4
    ]
    if not loose_candidates:
        return [line[:] for line in legacy_lines], {
            "line_evidence_axis_replace_enabled": True,
            "line_evidence_axis_replace_count": len(legacy_lines),
            "line_evidence_axis_replace_segments": [line[:] for line in legacy_lines],
            "line_evidence_axis_replace_replacements": [],
            "line_evidence_axis_replace_score": score_inner_wall_set(legacy_lines, polygon_np, source_mask, scale),
        }

    axis_tol = max(14, int(round(scale["opening_min_gap"] * 1.6)))
    max_axis_shift = max(38, int(round(scale["opening_max_gap"] * 0.55)))
    min_len = max(MIN_WALL_LENGTH_PX, int(round(scale["opening_min_gap"] * 1.8)))
    required_thickness = max(scale["min_wall_thickness"], int(round(scale["max_wall_thickness"] * 0.38)))
    outer_margin = max(scale["outer_edge_offset"] * 3, scale["max_wall_thickness"] * 2 + 6)
    span_pad = max(6, scale["opening_scan_step"])

    result_lines = [line[:] for line in legacy_lines]
    replacements: List[Dict[str, Any]] = []

    def span_overlap(a: Dict[str, Any], b: Dict[str, Any]) -> float:
        overlap = max(0, min(int(a["end"]), int(b["end"])) - max(int(a["start"]), int(b["start"])))
        shorter = max(1, min(int(a["end"]) - int(a["start"]), int(b["end"]) - int(b["start"])))
        return overlap / float(shorter)

    def candidate_score(line: List[int]) -> float:
        norm = normalize_line(line)
        support = compute_line_support_score(line, norm["orientation"], source_mask, scale)
        thickness = estimate_line_thickness(line, norm["orientation"], source_mask, scale["max_wall_thickness"])
        mx, my = midpoint_of_line(line)
        inner_dist = cv2.pointPolygonTest(polygon_np, (float(mx), float(my)), True)
        return float(support) + min(float(thickness), 12.0) + min(max(inner_dist - outer_margin, 0.0) * 0.08, 10.0)

    for idx, legacy in enumerate(list(result_lines)):
        legacy_norm = normalize_line(legacy)
        legacy_len = line_length(legacy)
        if legacy_len < min_len:
            continue
        base_score = candidate_score(legacy)
        best_line: Optional[List[int]] = None
        best_score = base_score
        best_meta: Optional[Dict[str, Any]] = None

        for cand in loose_candidates:
            cand_norm = normalize_line(cand)
            if cand_norm["orientation"] != legacy_norm["orientation"]:
                continue
            axis_shift = abs(int(cand_norm["fixed"]) - int(legacy_norm["fixed"]))
            if axis_shift < 3 or axis_shift > max_axis_shift:
                continue
            if span_overlap(legacy_norm, cand_norm) < 0.55:
                continue
            if line_length(cand) < legacy_len * 0.8:
                continue
            if line_length(cand) > legacy_len * 1.8:
                continue
            thickness = estimate_line_thickness(cand, cand_norm["orientation"], source_mask, scale["max_wall_thickness"])
            if thickness < required_thickness:
                continue

            start = max(int(legacy_norm["start"]), int(cand_norm["start"]) - span_pad)
            end = min(int(legacy_norm["end"]), int(cand_norm["end"]) + span_pad)
            if end - start < min_len:
                start = max(int(cand_norm["start"]), int(legacy_norm["start"]))
                end = min(int(cand_norm["end"]), int(legacy_norm["end"]))
            if end - start < min_len:
                continue
            replaced_norm = {
                "orientation": legacy_norm["orientation"],
                "fixed": int(cand_norm["fixed"]),
                "start": int(start),
                "end": int(end),
            }
            replaced = denormalize_line(replaced_norm)
            mx, my = midpoint_of_line(replaced)
            if cv2.pointPolygonTest(polygon_np, (float(mx), float(my)), True) <= outer_margin:
                continue
            score = candidate_score(replaced)
            if score <= base_score + 0.3:
                continue
            if score > best_score:
                best_score = score
                best_line = replaced
                best_meta = {
                    "legacy_line": legacy[:],
                    "replacement_line": replaced[:],
                    "source_candidate": cand[:],
                    "axis_shift": axis_shift,
                    "score_gain": round(score - base_score, 3),
                }

        if best_line is not None and best_meta is not None:
            result_lines[idx] = best_line
            replacements.append(best_meta)

    result_lines, _ = merge_collinear_lines(
        result_lines,
        pos_tol=8,
        gap_tol=max(14, min(scale["opening_max_gap"], 110)),
        return_stats=True,
    )
    result_lines = remove_duplicate_lines(result_lines, coord_tol=8, length_tol=12)
    return result_lines, {
        "line_evidence_axis_replace_enabled": True,
        "line_evidence_axis_replace_count": len(result_lines),
        "line_evidence_axis_replace_segments": [line[:] for line in result_lines],
        "line_evidence_axis_replace_replacements": replacements,
        "line_evidence_axis_replace_score": score_inner_wall_set(result_lines, polygon_np, source_mask, scale),
    }


def build_raw_axis_hybrid_inner_walls(
    legacy_lines: List[List[int]],
    raw_axis_lines: List[List[int]],
    polygon_np: np.ndarray,
    source_mask: np.ndarray,
    scale: Dict[str, int],
) -> Tuple[List[List[int]], Dict[str, Any]]:
    if not legacy_lines:
        return [], {
            "raw_axis_hybrid_enabled": True,
            "raw_axis_hybrid_count": 0,
            "raw_axis_hybrid_segments": [],
            "raw_axis_hybrid_replacements": [],
            "raw_axis_hybrid_score": float("-inf"),
        }
    if not raw_axis_lines:
        return [line[:] for line in legacy_lines], {
            "raw_axis_hybrid_enabled": True,
            "raw_axis_hybrid_count": len(legacy_lines),
            "raw_axis_hybrid_segments": [line[:] for line in legacy_lines],
            "raw_axis_hybrid_replacements": [],
            "raw_axis_hybrid_score": score_inner_wall_set(legacy_lines, polygon_np, source_mask, scale),
        }

    axis_tol = max(16, int(round(scale["opening_min_gap"] * 1.4)))
    max_axis_shift = max(42, int(round(scale["opening_max_gap"] * 0.65)))
    min_len = max(MIN_WALL_LENGTH_PX, int(round(scale["opening_min_gap"] * 1.8)))
    required_thickness = max(scale["min_wall_thickness"], int(round(scale["max_wall_thickness"] * 0.34)))
    outer_margin = max(scale["outer_edge_offset"] * 3, scale["max_wall_thickness"] * 2 + 6)
    span_pad = max(8, scale["opening_scan_step"] * 2)

    result_lines = [line[:] for line in legacy_lines]
    replacements: List[Dict[str, Any]] = []

    def span_overlap(a: Dict[str, Any], b: Dict[str, Any]) -> float:
        overlap = max(0, min(int(a["end"]), int(b["end"])) - max(int(a["start"]), int(b["start"])))
        shorter = max(1, min(int(a["end"]) - int(a["start"]), int(b["end"]) - int(b["start"])))
        return overlap / float(shorter)

    def candidate_score(line: List[int]) -> float:
        norm = normalize_line(line)
        support = compute_line_support_score(line, norm["orientation"], source_mask, scale)
        thickness = estimate_line_thickness(line, norm["orientation"], source_mask, scale["max_wall_thickness"])
        mx, my = midpoint_of_line(line)
        inner_dist = cv2.pointPolygonTest(polygon_np, (float(mx), float(my)), True)
        return float(support) + min(float(thickness), 12.0) + min(max(inner_dist - outer_margin, 0.0) * 0.08, 10.0)

    for idx, legacy in enumerate(list(result_lines)):
        legacy_norm = normalize_line(legacy)
        legacy_len = line_length(legacy)
        if legacy_len < min_len:
            continue
        base_score = candidate_score(legacy)
        best_line: Optional[List[int]] = None
        best_score = base_score
        best_meta: Optional[Dict[str, Any]] = None

        for cand in raw_axis_lines:
            cand_norm = normalize_line(cand)
            if cand_norm["orientation"] != legacy_norm["orientation"]:
                continue
            axis_shift = abs(int(cand_norm["fixed"]) - int(legacy_norm["fixed"]))
            if axis_shift > max_axis_shift:
                continue
            if span_overlap(legacy_norm, cand_norm) < 0.58:
                continue
            if line_length(cand) < legacy_len * 0.55:
                continue
            thickness = estimate_line_thickness(cand, cand_norm["orientation"], source_mask, scale["max_wall_thickness"])
            if thickness < required_thickness:
                continue

            start = max(int(legacy_norm["start"]), int(cand_norm["start"]) - span_pad)
            end = min(int(legacy_norm["end"]), int(cand_norm["end"]) + span_pad)
            if end - start < min_len:
                start = int(legacy_norm["start"])
                end = int(legacy_norm["end"])
            replaced_norm = {
                "orientation": legacy_norm["orientation"],
                "fixed": int(cand_norm["fixed"]),
                "start": int(start),
                "end": int(end),
            }
            replaced = denormalize_line(replaced_norm)
            mx, my = midpoint_of_line(replaced)
            if cv2.pointPolygonTest(polygon_np, (float(mx), float(my)), True) <= outer_margin:
                continue
            score = candidate_score(replaced)
            # Prefer materially better support or a real axis move on comparable support.
            if score <= base_score + 0.18 and axis_shift < 6:
                continue
            if score > best_score + 0.05:
                best_score = score
                best_line = replaced
                best_meta = {
                    "legacy_line": legacy[:],
                    "replacement_line": replaced[:],
                    "source_candidate": cand[:],
                    "axis_shift": axis_shift,
                    "score_gain": round(score - base_score, 3),
                }

        if best_line is not None and best_meta is not None:
            result_lines[idx] = best_line
            replacements.append(best_meta)

    result_lines, _ = merge_collinear_lines(
        result_lines,
        pos_tol=8,
        gap_tol=max(14, min(scale["opening_max_gap"], 110)),
        return_stats=True,
    )
    result_lines = remove_duplicate_lines(result_lines, coord_tol=8, length_tol=12)
    return result_lines, {
        "raw_axis_hybrid_enabled": True,
        "raw_axis_hybrid_count": len(result_lines),
        "raw_axis_hybrid_segments": [line[:] for line in result_lines],
        "raw_axis_hybrid_replacements": replacements,
        "raw_axis_hybrid_score": score_inner_wall_set(result_lines, polygon_np, source_mask, scale),
    }


def build_line_evidence_partition_rescue_inner_walls(
    legacy_lines: List[List[int]],
    line_evidence_lines: List[List[int]],
    polygon_np: np.ndarray,
    source_mask: np.ndarray,
    scale: Dict[str, int],
) -> Tuple[List[List[int]], Dict[str, Any]]:
    if not legacy_lines or not line_evidence_lines:
        return [line[:] for line in legacy_lines], {
            "line_evidence_partition_enabled": True,
            "line_evidence_partition_count": len(legacy_lines),
            "line_evidence_partition_segments": [line[:] for line in legacy_lines],
            "line_evidence_partition_replacements": [],
            "line_evidence_partition_score": score_inner_wall_set(legacy_lines, polygon_np, source_mask, scale) if legacy_lines else float("-inf"),
        }

    axis_tol = max(10, int(round(scale["max_wall_thickness"] * 1.5)))
    min_piece_len = max(MIN_WALL_LENGTH_PX, int(round(scale["opening_min_gap"] * 1.4)))
    min_group_size = 2
    overlap_min = 0.35
    outer_margin = max(scale["outer_edge_offset"] * 3, scale["max_wall_thickness"] * 2 + 6)
    result_lines = [line[:] for line in legacy_lines]
    replacements: List[Dict[str, Any]] = []
    outer_segments = polygon_to_axis_aligned_segments(
        [[int(pt[0]), int(pt[1])] for pt in polygon_np.reshape(-1, 2)],
        axis_tol=10,
        min_length=max(32, scale["opening_min_gap"]),
    )

    def group_for_evidence(evidence_norm: Dict[str, Any]) -> List[int]:
        indices: List[int] = []
        for idx, line in enumerate(result_lines):
            norm = normalize_line(line)
            if norm["orientation"] != evidence_norm["orientation"]:
                continue
            if abs(int(norm["fixed"]) - int(evidence_norm["fixed"])) > axis_tol:
                continue
            overlap = max(0, min(int(norm["end"]), int(evidence_norm["end"])) - max(int(norm["start"]), int(evidence_norm["start"])))
            shorter = max(1, min(int(norm["end"]) - int(norm["start"]), int(evidence_norm["end"]) - int(evidence_norm["start"])))
            if overlap / float(shorter) >= overlap_min:
                indices.append(idx)
        return indices

    def anchor_positions(evidence_norm: Dict[str, Any]) -> List[int]:
        anchors = [int(evidence_norm["start"]), int(evidence_norm["end"])]
        for other in result_lines + outer_segments:
            other_norm = normalize_line(other)
            if other_norm["orientation"] == evidence_norm["orientation"]:
                continue
            if int(other_norm["start"]) - axis_tol <= int(evidence_norm["fixed"]) <= int(other_norm["end"]) + axis_tol:
                pos = int(other_norm["fixed"])
                if int(evidence_norm["start"]) + min_piece_len // 3 < pos < int(evidence_norm["end"]) - min_piece_len // 3:
                    anchors.append(pos)
        return sorted(set(anchors))

    def build_piece(evidence_norm: Dict[str, Any], start: int, end: int) -> Optional[List[int]]:
        if end - start < min_piece_len:
            return None
        piece_norm = {
            "orientation": evidence_norm["orientation"],
            "fixed": int(evidence_norm["fixed"]),
            "start": int(start),
            "end": int(end),
        }
        piece = denormalize_line(piece_norm)
        if compute_line_support_score(piece, evidence_norm["orientation"], source_mask, scale) <= 0:
            return None
        mx, my = midpoint_of_line(piece)
        if cv2.pointPolygonTest(polygon_np, (float(mx), float(my)), True) <= outer_margin:
            return None
        return piece

    for evidence in sorted(line_evidence_lines, key=line_length, reverse=True):
        evidence_norm = normalize_line(evidence)
        group_indices = group_for_evidence(evidence_norm)
        if len(group_indices) < min_group_size:
            continue
        anchors = anchor_positions(evidence_norm)
        if len(anchors) < 3:
            continue

        pieces: List[List[int]] = []
        for start, end in zip(anchors[:-1], anchors[1:]):
            piece = build_piece(evidence_norm, start, end)
            if piece is not None:
                pieces.append(piece)
        pieces = remove_duplicate_lines(pieces, coord_tol=8, length_tol=12)
        if len(pieces) < len(group_indices):
            continue

        candidate_lines = [
            line[:] for idx, line in enumerate(result_lines)
            if idx not in group_indices
        ] + [line[:] for line in pieces]
        candidate_lines, _ = merge_collinear_lines(
            candidate_lines,
            pos_tol=8,
            gap_tol=max(14, min(scale["opening_max_gap"], 110)),
            return_stats=True,
        )
        candidate_lines = remove_duplicate_lines(candidate_lines, coord_tol=8, length_tol=12)
        current_score = score_inner_wall_set(result_lines, polygon_np, source_mask, scale)
        candidate_score = score_inner_wall_set(candidate_lines, polygon_np, source_mask, scale)
        if candidate_score < current_score - 0.05:
            continue

        replaced_lines = [result_lines[idx][:] for idx in group_indices]
        result_lines = candidate_lines
        replacements.append({
            "evidence_line": evidence[:],
            "replaced_legacy_lines": replaced_lines,
            "partition_segments": [line[:] for line in pieces],
        })

    return result_lines, {
        "line_evidence_partition_enabled": True,
        "line_evidence_partition_count": len(result_lines),
        "line_evidence_partition_segments": [line[:] for line in result_lines],
        "line_evidence_partition_replacements": replacements,
        "line_evidence_partition_score": score_inner_wall_set(result_lines, polygon_np, source_mask, scale),
    }


def conservative_axis_snap_segments(
    lines: List[List[int]],
    polygon_np: np.ndarray,
    source_mask: np.ndarray,
    scale: Dict[str, int],
) -> Tuple[List[List[int]], Dict[str, Any]]:
    if not lines:
        return [], {
            "axis_snap_enabled": True,
            "axis_snap_tol": 0,
            "axis_snap_cluster_count_vertical": 0,
            "axis_snap_cluster_count_horizontal": 0,
            "axis_grid_vertical": [],
            "axis_grid_horizontal": [],
            "axis_grid_scores": {"v": [], "h": []},
            "axis_snap_candidates": [],
            "axis_snapped_segment_count": 0,
            "axis_snap_aborted_reason": None,
            "axis_snapped_segments_before_after": [],
            "axis_target_lines": [],
        }

    axis_cluster_tol = max(10, int(round(scale["max_wall_thickness"] * 1.5)))
    snap_tol = axis_cluster_tol
    max_snap_distance = max(28, int(round(scale["max_wall_thickness"] * 4.0)))
    min_meaningful_length = max(scale["opening_min_gap"] * 2, 60)
    required_thickness = max(scale["min_wall_thickness"], int(round(scale["max_wall_thickness"] * 0.5)))
    boundary_safe_margin = max(scale["outer_edge_offset"] * 3, scale["max_wall_thickness"] * 2 + 4)
    step = max(4, scale["opening_scan_step"])
    max_snapped_ratio = 0.4

    normalized = [normalize_line(line) for line in lines]
    orientation_groups = {
        "h": [idx for idx, item in enumerate(normalized) if item["orientation"] == "h"],
        "v": [idx for idx, item in enumerate(normalized) if item["orientation"] == "v"],
    }

    def span_overlap_ratio(item_a: Dict[str, Any], item_b: Dict[str, Any]) -> float:
        overlap = max(0, min(int(item_a["end"]), int(item_b["end"])) - max(int(item_a["start"]), int(item_b["start"])))
        shorter = max(1, min(int(item_a["end"]) - int(item_a["start"]), int(item_b["end"]) - int(item_b["start"])))
        return overlap / float(shorter)

    def is_near_outer_boundary(line: List[int]) -> bool:
        mx, my = midpoint_of_line(line)
        return cv2.pointPolygonTest(polygon_np, (float(mx), float(my)), True) <= boundary_safe_margin

    def local_support_score(line: List[int], orientation: str) -> float:
        return compute_line_support_score(line, orientation, source_mask, scale)

    def shift_to_fixed(item: Dict[str, Any], candidate_fixed: int) -> List[int]:
        shifted = item.copy()
        shifted["fixed"] = int(candidate_fixed)
        return denormalize_line(shifted)

    polygon_points = [[int(pt[0]), int(pt[1])] for pt in polygon_np.reshape(-1, 2)]
    outer_segments = polygon_to_axis_aligned_segments(
        polygon_points,
        axis_tol=10,
        min_length=max(32, scale["opening_min_gap"]),
    )
    outer_axis_values = {
        "v": [normalize_line(line)["fixed"] for line in outer_segments if normalize_line(line)["orientation"] == "v"],
        "h": [normalize_line(line)["fixed"] for line in outer_segments if normalize_line(line)["orientation"] == "h"],
    }

    def is_strong_segment(line: List[int], item: Dict[str, Any]) -> bool:
        if line_length(line) < min_meaningful_length:
            return False
        thickness = estimate_line_thickness(
            line,
            orientation=item["orientation"],
            source_mask=source_mask,
            max_radius=scale["max_wall_thickness"],
        )
        return thickness >= required_thickness

    def projection_peaks(orientation: str) -> List[int]:
        mask = (source_mask > 0).astype(np.uint8)
        if orientation == "v":
            projection = np.sum(mask, axis=0).astype(np.float32)
        else:
            projection = np.sum(mask, axis=1).astype(np.float32)
        if projection.size == 0:
            return []
        kernel = np.ones(max(3, scale["max_wall_thickness"]), dtype=np.float32)
        smooth = np.convolve(projection, kernel / kernel.size, mode="same")
        threshold = max(8.0, float(np.percentile(smooth, 75) * 0.45))
        peaks: List[int] = []
        for idx in range(1, len(smooth) - 1):
            if smooth[idx] >= threshold and smooth[idx] >= smooth[idx - 1] and smooth[idx] >= smooth[idx + 1]:
                peaks.append(int(idx))
        return peaks

    def build_axis_clusters(orientation: str, indices: List[int]) -> List[Dict[str, Any]]:
        axis_points: List[Dict[str, Any]] = []
        peak_values = projection_peaks(orientation)

        for idx in indices:
            item = normalized[idx]
            axis_points.append({
                "fixed": int(item["fixed"]),
                "weight": max(line_length(lines[idx]), 1.0),
                "kind": "final",
            })

        raw_orientation = "v" if orientation == "v" else "h"
        positive_coords = np.where(source_mask > 0)
        coord_values = positive_coords[1] if orientation == "v" else positive_coords[0]
        if coord_values.size:
            unique_vals, counts = np.unique(coord_values, return_counts=True)
            threshold = max(6, int(np.percentile(counts, 70)))
            for fixed, count in zip(unique_vals.tolist(), counts.tolist()):
                if count >= threshold:
                    axis_points.append({
                        "fixed": int(fixed),
                        "weight": float(count) * 0.3,
                        "kind": "raw",
                    })

        for peak in peak_values:
            axis_points.append({
                "fixed": int(peak),
                "weight": float(scale["max_wall_thickness"] * 3),
                "kind": "peak",
            })

        if not axis_points:
            return []

        axis_points.sort(key=lambda item: item["fixed"])
        clusters: List[List[Dict[str, Any]]] = [[axis_points[0]]]
        for point in axis_points[1:]:
            current_cluster = clusters[-1]
            cluster_center = int(round(sum(entry["fixed"] * entry["weight"] for entry in current_cluster) / max(1.0, sum(entry["weight"] for entry in current_cluster))))
            if abs(point["fixed"] - cluster_center) <= axis_cluster_tol:
                current_cluster.append(point)
            else:
                clusters.append([point])

        results: List[Dict[str, Any]] = []
        for cluster in clusters:
            total_weight = sum(entry["weight"] for entry in cluster)
            axis_value = int(round(sum(entry["fixed"] * entry["weight"] for entry in cluster) / max(1.0, total_weight)))
            near_outer = min((abs(axis_value - outer_val) for outer_val in outer_axis_values[orientation]), default=9999)
            support_segments = [
                idx for idx in indices
                if abs(int(normalized[idx]["fixed"]) - axis_value) <= max_snap_distance
            ]
            strong_segment_count = 0
            total_segment_length = 0.0
            for idx in support_segments:
                candidate_line = shift_to_fixed(normalized[idx], axis_value)
                support = local_support_score(candidate_line, orientation)
                if support > 0:
                    total_segment_length += line_length(lines[idx])
                if is_strong_segment(candidate_line, normalize_line(candidate_line)):
                    strong_segment_count += 1

            peak_bonus = sum(entry["weight"] for entry in cluster if entry["kind"] == "peak")
            raw_bonus = sum(entry["weight"] for entry in cluster if entry["kind"] == "raw")
            boundary_penalty = 45.0 if near_outer <= boundary_safe_margin else (18.0 if near_outer <= boundary_safe_margin * 1.6 else 0.0)
            score = (total_segment_length * 0.035) + (raw_bonus * 0.25) + (peak_bonus * 0.3) + (strong_segment_count * 12.0) - boundary_penalty
            results.append({
                "axis": axis_value,
                "score": round(score, 3),
                "support_segment_count": len(support_segments),
                "strong_segment_count": strong_segment_count,
                "near_outer_distance": near_outer,
            })
        return results

    snapped_lines = [line[:] for line in lines]
    snapped_segments_before_after: List[Dict[str, Any]] = []
    axis_snap_candidates: List[Dict[str, Any]] = []
    axis_target_lines: List[List[int]] = []
    axis_clusters = {
        "v": build_axis_clusters("v", orientation_groups["v"]),
        "h": build_axis_clusters("h", orientation_groups["h"]),
    }
    aborted_reason: Optional[str] = None

    for orientation in ("v", "h"):
        clusters = axis_clusters[orientation]
        for cluster in clusters:
            if orientation == "v":
                axis_target_lines.append([int(cluster["axis"]), 0, int(cluster["axis"]), int(source_mask.shape[0] - 1)])
            else:
                axis_target_lines.append([0, int(cluster["axis"]), int(source_mask.shape[1] - 1), int(cluster["axis"])])

    proposed_changes: List[Dict[str, Any]] = []
    for orientation, indices in orientation_groups.items():
        clusters = axis_clusters[orientation]
        if not clusters:
            continue
        for idx in indices:
            item = normalized[idx]
            current_line = snapped_lines[idx]
            current_support = local_support_score(current_line, orientation)
            current_axis_score = 0.0
            for cluster in clusters:
                if abs(int(item["fixed"]) - int(cluster["axis"])) <= axis_cluster_tol:
                    current_axis_score = max(current_axis_score, float(cluster["score"]))

            best_option: Optional[Dict[str, Any]] = None
            for cluster in clusters:
                target_axis = int(cluster["axis"])
                snap_distance = abs(target_axis - int(item["fixed"]))
                if snap_distance < 4 or snap_distance > max_snap_distance:
                    continue
                if cluster["near_outer_distance"] <= boundary_safe_margin:
                    continue
                if line_length(current_line) < min_meaningful_length:
                    continue
                if is_near_outer_boundary(current_line):
                    continue

                candidate_line = shift_to_fixed(item, target_axis)
                candidate_support = local_support_score(candidate_line, orientation)
                segment_is_strong = is_strong_segment(current_line, item)
                if cluster["strong_segment_count"] < 2 and not segment_is_strong:
                    continue
                if candidate_support <= 0:
                    continue
                if candidate_support < current_support * 0.88:
                    continue

                support_gain = candidate_support - current_support
                cluster_gain = float(cluster["score"]) - current_axis_score
                decision_score = (cluster_gain * 1.35) + (support_gain * 0.65) - (snap_distance * 0.12)
                candidate_meta = {
                    "idx": idx,
                    "target_axis": target_axis,
                    "snap_distance": snap_distance,
                    "candidate_support": round(candidate_support, 3),
                    "current_support": round(current_support, 3),
                    "target_axis_score": round(float(cluster["score"]), 3),
                    "current_axis_score": round(current_axis_score, 3),
                    "decision_score": round(decision_score, 3),
                    "before": current_line[:],
                    "after": candidate_line[:],
                }
                if best_option is None or decision_score > best_option["decision_score"]:
                    best_option = candidate_meta

            if best_option is None:
                continue

            axis_snap_candidates.append({
                "segment": current_line[:],
                "target_axis": best_option["target_axis"],
                "snap_distance": best_option["snap_distance"],
                "target_axis_score": best_option["target_axis_score"],
                "current_axis_score": best_option["current_axis_score"],
                "decision_score": best_option["decision_score"],
                "accepted": False,
            })

            if best_option["target_axis_score"] <= best_option["current_axis_score"] + 0.75 and best_option["candidate_support"] <= best_option["current_support"] + 0.2:
                continue
            if best_option["decision_score"] <= 0.35:
                continue

            proposed_changes.append(best_option)

    if proposed_changes and len(proposed_changes) / float(max(1, len(lines))) > max_snapped_ratio:
        aborted_reason = "snap_aborted_too_many_changes"
        proposed_changes = []

    for change in proposed_changes:
        idx = int(change["idx"])
        snapped_lines[idx] = change["after"][:]
        normalized[idx] = normalize_line(change["after"])
        snapped_segments_before_after.append({
            "before": change["before"][:],
            "after": change["after"][:],
            "target_axis": int(change["target_axis"]),
        })

    accepted_set = {
        (tuple(item["before"]), tuple(item["after"]))
        for item in snapped_segments_before_after
    }
    for candidate in axis_snap_candidates:
        key = (tuple(candidate["segment"]), tuple(shift_to_fixed(normalize_line(candidate["segment"]), int(candidate["target_axis"]))))
        candidate["accepted"] = key in accepted_set

    return snapped_lines, {
        "axis_snap_enabled": True,
        "axis_snap_tol": snap_tol,
        "axis_snap_cluster_count_vertical": len(axis_clusters["v"]),
        "axis_snap_cluster_count_horizontal": len(axis_clusters["h"]),
        "axis_grid_vertical": [int(item["axis"]) for item in axis_clusters["v"]],
        "axis_grid_horizontal": [int(item["axis"]) for item in axis_clusters["h"]],
        "axis_grid_scores": {
            "v": axis_clusters["v"],
            "h": axis_clusters["h"],
        },
        "axis_snap_candidates": axis_snap_candidates,
        "axis_snapped_segment_count": len(snapped_segments_before_after),
        "axis_snap_aborted_reason": aborted_reason,
        "axis_snapped_segments_before_after": snapped_segments_before_after,
        "axis_target_lines": axis_target_lines,
    }


def line_endpoints(line: List[int]) -> List[Tuple[int, int]]:
    return [(int(line[0]), int(line[1])), (int(line[2]), int(line[3]))]


def points_close(a: Tuple[int, int], b: Tuple[int, int], tol: int) -> bool:
    return hypot(a[0] - b[0], a[1] - b[1]) <= tol


def point_lies_on_axis_line(point: Tuple[int, int], line: List[int], tol: int = 8) -> bool:
    px, py = point
    x1, y1, x2, y2 = line

    if y1 == y2:
        return abs(py - y1) <= tol and min(x1, x2) - tol <= px <= max(x1, x2) + tol

    if x1 == x2:
        return abs(px - x1) <= tol and min(y1, y2) - tol <= py <= max(y1, y2) + tol

    return False


def count_line_connections(line: List[int], lines: List[List[int]], tol: int = 10) -> int:
    endpoints = line_endpoints(line)
    connections = 0

    for endpoint in endpoints:
        endpoint_connected = False
        for other in lines:
            if other is line or other == line:
                continue

            other_endpoints = line_endpoints(other)
            if any(points_close(endpoint, other_pt, tol) for other_pt in other_endpoints):
                endpoint_connected = True
                break

            if point_lies_on_axis_line(endpoint, other, tol=tol):
                endpoint_connected = True
                break

        if endpoint_connected:
            connections += 1

    return connections


def prune_spurious_inner_walls(
    lines: List[List[int]],
    polygon_np: np.ndarray,
    scale: Dict[str, int],
) -> Tuple[List[List[int]], Dict[str, int]]:
    if not lines:
        return [], {"pruned_spurious_short_or_disconnected": 0}

    kept: List[List[int]] = []
    stats = {"pruned_spurious_short_or_disconnected": 0}
    long_len = max(140, scale["opening_max_gap"])
    medium_len = max(85, scale["opening_min_gap"] * 3)
    short_len = max(58, scale["opening_min_gap"] * 2)

    for line in lines:
        length = line_length(line)
        connections = count_line_connections(line, lines, tol=12)
        mx, my = midpoint_of_line(line)
        signed_dist = cv2.pointPolygonTest(polygon_np, (float(mx), float(my)), True)

        if length >= long_len and connections >= 1:
            kept.append(line)
            continue

        if length >= medium_len and connections >= 1 and signed_dist >= scale["outer_edge_offset"] * 1.5:
            kept.append(line)
            continue

        if (
            length >= short_len
            and connections >= 1
            and signed_dist >= max(scale["outer_edge_offset"] * 3.0, scale["opening_min_gap"] * 1.2)
        ):
            kept.append(line)
            continue

        stats["pruned_spurious_short_or_disconnected"] += 1

    return kept, stats


def filter_symbolic_vertical_walls(
    lines: List[List[int]],
    polygon_np: np.ndarray,
    source_mask: np.ndarray,
    scale: Dict[str, int],
) -> Tuple[List[List[int]], Dict[str, int]]:
    if not lines:
        return [], {"rejected_symbolic_vertical_lines": 0}

    filtered: List[List[int]] = []
    stats = {"rejected_symbolic_vertical_lines": 0}
    top_symbol_band = max(scale["opening_max_gap"], 110)
    thin_symbol_thickness = max(scale["min_wall_thickness"] + 1, int(round(scale["max_wall_thickness"] * 0.32)))
    max_symbol_length = max(scale["opening_max_gap"] * 2, 250)

    def count_horizontal_crossings(line_to_check: List[int]) -> int:
        px = int(line_to_check[0])
        y_start = int(min(line_to_check[1], line_to_check[3]))
        y_end = int(max(line_to_check[1], line_to_check[3]))
        h, w = source_mask.shape[:2]
        hits: List[int] = []
        span_threshold = max(scale["opening_min_gap"], 24)

        for py in range(max(0, y_start), min(h, y_end + 1), max(4, scale["opening_scan_step"])):
            left = px
            while left >= 0 and source_mask[py, left] > 0:
                left -= 1
            right = px
            while right < w and source_mask[py, right] > 0:
                right += 1
            span = right - left - 1
            if span >= span_threshold:
                hits.append(py)

        if not hits:
            return 0

        clusters = 1
        previous = hits[0]
        for current in hits[1:]:
            if current - previous > max(8, scale["opening_scan_step"] * 2):
                clusters += 1
            previous = current
        return clusters

    for line in lines:
        x1, y1, x2, y2 = line
        if x1 != x2:
            filtered.append(line)
            continue

        mx, my = midpoint_of_line(line)
        signed_dist = cv2.pointPolygonTest(polygon_np, (float(mx), float(my)), True)
        length = line_length(line)
        connections = count_line_connections(line, lines, tol=12)
        thickness = estimate_line_thickness(
            line,
            orientation="v",
            source_mask=source_mask,
            max_radius=scale["max_wall_thickness"],
        )
        horizontal_crossings = count_horizontal_crossings(line)

        if (
            signed_dist <= top_symbol_band
            and length <= max_symbol_length
            and thickness <= thin_symbol_thickness
            and connections <= 2
            and horizontal_crossings >= 4
        ):
            stats["rejected_symbolic_vertical_lines"] += 1
            continue

        filtered.append(line)

    return filtered, stats


def filter_symbolic_line_clusters(
    candidate_mask: np.ndarray,
    polygon_np: np.ndarray,
    scale: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    filtered_mask = candidate_mask.copy()
    removed_mask = np.zeros_like(candidate_mask)
    stats: Dict[str, Any] = {
        "symbolic_cluster_count": 0,
        "stair_like_cluster_count": 0,
        "removed_symbolic_pixels": 0,
        "rejected_symbolic_clusters": 0,
        "reject_reasons": {},
        "clusters": [],
    }

    def reject(reason: str, info: Optional[Dict[str, Any]] = None) -> None:
        stats["rejected_symbolic_clusters"] += 1
        stats["reject_reasons"][reason] = stats["reject_reasons"].get(reason, 0) + 1
        if info is not None:
            stats["clusters"].append({"accepted": False, "reason": reason, **info})

    num_labels, labels, cc_stats, _ = cv2.connectedComponentsWithStats(candidate_mask, connectivity=8)
    min_line_len = max(16, int(round(scale["opening_min_gap"] * 0.55)))
    max_line_len = max(52, int(round(scale["opening_max_gap"] * 1.15)))
    min_cluster_lines = 4
    regularity_tol = max(4.0, scale["min_wall_thickness"] * 1.5)
    min_gap = max(5, scale["min_wall_thickness"])
    max_gap = max(18, scale["max_wall_thickness"] * 4)
    near_outer_tol = max(scale["outer_edge_offset"] * 2, scale["max_wall_thickness"] + 6)

    for label in range(1, num_labels):
        area = int(cc_stats[label, cv2.CC_STAT_AREA])
        x = int(cc_stats[label, cv2.CC_STAT_LEFT])
        y = int(cc_stats[label, cv2.CC_STAT_TOP])
        w = int(cc_stats[label, cv2.CC_STAT_WIDTH])
        h = int(cc_stats[label, cv2.CC_STAT_HEIGHT])
        cx = x + w / 2.0
        cy = y + h / 2.0
        info = {"bbox": [x, y, w, h], "area": area}

        if area < max(18, scale["min_wall_thickness"] * 4):
            reject("cluster_area_too_small", info)
            continue
        if w < min_line_len and h < min_line_len:
            reject("cluster_bbox_too_small", info)
            continue
        if cv2.pointPolygonTest(polygon_np, (float(cx), float(cy)), True) < near_outer_tol:
            reject("cluster_near_outer_polygon", info)
            continue

        component_mask = np.zeros_like(candidate_mask)
        component_mask[labels == label] = 255
        lines = cv2.HoughLinesP(
            component_mask,
            1,
            np.pi / 180.0,
            threshold=max(8, scale["opening_scan_step"] * 2),
            minLineLength=min_line_len,
            maxLineGap=max(4, scale["opening_scan_step"]),
        )
        if lines is None or len(lines) < min_cluster_lines:
            reject("cluster_not_enough_lines", info)
            continue

        horizontal_positions: List[float] = []
        vertical_positions: List[float] = []
        horizontal_lengths: List[float] = []
        vertical_lengths: List[float] = []

        for raw in lines:
            x1, y1, x2, y2 = [int(v) for v in raw[0]]
            dx = x2 - x1
            dy = y2 - y1
            length = float(np.hypot(dx, dy))
            if length < min_line_len or length > max_line_len:
                continue
            if abs(dy) <= max(3, scale["min_wall_thickness"]):
                horizontal_positions.append((y1 + y2) / 2.0)
                horizontal_lengths.append(length)
            elif abs(dx) <= max(3, scale["min_wall_thickness"]):
                vertical_positions.append((x1 + x2) / 2.0)
                vertical_lengths.append(length)

        dominant_orientation: Optional[str] = None
        dominant_positions: List[float] = []
        dominant_lengths: List[float] = []
        if len(horizontal_positions) >= min_cluster_lines and len(horizontal_positions) >= len(vertical_positions):
            dominant_orientation = "h"
            dominant_positions = sorted(horizontal_positions)
            dominant_lengths = horizontal_lengths
        elif len(vertical_positions) >= min_cluster_lines:
            dominant_orientation = "v"
            dominant_positions = sorted(vertical_positions)
            dominant_lengths = vertical_lengths
        else:
            reject("cluster_not_parallel_enough", {**info, "h_count": len(horizontal_positions), "v_count": len(vertical_positions)})
            continue

        if len(dominant_positions) < min_cluster_lines:
            reject("cluster_not_enough_parallel_lines", {**info, "orientation": dominant_orientation})
            continue

        gaps = [dominant_positions[i + 1] - dominant_positions[i] for i in range(len(dominant_positions) - 1)]
        usable_gaps = [gap for gap in gaps if gap > 0]
        if len(usable_gaps) < min_cluster_lines - 1:
            reject("cluster_gap_count_too_small", {**info, "orientation": dominant_orientation})
            continue

        median_gap = float(np.median(usable_gaps))
        mean_gap = float(np.mean(usable_gaps))
        if median_gap < min_gap or median_gap > max_gap:
            reject("cluster_gap_out_of_range", {**info, "orientation": dominant_orientation, "median_gap": median_gap})
            continue
        if float(np.std(usable_gaps)) > regularity_tol:
            reject("cluster_gap_irregular", {**info, "orientation": dominant_orientation, "gaps": usable_gaps})
            continue

        if dominant_lengths:
            median_length = float(np.median(dominant_lengths))
            if median_length > max_line_len:
                reject("cluster_line_length_too_large", {**info, "orientation": dominant_orientation, "median_length": median_length})
                continue
            if any(abs(length - median_length) > max(12.0, median_length * 0.45) for length in dominant_lengths):
                reject("cluster_line_length_irregular", {**info, "orientation": dominant_orientation, "median_length": median_length})
                continue

        stats["symbolic_cluster_count"] += 1
        stats["stair_like_cluster_count"] += 1
        removed_mask[component_mask > 0] = 255
        filtered_mask[component_mask > 0] = 0
        stats["clusters"].append(
            {
                "accepted": True,
                "kind": "stair_like_parallel_cluster",
                "bbox": [x, y, w, h],
                "area": area,
                "orientation": dominant_orientation,
                "line_count": len(dominant_positions),
                "median_gap": round(median_gap, 3),
                "mean_gap": round(mean_gap, 3),
            }
        )

    stats["removed_symbolic_pixels"] = int(np.sum(removed_mask > 0))
    return filtered_mask, removed_mask, stats


def filter_repeated_line_patterns(
    candidate_mask: np.ndarray,
    polygon_np: np.ndarray,
    scale: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    filtered_mask = candidate_mask.copy()
    removed_mask = np.zeros_like(candidate_mask)
    stats: Dict[str, Any] = {
        "repeated_line_pattern_count": 0,
        "removed_repeated_line_pixels": 0,
        "rejected_repeated_patterns": 0,
        "reject_reasons": {},
        "patterns": [],
    }

    def reject(reason: str, info: Optional[Dict[str, Any]] = None) -> None:
        stats["rejected_repeated_patterns"] += 1
        stats["reject_reasons"][reason] = stats["reject_reasons"].get(reason, 0) + 1
        if info is not None:
            stats["patterns"].append({"accepted": False, "reason": reason, **info})

    min_len = max(18, int(round(scale["opening_min_gap"] * 0.6)))
    max_len = max(64, int(round(scale["opening_max_gap"] * 0.85)))
    h, w = candidate_mask.shape[:2]
    lines = cv2.HoughLinesP(
        candidate_mask,
        1,
        np.pi / 180.0,
        threshold=max(12, scale["opening_scan_step"] * 3),
        minLineLength=min_len,
        maxLineGap=max(5, scale["opening_scan_step"]),
    )
    if lines is None:
        return filtered_mask, removed_mask, stats

    short_lines: List[Dict[str, Any]] = []
    tol_axis = max(4, scale["max_wall_thickness"])
    for raw in lines:
        x1, y1, x2, y2 = [int(v) for v in raw[0]]
        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))
        if length < min_len or length > max_len:
            continue
        if abs(dy) <= tol_axis:
            short_lines.append(
                {
                    "orientation": "h",
                    "coords": [min(x1, x2), int(round((y1 + y2) / 2)), max(x1, x2), int(round((y1 + y2) / 2))],
                    "position": float((y1 + y2) / 2.0),
                    "span_start": float(min(x1, x2)),
                    "span_end": float(max(x1, x2)),
                    "length": length,
                }
            )
        elif abs(dx) <= tol_axis:
            short_lines.append(
                {
                    "orientation": "v",
                    "coords": [int(round((x1 + x2) / 2)), min(y1, y2), int(round((x1 + x2) / 2)), max(y1, y2)],
                    "position": float((x1 + x2) / 2.0),
                    "span_start": float(min(y1, y2)),
                    "span_end": float(max(y1, y2)),
                    "length": length,
                }
            )

    if not short_lines:
        return filtered_mask, removed_mask, stats

    min_cluster_lines = 4
    near_outer_tol = max(scale["outer_edge_offset"] * 2, scale["max_wall_thickness"] + 6)
    min_gap = max(5.0, float(scale["min_wall_thickness"]))
    max_gap = max(18.0, float(scale["max_wall_thickness"] * 4))
    max_span_mismatch = max(14.0, float(scale["opening_min_gap"]))
    regularity_tol = max(4.0, float(scale["min_wall_thickness"] * 1.6))

    long_h_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(max_len + 10, scale["opening_max_gap"]), 1),
    )
    long_v_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(max_len + 10, scale["opening_max_gap"])),
    )
    long_support = cv2.bitwise_or(
        cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, long_h_kernel),
        cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, long_v_kernel),
    )

    used: set[int] = set()
    for orientation in ("h", "v"):
        oriented = [line for line in short_lines if line["orientation"] == orientation]
        for idx, base in enumerate(oriented):
            if idx in used:
                continue
            cluster = [idx]
            used.add(idx)
            for j in range(idx + 1, len(oriented)):
                if j in used:
                    continue
                candidate = oriented[j]
                span_overlap = min(base["span_end"], candidate["span_end"]) - max(base["span_start"], candidate["span_start"])
                if span_overlap < min(base["length"], candidate["length"]) * 0.45:
                    continue
                if abs(base["span_start"] - candidate["span_start"]) > max_span_mismatch:
                    continue
                if abs(base["span_end"] - candidate["span_end"]) > max_span_mismatch:
                    continue
                cluster.append(j)
                used.add(j)

            if len(cluster) < min_cluster_lines:
                reject("pattern_not_enough_parallel_lines", {"orientation": orientation, "line_count": len(cluster)})
                continue

            cluster_lines = [oriented[i] for i in cluster]
            positions = sorted(line["position"] for line in cluster_lines)
            lengths = [line["length"] for line in cluster_lines]
            gaps = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
            if not gaps:
                reject("pattern_missing_gaps", {"orientation": orientation, "line_count": len(cluster_lines)})
                continue
            median_gap = float(np.median(gaps))
            if median_gap < min_gap or median_gap > max_gap:
                reject("pattern_gap_out_of_range", {"orientation": orientation, "median_gap": round(median_gap, 3), "line_count": len(cluster_lines)})
                continue
            if float(np.std(gaps)) > regularity_tol:
                reject("pattern_gap_irregular", {"orientation": orientation, "gaps": [round(v, 3) for v in gaps]})
                continue

            median_length = float(np.median(lengths))
            if any(abs(length - median_length) > max(12.0, median_length * 0.35) for length in lengths):
                reject("pattern_length_irregular", {"orientation": orientation, "median_length": round(median_length, 3)})
                continue

            xs: List[float] = []
            ys: List[float] = []
            for line in cluster_lines:
                x1, y1, x2, y2 = line["coords"]
                xs.extend([x1, x2])
                ys.extend([y1, y2])
            min_x = max(0, int(np.floor(min(xs) - scale["max_wall_thickness"] * 2)))
            max_x = min(w - 1, int(np.ceil(max(xs) + scale["max_wall_thickness"] * 2)))
            min_y = max(0, int(np.floor(min(ys) - scale["max_wall_thickness"] * 2)))
            max_y = min(h - 1, int(np.ceil(max(ys) + scale["max_wall_thickness"] * 2)))
            center = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
            if cv2.pointPolygonTest(polygon_np, center, True) < near_outer_tol:
                reject("pattern_near_outer_polygon", {"orientation": orientation, "bbox": [min_x, min_y, max_x - min_x + 1, max_y - min_y + 1]})
                continue

            bbox_w = max_x - min_x + 1
            bbox_h = max_y - min_y + 1
            if orientation == "h":
                if bbox_h > max_gap * (len(cluster_lines) + 2) or bbox_w > median_length * 1.6:
                    reject("pattern_bbox_too_large", {"orientation": orientation, "bbox": [min_x, min_y, bbox_w, bbox_h]})
                    continue
            else:
                if bbox_w > max_gap * (len(cluster_lines) + 2) or bbox_h > median_length * 1.6:
                    reject("pattern_bbox_too_large", {"orientation": orientation, "bbox": [min_x, min_y, bbox_w, bbox_h]})
                    continue

            local_candidate = filtered_mask[min_y:max_y + 1, min_x:max_x + 1]
            local_long_support = long_support[min_y:max_y + 1, min_x:max_x + 1]
            removable_local = cv2.bitwise_and(local_candidate, cv2.bitwise_not(local_long_support))
            removed_pixels = int(np.sum(removable_local > 0))
            if removed_pixels < max(20, scale["min_wall_thickness"] * 6):
                reject("pattern_not_enough_removable_pixels", {"orientation": orientation, "removed_pixels": removed_pixels})
                continue

            removed_mask[min_y:max_y + 1, min_x:max_x + 1][removable_local > 0] = 255
            filtered_mask[min_y:max_y + 1, min_x:max_x + 1][removable_local > 0] = 0
            stats["repeated_line_pattern_count"] += 1
            stats["patterns"].append(
                {
                    "accepted": True,
                    "orientation": orientation,
                    "bbox": [min_x, min_y, bbox_w, bbox_h],
                    "line_count": len(cluster_lines),
                    "median_gap": round(median_gap, 3),
                    "median_length": round(median_length, 3),
                    "removed_pixels": removed_pixels,
                }
            )

    stats["removed_repeated_line_pixels"] = int(np.sum(removed_mask > 0))
    return filtered_mask, removed_mask, stats


def draw_lines_mask(shape: Tuple[int, int], lines: List[List[int]], thickness: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for x1, y1, x2, y2 in lines:
        cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, thickness=thickness)
    return mask


def estimate_rooms(
    polygon_np: np.ndarray,
    inner_walls: List[List[int]],
    doors: List[Dict[str, int]],
    mask_shape: Tuple[int, int],
    scale: Dict[str, int],
    project_debug_dir: Optional[Path] = None,
    floor_name: Optional[str] = None,
) -> List[Dict[str, int]]:
    room_mask = np.zeros(mask_shape, dtype=np.uint8)
    cv2.fillPoly(room_mask, [polygon_np], 255)
    polygon_mask = np.zeros(mask_shape, dtype=np.uint8)
    cv2.fillPoly(polygon_mask, [polygon_np], 255)

    wall_mask = draw_lines_mask(
        mask_shape,
        inner_walls,
        thickness=max(8, scale["max_wall_thickness"]),
    )
    room_mask[wall_mask > 0] = 0

    # Door points represent openings between rooms; for room segmentation they must
    # act as temporary barriers instead of reconnecting the free-space mask.
    if doors:
        door_close_mask = np.zeros(mask_shape, dtype=np.uint8)
        close_thickness = max(scale["max_wall_thickness"] + 6, scale["opening_min_gap"] // 2)
        for door in doors:
            cx, cy = int(door["x"]), int(door["y"])
            cv2.circle(door_close_mask, (cx, cy), close_thickness // 2, 255, -1)

        room_mask[door_close_mask > 0] = 0
        room_mask = cv2.bitwise_and(room_mask, polygon_mask)

    room_mask = cv2.bitwise_and(room_mask, polygon_mask)
    room_mask = cv2.morphologyEx(room_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(room_mask, connectivity=8)
    rooms: List[Dict[str, int]] = []
    min_room_area = max(1400, (scale["opening_min_gap"] ** 2) * 3)

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_room_area:
            continue

        cx, cy = centroids[label]
        rooms.append({
            "id": len(rooms) + 1,
            "x": int(round(cx)),
            "y": int(round(cy)),
            "area": area,
        })

    if project_debug_dir is not None and floor_name is not None:
        debug = np.zeros((mask_shape[0], mask_shape[1], 3), dtype=np.uint8)
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_room_area:
                continue
            color = (
                int((53 * label) % 255),
                int((97 * label) % 255),
                int((149 * label) % 255),
            )
            debug[labels == label] = color

        wall_mask_bgr = cv2.cvtColor(wall_mask, cv2.COLOR_GRAY2BGR)
        debug = cv2.addWeighted(debug, 0.82, wall_mask_bgr, 0.18, 0)
        save_debug(project_debug_dir, floor_name, "rooms.png", debug)

    return rooms


def estimate_rooms_clean_plan(
    polygon_np: np.ndarray,
    inner_walls: List[List[int]],
    doors: List[Dict[str, int]],
    mask_shape: Tuple[int, int],
    scale: Dict[str, int],
    project_debug_dir: Optional[Path] = None,
    floor_name: Optional[str] = None,
) -> List[Dict[str, int]]:
    room_mask = np.zeros(mask_shape, dtype=np.uint8)
    cv2.fillPoly(room_mask, [polygon_np], 255)
    polygon_mask = np.zeros(mask_shape, dtype=np.uint8)
    cv2.fillPoly(polygon_mask, [polygon_np], 255)

    barrier_thickness = max(12, scale["max_wall_thickness"] + 6, scale["opening_min_gap"] // 2)
    barrier_mask = draw_lines_mask(
        mask_shape,
        inner_walls,
        thickness=barrier_thickness,
    )
    endpoint_radius = max(3, barrier_thickness // 2)
    axis_tol = max(8, int(round(scale["max_wall_thickness"] * 1.6)))
    anchor_tol = max(14, scale["opening_min_gap"] // 2, scale["max_wall_thickness"] * 2 + 2)
    bridge_tol = max(10, scale["max_wall_thickness"] * 2 + 2)
    outer_segments = polygon_to_axis_aligned_segments(
        [[int(pt[0]), int(pt[1])] for pt in polygon_np.reshape(-1, 2)],
        axis_tol=8,
        min_length=max(24, scale["opening_min_gap"]),
    )

    def projected_anchor(endpoint: Tuple[int, int], line: List[int], anchors: List[List[int]]) -> Optional[Tuple[int, int]]:
        normalized = normalize_line(line)
        orth = "v" if normalized["orientation"] == "h" else "h"
        px, py = endpoint
        best: Optional[Tuple[float, Tuple[int, int]]] = None
        for other in anchors:
            other_norm = normalize_line(other)
            if other_norm["orientation"] != orth:
                continue
            if orth == "h":
                tx = int(np.clip(px, int(other_norm["start"]), int(other_norm["end"])))
                ty = int(other_norm["fixed"])
            else:
                tx = int(other_norm["fixed"])
                ty = int(np.clip(py, int(other_norm["start"]), int(other_norm["end"])))
            dist = hypot(float(px - tx), float(py - ty))
            if dist > anchor_tol:
                continue
            if best is None or dist < best[0]:
                best = (dist, (tx, ty))
        return best[1] if best is not None else None

    for idx, line in enumerate(inner_walls):
        for endpoint in line_endpoints(line):
            cv2.circle(barrier_mask, endpoint, endpoint_radius, 255, -1)
            anchor_lines = [other for j, other in enumerate(inner_walls) if j != idx] + outer_segments
            anchor_point = projected_anchor(endpoint, line, anchor_lines)
            if anchor_point is not None:
                cv2.line(barrier_mask, endpoint, anchor_point, 255, max(2, barrier_thickness // 3))

    horizontal_lines = sorted(
        [line[:] for line in inner_walls if normalize_line(line)["orientation"] == "h"],
        key=lambda line: (normalize_line(line)["fixed"], normalize_line(line)["start"]),
    )
    vertical_lines = sorted(
        [line[:] for line in inner_walls if normalize_line(line)["orientation"] == "v"],
        key=lambda line: (normalize_line(line)["fixed"], normalize_line(line)["start"]),
    )

    def bridge_collinear(lines: List[List[int]], orientation: str) -> None:
        for first, second in zip(lines, lines[1:]):
            a = normalize_line(first)
            b = normalize_line(second)
            if abs(int(a["fixed"]) - int(b["fixed"])) > axis_tol:
                continue
            gap = int(b["start"]) - int(a["end"])
            if gap <= 0 or gap > bridge_tol:
                continue
            if orientation == "h":
                mid_x = int(round((int(a["end"]) + int(b["start"])) / 2.0))
                mid_y = int(a["fixed"])
            else:
                mid_x = int(a["fixed"])
                mid_y = int(round((int(a["end"]) + int(b["start"])) / 2.0))
            if any(hypot(float(door["x"] - mid_x), float(door["y"] - mid_y)) <= max(12, scale["opening_min_gap"] // 2) for door in doors):
                continue
            bridge_line = denormalize_line({
                "orientation": orientation,
                "fixed": int(a["fixed"]),
                "start": int(a["end"]),
                "end": int(b["start"]),
            })
            cv2.line(
                barrier_mask,
                (int(bridge_line[0]), int(bridge_line[1])),
                (int(bridge_line[2]), int(bridge_line[3])),
                255,
                max(2, barrier_thickness // 3),
            )

    bridge_collinear(horizontal_lines, "h")
    bridge_collinear(vertical_lines, "v")

    barrier_mask = cv2.dilate(barrier_mask, np.ones((3, 3), np.uint8), iterations=1)
    room_mask[barrier_mask > 0] = 0

    if doors:
        door_close_mask = np.zeros(mask_shape, dtype=np.uint8)
        close_thickness = max(scale["max_wall_thickness"] + 8, scale["opening_min_gap"] // 2 + 2)
        for door in doors:
            cx, cy = int(door["x"]), int(door["y"])
            cv2.circle(door_close_mask, (cx, cy), close_thickness // 2, 255, -1)
        room_mask[door_close_mask > 0] = 0
        room_mask = cv2.bitwise_and(room_mask, polygon_mask)

    room_mask = cv2.bitwise_and(room_mask, polygon_mask)
    room_mask = cv2.morphologyEx(room_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(room_mask, connectivity=8)
    rooms: List[Dict[str, int]] = []
    min_room_area = max(1000, int((scale["opening_min_gap"] ** 2) * 2.2))

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_room_area:
            continue
        cx, cy = centroids[label]
        rooms.append({
            "id": len(rooms) + 1,
            "x": int(round(cx)),
            "y": int(round(cy)),
            "area": area,
        })

    if project_debug_dir is not None and floor_name is not None:
        debug = np.zeros((mask_shape[0], mask_shape[1], 3), dtype=np.uint8)
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_room_area:
                continue
            color = (
                int((53 * label) % 255),
                int((97 * label) % 255),
                int((149 * label) % 255),
            )
            debug[labels == label] = color

        barrier_bgr = cv2.cvtColor(barrier_mask, cv2.COLOR_GRAY2BGR)
        debug = cv2.addWeighted(debug, 0.82, barrier_bgr, 0.18, 0)
        save_debug(project_debug_dir, floor_name, "rooms.png", debug)

    return rooms


def build_floor_summary(
    polygon: List[List[int]],
    inner_walls: List[List[int]],
    doors: List[Dict[str, int]],
    windows: List[Dict[str, int]],
    rooms: List[Dict[str, int]],
) -> Dict[str, int]:
    return {
        "polygon_points": len(polygon),
        "inner_wall_count": len(inner_walls),
        "door_count": len(doors),
        "window_count": len(windows),
        "room_count": len(rooms),
    }


def build_inner_wall_mask_from_segments(
    mask_shape: Tuple[int, int],
    inner_walls: List[List[int]],
    scale: Dict[str, int],
) -> np.ndarray:
    mask = np.zeros(mask_shape, dtype=np.uint8)
    draw_thickness = max(2, scale["max_wall_thickness"] // 2)
    for line in inner_walls:
        cv2.line(mask, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), 255, draw_thickness)
    if inner_walls:
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return mask


def score_fullpath_wall_candidate(
    wall_score: float,
    inner_walls: List[List[int]],
    doors: List[Dict[str, int]],
    windows: List[Dict[str, int]],
    rooms: List[Dict[str, Any]],
) -> float:
    return (
        float(wall_score)
        + len(doors) * 1.8
        + len(windows) * 1.1
        + len(rooms) * 2.6
        - max(0, len(inner_walls) - 8) * 0.2
    )


def extract_inner_walls(
    binary_mask: np.ndarray,
    polygon_np: np.ndarray,
    debug_img: np.ndarray,
    project_debug_dir: Path,
    floor_name: str,
    scale: Dict[str, int],
) -> Tuple[List[List[int]], np.ndarray, Dict[str, Any]]:
    debug_stats: Dict[str, Any] = {
        "raw_candidate_wall_count": 0,
        "filtered_wall_count": 0,
        "rejected_segment_count": 0,
        "reject_reasons": {},
        "stages": {},
    }

    def merge_reasons(reason_map: Dict[str, int]) -> None:
        for key, value in reason_map.items():
            debug_stats["reject_reasons"][key] = debug_stats["reject_reasons"].get(key, 0) + int(value)
            debug_stats["rejected_segment_count"] += int(value)

    horizontal_mask, vertical_mask, combined_mask, inner_candidates = extract_oriented_wall_masks(
        binary_mask,
        polygon_np,
        scale,
    )
    raw_inner_candidates_mask = inner_candidates.copy()
    inner_candidates, symbolic_removed_mask, symbolic_cluster_stats = filter_symbolic_line_clusters(
        inner_candidates,
        polygon_np,
        scale,
    )
    debug_stats["symbolic_cluster_count"] = int(symbolic_cluster_stats.get("symbolic_cluster_count", 0))
    debug_stats["stair_like_cluster_count"] = int(symbolic_cluster_stats.get("stair_like_cluster_count", 0))
    debug_stats["removed_symbolic_pixels"] = int(symbolic_cluster_stats.get("removed_symbolic_pixels", 0))
    debug_stats["rejected_symbolic_clusters"] = int(symbolic_cluster_stats.get("rejected_symbolic_clusters", 0))
    if symbolic_cluster_stats.get("reject_reasons"):
        merge_reasons(symbolic_cluster_stats["reject_reasons"])
    debug_stats["symbolic_cluster_debug"] = symbolic_cluster_stats.get("clusters", [])
    debug_stats["stages"]["after_symbolic_cluster_filter"] = int(np.count_nonzero(inner_candidates > 0))

    inner_candidates, repeated_removed_mask, repeated_pattern_stats = filter_repeated_line_patterns(
        inner_candidates,
        polygon_np,
        scale,
    )
    debug_stats["repeated_line_pattern_count"] = int(repeated_pattern_stats.get("repeated_line_pattern_count", 0))
    debug_stats["removed_repeated_line_pixels"] = int(repeated_pattern_stats.get("removed_repeated_line_pixels", 0))
    debug_stats["rejected_repeated_patterns"] = int(repeated_pattern_stats.get("rejected_repeated_patterns", 0))
    if repeated_pattern_stats.get("reject_reasons"):
        merge_reasons(repeated_pattern_stats["reject_reasons"])
    debug_stats["repeated_pattern_debug"] = repeated_pattern_stats.get("patterns", [])
    debug_stats["stages"]["after_repeated_line_filter"] = int(np.count_nonzero(inner_candidates > 0))
    skeleton_debug = extract_skeleton_centerline_candidates(
        inner_candidates,
        polygon_np,
        scale,
    )
    debug_stats["skeleton_enabled"] = bool(skeleton_debug.get("skeleton_enabled", False))
    debug_stats["skeleton_method"] = skeleton_debug.get("skeleton_method", "not_available")
    debug_stats["skeleton_candidate_count"] = int(skeleton_debug.get("skeleton_candidate_count", 0))
    debug_stats["skeleton_candidates"] = skeleton_debug.get("skeleton_candidates", [])
    debug_stats["skeleton_candidate_vertical_count"] = int(skeleton_debug.get("skeleton_candidate_vertical_count", 0))
    debug_stats["skeleton_candidate_horizontal_count"] = int(skeleton_debug.get("skeleton_candidate_horizontal_count", 0))
    debug_stats["skeleton_mask_nonzero_count"] = int(skeleton_debug.get("skeleton_mask_nonzero_count", 0))
    projection_debug = extract_projection_grid_candidates(
        inner_candidates,
        polygon_np,
        scale,
    )
    debug_stats["projection_grid_enabled"] = bool(projection_debug.get("projection_grid_enabled", False))
    debug_stats["projection_vertical_axis_count"] = int(projection_debug.get("projection_vertical_axis_count", 0))
    debug_stats["projection_horizontal_axis_count"] = int(projection_debug.get("projection_horizontal_axis_count", 0))
    debug_stats["projection_vertical_axes"] = projection_debug.get("projection_vertical_axes", [])
    debug_stats["projection_horizontal_axes"] = projection_debug.get("projection_horizontal_axes", [])
    debug_stats["projection_candidate_count"] = int(projection_debug.get("projection_candidate_count", 0))
    debug_stats["projection_candidate_vertical_count"] = int(projection_debug.get("projection_candidate_vertical_count", 0))
    debug_stats["projection_candidate_horizontal_count"] = int(projection_debug.get("projection_candidate_horizontal_count", 0))
    debug_stats["projection_candidates"] = projection_debug.get("projection_candidates", [])

    hk = scale["horizontal_kernel"]
    vk = scale["vertical_kernel"]
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk))

    horizontal_mask = cv2.morphologyEx(inner_candidates, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_mask = cv2.morphologyEx(inner_candidates, cv2.MORPH_OPEN, vertical_kernel)
    horizontal_mask = cv2.morphologyEx(horizontal_mask, cv2.MORPH_CLOSE, np.ones((9, 3), np.uint8), iterations=1)
    vertical_mask = cv2.morphologyEx(vertical_mask, cv2.MORPH_CLOSE, np.ones((3, 9), np.uint8), iterations=1)
    combined_mask = cv2.bitwise_or(horizontal_mask, vertical_mask)

    save_debug(project_debug_dir, floor_name, "inner_candidates.png", inner_candidates)
    save_debug(project_debug_dir, floor_name, "horizontal_walls_mask.png", horizontal_mask)
    save_debug(project_debug_dir, floor_name, "vertical_walls_mask.png", vertical_mask)
    save_debug(project_debug_dir, floor_name, "combined_walls_mask.png", combined_mask)
    symbol_overlay = debug_img.copy()
    symbol_overlay[symbolic_removed_mask > 0] = (0, 0, 255)
    save_debug(project_debug_dir, floor_name, "symbol_filter_overlay.png", symbol_overlay)
    repeated_overlay = debug_img.copy()
    repeated_overlay[repeated_removed_mask > 0] = (255, 0, 255)
    save_debug(project_debug_dir, floor_name, "repeated_line_overlay.png", repeated_overlay)

    horizontal_segments, horizontal_stats = segments_from_oriented_mask(
        horizontal_mask, "h", polygon_np, inner_candidates, scale
    )
    vertical_segments, vertical_stats = segments_from_oriented_mask(
        vertical_mask, "v", polygon_np, inner_candidates, scale
    )
    raw_candidate_analysis = analyze_raw_inner_wall_candidates(
        horizontal_segments + vertical_segments,
        inner_candidates,
        polygon_np,
        scale,
    )

    debug_stats["raw_candidate_wall_count"] = int(
        horizontal_stats["raw_candidate_wall_count"] + vertical_stats["raw_candidate_wall_count"]
    )
    debug_stats["raw_candidate_segment_details"] = raw_candidate_analysis.get("candidate_details", [])
    debug_stats["parallel_edge_pair_count"] = int(raw_candidate_analysis.get("parallel_edge_pair_count", 0))
    debug_stats["estimated_centerline_axis_count"] = int(raw_candidate_analysis.get("estimated_centerline_axis_count", 0))
    debug_stats["estimated_centerline_axes"] = raw_candidate_analysis.get("estimated_centerline_axes", [])
    debug_stats["edge_like_candidate_count"] = int(raw_candidate_analysis.get("edge_like_candidate_count", 0))
    debug_stats["centerline_like_candidate_count"] = int(raw_candidate_analysis.get("centerline_like_candidate_count", 0))
    debug_stats["parallel_edge_pairs"] = raw_candidate_analysis.get("parallel_edge_pairs", [])
    merge_reasons(horizontal_stats["reject_reasons"])
    merge_reasons(vertical_stats["reject_reasons"])
    debug_stats["stages"]["oriented_segments"] = int(len(horizontal_segments) + len(vertical_segments))

    wall_segments = horizontal_segments + vertical_segments
    debug_stats["oriented_wall_segments"] = [line[:] for line in wall_segments]
    raw_candidate_based_segments, raw_candidate_based_stats = build_raw_candidate_based_inner_walls(
        wall_segments,
        polygon_np,
        inner_candidates,
        scale,
    )
    raw_axis_recon_segments: List[List[int]] = []
    raw_axis_recon_mask = np.zeros_like(inner_candidates)
    raw_axis_recon_stats: Dict[str, Any] = {
        "raw_axis_recon_enabled": False,
        "raw_axis_recon_count": 0,
        "raw_axis_recon_segments": [],
        "raw_axis_recon_score": float("-inf"),
    }
    if USE_RAW_AXIS_RECON_EXTRACTOR:
        raw_axis_recon_segments, raw_axis_recon_mask, raw_axis_recon_stats = build_raw_axis_reconstructed_inner_walls(
            wall_segments,
            polygon_np,
            inner_candidates,
            scale,
        )
    raw_axis_hybrid_segments: List[List[int]] = []
    raw_axis_hybrid_mask = np.zeros_like(inner_candidates)
    raw_axis_hybrid_stats: Dict[str, Any] = {
        "raw_axis_hybrid_enabled": False,
        "raw_axis_hybrid_count": 0,
        "raw_axis_hybrid_segments": [],
        "raw_axis_hybrid_score": float("-inf"),
    }
    wall_region_graph_segments: List[List[int]] = []
    wall_region_graph_mask = np.zeros_like(inner_candidates)
    wall_region_graph_stats: Dict[str, Any] = {
        "wall_region_graph_enabled": False,
        "wall_region_graph_count": 0,
        "wall_region_graph_segments": [],
        "wall_region_graph_score": float("-inf"),
    }
    if USE_WALL_REGION_GRAPH_EXTRACTOR:
        wall_region_graph_segments, wall_region_graph_mask, wall_region_graph_stats = build_wall_region_graph_inner_walls(
            inner_candidates,
            polygon_np,
            scale,
        )
    orthogonal_clean_segments: List[List[int]] = []
    orthogonal_clean_mask = np.zeros_like(inner_candidates)
    orthogonal_clean_stats: Dict[str, Any] = {
        "orthogonal_clean_enabled": False,
        "orthogonal_clean_count": 0,
        "orthogonal_clean_segments": [],
        "orthogonal_clean_score": float("-inf"),
    }
    if USE_ORTHOGONAL_CLEAN_PLAN_EXTRACTOR:
        orthogonal_clean_segments, orthogonal_clean_mask, orthogonal_clean_stats = build_orthogonal_clean_plan_inner_walls(
            raw_inner_candidates_mask,
            polygon_np,
            scale,
        )
    line_evidence_segments: List[List[int]] = []
    line_evidence_mask = np.zeros_like(inner_candidates)
    line_evidence_stats: Dict[str, Any] = {
        "line_evidence_enabled": False,
        "line_evidence_count": 0,
        "line_evidence_segments": [],
        "line_evidence_score": float("-inf"),
    }
    if USE_LINE_EVIDENCE_EXTRACTOR:
        line_evidence_segments, line_evidence_mask, line_evidence_stats = build_line_evidence_inner_walls(
            raw_inner_candidates_mask,
            polygon_np,
            scale,
        )
    semantic_inner_mask = np.zeros_like(inner_candidates)
    semantic_inner_mask_stats: Dict[str, Any] = {
        "semantic_inner_mask_enabled": False,
        "semantic_inner_mask_pixels": 0,
        "semantic_non_axis_pixels": 0,
        "semantic_axis_seed_pixels": 0,
    }
    semantic_inner_segments: List[List[int]] = []
    semantic_inner_line_mask = np.zeros_like(inner_candidates)
    semantic_inner_stats: Dict[str, Any] = {
        "semantic_segment_count": 0,
        "semantic_filtered_pixels": 0,
        "semantic_symbolic_rejects": 0,
        "semantic_repeated_rejects": 0,
        "semantic_score": float("-inf"),
    }
    if USE_SEMANTIC_INNER_MASK_EXTRACTOR:
        semantic_inner_mask, semantic_inner_mask_stats = build_semantic_inner_wall_mask(
            binary_mask,
            polygon_np,
            scale,
        )
        semantic_inner_segments, semantic_inner_line_mask, semantic_inner_stats = extract_inner_wall_segments_from_candidate_mask(
            semantic_inner_mask,
            polygon_np,
            scale,
        )
        semantic_inner_stats["semantic_score"] = score_inner_wall_set(
            semantic_inner_segments,
            polygon_np,
            semantic_inner_mask,
            scale,
        )
    clean_plan_mode_segments: List[List[int]] = []
    clean_plan_mode_mask = np.zeros_like(inner_candidates)
    clean_plan_mode_stats: Dict[str, Any] = {
        "clean_plan_mode_enabled": False,
        "clean_plan_mode_count": 0,
        "clean_plan_mode_segments": [],
        "clean_plan_mode_score": float("-inf"),
    }
    if USE_CLEAN_PLAN_MODE_EXTRACTOR:
        clean_plan_mode_segments, clean_plan_mode_mask, clean_plan_mode_stats = build_clean_plan_mode_inner_walls(
            polygon_np,
            raw_inner_candidates_mask,
            scale,
            raw_axis_recon_segments,
            line_evidence_segments,
            orthogonal_clean_segments,
            semantic_inner_segments,
            raw_candidate_based_segments,
        )
    line_evidence_supplement_segments: List[List[int]] = []
    line_evidence_supplement_mask = np.zeros_like(inner_candidates)
    line_evidence_supplement_stats: Dict[str, Any] = {
        "line_evidence_supplement_enabled": False,
        "line_evidence_supplement_count": 0,
        "line_evidence_supplement_segments": [],
        "line_evidence_supplement_added_segments": [],
        "line_evidence_supplement_score": float("-inf"),
        "line_evidence_supplement_runtime_score": float("-inf"),
    }
    line_evidence_gap_rescue_segments: List[List[int]] = []
    line_evidence_gap_rescue_mask = np.zeros_like(inner_candidates)
    line_evidence_gap_rescue_stats: Dict[str, Any] = {
        "line_evidence_gap_rescue_enabled": False,
        "line_evidence_gap_rescue_count": 0,
        "line_evidence_gap_rescue_segments": [],
        "line_evidence_gap_rescue_replacements": [],
        "line_evidence_gap_rescue_score": float("-inf"),
    }
    line_evidence_axis_replace_segments: List[List[int]] = []
    line_evidence_axis_replace_mask = np.zeros_like(inner_candidates)
    line_evidence_axis_replace_stats: Dict[str, Any] = {
        "line_evidence_axis_replace_enabled": False,
        "line_evidence_axis_replace_count": 0,
        "line_evidence_axis_replace_segments": [],
        "line_evidence_axis_replace_replacements": [],
        "line_evidence_axis_replace_score": float("-inf"),
    }
    line_evidence_partition_segments: List[List[int]] = []
    line_evidence_partition_mask = np.zeros_like(inner_candidates)
    line_evidence_partition_stats: Dict[str, Any] = {
        "line_evidence_partition_enabled": False,
        "line_evidence_partition_count": 0,
        "line_evidence_partition_segments": [],
        "line_evidence_partition_replacements": [],
        "line_evidence_partition_score": float("-inf"),
    }
    debug_stats["raw_candidate_based_enabled"] = bool(raw_candidate_based_stats.get("raw_candidate_based_enabled", False))
    debug_stats["raw_candidate_based_count"] = int(raw_candidate_based_stats.get("raw_candidate_based_count", 0))
    debug_stats["raw_candidate_based_segments"] = raw_candidate_based_stats.get("raw_candidate_based_segments", [])
    debug_stats["raw_axis_recon_enabled"] = bool(raw_axis_recon_stats.get("raw_axis_recon_enabled", False))
    debug_stats["raw_axis_recon_count"] = int(raw_axis_recon_stats.get("raw_axis_recon_count", 0))
    debug_stats["raw_axis_recon_segments"] = raw_axis_recon_stats.get("raw_axis_recon_segments", [])
    debug_stats["raw_axis_recon_cluster_count_h"] = int(raw_axis_recon_stats.get("raw_axis_recon_cluster_count_h", 0))
    debug_stats["raw_axis_recon_cluster_count_v"] = int(raw_axis_recon_stats.get("raw_axis_recon_cluster_count_v", 0))
    debug_stats["raw_axis_recon_clusters"] = raw_axis_recon_stats.get("raw_axis_recon_clusters", [])
    debug_stats["raw_axis_hybrid_enabled"] = bool(raw_axis_hybrid_stats.get("raw_axis_hybrid_enabled", False))
    debug_stats["raw_axis_hybrid_count"] = int(raw_axis_hybrid_stats.get("raw_axis_hybrid_count", 0))
    debug_stats["raw_axis_hybrid_segments"] = raw_axis_hybrid_stats.get("raw_axis_hybrid_segments", [])
    debug_stats["raw_axis_hybrid_replacements"] = raw_axis_hybrid_stats.get("raw_axis_hybrid_replacements", [])
    debug_stats["wall_region_graph_enabled"] = bool(wall_region_graph_stats.get("wall_region_graph_enabled", False))
    debug_stats["wall_region_graph_count"] = int(wall_region_graph_stats.get("wall_region_graph_count", 0))
    debug_stats["wall_region_graph_segments"] = wall_region_graph_stats.get("wall_region_graph_segments", [])
    debug_stats["wall_region_component_count"] = int(wall_region_graph_stats.get("wall_region_component_count", 0))
    debug_stats["wall_region_stripe_axis_count_vertical"] = int(wall_region_graph_stats.get("wall_region_stripe_axis_count_vertical", 0))
    debug_stats["wall_region_stripe_axis_count_horizontal"] = int(wall_region_graph_stats.get("wall_region_stripe_axis_count_horizontal", 0))
    debug_stats["wall_region_stripe_axes"] = wall_region_graph_stats.get("wall_region_stripe_axes", [])
    debug_stats["wall_region_rejected_short_count"] = int(wall_region_graph_stats.get("wall_region_rejected_short_count", 0))
    debug_stats["wall_region_rejected_outer_count"] = int(wall_region_graph_stats.get("wall_region_rejected_outer_count", 0))
    debug_stats["wall_region_candidate_generation_notes"] = wall_region_graph_stats.get("wall_region_candidate_generation_notes", [])
    debug_stats["wall_region_axis_refinement_enabled"] = bool(wall_region_graph_stats.get("wall_region_axis_refinement_enabled", False))
    debug_stats["wall_region_axis_refined_count"] = int(wall_region_graph_stats.get("wall_region_axis_refined_count", 0))
    debug_stats["wall_region_axis_refinement_candidates"] = wall_region_graph_stats.get("wall_region_axis_refinement_candidates", [])
    debug_stats["wall_region_axis_refinement_rejected_count"] = int(wall_region_graph_stats.get("wall_region_axis_refinement_rejected_count", 0))
    debug_stats["wall_region_axis_refinement_notes"] = wall_region_graph_stats.get("wall_region_axis_refinement_notes", [])
    debug_stats["wall_region_original_stripe_axes"] = wall_region_graph_stats.get("wall_region_original_stripe_axes", [])
    debug_stats["wall_region_refined_stripe_axes"] = wall_region_graph_stats.get("wall_region_refined_stripe_axes", [])
    debug_stats["orthogonal_clean_enabled"] = bool(orthogonal_clean_stats.get("orthogonal_clean_enabled", False))
    debug_stats["orthogonal_clean_count"] = int(orthogonal_clean_stats.get("orthogonal_clean_count", 0))
    debug_stats["orthogonal_clean_segments"] = orthogonal_clean_stats.get("orthogonal_clean_segments", [])
    debug_stats["orthogonal_clean_rejected_short"] = int(orthogonal_clean_stats.get("orthogonal_clean_rejected_short", 0))
    debug_stats["orthogonal_clean_rejected_outer"] = int(orthogonal_clean_stats.get("orthogonal_clean_rejected_outer", 0))
    debug_stats["orthogonal_clean_rejected_symbolic"] = int(orthogonal_clean_stats.get("orthogonal_clean_rejected_symbolic", 0))
    debug_stats["line_evidence_enabled"] = bool(line_evidence_stats.get("line_evidence_enabled", False))
    debug_stats["line_evidence_count"] = int(line_evidence_stats.get("line_evidence_count", 0))
    debug_stats["line_evidence_segments"] = line_evidence_stats.get("line_evidence_segments", [])
    debug_stats["line_evidence_rejected_short"] = int(line_evidence_stats.get("line_evidence_rejected_short", 0))
    debug_stats["line_evidence_rejected_outer"] = int(line_evidence_stats.get("line_evidence_rejected_outer", 0))
    debug_stats["line_evidence_rejected_symbolic"] = int(line_evidence_stats.get("line_evidence_rejected_symbolic", 0))
    debug_stats["semantic_inner_mask_enabled"] = bool(semantic_inner_mask_stats.get("semantic_inner_mask_enabled", False))
    debug_stats["semantic_inner_mask_pixels"] = int(semantic_inner_mask_stats.get("semantic_inner_mask_pixels", 0))
    debug_stats["semantic_non_axis_pixels"] = int(semantic_inner_mask_stats.get("semantic_non_axis_pixels", 0))
    debug_stats["semantic_axis_seed_pixels"] = int(semantic_inner_mask_stats.get("semantic_axis_seed_pixels", 0))
    debug_stats["semantic_inner_count"] = int(len(semantic_inner_segments))
    debug_stats["semantic_inner_segments"] = [line[:] for line in semantic_inner_segments]
    debug_stats["semantic_filtered_pixels"] = int(semantic_inner_stats.get("semantic_filtered_pixels", 0))
    debug_stats["semantic_symbolic_rejects"] = int(semantic_inner_stats.get("semantic_symbolic_rejects", 0))
    debug_stats["semantic_repeated_rejects"] = int(semantic_inner_stats.get("semantic_repeated_rejects", 0))
    debug_stats["clean_plan_mode_enabled"] = bool(clean_plan_mode_stats.get("clean_plan_mode_enabled", False))
    debug_stats["clean_plan_mode_count"] = int(clean_plan_mode_stats.get("clean_plan_mode_count", 0))
    debug_stats["clean_plan_mode_segments"] = clean_plan_mode_stats.get("clean_plan_mode_segments", [])
    debug_stats["clean_plan_mode_group_count"] = int(clean_plan_mode_stats.get("clean_plan_mode_group_count", 0))
    debug_stats["clean_plan_mode_consensus_kept"] = int(clean_plan_mode_stats.get("clean_plan_mode_consensus_kept", 0))
    debug_stats["clean_plan_mode_single_source_kept"] = int(clean_plan_mode_stats.get("clean_plan_mode_single_source_kept", 0))
    debug_stats["clean_plan_mode_anchor_supplement_kept"] = int(clean_plan_mode_stats.get("clean_plan_mode_anchor_supplement_kept", 0))
    debug_stats["clean_plan_mode_clean_like"] = bool(clean_plan_mode_stats.get("clean_plan_mode_clean_like", False))
    debug_stats["clean_plan_mode_rejected_outer"] = int(clean_plan_mode_stats.get("clean_plan_mode_rejected_outer", 0))
    debug_stats["clean_plan_mode_rejected_symbolic"] = int(clean_plan_mode_stats.get("clean_plan_mode_rejected_symbolic", 0))
    debug_stats["clean_plan_mode_sources"] = clean_plan_mode_stats.get("clean_plan_mode_sources", {})
    wall_segments = collapse_parallel_double_lines(wall_segments, pair_tol=scale["pair_merge_tol"])
    debug_stats["stages"]["after_collapse_parallel_double_lines"] = int(len(wall_segments))
    collinear_gap_tol = max(14, min(scale["opening_max_gap"], 110))
    wall_segments, merge_stats_1 = merge_collinear_lines(
        wall_segments,
        pos_tol=8,
        gap_tol=collinear_gap_tol,
        return_stats=True,
    )
    debug_stats["junction_points_count"] = int(merge_stats_1.get("junction_points_count", 0))
    debug_stats["merge_skipped_due_to_junction"] = int(merge_stats_1.get("merge_skipped_due_to_junction", 0))
    debug_stats["merge_allowed_simple_collinear"] = int(merge_stats_1.get("merge_allowed_simple_collinear", 0))
    debug_stats["preserved_short_connectors"] = int(merge_stats_1.get("preserved_short_connectors", 0))
    debug_stats["stages"]["after_merge_collinear_lines_1"] = int(len(wall_segments))
    wall_segments = remove_duplicate_lines(wall_segments, coord_tol=8, length_tol=12)
    debug_stats["stages"]["after_remove_duplicate_lines_1"] = int(len(wall_segments))
    outer_margin = max(scale["outer_edge_offset"] * 4, scale["max_wall_thickness"] * 2 + 12)
    wall_segments, outer_filter_stats = filter_lines_inside_building(
        wall_segments,
        polygon_np,
        outer_margin=outer_margin,
    )
    merge_reasons(outer_filter_stats)
    debug_stats["stages"]["after_filter_lines_inside_building"] = int(len(wall_segments))
    wall_segments, symbolic_stats = filter_symbolic_vertical_walls(
        wall_segments,
        polygon_np,
        source_mask=inner_candidates,
        scale=scale,
    )
    merge_reasons(symbolic_stats)
    debug_stats["stages"]["after_filter_symbolic_vertical_walls"] = int(len(wall_segments))
    wall_segments, outer_boundary_stats = reject_outer_boundary_parallel_candidates(
        wall_segments,
        polygon_np,
        scale,
    )
    debug_stats["outer_boundary_rejected_count"] = int(outer_boundary_stats.get("outer_boundary_rejected_count", 0))
    debug_stats["outer_boundary_rejected_segments"] = outer_boundary_stats.get("outer_boundary_rejected_segments", [])
    debug_stats["stages"]["after_outer_boundary_candidate_rejection"] = int(len(wall_segments))
    axis_snap_before_segments = [line[:] for line in wall_segments]
    wall_segments, axis_snap_stats = conservative_axis_snap_segments(
        wall_segments,
        polygon_np,
        inner_candidates,
        scale,
    )
    debug_stats["axis_snap_enabled"] = bool(axis_snap_stats.get("axis_snap_enabled", False))
    debug_stats["axis_snap_tol"] = int(axis_snap_stats.get("axis_snap_tol", 0))
    debug_stats["axis_snap_cluster_count_vertical"] = int(axis_snap_stats.get("axis_snap_cluster_count_vertical", 0))
    debug_stats["axis_snap_cluster_count_horizontal"] = int(axis_snap_stats.get("axis_snap_cluster_count_horizontal", 0))
    debug_stats["axis_grid_vertical"] = axis_snap_stats.get("axis_grid_vertical", [])
    debug_stats["axis_grid_horizontal"] = axis_snap_stats.get("axis_grid_horizontal", [])
    debug_stats["axis_grid_scores"] = axis_snap_stats.get("axis_grid_scores", {"v": [], "h": []})
    debug_stats["axis_snap_candidates"] = axis_snap_stats.get("axis_snap_candidates", [])
    debug_stats["axis_snapped_segment_count"] = int(axis_snap_stats.get("axis_snapped_segment_count", 0))
    debug_stats["axis_snap_aborted_reason"] = axis_snap_stats.get("axis_snap_aborted_reason")
    debug_stats["axis_snapped_segments_before_after"] = axis_snap_stats.get("axis_snapped_segments_before_after", [])
    debug_stats["stages"]["after_axis_snapping"] = int(len(wall_segments))
    debug_stats["stages"]["after_prune_spurious_inner_walls"] = int(len(wall_segments))
    filtered_wall_segments = [line[:] for line in wall_segments]
    wall_segments, merge_stats_2 = merge_collinear_lines(
        wall_segments,
        pos_tol=8,
        gap_tol=collinear_gap_tol,
        return_stats=True,
    )
    debug_stats["junction_points_count"] += int(merge_stats_2.get("junction_points_count", 0))
    debug_stats["merge_skipped_due_to_junction"] += int(merge_stats_2.get("merge_skipped_due_to_junction", 0))
    debug_stats["merge_allowed_simple_collinear"] += int(merge_stats_2.get("merge_allowed_simple_collinear", 0))
    debug_stats["preserved_short_connectors"] += int(merge_stats_2.get("preserved_short_connectors", 0))
    debug_stats["stages"]["after_merge_collinear_lines_2"] = int(len(wall_segments))
    wall_segments = remove_duplicate_lines(wall_segments, coord_tol=8, length_tol=12)
    debug_stats["stages"]["after_remove_duplicate_lines_2"] = int(len(wall_segments))
    legacy_final_segments = [line[:] for line in wall_segments]
    legacy_final_score = score_inner_wall_set(legacy_final_segments, polygon_np, inner_candidates, scale)
    legacy_opening_eval = evaluate_opening_compatibility_for_wall_set(
        legacy_final_segments,
        combined_mask,
        binary_mask,
        polygon_np,
        scale,
    )
    raw_axis_opening_eval = evaluate_opening_compatibility_for_wall_set(
        raw_axis_recon_segments,
        raw_axis_recon_mask,
        binary_mask,
        polygon_np,
        scale,
    ) if raw_axis_recon_segments else {
        "opening_scan_count": 0,
        "gap_scan_count": 0,
        "hostable_line_count": 0,
        "opening_compatibility_score": float("-inf"),
    }
    if USE_RAW_AXIS_RECON_EXTRACTOR:
        raw_axis_hybrid_segments, raw_axis_hybrid_stats = build_raw_axis_hybrid_inner_walls(
            legacy_final_segments,
            raw_axis_recon_segments,
            polygon_np,
            inner_candidates,
            scale,
        )
        if raw_axis_hybrid_segments:
            draw_thickness = max(2, scale["max_wall_thickness"] // 2)
            for line in raw_axis_hybrid_segments:
                cv2.line(
                    raw_axis_hybrid_mask,
                    (int(line[0]), int(line[1])),
                    (int(line[2]), int(line[3])),
                    255,
                    draw_thickness,
                )
            raw_axis_hybrid_mask = cv2.dilate(
                raw_axis_hybrid_mask,
                np.ones((3, 3), np.uint8),
                iterations=1,
            )
    raw_axis_hybrid_opening_eval = evaluate_opening_compatibility_for_wall_set(
        raw_axis_hybrid_segments,
        raw_axis_hybrid_mask,
        binary_mask,
        polygon_np,
        scale,
    ) if raw_axis_hybrid_segments else {
        "opening_scan_count": 0,
        "gap_scan_count": 0,
        "hostable_line_count": 0,
        "opening_compatibility_score": float("-inf"),
    }
    if USE_LINE_EVIDENCE_EXTRACTOR and USE_LINE_EVIDENCE_SUPPLEMENT:
        line_evidence_supplement_segments, line_evidence_supplement_stats = build_line_evidence_supplemented_inner_walls(
            legacy_final_segments,
            line_evidence_stats,
            polygon_np,
            raw_inner_candidates_mask,
            scale,
        )
        if line_evidence_supplement_segments:
            draw_thickness = max(2, scale["max_wall_thickness"] // 2)
            for line in line_evidence_supplement_segments:
                cv2.line(
                    line_evidence_supplement_mask,
                    (int(line[0]), int(line[1])),
                    (int(line[2]), int(line[3])),
                    255,
                    draw_thickness,
                )
            line_evidence_supplement_mask = cv2.dilate(
                line_evidence_supplement_mask,
                np.ones((3, 3), np.uint8),
                iterations=1,
            )
        line_evidence_gap_rescue_segments, line_evidence_gap_rescue_stats = build_line_evidence_gap_rescue_inner_walls(
            legacy_final_segments,
            line_evidence_segments,
            polygon_np,
            raw_inner_candidates_mask,
            scale,
        )
        if line_evidence_gap_rescue_segments:
            draw_thickness = max(2, scale["max_wall_thickness"] // 2)
            for line in line_evidence_gap_rescue_segments:
                cv2.line(
                    line_evidence_gap_rescue_mask,
                    (int(line[0]), int(line[1])),
                    (int(line[2]), int(line[3])),
                    255,
                    draw_thickness,
                )
            line_evidence_gap_rescue_mask = cv2.dilate(
                line_evidence_gap_rescue_mask,
                np.ones((3, 3), np.uint8),
                iterations=1,
            )
        line_evidence_axis_replace_segments, line_evidence_axis_replace_stats = build_line_evidence_axis_replacement_inner_walls(
            legacy_final_segments,
            line_evidence_stats,
            polygon_np,
            raw_inner_candidates_mask,
            scale,
        )
        if line_evidence_axis_replace_segments:
            draw_thickness = max(2, scale["max_wall_thickness"] // 2)
            for line in line_evidence_axis_replace_segments:
                cv2.line(
                    line_evidence_axis_replace_mask,
                    (int(line[0]), int(line[1])),
                    (int(line[2]), int(line[3])),
                    255,
                    draw_thickness,
                )
            line_evidence_axis_replace_mask = cv2.dilate(
                line_evidence_axis_replace_mask,
                np.ones((3, 3), np.uint8),
                iterations=1,
            )
        line_evidence_partition_segments, line_evidence_partition_stats = build_line_evidence_partition_rescue_inner_walls(
            legacy_final_segments,
            line_evidence_segments,
            polygon_np,
            raw_inner_candidates_mask,
            scale,
        )
        if line_evidence_partition_segments:
            draw_thickness = max(2, scale["max_wall_thickness"] // 2)
            for line in line_evidence_partition_segments:
                cv2.line(
                    line_evidence_partition_mask,
                    (int(line[0]), int(line[1])),
                    (int(line[2]), int(line[3])),
                    255,
                    draw_thickness,
                )
            line_evidence_partition_mask = cv2.dilate(
                line_evidence_partition_mask,
                np.ones((3, 3), np.uint8),
                iterations=1,
            )
    raw_candidate_based_score = float(raw_candidate_based_stats.get("raw_candidate_based_score", float("-inf")))
    wall_region_graph_score = float(wall_region_graph_stats.get("wall_region_graph_score", float("-inf")))
    orthogonal_clean_score = float(orthogonal_clean_stats.get("orthogonal_clean_score", float("-inf")))
    line_evidence_score = float(line_evidence_stats.get("line_evidence_score", float("-inf")))
    semantic_inner_score = float(semantic_inner_stats.get("semantic_score", float("-inf")))
    line_evidence_supplement_score = float(line_evidence_supplement_stats.get("line_evidence_supplement_score", float("-inf")))
    line_evidence_supplement_runtime_score = float("-inf")
    line_evidence_gap_rescue_score = float(line_evidence_gap_rescue_stats.get("line_evidence_gap_rescue_score", float("-inf")))
    line_evidence_axis_replace_score = float(line_evidence_axis_replace_stats.get("line_evidence_axis_replace_score", float("-inf")))
    line_evidence_partition_score = float(line_evidence_partition_stats.get("line_evidence_partition_score", float("-inf")))
    supplement_added_count = len(line_evidence_supplement_stats.get("line_evidence_supplement_added_segments", []))
    gap_rescue_replacement_count = len(line_evidence_gap_rescue_stats.get("line_evidence_gap_rescue_replacements", []))
    axis_replace_replacement_count = len(line_evidence_axis_replace_stats.get("line_evidence_axis_replace_replacements", []))
    partition_replacement_count = len(line_evidence_partition_stats.get("line_evidence_partition_replacements", []))
    if line_evidence_supplement_segments:
        small_addition_bonus = min(2, supplement_added_count) * 1.1
        count_penalty = max(0, len(line_evidence_supplement_segments) - len(legacy_final_segments)) * 0.08
        line_evidence_supplement_runtime_score = line_evidence_supplement_score + small_addition_bonus - count_penalty
        line_evidence_supplement_stats["line_evidence_supplement_runtime_score"] = line_evidence_supplement_runtime_score
    raw_candidate_based_reason = "feature_flag_disabled"
    raw_candidate_based_used_as_final = False
    raw_axis_recon_reason = "feature_flag_disabled"
    raw_axis_recon_used_as_final = False
    raw_axis_hybrid_reason = "feature_flag_disabled"
    raw_axis_hybrid_used_as_final = False
    wall_region_graph_reason = "feature_flag_disabled"
    wall_region_graph_used_as_final = False
    orthogonal_clean_reason = "feature_flag_disabled"
    orthogonal_clean_used_as_final = False
    line_evidence_reason = "feature_flag_disabled"
    line_evidence_used_as_final = False
    semantic_inner_reason = "feature_flag_disabled"
    semantic_inner_used_as_final = False
    clean_plan_mode_reason = "parallel_candidate_only"
    clean_plan_mode_used_as_final = False
    line_evidence_supplement_reason = "feature_flag_disabled"
    line_evidence_supplement_used_as_final = False
    line_evidence_gap_rescue_reason = "feature_flag_disabled"
    line_evidence_gap_rescue_used_as_final = False
    line_evidence_axis_replace_reason = "feature_flag_disabled"
    line_evidence_axis_replace_used_as_final = False
    line_evidence_partition_reason = "feature_flag_disabled"
    line_evidence_partition_used_as_final = False
    selected_segments = [line[:] for line in legacy_final_segments]
    selected_score = legacy_final_score
    selected_mask = combined_mask.copy()
    if USE_RAW_CANDIDATE_INNER_WALLS:
        if not raw_candidate_based_segments:
            raw_candidate_based_reason = "raw_candidate_based_empty"
        elif raw_candidate_based_score >= selected_score + 2.5:
            selected_segments = [line[:] for line in raw_candidate_based_segments]
            selected_score = raw_candidate_based_score
            raw_candidate_based_used_as_final = True
            raw_candidate_based_reason = "raw_candidate_based_score_better"
        else:
            raw_candidate_based_reason = "legacy_final_score_better"
    raw_axis_recon_score = float(raw_axis_recon_stats.get("raw_axis_recon_score", float("-inf")))
    if USE_RAW_AXIS_RECON_EXTRACTOR:
        if not raw_axis_recon_segments:
            raw_axis_recon_reason = "raw_axis_recon_empty"
        elif (
            len(raw_axis_recon_segments) != len(legacy_final_segments)
            and max(
                legacy_opening_eval["opening_compatibility_score"],
                raw_axis_opening_eval["opening_compatibility_score"],
            ) <= 0.0
        ):
            raw_axis_recon_reason = "raw_axis_recon_insufficient_opening_signal"
        elif raw_axis_opening_eval["opening_compatibility_score"] + 0.25 < legacy_opening_eval["opening_compatibility_score"]:
            raw_axis_recon_reason = "raw_axis_recon_opening_compat_worse"
        elif len(raw_axis_recon_segments) > max(12, len(legacy_final_segments) + 2):
            raw_axis_recon_reason = "raw_axis_recon_count_not_safe"
        elif raw_axis_recon_score >= selected_score + 0.6:
            selected_segments = [line[:] for line in raw_axis_recon_segments]
            selected_score = raw_axis_recon_score
            selected_mask = raw_axis_recon_mask.copy()
            raw_axis_recon_used_as_final = True
            raw_candidate_based_used_as_final = False
            raw_axis_recon_reason = "raw_axis_recon_score_better"
        else:
            raw_axis_recon_reason = "legacy_or_alt_score_better"
    raw_axis_hybrid_score = float(raw_axis_hybrid_stats.get("raw_axis_hybrid_score", float("-inf")))
    raw_axis_hybrid_replacement_count = len(raw_axis_hybrid_stats.get("raw_axis_hybrid_replacements", []))
    if USE_RAW_AXIS_RECON_EXTRACTOR:
        if not raw_axis_hybrid_segments:
            raw_axis_hybrid_reason = "raw_axis_hybrid_empty"
        elif raw_axis_hybrid_replacement_count < 1:
            raw_axis_hybrid_reason = "raw_axis_hybrid_no_replacements"
        elif len(raw_axis_hybrid_segments) != len(legacy_final_segments):
            raw_axis_hybrid_reason = "raw_axis_hybrid_count_changed"
        elif raw_axis_hybrid_opening_eval["opening_compatibility_score"] + 0.1 < legacy_opening_eval["opening_compatibility_score"]:
            raw_axis_hybrid_reason = "raw_axis_hybrid_opening_compat_worse"
        elif raw_axis_hybrid_score >= selected_score + 0.2:
            selected_segments = [line[:] for line in raw_axis_hybrid_segments]
            selected_score = raw_axis_hybrid_score
            selected_mask = raw_axis_hybrid_mask.copy()
            raw_axis_hybrid_used_as_final = True
            raw_candidate_based_used_as_final = False
            raw_axis_recon_used_as_final = False
            raw_axis_hybrid_reason = "raw_axis_hybrid_score_better"
        else:
            raw_axis_hybrid_reason = "legacy_or_alt_score_better"
    if USE_WALL_REGION_GRAPH_EXTRACTOR:
        if not wall_region_graph_segments:
            wall_region_graph_reason = "wall_region_graph_empty"
        elif wall_region_graph_score >= selected_score + 1.5:
            selected_segments = [line[:] for line in wall_region_graph_segments]
            selected_score = wall_region_graph_score
            selected_mask = wall_region_graph_mask.copy()
            wall_region_graph_used_as_final = True
            raw_candidate_based_used_as_final = False
            raw_axis_recon_used_as_final = False
            raw_axis_hybrid_used_as_final = False
            wall_region_graph_reason = "wall_region_graph_score_better"
        else:
            wall_region_graph_reason = "legacy_or_raw_score_better"
    if USE_ORTHOGONAL_CLEAN_PLAN_EXTRACTOR:
        if not orthogonal_clean_segments:
            orthogonal_clean_reason = "orthogonal_clean_empty"
        elif orthogonal_clean_score >= selected_score + 1.0:
            selected_segments = [line[:] for line in orthogonal_clean_segments]
            selected_score = orthogonal_clean_score
            selected_mask = orthogonal_clean_mask.copy()
            orthogonal_clean_used_as_final = True
            raw_candidate_based_used_as_final = False
            wall_region_graph_used_as_final = False
            raw_axis_recon_used_as_final = False
            raw_axis_hybrid_used_as_final = False
            orthogonal_clean_reason = "orthogonal_clean_score_better"
        else:
            orthogonal_clean_reason = "legacy_or_alt_score_better"
    if USE_LINE_EVIDENCE_EXTRACTOR:
        if not line_evidence_segments:
            line_evidence_reason = "line_evidence_empty"
        else:
            legacy_count = max(1, len(legacy_final_segments))
            count_ok = (
                len(line_evidence_segments) >= legacy_count
                and len(line_evidence_segments) <= max(12, int(round(legacy_count * 1.6)))
            )
            if not count_ok:
                line_evidence_reason = "line_evidence_count_not_safe"
            elif line_evidence_score >= selected_score + 1.0:
                selected_segments = [line[:] for line in line_evidence_segments]
                selected_score = line_evidence_score
                selected_mask = line_evidence_mask.copy()
                line_evidence_used_as_final = True
                raw_candidate_based_used_as_final = False
                wall_region_graph_used_as_final = False
                orthogonal_clean_used_as_final = False
                raw_axis_recon_used_as_final = False
                raw_axis_hybrid_used_as_final = False
                line_evidence_reason = "line_evidence_score_better"
            else:
                line_evidence_reason = "legacy_or_alt_score_better"
    if USE_SEMANTIC_INNER_MASK_EXTRACTOR:
        if not semantic_inner_segments:
            semantic_inner_reason = "semantic_inner_empty"
        elif len(semantic_inner_segments) == len(legacy_final_segments):
            semantic_inner_reason = "semantic_inner_same_as_legacy"
        elif len(semantic_inner_segments) > max(12, len(legacy_final_segments) + 2):
            semantic_inner_reason = "semantic_inner_count_not_safe"
        elif semantic_inner_score >= selected_score + 1.0:
            selected_segments = [line[:] for line in semantic_inner_segments]
            selected_score = semantic_inner_score
            selected_mask = semantic_inner_line_mask.copy()
            semantic_inner_used_as_final = True
            raw_candidate_based_used_as_final = False
            wall_region_graph_used_as_final = False
            orthogonal_clean_used_as_final = False
            line_evidence_used_as_final = False
            raw_axis_recon_used_as_final = False
            raw_axis_hybrid_used_as_final = False
            semantic_inner_reason = "semantic_inner_score_better"
        else:
            semantic_inner_reason = "legacy_or_alt_score_better"
    if USE_LINE_EVIDENCE_EXTRACTOR and USE_LINE_EVIDENCE_SUPPLEMENT:
        if not line_evidence_supplement_segments:
            line_evidence_supplement_reason = "line_evidence_supplement_empty"
        else:
            legacy_count = max(1, len(legacy_final_segments))
            count_delta = len(line_evidence_supplement_segments) - legacy_count
            count_ok = count_delta >= 0 and count_delta <= 3 and supplement_added_count <= 3
            if not count_ok:
                line_evidence_supplement_reason = "line_evidence_supplement_count_not_safe"
            elif supplement_added_count < 2:
                line_evidence_supplement_reason = "line_evidence_supplement_not_enough_additions"
            elif line_evidence_supplement_runtime_score >= selected_score + 0.5:
                selected_segments = [line[:] for line in line_evidence_supplement_segments]
                selected_score = line_evidence_supplement_runtime_score
                selected_mask = line_evidence_supplement_mask.copy()
                line_evidence_supplement_used_as_final = True
                raw_candidate_based_used_as_final = False
                wall_region_graph_used_as_final = False
                orthogonal_clean_used_as_final = False
                line_evidence_used_as_final = False
                raw_axis_recon_used_as_final = False
                raw_axis_hybrid_used_as_final = False
                line_evidence_supplement_reason = "line_evidence_supplement_runtime_score_better"
            else:
                line_evidence_supplement_reason = "legacy_or_alt_score_better"
        if not line_evidence_gap_rescue_segments:
            line_evidence_gap_rescue_reason = "line_evidence_gap_rescue_empty"
        elif gap_rescue_replacement_count < 2:
            line_evidence_gap_rescue_reason = "line_evidence_gap_rescue_no_replacements"
        elif len(line_evidence_gap_rescue_segments) > len(legacy_final_segments) + 2:
            line_evidence_gap_rescue_reason = "line_evidence_gap_rescue_count_not_safe"
        elif line_evidence_gap_rescue_score >= selected_score + 0.25:
            selected_segments = [line[:] for line in line_evidence_gap_rescue_segments]
            selected_score = line_evidence_gap_rescue_score
            selected_mask = line_evidence_gap_rescue_mask.copy()
            line_evidence_gap_rescue_used_as_final = True
            raw_candidate_based_used_as_final = False
            wall_region_graph_used_as_final = False
            orthogonal_clean_used_as_final = False
            line_evidence_used_as_final = False
            line_evidence_supplement_used_as_final = False
            raw_axis_recon_used_as_final = False
            raw_axis_hybrid_used_as_final = False
            line_evidence_gap_rescue_reason = "line_evidence_gap_rescue_score_better"
        else:
            line_evidence_gap_rescue_reason = "legacy_or_alt_score_better"
        if not line_evidence_axis_replace_segments:
            line_evidence_axis_replace_reason = "line_evidence_axis_replace_empty"
        elif axis_replace_replacement_count < 3:
            line_evidence_axis_replace_reason = "line_evidence_axis_replace_no_replacements"
        elif len(line_evidence_axis_replace_segments) > len(legacy_final_segments) + 1:
            line_evidence_axis_replace_reason = "line_evidence_axis_replace_count_not_safe"
        elif line_evidence_axis_replace_score >= selected_score + 0.45:
            selected_segments = [line[:] for line in line_evidence_axis_replace_segments]
            selected_score = line_evidence_axis_replace_score
            selected_mask = line_evidence_axis_replace_mask.copy()
            line_evidence_axis_replace_used_as_final = True
            raw_candidate_based_used_as_final = False
            wall_region_graph_used_as_final = False
            orthogonal_clean_used_as_final = False
            line_evidence_used_as_final = False
            line_evidence_supplement_used_as_final = False
            line_evidence_gap_rescue_used_as_final = False
            raw_axis_recon_used_as_final = False
            raw_axis_hybrid_used_as_final = False
            line_evidence_axis_replace_reason = "line_evidence_axis_replace_score_better"
        else:
            line_evidence_axis_replace_reason = "legacy_or_alt_score_better"
        if not line_evidence_partition_segments:
            line_evidence_partition_reason = "line_evidence_partition_empty"
        elif partition_replacement_count < 2:
            line_evidence_partition_reason = "line_evidence_partition_no_replacements"
        elif len(line_evidence_partition_segments) > len(legacy_final_segments) + 2:
            line_evidence_partition_reason = "line_evidence_partition_count_not_safe"
        elif line_evidence_partition_score >= selected_score + 0.45:
            selected_segments = [line[:] for line in line_evidence_partition_segments]
            selected_score = line_evidence_partition_score
            selected_mask = line_evidence_partition_mask.copy()
            line_evidence_partition_used_as_final = True
            raw_candidate_based_used_as_final = False
            wall_region_graph_used_as_final = False
            orthogonal_clean_used_as_final = False
            line_evidence_used_as_final = False
            line_evidence_supplement_used_as_final = False
            line_evidence_gap_rescue_used_as_final = False
            line_evidence_axis_replace_used_as_final = False
            raw_axis_recon_used_as_final = False
            raw_axis_hybrid_used_as_final = False
            line_evidence_partition_reason = "line_evidence_partition_score_better"
        else:
            line_evidence_partition_reason = "legacy_or_alt_score_better"
    wall_segments = selected_segments
    combined_mask = selected_mask
    debug_stats["raw_candidate_based_used_as_final"] = raw_candidate_based_used_as_final
    debug_stats["raw_candidate_based_reason"] = raw_candidate_based_reason
    debug_stats["raw_candidate_based_score"] = raw_candidate_based_score
    debug_stats["raw_axis_recon_used_as_final"] = raw_axis_recon_used_as_final
    debug_stats["raw_axis_recon_reason"] = raw_axis_recon_reason
    debug_stats["raw_axis_recon_score"] = raw_axis_recon_score
    debug_stats["raw_axis_hybrid_enabled"] = bool(raw_axis_hybrid_stats.get("raw_axis_hybrid_enabled", False))
    debug_stats["raw_axis_hybrid_count"] = int(raw_axis_hybrid_stats.get("raw_axis_hybrid_count", 0))
    debug_stats["raw_axis_hybrid_segments"] = raw_axis_hybrid_stats.get("raw_axis_hybrid_segments", [])
    debug_stats["raw_axis_hybrid_replacements"] = raw_axis_hybrid_stats.get("raw_axis_hybrid_replacements", [])
    debug_stats["raw_axis_hybrid_used_as_final"] = raw_axis_hybrid_used_as_final
    debug_stats["raw_axis_hybrid_reason"] = raw_axis_hybrid_reason
    debug_stats["raw_axis_hybrid_score"] = raw_axis_hybrid_score
    debug_stats["wall_region_graph_used_as_final"] = wall_region_graph_used_as_final
    debug_stats["wall_region_graph_reason"] = wall_region_graph_reason
    debug_stats["wall_region_graph_score"] = wall_region_graph_score
    debug_stats["orthogonal_clean_used_as_final"] = orthogonal_clean_used_as_final
    debug_stats["orthogonal_clean_reason"] = orthogonal_clean_reason
    debug_stats["orthogonal_clean_score"] = orthogonal_clean_score
    debug_stats["line_evidence_used_as_final"] = line_evidence_used_as_final
    debug_stats["line_evidence_reason"] = line_evidence_reason
    debug_stats["line_evidence_score"] = line_evidence_score
    debug_stats["semantic_inner_used_as_final"] = semantic_inner_used_as_final
    debug_stats["semantic_inner_reason"] = semantic_inner_reason
    debug_stats["semantic_inner_score"] = semantic_inner_score
    debug_stats["clean_plan_mode_used_as_final"] = clean_plan_mode_used_as_final
    debug_stats["clean_plan_mode_reason"] = clean_plan_mode_reason
    debug_stats["clean_plan_mode_score"] = float(clean_plan_mode_stats.get("clean_plan_mode_score", float("-inf")))
    debug_stats["line_evidence_supplement_enabled"] = bool(line_evidence_supplement_stats.get("line_evidence_supplement_enabled", False))
    debug_stats["line_evidence_supplement_count"] = int(line_evidence_supplement_stats.get("line_evidence_supplement_count", 0))
    debug_stats["line_evidence_supplement_segments"] = line_evidence_supplement_stats.get("line_evidence_supplement_segments", [])
    debug_stats["line_evidence_supplement_added_segments"] = line_evidence_supplement_stats.get("line_evidence_supplement_added_segments", [])
    debug_stats["line_evidence_supplement_used_as_final"] = line_evidence_supplement_used_as_final
    debug_stats["line_evidence_supplement_reason"] = line_evidence_supplement_reason
    debug_stats["line_evidence_supplement_score"] = line_evidence_supplement_score
    debug_stats["line_evidence_supplement_runtime_score"] = line_evidence_supplement_runtime_score
    debug_stats["line_evidence_gap_rescue_enabled"] = bool(line_evidence_gap_rescue_stats.get("line_evidence_gap_rescue_enabled", False))
    debug_stats["line_evidence_gap_rescue_count"] = int(line_evidence_gap_rescue_stats.get("line_evidence_gap_rescue_count", 0))
    debug_stats["line_evidence_gap_rescue_segments"] = line_evidence_gap_rescue_stats.get("line_evidence_gap_rescue_segments", [])
    debug_stats["line_evidence_gap_rescue_replacements"] = line_evidence_gap_rescue_stats.get("line_evidence_gap_rescue_replacements", [])
    debug_stats["line_evidence_gap_rescue_used_as_final"] = line_evidence_gap_rescue_used_as_final
    debug_stats["line_evidence_gap_rescue_reason"] = line_evidence_gap_rescue_reason
    debug_stats["line_evidence_gap_rescue_score"] = line_evidence_gap_rescue_score
    debug_stats["line_evidence_axis_replace_enabled"] = bool(line_evidence_axis_replace_stats.get("line_evidence_axis_replace_enabled", False))
    debug_stats["line_evidence_axis_replace_count"] = int(line_evidence_axis_replace_stats.get("line_evidence_axis_replace_count", 0))
    debug_stats["line_evidence_axis_replace_segments"] = line_evidence_axis_replace_stats.get("line_evidence_axis_replace_segments", [])
    debug_stats["line_evidence_axis_replace_replacements"] = line_evidence_axis_replace_stats.get("line_evidence_axis_replace_replacements", [])
    debug_stats["line_evidence_axis_replace_used_as_final"] = line_evidence_axis_replace_used_as_final
    debug_stats["line_evidence_axis_replace_reason"] = line_evidence_axis_replace_reason
    debug_stats["line_evidence_axis_replace_score"] = line_evidence_axis_replace_score
    debug_stats["line_evidence_partition_enabled"] = bool(line_evidence_partition_stats.get("line_evidence_partition_enabled", False))
    debug_stats["line_evidence_partition_count"] = int(line_evidence_partition_stats.get("line_evidence_partition_count", 0))
    debug_stats["line_evidence_partition_segments"] = line_evidence_partition_stats.get("line_evidence_partition_segments", [])
    debug_stats["line_evidence_partition_replacements"] = line_evidence_partition_stats.get("line_evidence_partition_replacements", [])
    debug_stats["line_evidence_partition_used_as_final"] = line_evidence_partition_used_as_final
    debug_stats["line_evidence_partition_reason"] = line_evidence_partition_reason
    debug_stats["line_evidence_partition_score"] = line_evidence_partition_score
    debug_stats["legacy_final_count"] = int(len(legacy_final_segments))
    debug_stats["legacy_final_segments"] = [line[:] for line in legacy_final_segments]
    debug_stats["legacy_final_score"] = legacy_final_score
    debug_stats["legacy_opening_compatibility"] = legacy_opening_eval
    debug_stats["raw_axis_recon_opening_compatibility"] = raw_axis_opening_eval
    debug_stats["raw_axis_hybrid_opening_compatibility"] = raw_axis_hybrid_opening_eval
    debug_stats["filtered_wall_count"] = int(len(horizontal_segments) + len(vertical_segments))
    debug_stats["final_inner_wall_count"] = int(len(wall_segments))

    raw_candidates_overlay = overlay_mask_on_image(
        debug_img,
        raw_inner_candidates_mask,
        color=(0, 0, 255),
        alpha=0.38,
    )
    skeleton_centerline_overlay = overlay_mask_on_image(
        debug_img,
        skeleton_debug.get("skeleton_mask", np.zeros_like(inner_candidates)),
        color=(0, 180, 255),
        alpha=0.35,
    )
    for line in skeleton_debug.get("skeleton_candidates", []):
        cv2.line(
            skeleton_centerline_overlay,
            (int(line[0]), int(line[1])),
            (int(line[2]), int(line[3])),
            (0, 255, 0),
            2,
        )
    projection_grid_overlay = overlay_mask_on_image(
        debug_img,
        projection_debug.get("projection_roi_mask", np.zeros_like(inner_candidates)),
        color=(255, 200, 0),
        alpha=0.2,
    )
    for axis in projection_debug.get("projection_vertical_axes", []):
        cv2.line(projection_grid_overlay, (int(axis), 0), (int(axis), int(debug_img.shape[0] - 1)), (180, 180, 180), 1)
    for axis in projection_debug.get("projection_horizontal_axes", []):
        cv2.line(projection_grid_overlay, (0, int(axis)), (int(debug_img.shape[1] - 1), int(axis)), (180, 180, 180), 1)
    for line in projection_debug.get("projection_candidates", []):
        cv2.line(
            projection_grid_overlay,
            (int(line[0]), int(line[1])),
            (int(line[2]), int(line[3])),
            (0, 255, 0),
            2,
        )
    raw_candidate_based_overlay = overlay_lines_on_image(
        debug_img,
        raw_candidate_based_segments,
        color=(0, 255, 0),
        thickness=2,
    )
    if raw_candidate_based_used_as_final:
        cv2.putText(
            raw_candidate_based_overlay,
            "USED AS FINAL",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 180, 0),
            2,
            cv2.LINE_AA,
        )
    raw_axis_recon_overlay = overlay_lines_on_image(
        debug_img,
        raw_axis_recon_segments,
        color=(0, 220, 255),
        thickness=2,
    )
    if raw_axis_recon_used_as_final:
        cv2.putText(
            raw_axis_recon_overlay,
            "USED AS FINAL",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 180, 0),
            2,
            cv2.LINE_AA,
        )
    raw_axis_hybrid_overlay = overlay_lines_on_image(
        debug_img,
        raw_axis_hybrid_segments,
        color=(60, 255, 180),
        thickness=2,
    )
    if raw_axis_hybrid_used_as_final:
        cv2.putText(
            raw_axis_hybrid_overlay,
            "USED AS FINAL",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 180, 0),
            2,
            cv2.LINE_AA,
        )
    orthogonal_clean_overlay = overlay_mask_on_image(
        debug_img,
        orthogonal_clean_stats.get("orthogonal_clean_mask", np.zeros_like(inner_candidates)),
        color=(120, 220, 120),
        alpha=0.16,
    )
    for line in orthogonal_clean_segments:
        cv2.line(
            orthogonal_clean_overlay,
            (int(line[0]), int(line[1])),
            (int(line[2]), int(line[3])),
            (0, 255, 0),
            2,
        )
    if orthogonal_clean_used_as_final:
        cv2.putText(
            orthogonal_clean_overlay,
            "USED AS FINAL",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 180, 0),
            2,
            cv2.LINE_AA,
        )
    line_evidence_overlay = overlay_mask_on_image(
        debug_img,
        line_evidence_stats.get("line_evidence_mask", np.zeros_like(inner_candidates)),
        color=(80, 180, 255),
        alpha=0.18,
    )
    for line in line_evidence_segments:
        cv2.line(
            line_evidence_overlay,
            (int(line[0]), int(line[1])),
            (int(line[2]), int(line[3])),
            (0, 255, 0),
            2,
        )
    if line_evidence_used_as_final:
        cv2.putText(
            line_evidence_overlay,
            "USED AS FINAL",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 180, 0),
            2,
            cv2.LINE_AA,
        )
    semantic_inner_overlay = overlay_mask_on_image(
        debug_img,
        semantic_inner_mask,
        color=(120, 255, 180),
        alpha=0.18,
    )
    for line in semantic_inner_segments:
        cv2.line(
            semantic_inner_overlay,
            (int(line[0]), int(line[1])),
            (int(line[2]), int(line[3])),
            (0, 255, 0),
            2,
        )
    if semantic_inner_used_as_final:
        cv2.putText(
            semantic_inner_overlay,
            "USED AS FINAL",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 180, 0),
            2,
            cv2.LINE_AA,
        )
    clean_plan_mode_overlay = overlay_mask_on_image(
        debug_img,
        clean_plan_mode_mask,
        color=(255, 210, 90),
        alpha=0.16,
    )
    for line in clean_plan_mode_segments:
        cv2.line(
            clean_plan_mode_overlay,
            (int(line[0]), int(line[1])),
            (int(line[2]), int(line[3])),
            (0, 255, 0),
            2,
        )
    if clean_plan_mode_used_as_final:
        cv2.putText(
            clean_plan_mode_overlay,
            "USED AS FINAL",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 180, 0),
            2,
            cv2.LINE_AA,
        )
    line_evidence_supplement_overlay = overlay_mask_on_image(
        debug_img,
        line_evidence_supplement_mask,
        color=(160, 210, 255),
        alpha=0.16,
    )
    for line in legacy_final_segments:
        cv2.line(
            line_evidence_supplement_overlay,
            (int(line[0]), int(line[1])),
            (int(line[2]), int(line[3])),
            (255, 120, 0),
            2,
        )
    for line in line_evidence_supplement_stats.get("line_evidence_supplement_added_segments", []):
        cv2.line(
            line_evidence_supplement_overlay,
            (int(line[0]), int(line[1])),
            (int(line[2]), int(line[3])),
            (0, 220, 255),
            2,
        )
    for line in line_evidence_supplement_segments:
        cv2.line(
            line_evidence_supplement_overlay,
            (int(line[0]), int(line[1])),
            (int(line[2]), int(line[3])),
            (0, 255, 0),
            2,
        )
    if line_evidence_supplement_used_as_final:
        cv2.putText(
            line_evidence_supplement_overlay,
            "USED AS FINAL",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 180, 0),
            2,
            cv2.LINE_AA,
        )
    line_evidence_gap_rescue_overlay = overlay_mask_on_image(
        debug_img,
        line_evidence_gap_rescue_mask,
        color=(120, 255, 180),
        alpha=0.16,
    )
    for line in legacy_final_segments:
        cv2.line(
            line_evidence_gap_rescue_overlay,
            (int(line[0]), int(line[1])),
            (int(line[2]), int(line[3])),
            (255, 120, 0),
            2,
        )
    for item in line_evidence_gap_rescue_stats.get("line_evidence_gap_rescue_replacements", []):
        for line in item.get("rescued_segments", []):
            cv2.line(
                line_evidence_gap_rescue_overlay,
                (int(line[0]), int(line[1])),
                (int(line[2]), int(line[3])),
                (0, 255, 255),
                2,
            )
    for line in line_evidence_gap_rescue_segments:
        cv2.line(
            line_evidence_gap_rescue_overlay,
            (int(line[0]), int(line[1])),
            (int(line[2]), int(line[3])),
            (0, 255, 0),
            2,
        )
    if line_evidence_gap_rescue_used_as_final:
        cv2.putText(
            line_evidence_gap_rescue_overlay,
            "USED AS FINAL",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 180, 0),
            2,
            cv2.LINE_AA,
        )
    line_evidence_axis_replace_overlay = overlay_mask_on_image(
        debug_img,
        line_evidence_axis_replace_mask,
        color=(255, 180, 120),
        alpha=0.16,
    )
    for line in legacy_final_segments:
        cv2.line(
            line_evidence_axis_replace_overlay,
            (int(line[0]), int(line[1])),
            (int(line[2]), int(line[3])),
            (255, 120, 0),
            2,
        )
    for item in line_evidence_axis_replace_stats.get("line_evidence_axis_replace_replacements", []):
        repl = item.get("replacement_line")
        if repl:
            cv2.line(
                line_evidence_axis_replace_overlay,
                (int(repl[0]), int(repl[1])),
                (int(repl[2]), int(repl[3])),
                (0, 255, 255),
                2,
            )
    for line in line_evidence_axis_replace_segments:
        cv2.line(
            line_evidence_axis_replace_overlay,
            (int(line[0]), int(line[1])),
            (int(line[2]), int(line[3])),
            (0, 255, 0),
            2,
        )
    if line_evidence_axis_replace_used_as_final:
        cv2.putText(
            line_evidence_axis_replace_overlay,
            "USED AS FINAL",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 180, 0),
            2,
            cv2.LINE_AA,
        )
    line_evidence_partition_overlay = overlay_mask_on_image(
        debug_img,
        line_evidence_partition_mask,
        color=(180, 180, 255),
        alpha=0.16,
    )
    for line in legacy_final_segments:
        cv2.line(
            line_evidence_partition_overlay,
            (int(line[0]), int(line[1])),
            (int(line[2]), int(line[3])),
            (255, 120, 0),
            2,
        )
    for item in line_evidence_partition_stats.get("line_evidence_partition_replacements", []):
        for line in item.get("partition_segments", []):
            cv2.line(
                line_evidence_partition_overlay,
                (int(line[0]), int(line[1])),
                (int(line[2]), int(line[3])),
                (0, 255, 255),
                2,
            )
    for line in line_evidence_partition_segments:
        cv2.line(
            line_evidence_partition_overlay,
            (int(line[0]), int(line[1])),
            (int(line[2]), int(line[3])),
            (0, 255, 0),
            2,
        )
    if line_evidence_partition_used_as_final:
        cv2.putText(
            line_evidence_partition_overlay,
            "USED AS FINAL",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 180, 0),
            2,
            cv2.LINE_AA,
        )
    wall_region_graph_overlay = overlay_mask_on_image(
        debug_img,
        wall_region_graph_stats.get("wall_region_graph_region_mask", np.zeros_like(inner_candidates)),
        color=(255, 220, 120),
        alpha=0.18,
    )
    for item in wall_region_graph_stats.get("wall_region_graph_component_debug", []):
        bbox = item.get("bbox")
        if bbox and len(bbox) == 4:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(wall_region_graph_overlay, (x, y), (x + w, y + h), (180, 180, 180), 1)
        for sub_bbox in item.get("subregion_bboxes", []):
            if sub_bbox and len(sub_bbox) == 4:
                sx, sy, sw, sh = [int(v) for v in sub_bbox]
                cv2.rectangle(wall_region_graph_overlay, (sx, sy), (sx + sw, sy + sh), (0, 190, 190), 1)
    for axis_item in wall_region_graph_stats.get("wall_region_original_stripe_axes", []):
        orientation = axis_item.get("orientation")
        axis = int(axis_item.get("axis", 0))
        bbox = axis_item.get("component_bbox", [0, 0, 0, 0])
        bx, by, bw, bh = [int(v) for v in bbox]
        if orientation == "v":
            cv2.line(wall_region_graph_overlay, (axis, by), (axis, by + bh), (160, 160, 160), 1)
        else:
            cv2.line(wall_region_graph_overlay, (bx, axis), (bx + bw, axis), (160, 160, 160), 1)
    for axis_item in wall_region_graph_stats.get("wall_region_refined_stripe_axes", []):
        orientation = axis_item.get("orientation")
        axis = int(axis_item.get("axis", 0))
        bbox = axis_item.get("component_bbox", [0, 0, 0, 0])
        bx, by, bw, bh = [int(v) for v in bbox]
        if orientation == "v":
            cv2.line(wall_region_graph_overlay, (axis, by), (axis, by + bh), (120, 120, 255), 1)
        else:
            cv2.line(wall_region_graph_overlay, (bx, axis), (bx + bw, axis), (120, 120, 255), 1)
    for line in wall_region_graph_segments:
        cv2.line(
            wall_region_graph_overlay,
            (int(line[0]), int(line[1])),
            (int(line[2]), int(line[3])),
            (0, 255, 0),
            2,
        )
    if wall_region_graph_used_as_final:
        cv2.putText(
            wall_region_graph_overlay,
            "USED AS FINAL",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 180, 0),
            2,
            cv2.LINE_AA,
        )
    edge_pair_centerlines_overlay = overlay_lines_on_image(
        debug_img,
        horizontal_segments + vertical_segments,
        color=(0, 200, 255),
        thickness=2,
    )
    for pair in raw_candidate_analysis.get("parallel_edge_pairs", []):
        seg_a = pair.get("segment_a")
        seg_b = pair.get("segment_b")
        center_seg = pair.get("centerline_segment")
        if seg_a:
            cv2.line(edge_pair_centerlines_overlay, (int(seg_a[0]), int(seg_a[1])), (int(seg_a[2]), int(seg_a[3])), (255, 0, 255), 2)
        if seg_b:
            cv2.line(edge_pair_centerlines_overlay, (int(seg_b[0]), int(seg_b[1])), (int(seg_b[2]), int(seg_b[3])), (255, 0, 255), 2)
        if center_seg:
            cv2.line(edge_pair_centerlines_overlay, (int(center_seg[0]), int(center_seg[1])), (int(center_seg[2]), int(center_seg[3])), (0, 255, 0), 2)
    outer_boundary_rejected_overlay = overlay_lines_on_image(
        debug_img,
        debug_stats.get("outer_boundary_rejected_segments", []),
        color=(255, 0, 255),
        thickness=2,
    )
    axis_snapping_overlay = debug_img.copy()
    for line in axis_snap_stats.get("axis_target_lines", []):
        x1, y1, x2, y2 = line
        cv2.line(axis_snapping_overlay, (int(x1), int(y1)), (int(x2), int(y2)), (180, 180, 180), 1)
    for line in axis_snap_before_segments:
        x1, y1, x2, y2 = line
        cv2.line(axis_snapping_overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 255), 2)
    for item in debug_stats.get("axis_snapped_segments_before_after", []):
        before = item.get("before")
        after = item.get("after")
        if before:
            cv2.line(axis_snapping_overlay, (int(before[0]), int(before[1])), (int(before[2]), int(before[3])), (0, 200, 255), 2)
        if after:
            cv2.line(axis_snapping_overlay, (int(after[0]), int(after[1])), (int(after[2]), int(after[3])), (255, 0, 0), 2)
    filtered_debug = overlay_lines_on_image(debug_img, filtered_wall_segments, color=(0, 200, 255), thickness=2)
    debug = overlay_lines_on_image(debug_img, wall_segments, color=(0, 0, 255), thickness=2)

    save_debug(project_debug_dir, floor_name, "raw_inner_candidates_overlay.png", raw_candidates_overlay)
    save_debug(project_debug_dir, floor_name, "skeleton_centerline_candidates_overlay.png", skeleton_centerline_overlay)
    save_debug(project_debug_dir, floor_name, "projection_grid_candidates_overlay.png", projection_grid_overlay)
    save_debug(project_debug_dir, floor_name, "raw_candidate_based_final_overlay.png", raw_candidate_based_overlay)
    save_debug(project_debug_dir, floor_name, "raw_axis_recon_final_overlay.png", raw_axis_recon_overlay)
    save_debug(project_debug_dir, floor_name, "raw_axis_hybrid_final_overlay.png", raw_axis_hybrid_overlay)
    save_debug(project_debug_dir, floor_name, "orthogonal_clean_plan_final_overlay.png", orthogonal_clean_overlay)
    save_debug(project_debug_dir, floor_name, "line_evidence_final_overlay.png", line_evidence_overlay)
    save_debug(project_debug_dir, floor_name, "semantic_inner_mask_final_overlay.png", semantic_inner_overlay)
    save_debug(project_debug_dir, floor_name, "clean_plan_mode_final_overlay.png", clean_plan_mode_overlay)
    save_debug(project_debug_dir, floor_name, "line_evidence_supplement_final_overlay.png", line_evidence_supplement_overlay)
    save_debug(project_debug_dir, floor_name, "line_evidence_gap_rescue_final_overlay.png", line_evidence_gap_rescue_overlay)
    save_debug(project_debug_dir, floor_name, "line_evidence_axis_replace_final_overlay.png", line_evidence_axis_replace_overlay)
    save_debug(project_debug_dir, floor_name, "line_evidence_partition_final_overlay.png", line_evidence_partition_overlay)
    save_debug(project_debug_dir, floor_name, "wall_region_graph_final_overlay.png", wall_region_graph_overlay)
    save_debug(project_debug_dir, floor_name, "edge_pair_centerlines_overlay.png", edge_pair_centerlines_overlay)
    save_debug(project_debug_dir, floor_name, "filtered_inner_walls_overlay.png", filtered_debug)
    save_debug(project_debug_dir, floor_name, "outer_boundary_rejected_overlay.png", outer_boundary_rejected_overlay)
    save_debug(project_debug_dir, floor_name, "axis_snapping_overlay.png", axis_snapping_overlay)
    save_debug(project_debug_dir, floor_name, "inner_walls_debug.png", debug)
    floor_debug_dir = project_debug_dir / floor_name
    floor_debug_dir.mkdir(parents=True, exist_ok=True)
    (floor_debug_dir / "inner_walls_stats.json").write_text(json.dumps(debug_stats, indent=2), encoding="utf-8")
    print(
        "[extract_inner_walls] "
        f"floor={floor_name} raw_candidate_wall_count={debug_stats['raw_candidate_wall_count']} "
        f"filtered_wall_count={debug_stats['filtered_wall_count']} "
        f"rejected_segment_count={debug_stats['rejected_segment_count']} "
        f"final_inner_wall_count={debug_stats['final_inner_wall_count']}"
    )
    print(f"[extract_inner_walls] floor={floor_name} reject_reasons={debug_stats['reject_reasons']}")
    return wall_segments, combined_mask, {
        "stats": debug_stats,
        "raw_inner_candidates_overlay": raw_candidates_overlay,
        "skeleton_centerline_candidates_overlay": skeleton_centerline_overlay,
        "projection_grid_candidates_overlay": projection_grid_overlay,
        "raw_candidate_based_final_overlay": raw_candidate_based_overlay,
        "raw_axis_recon_final_overlay": raw_axis_recon_overlay,
        "raw_axis_hybrid_final_overlay": raw_axis_hybrid_overlay,
        "orthogonal_clean_plan_final_overlay": orthogonal_clean_overlay,
        "line_evidence_final_overlay": line_evidence_overlay,
        "semantic_inner_mask_final_overlay": semantic_inner_overlay,
        "clean_plan_mode_final_overlay": clean_plan_mode_overlay,
        "line_evidence_supplement_final_overlay": line_evidence_supplement_overlay,
        "line_evidence_gap_rescue_final_overlay": line_evidence_gap_rescue_overlay,
        "line_evidence_axis_replace_final_overlay": line_evidence_axis_replace_overlay,
        "line_evidence_partition_final_overlay": line_evidence_partition_overlay,
        "wall_region_graph_final_overlay": wall_region_graph_overlay,
        "edge_pair_centerlines_overlay": edge_pair_centerlines_overlay,
        "filtered_inner_walls_overlay": filtered_debug,
        "final_inner_walls_overlay": debug,
        "outer_boundary_rejected_overlay": outer_boundary_rejected_overlay,
        "axis_snapping_overlay": axis_snapping_overlay,
    }


def sample_along_segment(line: List[int], step: int) -> List[Tuple[int, int]]:
    x1, y1, x2, y2 = line
    samples: List[Tuple[int, int]] = []

    if y1 == y2:
        for x in range(min(x1, x2), max(x1, x2) + 1, step):
            samples.append((x, y1))
    elif x1 == x2:
        for y in range(min(y1, y2), max(y1, y2) + 1, step):
            samples.append((x1, y))

    return samples


def wall_support_at_point(
    point: Tuple[int, int],
    line: List[int],
    wall_mask: np.ndarray,
    band_radius: int,
) -> int:
    px, py = point
    h, w = wall_mask.shape[:2]

    if line[1] == line[3]:
        y1 = max(0, py - band_radius)
        y2 = min(h, py + band_radius + 1)
        x1 = max(0, px - 1)
        x2 = min(w, px + 2)
    else:
        y1 = max(0, py - 1)
        y2 = min(h, py + 2)
        x1 = max(0, px - band_radius)
        x2 = min(w, px + band_radius + 1)

    patch = wall_mask[y1:y2, x1:x2]
    if patch.size == 0:
        return 0

    return int(np.sum(patch > 0))


def scan_openings_on_line(
    line: List[int],
    wall_mask: np.ndarray,
    scale: Dict[str, int],
) -> List[Dict[str, int]]:
    step = scale["opening_scan_step"]
    min_gap = scale["opening_min_gap"]
    max_gap = scale["opening_max_gap"]
    band_radius = max(3, scale["max_wall_thickness"] // 2)

    samples = sample_along_segment(line, step=step)
    if len(samples) < 3:
        return []

    values = [wall_support_at_point(pt, line, wall_mask, band_radius) for pt in samples]
    positive_values = [v for v in values if v > 0]

    if not positive_values:
        return []

    support_threshold = max(2, int(np.percentile(positive_values, 35) * 0.45))

    openings: List[Dict[str, int]] = []
    gap_start_idx: Optional[int] = None

    for i, val in enumerate(values):
        is_gap = val <= support_threshold

        if is_gap and gap_start_idx is None:
            gap_start_idx = i
        elif not is_gap and gap_start_idx is not None:
            gap_len = (i - gap_start_idx) * step
            if min_gap <= gap_len <= max_gap:
                mid_idx = (gap_start_idx + i) // 2
                mx, my = samples[mid_idx]
                openings.append({"x": int(mx), "y": int(my), "width": int(gap_len)})
            gap_start_idx = None

    if gap_start_idx is not None:
        gap_len = (len(values) - gap_start_idx) * step
        if min_gap <= gap_len <= max_gap:
            mid_idx = (gap_start_idx + len(values) - 1) // 2
            mx, my = samples[mid_idx]
            openings.append({"x": int(mx), "y": int(my), "width": int(gap_len)})

    return openings


def scan_inner_wall_gaps_with_symbol_support(
    line: List[int],
    wall_mask: np.ndarray,
    symbol_mask: np.ndarray,
    polygon_contour: np.ndarray,
    scale: Dict[str, int],
) -> Tuple[List[Dict[str, int]], Dict[str, Any]]:
    step = max(4, scale["opening_scan_step"])
    min_gap = max(12, int(round(scale["opening_min_gap"] * 0.45)))
    max_gap = int(round(scale["opening_max_gap"] * 1.25))
    band_radius = max(3, scale["max_wall_thickness"] // 2)
    near_outer_tol = max(24, scale["outer_edge_offset"] * 3)
    symbol_band = max(10, scale["max_wall_thickness"] * 3)
    min_symbol_pixels = max(10, scale["min_wall_thickness"] * 2)

    stats: Dict[str, Any] = {
        "gap_candidates": [],
        "accepted_candidates": [],
        "reject_reasons": {},
        "evaluations": [],
    }

    def reject(reason: str) -> None:
        stats["reject_reasons"][reason] = stats["reject_reasons"].get(reason, 0) + 1

    samples = sample_along_segment(line, step=step)
    if len(samples) < 4:
        reject("line_too_short_for_gap_scan")
        return [], stats

    values = [wall_support_at_point(pt, line, wall_mask, band_radius) for pt in samples]
    positive_values = [v for v in values if v > 0]
    if not positive_values:
        reject("no_positive_wall_support")
        return [], stats

    support_threshold = max(2, int(np.percentile(positive_values, 55) * 0.8))
    candidates: List[Dict[str, int]] = []
    gap_start_idx: Optional[int] = None
    h, w = symbol_mask.shape[:2]

    for i, val in enumerate(values):
        is_gap = val <= support_threshold
        if is_gap and gap_start_idx is None:
            gap_start_idx = i
        elif not is_gap and gap_start_idx is not None:
            gap_len = (i - gap_start_idx) * step
            if min_gap <= gap_len <= max_gap:
                mid_idx = (gap_start_idx + i) // 2
                mx, my = samples[mid_idx]
                point = {"x": int(mx), "y": int(my), "width": int(gap_len)}
                stats["gap_candidates"].append(point)
                search_rect = {
                    "x1": int(max(0, mx - symbol_band)),
                    "y1": int(max(0, my - symbol_band)),
                    "x2": int(min(w, mx + symbol_band + 1)),
                    "y2": int(min(h, my + symbol_band + 1)),
                }
                evaluation: Dict[str, Any] = {
                    "center": point,
                    "host_line": line,
                    "search_rect": search_rect,
                    "accepted": False,
                    "reject_reason": None,
                    "symbol_pixels": 0,
                }

                if point_near_polygon((mx, my), polygon_contour, dist_tol=near_outer_tol):
                    evaluation["reject_reason"] = "gap_near_outer_polygon"
                    stats["evaluations"].append(evaluation)
                    reject("gap_near_outer_polygon")
                elif point_near_any_line_endpoint((mx, my), [line], tol=14):
                    evaluation["reject_reason"] = "gap_near_line_endpoint"
                    stats["evaluations"].append(evaluation)
                    reject("gap_near_line_endpoint")
                else:
                    if line[1] == line[3]:
                        left_x1 = max(0, mx - symbol_band)
                        left_x2 = max(0, mx - 2)
                        right_x1 = min(w, mx + 2)
                        right_x2 = min(w, mx + symbol_band)
                        y1 = max(0, my - band_radius - 4)
                        y2 = min(h, my + band_radius + 5)
                        side_a = symbol_mask[y1:y2, left_x1:left_x2]
                        side_b = symbol_mask[y1:y2, right_x1:right_x2]
                    else:
                        top_y1 = max(0, my - symbol_band)
                        top_y2 = max(0, my - 2)
                        bottom_y1 = min(h, my + 2)
                        bottom_y2 = min(h, my + symbol_band)
                        x1 = max(0, mx - band_radius - 4)
                        x2 = min(w, mx + band_radius + 5)
                        side_a = symbol_mask[top_y1:top_y2, x1:x2]
                        side_b = symbol_mask[bottom_y1:bottom_y2, x1:x2]

                    symbol_pixels = max(
                        int(np.sum(side_a > 0)) if side_a.size else 0,
                        int(np.sum(side_b > 0)) if side_b.size else 0,
                    )
                    evaluation["symbol_pixels"] = symbol_pixels
                    if symbol_pixels < min_symbol_pixels:
                        evaluation["reject_reason"] = "gap_missing_nearby_symbol"
                        stats["evaluations"].append(evaluation)
                        reject("gap_missing_nearby_symbol")
                    else:
                        candidates.append(point)
                        stats["accepted_candidates"].append({**point, "symbol_pixels": symbol_pixels})
                        evaluation["accepted"] = True
                        stats["evaluations"].append(evaluation)
            gap_start_idx = None

    if gap_start_idx is not None:
        gap_len = (len(values) - gap_start_idx) * step
        if min_gap <= gap_len <= max_gap:
            mid_idx = (gap_start_idx + len(values) - 1) // 2
            mx, my = samples[mid_idx]
            point = {"x": int(mx), "y": int(my), "width": int(gap_len)}
            stats["gap_candidates"].append(point)
            search_rect = {
                "x1": int(max(0, mx - symbol_band)),
                "y1": int(max(0, my - symbol_band)),
                "x2": int(min(w, mx + symbol_band + 1)),
                "y2": int(min(h, my + symbol_band + 1)),
            }
            evaluation = {
                "center": point,
                "host_line": line,
                "search_rect": search_rect,
                "accepted": False,
                "reject_reason": None,
                "symbol_pixels": 0,
            }
            if point_near_polygon((mx, my), polygon_contour, dist_tol=near_outer_tol):
                evaluation["reject_reason"] = "gap_near_outer_polygon"
                stats["evaluations"].append(evaluation)
                reject("gap_near_outer_polygon")
            elif point_near_any_line_endpoint((mx, my), [line], tol=14):
                evaluation["reject_reason"] = "gap_near_line_endpoint"
                stats["evaluations"].append(evaluation)
                reject("gap_near_line_endpoint")
            else:
                if line[1] == line[3]:
                    left_x1 = max(0, mx - symbol_band)
                    left_x2 = max(0, mx - 2)
                    right_x1 = min(w, mx + 2)
                    right_x2 = min(w, mx + symbol_band)
                    y1 = max(0, my - band_radius - 4)
                    y2 = min(h, my + band_radius + 5)
                    side_a = symbol_mask[y1:y2, left_x1:left_x2]
                    side_b = symbol_mask[y1:y2, right_x1:right_x2]
                else:
                    top_y1 = max(0, my - symbol_band)
                    top_y2 = max(0, my - 2)
                    bottom_y1 = min(h, my + 2)
                    bottom_y2 = min(h, my + symbol_band)
                    x1 = max(0, mx - band_radius - 4)
                    x2 = min(w, mx + band_radius + 5)
                    side_a = symbol_mask[top_y1:top_y2, x1:x2]
                    side_b = symbol_mask[bottom_y1:bottom_y2, x1:x2]
                symbol_pixels = max(
                    int(np.sum(side_a > 0)) if side_a.size else 0,
                    int(np.sum(side_b > 0)) if side_b.size else 0,
                )
                evaluation["symbol_pixels"] = symbol_pixels
                if symbol_pixels < min_symbol_pixels:
                    evaluation["reject_reason"] = "gap_missing_nearby_symbol"
                    stats["evaluations"].append(evaluation)
                    reject("gap_missing_nearby_symbol")
                else:
                    candidates.append(point)
                    stats["accepted_candidates"].append({**point, "symbol_pixels": symbol_pixels})
                    evaluation["accepted"] = True
                    stats["evaluations"].append(evaluation)

    return candidates, stats


def evaluate_opening_compatibility_for_wall_set(
    inner_walls: List[List[int]],
    inner_wall_mask: np.ndarray,
    structural_mask: np.ndarray,
    polygon_contour: np.ndarray,
    scale: Dict[str, int],
) -> Dict[str, Any]:
    if not inner_walls:
        return {
            "opening_scan_count": 0,
            "gap_scan_count": 0,
            "hostable_line_count": 0,
            "opening_compatibility_score": float("-inf"),
        }

    axis_kernel = max(10, scale["opening_min_gap"] // 2)
    horizontal = cv2.morphologyEx(
        structural_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (axis_kernel, 1)),
    )
    vertical = cv2.morphologyEx(
        structural_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, axis_kernel)),
    )
    axis_aligned = cv2.bitwise_or(horizontal, vertical)
    axis_aligned = cv2.dilate(axis_aligned, np.ones((3, 3), np.uint8), iterations=1)
    symbol_mask = cv2.bitwise_and(structural_mask, cv2.bitwise_not(axis_aligned))
    symbol_mask = cv2.morphologyEx(symbol_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    symbol_mask = cv2.morphologyEx(symbol_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    opening_scan_count = 0
    gap_scan_count = 0
    hostable_line_count = 0

    for line in inner_walls:
        opening_points = scan_openings_on_line(line, inner_wall_mask, scale)
        gap_points, _ = scan_inner_wall_gaps_with_symbol_support(
            line,
            wall_mask=inner_wall_mask,
            symbol_mask=symbol_mask,
            polygon_contour=polygon_contour,
            scale=scale,
        )
        opening_scan_count += len(opening_points)
        gap_scan_count += len(gap_points)
        if opening_points or gap_points:
            hostable_line_count += 1

    score = (
        hostable_line_count * 2.5
        + opening_scan_count * 1.2
        + gap_scan_count * 1.8
        - max(0, len(inner_walls) - hostable_line_count) * 0.35
    )
    return {
        "opening_scan_count": int(opening_scan_count),
        "gap_scan_count": int(gap_scan_count),
        "hostable_line_count": int(hostable_line_count),
        "opening_compatibility_score": float(score),
    }


def point_near_any_line_endpoint(
    point: Tuple[int, int],
    lines: List[List[int]],
    tol: int = 18,
) -> bool:
    px, py = point

    for line in lines:
        x1, y1, x2, y2 = line
        if hypot(px - x1, py - y1) <= tol or hypot(px - x2, py - y2) <= tol:
            return True

    return False


def classify_outer_opening(
    point: Dict[str, int],
    segment: List[int],
    outer_polygon: List[List[int]],
    binary_mask: np.ndarray,
    floor_index: int,
    total_floors: int,
    scale: Dict[str, int],
    prefer_window: bool = False,
) -> str:
    h, w = binary_mask.shape[:2]
    px, py = int(point["x"]), int(point["y"])
    width = max(scale["opening_min_gap"], int(point.get("width", 0)))
    band = max(10, scale["max_wall_thickness"] * 2)
    reach = max(18, int(width * 0.8))

    xs = [pt[0] for pt in outer_polygon]
    ys = [pt[1] for pt in outer_polygon]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    if abs(segment[1] - segment[3]) <= abs(segment[0] - segment[2]):
        near_top = py <= min_y + band
        near_bottom = py >= max_y - band

        if near_top:
            y1 = max(0, py)
            y2 = min(h, py + reach)
        elif near_bottom:
            y1 = max(0, py - reach)
            y2 = min(h, py)
        else:
            y1 = max(0, py - reach // 2)
            y2 = min(h, py + reach // 2)

        x1 = max(0, px - width)
        x2 = min(w, px + width)
    else:
        near_left = px <= min_x + band
        near_right = px >= max_x - band

        if near_left:
            x1 = max(0, px)
            x2 = min(w, px + reach)
        elif near_right:
            x1 = max(0, px - reach)
            x2 = min(w, px)
        else:
            x1 = max(0, px - reach // 2)
            x2 = min(w, px + reach // 2)

        y1 = max(0, py - width)
        y2 = min(h, py + width)

    patch = binary_mask[y1:y2, x1:x2]
    if patch.size == 0:
        return "window"

    # Window symbols are mostly axis-aligned short strokes; door symbols introduce diagonals/curves.
    horizontal = cv2.morphologyEx(
        patch,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, width // 2), 1)),
    )
    vertical = cv2.morphologyEx(
        patch,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(9, width // 2))),
    )
    axis_aligned = cv2.bitwise_or(horizontal, vertical)
    non_axis = cv2.bitwise_and(patch, cv2.bitwise_not(axis_aligned))

    axis_ratio = float(np.mean(axis_aligned > 0))
    non_axis_ratio = float(np.mean(non_axis > 0))

    if non_axis_ratio > 0.035:
        return "door"

    if axis_ratio > 0.06:
        return "window"

    if prefer_window and non_axis_ratio <= 0.02 and axis_ratio >= 0.015:
        return "window"

    return "door" if floor_index == 1 and width >= max(56, scale["opening_min_gap"] * 2) else "window"


def scan_outer_openings_on_line(
    line: List[int],
    binary_mask: np.ndarray,
    scale: Dict[str, int],
    threshold_scale: float = 0.72,
    min_threshold: float = 0.18,
) -> List[Dict[str, int]]:
    step = max(4, scale["opening_scan_step"])
    min_width = scale["opening_min_gap"]
    max_width = scale["opening_max_gap"]
    band = max(4, scale["max_wall_thickness"] // 2 + 2)
    samples = sample_along_segment(line, step=step)
    if len(samples) < 4:
        return []

    values: List[float] = []
    h, w = binary_mask.shape[:2]
    horizontal = line[1] == line[3]

    for px, py in samples:
        if horizontal:
            y1 = max(0, py - band)
            y2 = min(h, py + band + 1)
            x1 = max(0, px - step)
            x2 = min(w, px + step + 1)
        else:
            y1 = max(0, py - step)
            y2 = min(h, py + step + 1)
            x1 = max(0, px - band)
            x2 = min(w, px + band + 1)

        patch = binary_mask[y1:y2, x1:x2]
        fill_ratio = float(np.mean(patch > 0)) if patch.size else 0.0
        values.append(fill_ratio)

    if not values:
        return []

    baseline = float(np.percentile(values, 70))
    threshold = max(min_threshold, baseline * threshold_scale)

    candidates: List[Dict[str, int]] = []
    run_start: Optional[int] = None

    for i, value in enumerate(values):
        is_opening_like = value <= threshold

        if is_opening_like and run_start is None:
            run_start = i
        elif not is_opening_like and run_start is not None:
            run_width = (i - run_start) * step
            if min_width <= run_width <= max_width:
                mid_idx = (run_start + i) // 2
                mx, my = samples[mid_idx]
                candidates.append({"x": int(mx), "y": int(my), "width": int(run_width)})
            run_start = None

    if run_start is not None:
        run_width = (len(values) - run_start) * step
        if min_width <= run_width <= max_width:
            mid_idx = (run_start + len(values) - 1) // 2
            mx, my = samples[mid_idx]
            candidates.append({"x": int(mx), "y": int(my), "width": int(run_width)})

    return candidates


def collect_fallback_outer_windows(
    outer_segments: List[List[int]],
    binary_mask: np.ndarray,
    outer_polygon: List[List[int]],
    outer_polygon_contour: np.ndarray,
    floor_index: int,
    total_floors: int,
    scale: Dict[str, int],
) -> List[Dict[str, int]]:
    fallback_windows: List[Dict[str, int]] = []

    for seg in outer_segments:
        if line_length(seg) < max(90, scale["opening_min_gap"] * 3):
            continue

        permissive_candidates = scan_outer_openings_on_line(
            seg,
            binary_mask,
            scale,
            threshold_scale=0.9,
            min_threshold=0.12,
        )

        for point in permissive_candidates:
            px, py = point["x"], point["y"]
            if not point_near_polygon((px, py), outer_polygon_contour, dist_tol=28):
                continue

            opening_type = classify_outer_opening(
                point,
                seg,
                outer_polygon,
                binary_mask,
                floor_index=floor_index,
                total_floors=total_floors,
                scale=scale,
            )
            if opening_type == "window":
                fallback_windows.append(point)

    return merge_opening_points(fallback_windows, tol=24)


def collect_clean_plan_outer_windows(
    outer_segments: List[List[int]],
    binary_mask: np.ndarray,
    outer_polygon: List[List[int]],
    outer_polygon_contour: np.ndarray,
    floor_index: int,
    total_floors: int,
    scale: Dict[str, int],
    existing_doors: List[Dict[str, int]],
) -> List[Dict[str, int]]:
    fallback_windows: List[Dict[str, int]] = []
    endpoint_tol = max(18, scale["max_wall_thickness"] * 2)

    for seg in outer_segments:
        if line_length(seg) < max(72, scale["opening_min_gap"] * 2):
            continue

        permissive_candidates = scan_outer_openings_on_line(
            seg,
            binary_mask,
            scale,
            threshold_scale=1.0,
            min_threshold=0.20,
        )
        aggressive_candidates = scan_outer_openings_on_line(
            seg,
            binary_mask,
            scale,
            threshold_scale=1.12,
            min_threshold=0.24,
        )
        candidates = merge_opening_points(permissive_candidates + aggressive_candidates, tol=18)

        for point in candidates:
            px, py = point["x"], point["y"]
            if not point_near_polygon((px, py), outer_polygon_contour, dist_tol=34):
                continue
            if point_near_any_line_endpoint((px, py), [seg], tol=endpoint_tol):
                continue
            if any(hypot(float(px - door["x"]), float(py - door["y"])) <= max(30, scale["opening_min_gap"]) for door in existing_doors):
                continue

            opening_type = classify_outer_opening(
                point,
                seg,
                outer_polygon,
                binary_mask,
                floor_index=floor_index,
                total_floors=total_floors,
                scale=scale,
                prefer_window=True,
            )
            if opening_type == "window":
                fallback_windows.append(point)

    return merge_opening_points(fallback_windows, tol=20)


def project_openings_to_host_lines(
    points: List[Dict[str, int]],
    lines: List[List[int]],
    max_distance: float,
    endpoint_tol: int = 22,
    min_t: float = 0.08,
    max_t: float = 0.92,
) -> List[Dict[str, int]]:
    hosted: List[Dict[str, int]] = []

    for point in points:
        px, py = point["x"], point["y"]

        host = find_best_host_line((px, py), lines, max_distance=max_distance, min_t=min_t, max_t=max_t)
        if host is None:
            continue

        line = host["line"]
        proj = host["projection"]

        if point_near_any_line_endpoint((int(proj["x"]), int(proj["y"])), [line], tol=endpoint_tol):
            continue

        hosted.append({
            "x": int(round(proj["x"])),
            "y": int(round(proj["y"])),
            "width": int(point.get("width", 0)),
        })

    return hosted


def boosted_opening_width(width: int, scale: Dict[str, int], opening_type: str) -> int:
    base_width = max(int(width), scale["opening_min_gap"])
    if opening_type == "door":
        return max(base_width * 4, int(round(scale["opening_max_gap"] * 2.2)))
    if opening_type == "window":
        return max(base_width * 4, int(round(scale["opening_max_gap"] * 2.2)))
    return base_width


def normalize_clean_plan_opening_widths(
    openings: List[Dict[str, int]],
    scale: Dict[str, int],
    opening_type: str,
) -> List[Dict[str, int]]:
    normalized: List[Dict[str, int]] = []
    min_width = max(scale["opening_min_gap"], scale["max_wall_thickness"] * 3)
    if opening_type == "door":
        max_width = max(int(round(scale["opening_max_gap"] * 1.10)), min_width + 22)
    else:
        max_width = max(int(round(scale["opening_max_gap"] * 1.65)), min_width + 28)

    for point in openings:
        width = int(point.get("width", 0))
        if width <= 0:
            normalized.append(dict(point))
            continue
        normalized.append({
            **point,
            "width": int(max(min_width, min(width, max_width))),
        })
    return normalized


def recover_supported_clean_plan_symbolic_doors(
    openings_stats_path: Path,
    inner_walls: List[List[int]],
    windows: List[Dict[str, int]],
    scale: Dict[str, int],
) -> Tuple[List[Dict[str, int]], Dict[str, Any]]:
    meta: Dict[str, Any] = {
        "enabled": True,
        "raw_symbolic_count": 0,
        "projected_count": 0,
        "final_count": 0,
        "applied": False,
    }
    if not openings_stats_path.exists() or not inner_walls:
        meta["reason"] = "openings_stats_missing"
        return [], meta

    try:
        opening_stats = json.loads(openings_stats_path.read_text(encoding="utf-8"))
    except Exception:
        meta["reason"] = "openings_stats_unreadable"
        return [], meta

    raw_symbolic = opening_stats.get("raw_symbolic_inner_door_candidates", []) or []
    meta["raw_symbolic_count"] = len(raw_symbolic)
    rejected_symbolic_samples = opening_stats.get("symbolic_rejected_component_samples", []) or []
    paired_symbolic: List[Dict[str, int]] = []
    symbol_no_host_samples = [
        sample for sample in rejected_symbolic_samples
        if str(sample.get("reason", "")) == "symbol_no_inner_host"
    ]
    if len(symbol_no_host_samples) >= 2:
        remaining = [
            {
                "x": int(sample.get("x", 0)),
                "y": int(sample.get("y", 0)),
                "bbox": [int(v) for v in sample.get("bbox", [0, 0, 0, 0])],
                "area": int(sample.get("area", 0)),
            }
            for sample in symbol_no_host_samples
        ]
        used = [False] * len(remaining)
        for i, a in enumerate(remaining):
            if used[i]:
                continue
            best_j = -1
            best_d = 10**9
            for j in range(i + 1, len(remaining)):
                if used[j]:
                    continue
                b = remaining[j]
                dx = abs(a["x"] - b["x"])
                dy = abs(a["y"] - b["y"])
                if dx > 70 or dy > 70:
                    continue
                distance = dx + dy
                if distance < best_d:
                    best_d = distance
                    best_j = j
            if best_j < 0:
                continue
            used[i] = True
            used[best_j] = True
            b = remaining[best_j]
            avg_x = int(round((a["x"] + b["x"]) / 2.0))
            avg_y = int(round((a["y"] + b["y"]) / 2.0))
            max_dim = max(
                a["bbox"][2], a["bbox"][3],
                b["bbox"][2], b["bbox"][3],
            )
            paired_symbolic.append({
                "x": avg_x,
                "y": avg_y,
                "width": max(20, int(max_dim)),
            })

    meta["paired_symbol_no_host_count"] = len(paired_symbolic)
    if not raw_symbolic and not paired_symbolic:
        meta["reason"] = "no_symbolic_inner_doors"
        return [], meta

    all_symbolic = list(raw_symbolic) + paired_symbolic
    projected = project_openings_to_host_lines(
        all_symbolic,
        inner_walls,
        max_distance=max(36, scale["max_wall_thickness"] * 6.0),
        endpoint_tol=12,
        min_t=0.06,
        max_t=0.94,
    )
    meta["projected_count"] = len(projected)
    if not projected:
        meta["reason"] = "symbolic_doors_not_projectable"
        return [], meta

    projected = [
        {
            "x": int(item["x"]),
            "y": int(item["y"]),
            "width": boosted_opening_width(int(item.get("width", 0)), scale, "door"),
        }
        for item in projected
    ]
    projected = merge_opening_points(projected, tol=20)
    projected = normalize_clean_plan_opening_widths(projected, scale, "door")

    cleaned: List[Dict[str, int]] = []
    for door in projected:
        if any(hypot(float(door["x"] - win["x"]), float(door["y"] - win["y"])) <= 24 for win in windows):
            continue
        cleaned.append(door)

    meta["final_count"] = len(cleaned)
    meta["applied"] = bool(cleaned)
    meta["reason"] = "reused_symbolic_inner_doors" if cleaned else "all_symbolic_doors_too_close_to_windows"
    return cleaned, meta


def recover_supported_clean_plan_additional_inner_doors(
    openings_stats_path: Path,
    polygon: List[List[int]],
    inner_walls: List[List[int]],
    current_doors: List[Dict[str, int]],
    windows: List[Dict[str, int]],
    scale: Dict[str, int],
) -> Tuple[List[Dict[str, int]], Dict[str, Any]]:
    meta: Dict[str, Any] = {
        "enabled": True,
        "applied": False,
        "reason": "not_evaluated",
        "current_door_count": len(current_doors),
        "outer_like_current_door_count": 0,
        "projected_count": 0,
        "final_count": 0,
    }
    if not openings_stats_path.exists() or not inner_walls or not current_doors:
        meta["reason"] = "openings_stats_missing_or_no_current_doors"
        return [], meta

    try:
        opening_stats = json.loads(openings_stats_path.read_text(encoding="utf-8"))
    except Exception:
        meta["reason"] = "openings_stats_unreadable"
        return [], meta

    raw_door_candidates = opening_stats.get("raw_door_candidates", []) or []
    outer_like = [
        cand for cand in raw_door_candidates
        if str(cand.get("source", "")) == "outer_segment_scan"
    ]
    outer_like_matches = 0
    for door in current_doors:
        if any(
            hypot(float(int(door["x"]) - int(cand.get("x", 0))), float(int(door["y"]) - int(cand.get("y", 0)))) <= 36.0
            for cand in outer_like
        ):
            outer_like_matches += 1
    meta["outer_like_current_door_count"] = outer_like_matches
    if outer_like_matches != len(current_doors):
        meta["reason"] = "current_doors_not_outer_like"
        return [], meta

    rejected_candidates = opening_stats.get("rejected_candidates", []) or []
    rejected_room_support = [
        cand for cand in rejected_candidates
        if str(cand.get("reason", "")) == "door_failed_room_support"
    ]
    if not rejected_room_support:
        meta["reason"] = "no_room_support_rejected_doors"
        return [], meta

    raw_symbolic = opening_stats.get("raw_symbolic_inner_door_candidates", []) or []
    if not raw_symbolic:
        meta["reason"] = "no_symbolic_inner_doors"
        return [], meta

    projected = project_openings_to_host_lines(
        raw_symbolic,
        inner_walls,
        max_distance=max(30, scale["max_wall_thickness"] * 3.0),
        endpoint_tol=12,
        min_t=0.06,
        max_t=0.94,
    )
    meta["projected_count"] = len(projected)
    if not projected:
        meta["reason"] = "symbolic_doors_not_projectable"
        return [], meta

    polygon_np = np.array(polygon, dtype=np.int32) if len(polygon) >= 4 else None
    projected = merge_opening_points([
        {
            "x": int(item["x"]),
            "y": int(item["y"]),
            "width": boosted_opening_width(int(item.get("width", 0)), scale, "door"),
        }
        for item in projected
    ], tol=32)
    projected = normalize_clean_plan_opening_widths(projected, scale, "door")

    cleaned: List[Dict[str, int]] = []
    for door in projected:
        if any(hypot(float(door["x"] - win["x"]), float(door["y"] - win["y"])) <= 24.0 for win in windows):
            continue
        if any(hypot(float(door["x"] - cur["x"]), float(door["y"] - cur["y"])) <= 72.0 for cur in current_doors):
            continue
        if polygon_np is not None:
            boundary_dist = cv2.pointPolygonTest(polygon_np, (float(door["x"]), float(door["y"])), True)
            if boundary_dist < max(36.0, float(scale["max_wall_thickness"] * 2.0)):
                continue
        cleaned.append(door)

    if len(cleaned) > 2:
        cleaned = cleaned[:2]
    meta["final_count"] = len(cleaned)
    meta["applied"] = bool(cleaned)
    meta["reason"] = "added_symbolic_inner_doors" if cleaned else "all_projected_symbolic_doors_rejected"
    return cleaned, meta


def cleanup_supported_clean_plan_window_like_outer_doors(
    openings_stats_path: Path,
    doors: List[Dict[str, int]],
    windows: List[Dict[str, int]],
    polygon: List[List[int]],
    scale: Dict[str, int],
) -> Tuple[List[Dict[str, int]], Dict[str, Any]]:
    meta: Dict[str, Any] = {
        "enabled": True,
        "applied": False,
        "reason": "not_evaluated",
        "removed_count": 0,
        "removed_samples": [],
    }
    if not openings_stats_path.exists() or not doors or not windows or len(polygon) < 4:
        meta["reason"] = "missing_inputs_or_openings_stats"
        return doors, meta

    try:
        opening_stats = json.loads(openings_stats_path.read_text(encoding="utf-8"))
    except Exception:
        meta["reason"] = "openings_stats_unreadable"
        return doors, meta

    raw_door_candidates = opening_stats.get("raw_door_candidates", []) or []
    outer_like = [
        cand for cand in raw_door_candidates
        if str(cand.get("source", "")) == "outer_segment_scan"
    ]
    if not outer_like:
        meta["reason"] = "no_outer_like_raw_doors"
        return doors, meta

    xs = [int(p[0]) for p in polygon]
    ys = [int(p[1]) for p in polygon]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    boundary_tol = max(36.0, float(scale["max_wall_thickness"] * 2.0))

    def classify_facade(x: int, y: int) -> str:
        distances = {
            "left": abs(x - min_x),
            "right": abs(x - max_x),
            "top": abs(y - min_y),
            "bottom": abs(y - max_y),
        }
        return min(distances, key=distances.get)

    cleaned: List[Dict[str, int]] = []
    removed_samples: List[Dict[str, Any]] = []
    for door in doors:
        dx = int(door["x"])
        dy = int(door["y"])
        polygon_np = np.array(polygon, dtype=np.int32)
        boundary_dist = cv2.pointPolygonTest(polygon_np, (float(dx), float(dy)), True)
        if boundary_dist > boundary_tol:
            cleaned.append(door)
            continue

        matched_outer_like = any(
            hypot(float(dx - int(cand.get("x", 0))), float(dy - int(cand.get("y", 0)))) <= 40.0
            for cand in outer_like
        )
        if not matched_outer_like:
            cleaned.append(door)
            continue

        facade = classify_facade(dx, dy)
        should_remove = False
        for win in windows:
            wx = int(win["x"])
            wy = int(win["y"])
            if classify_facade(wx, wy) != facade:
                continue
            if facade in ("top", "bottom"):
                if abs(dx - wx) <= 20 and abs(dy - wy) <= 40:
                    should_remove = True
                    break
            else:
                if abs(dy - wy) <= 20 and abs(dx - wx) <= 40:
                    should_remove = True
                    break

        if should_remove:
            removed_samples.append({"x": dx, "y": dy, "facade": facade})
            continue
        cleaned.append(door)

    if len(cleaned) == len(doors):
        meta["reason"] = "no_window_like_outer_doors"
        return doors, meta

    meta["applied"] = True
    meta["removed_count"] = len(doors) - len(cleaned)
    meta["removed_samples"] = removed_samples[:5]
    meta["reason"] = "removed_window_like_outer_doors"
    return cleaned, meta


def recover_supported_clean_plan_stair_conflicted_gap_door(
    openings_stats_path: Path,
    polygon: List[List[int]],
    doors: List[Dict[str, int]],
    windows: List[Dict[str, int]],
    stairs: List[Dict[str, Any]],
    scale: Dict[str, int],
) -> Tuple[List[Dict[str, int]], Dict[str, Any]]:
    meta: Dict[str, Any] = {
        "enabled": True,
        "applied": False,
        "reason": "not_evaluated",
        "removed_count": 0,
        "added_count": 0,
    }
    if not openings_stats_path.exists() or not doors or not stairs or len(polygon) < 4:
        meta["reason"] = "missing_inputs_or_openings_stats"
        return doors, meta

    try:
        opening_stats = json.loads(openings_stats_path.read_text(encoding="utf-8"))
    except Exception:
        meta["reason"] = "openings_stats_unreadable"
        return doors, meta

    gap_candidates = opening_stats.get("inner_wall_gap_candidates", []) or []
    if not gap_candidates:
        meta["reason"] = "no_gap_candidates"
        return doors, meta

    xs = [int(p[0]) for p in polygon]
    ys = [int(p[1]) for p in polygon]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    boundary_tol = max(30, int(round(scale["max_wall_thickness"] * 1.5)))

    top_stair = None
    for stair in stairs:
        bounds = stair.get("bounds") or []
        if len(bounds) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bounds]
        if abs(y1 - min_y) <= 90:
            top_stair = (x1, y1, x2, y2)
            break
    if top_stair is None:
        meta["reason"] = "no_top_stair_bounds"
        return doors, meta

    stair_x1, _, stair_x2, _ = top_stair
    conflicted_top_doors = [
        door for door in doors
        if abs(int(door["y"]) - min_y) <= boundary_tol
        and stair_x1 <= int(door["x"]) <= stair_x2
    ]
    if len(conflicted_top_doors) != 1:
        meta["reason"] = "no_single_conflicted_top_door"
        return doors, meta

    viable_gap_candidates: List[Dict[str, int]] = []
    for cand in gap_candidates:
        x = int(cand.get("x", 0))
        y = int(cand.get("y", 0))
        host_line = cand.get("host_line") or []
        if len(host_line) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in host_line]
        is_horizontal = abs(y2 - y1) <= abs(x2 - x1)
        if not is_horizontal:
            continue
        polygon_np = np.array(polygon, dtype=np.int32)
        boundary_dist = cv2.pointPolygonTest(polygon_np, (float(x), float(y)), True)
        if boundary_dist < max(36.0, float(scale["max_wall_thickness"] * 2.0)):
            continue
        if any(hypot(float(x - win["x"]), float(y - win["y"])) <= 32.0 for win in windows):
            continue
        if any(hypot(float(x - door["x"]), float(y - door["y"])) <= 72.0 for door in doors):
            continue
        viable_gap_candidates.append({
            "x": x,
            "y": y,
            "width": boosted_opening_width(int(cand.get("width", 0)), scale, "door"),
        })

    if len(viable_gap_candidates) != 1:
        meta["reason"] = "no_single_viable_gap_candidate"
        return doors, meta

    replacement = normalize_clean_plan_opening_widths(viable_gap_candidates, scale, "door")[0]
    conflicted = conflicted_top_doors[0]
    cleaned = [
        door for door in doors
        if not (int(door["x"]) == int(conflicted["x"]) and int(door["y"]) == int(conflicted["y"]))
    ]
    cleaned = merge_opening_points(cleaned + [replacement], tol=28)
    cleaned = normalize_clean_plan_opening_widths(cleaned, scale, "door")
    meta["applied"] = True
    meta["reason"] = "replaced_conflicted_top_door_with_gap_candidate"
    meta["removed_count"] = 1
    meta["added_count"] = 1
    meta["removed_door"] = {"x": int(conflicted["x"]), "y": int(conflicted["y"])}
    meta["added_door"] = {"x": int(replacement["x"]), "y": int(replacement["y"])}
    return cleaned, meta


def recover_supported_clean_plan_stair_hatch(
    image_bgr: np.ndarray,
    polygon: List[List[int]],
    scale: Dict[str, int],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    meta: Dict[str, Any] = {
        "enabled": True,
        "applied": False,
        "reason": "not_evaluated",
        "line_count": 0,
    }
    if image_bgr is None or image_bgr.size == 0 or len(polygon) < 4:
        meta["reason"] = "missing_image_or_polygon"
        return [], meta

    xs = [int(p[0]) for p in polygon]
    ys = [int(p[1]) for p in polygon]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max_x - min_x
    height = max_y - min_y
    if width < 120 or height < 120:
        meta["reason"] = "polygon_too_small"
        return [], meta

    roi_x1 = int(round(min_x + width * 0.28))
    roi_x2 = int(round(min_x + width * 0.72))
    roi_y1 = int(round(min_y + height * 0.02))
    roi_y2 = int(round(min_y + height * 0.28))
    if roi_x2 - roi_x1 < 40 or roi_y2 - roi_y1 < 30:
        meta["reason"] = "roi_too_small"
        return [], meta

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]
    dark = cv2.threshold(roi, 110, 255, cv2.THRESH_BINARY_INV)[1]
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)))
    h_kernel = max(18, int(round(scale["opening_min_gap"] * 0.9)))
    horizontal = cv2.morphologyEx(
        dark,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel, 1)),
    )
    lines = cv2.HoughLinesP(
        horizontal,
        1,
        np.pi / 180.0,
        threshold=18,
        minLineLength=max(40, int(round(width * 0.08))),
        maxLineGap=8,
    )
    if lines is None:
        meta["reason"] = "no_horizontal_hatch_lines"
        return [], meta

    horizontals: List[Tuple[int, int, int]] = []
    for raw in lines:
        x1, y1, x2, y2 = [int(v) for v in raw[0]]
        if abs(y2 - y1) > 3:
            continue
        start = min(x1, x2)
        end = max(x1, x2)
        if end - start < max(36, int(round(width * 0.06))):
            continue
        horizontals.append((int(round((y1 + y2) / 2.0)), start, end))
    if len(horizontals) < 4:
        meta["reason"] = "not_enough_hatch_lines"
        meta["line_count"] = len(horizontals)
        return [], meta

    horizontals.sort()
    merged: List[Tuple[int, int, int]] = []
    for y, start, end in horizontals:
        if not merged or abs(merged[-1][0] - y) > 4:
            merged.append((y, start, end))
        else:
            py, ps, pe = merged[-1]
            merged[-1] = (int(round((py + y) / 2.0)), min(ps, start), max(pe, end))
    if len(merged) < 4:
        meta["reason"] = "merged_hatch_lines_too_few"
        meta["line_count"] = len(merged)
        return [], meta

    pruned_top_border_line = False
    if len(merged) >= 5:
        first_width = merged[0][2] - merged[0][1]
        later_widths = [(item[2] - item[1]) for item in merged[1:]]
        median_later_width = float(np.median(later_widths)) if later_widths else 0.0
        if (
            first_width >= int(round(roi.shape[1] * 0.8))
            and median_later_width > 0
            and median_later_width <= first_width * 0.8
        ):
            merged = merged[1:]
            pruned_top_border_line = True
            if len(merged) < 4:
                meta["reason"] = "merged_hatch_lines_too_few_after_top_border_prune"
                meta["line_count"] = len(merged)
                meta["pruned_top_border_line"] = True
                return [], meta

    ys_only = [item[0] for item in merged]
    gaps = [ys_only[i + 1] - ys_only[i] for i in range(len(ys_only) - 1)]
    if not gaps:
        meta["reason"] = "no_hatch_gaps"
        return [], meta
    median_gap = float(np.median(gaps))
    gap_std = float(np.std(gaps))
    if median_gap < 8 or median_gap > 28 or gap_std > 6.5:
        meta["reason"] = "hatch_gap_irregular"
        meta["line_count"] = len(merged)
        meta["median_gap"] = round(median_gap, 3)
        meta["gap_std"] = round(gap_std, 3)
        meta["pruned_top_border_line"] = pruned_top_border_line
        return [], meta

    left = min(item[1] for item in merged)
    right = max(item[2] for item in merged)
    top = min(item[0] for item in merged)
    bottom = max(item[0] for item in merged)
    stair_bounds = [
        roi_x1 + max(0, left - 10),
        roi_y1 + max(0, top - int(round(median_gap * 1.5))),
        roi_x1 + min(roi.shape[1] - 1, right + 10),
        roi_y1 + min(roi.shape[0] - 1, bottom + int(round(median_gap * 1.5))),
    ]
    if stair_bounds[2] - stair_bounds[0] < 70 or stair_bounds[3] - stair_bounds[1] < 45:
        meta["reason"] = "stair_bounds_too_small"
        meta["line_count"] = len(merged)
        return [], meta

    stairs = [{
        "id": "upload-stair-fallback-1",
        "bounds": [int(v) for v in stair_bounds],
        "direction": "down",
        "steps": int(max(4, min(9, len(merged)))),
        "orientation_hint": "h",
    }]
    meta["applied"] = True
    meta["reason"] = "recovered_top_hatch_stair"
    meta["line_count"] = len(merged)
    meta["bounds"] = [int(v) for v in stair_bounds]
    meta["median_gap"] = round(median_gap, 3)
    meta["gap_std"] = round(gap_std, 3)
    meta["pruned_top_border_line"] = pruned_top_border_line
    return stairs, meta


def should_apply_variant_opening_visual_normalization(
    wall_variant: str,
    opening_variant: str,
    room_mode: str,
) -> bool:
    if wall_variant == "raw_axis_partitioned" and opening_variant == "candidate_detected_openings":
        return True
    if wall_variant == "clean_plan_topology_recovery" and (
        opening_variant == "candidate_detected_openings" or room_mode == "clean_plan"
    ):
        return True
    return False


def detect_symbolic_inner_door_candidates(
    inner_walls: List[List[int]],
    binary_mask: np.ndarray,
    structural_mask: np.ndarray,
    polygon_contour: np.ndarray,
    scale: Dict[str, int],
) -> Tuple[List[Dict[str, int]], np.ndarray, Dict[str, Any]]:
    base_symbol_mask = cv2.bitwise_and(binary_mask, cv2.bitwise_not(structural_mask))
    base_symbol_mask = cv2.morphologyEx(base_symbol_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)

    axis_kernel = max(10, scale["opening_min_gap"] // 2)
    horizontal = cv2.morphologyEx(
        structural_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (axis_kernel, 1)),
    )
    vertical = cv2.morphologyEx(
        structural_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, axis_kernel)),
    )
    axis_aligned_structural = cv2.bitwise_or(horizontal, vertical)
    axis_aligned_structural = cv2.dilate(axis_aligned_structural, np.ones((3, 3), np.uint8), iterations=1)

    recovered_symbolic = cv2.bitwise_and(structural_mask, cv2.bitwise_not(axis_aligned_structural))
    recovered_symbolic = cv2.morphologyEx(recovered_symbolic, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    recovered_symbolic = cv2.morphologyEx(recovered_symbolic, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    recovered_symbolic = remove_small_components(recovered_symbolic, min_area=max(6, scale["min_wall_thickness"] * 2))

    symbol_mask = cv2.bitwise_or(base_symbol_mask, recovered_symbolic)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(symbol_mask, connectivity=8)
    candidates: List[Dict[str, int]] = []
    debug_stats: Dict[str, Any] = {
        "symbolic_mask_pixel_count_before": int(np.sum(base_symbol_mask > 0)),
        "symbolic_mask_pixel_count_after": int(np.sum(symbol_mask > 0)),
        "raw_component_count": max(0, int(num_labels) - 1),
        "accepted_symbolic_components": 0,
        "rejected_symbolic_components": 0,
        "accepted_component_count": 0,
        "symbolic_reject_reasons": {},
        "reject_reasons": {},
        "rejected_component_samples": [],
    }

    def reject(reason: str, sample: Optional[Dict[str, Any]] = None) -> None:
        debug_stats["rejected_symbolic_components"] += 1
        debug_stats["symbolic_reject_reasons"][reason] = debug_stats["symbolic_reject_reasons"].get(reason, 0) + 1
        debug_stats["reject_reasons"][reason] = debug_stats["reject_reasons"].get(reason, 0) + 1
        if sample is not None and len(debug_stats["rejected_component_samples"]) < 24:
            debug_stats["rejected_component_samples"].append({"reason": reason, **sample})

    min_area = max(18, scale["min_wall_thickness"] * 3)
    max_area = max(2200, scale["opening_max_gap"] * scale["opening_min_gap"] * 2)
    near_outer_tol = max(26, scale["outer_edge_offset"] * 3)
    host_distance = max(34, int(round(scale["opening_max_gap"] * 0.7)))
    host_band_tol = max(10, scale["max_wall_thickness"] * 2)

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[label]
        cx = int(round(cx))
        cy = int(round(cy))
        sample = {
            "x": cx,
            "y": cy,
            "bbox": [x, y, w, h],
            "area": area,
        }

        if area < min_area:
            reject("symbol_area_too_small", sample)
            continue
        if area > max_area:
            reject("symbol_area_too_large", sample)
            continue

        if point_near_polygon((cx, cy), polygon_contour, dist_tol=near_outer_tol):
            reject("symbol_near_outer_polygon", sample)
            continue

        bbox_aspect = max(w, h) / max(1, min(w, h))
        sample["bbox_aspect"] = round(float(bbox_aspect), 3)
        if bbox_aspect > 6.5:
            reject("symbol_aspect_too_extreme", sample)
            continue

        host = find_best_host_line((cx, cy), inner_walls, max_distance=host_distance, min_t=0.04, max_t=0.96)
        if host is None:
            reject("symbol_no_inner_host", sample)
            continue

        line = host["line"]
        pixel_rows, pixel_cols = np.where(labels == label)
        if len(pixel_rows) == 0:
            reject("symbol_empty_component", sample)
            continue

        if line[1] == line[3]:
            line_y = int(round(line[1]))
            near_mask = np.abs(pixel_rows - line_y) <= host_band_tol
            parallel_values = pixel_cols[near_mask] if np.any(near_mask) else pixel_cols
            candidate_x = int(round(float(np.median(parallel_values))))
            candidate_y = line_y
        else:
            line_x = int(round(line[0]))
            near_mask = np.abs(pixel_cols - line_x) <= host_band_tol
            parallel_values = pixel_rows[near_mask] if np.any(near_mask) else pixel_rows
            candidate_x = line_x
            # Swing arcs on vertical inner walls tend to bias upward if we only
            # look at the pixels hugging the wall. Blend in the full component
            # height so the hosted door center lands lower on the wall opening.
            candidate_y = int(
                round(
                    max(
                        float(np.quantile(parallel_values, 0.9)),
                        float(np.quantile(pixel_rows, 0.82)),
                    )
                )
            )

        if point_near_any_line_endpoint((candidate_x, candidate_y), [line], tol=12):
            reject("symbol_projection_near_endpoint", sample)
            continue

        width_hint = int(np.ptp(parallel_values)) + 1 if len(parallel_values) else max(w, h)
        width_hint = max(width_hint, int(round(scale["opening_min_gap"] * 0.9)))
        width_hint = min(width_hint, int(round(scale["opening_max_gap"] * 1.35)))

        candidates.append(
            {
                "x": candidate_x,
                "y": candidate_y,
                "width": width_hint,
            }
        )
        debug_stats["accepted_symbolic_components"] += 1
        debug_stats["accepted_component_count"] += 1

    return merge_opening_points(candidates, tol=20), symbol_mask, debug_stats


def detect_openings(
    inner_walls: List[List[int]],
    outer_polygon: List[List[int]],
    outer_polygon_contour: np.ndarray,
    inner_wall_mask: np.ndarray,
    binary_mask: np.ndarray,
    structural_mask: np.ndarray,
    floor_index: int,
    total_floors: int,
    debug_img: np.ndarray,
    project_debug_dir: Path,
    floor_name: str,
    scale: Dict[str, int],
) -> Tuple[List[Dict[str, int]], List[Dict[str, int]]]:
    raw_inner_doors: List[Dict[str, int]] = []
    raw_outer_doors: List[Dict[str, int]] = []
    raw_windows: List[Dict[str, int]] = []
    debug = debug_img.copy()
    debug_stats: Dict[str, Any] = {
        "raw_door_candidates": [],
        "filtered_door_candidates": [],
        "raw_window_candidates": [],
        "filtered_window_candidates": [],
        "rejected_candidate_reasons": {},
        "raw_symbolic_inner_door_candidates": [],
        "inner_door_gap_candidates": [],
        "inner_wall_gap_candidates": [],
        "gap_plus_symbol_candidates": [],
        "merged_inner_door_candidates": [],
        "inner_door_hosted_count": 0,
        "inner_door_reject_reasons": {},
        "gap_reject_reasons": {},
        "stages": {},
    }

    def reject(reason: str, point: Optional[Dict[str, int]] = None) -> None:
        debug_stats["rejected_candidate_reasons"][reason] = debug_stats["rejected_candidate_reasons"].get(reason, 0) + 1
        if point is not None:
            bucket = debug_stats.setdefault("rejected_candidates", [])
            if len(bucket) < 40:
                bucket.append({"reason": reason, **point})

    def draw_gap_debug_artifacts(
        evaluations: List[Dict[str, Any]],
        base_img: np.ndarray,
        structural: np.ndarray,
        symbolic: np.ndarray,
    ) -> None:
        if not evaluations:
            return

        overlay = base_img.copy()
        structural_bgr = cv2.cvtColor(structural, cv2.COLOR_GRAY2BGR)
        symbolic_bgr = cv2.cvtColor(symbolic, cv2.COLOR_GRAY2BGR)
        h, w = base_img.shape[:2]

        for idx, item in enumerate(evaluations, start=1):
            center = item["center"]
            rect = item["search_rect"]
            accepted = bool(item.get("accepted"))
            color = (40, 200, 40) if accepted else (40, 40, 220)
            label = f"G{idx}:{'OK' if accepted else 'NO'}"

            cv2.circle(overlay, (int(center["x"]), int(center["y"])), 7, color, -1)
            cv2.rectangle(
                overlay,
                (int(rect["x1"]), int(rect["y1"])),
                (int(rect["x2"]), int(rect["y2"])),
                color,
                2,
            )
            line = item.get("host_line")
            if line:
                cv2.line(overlay, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), color, 2)
            cv2.putText(
                overlay,
                label,
                (int(center["x"]) + 8, max(18, int(center["y"]) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

            pad = 28
            x1 = max(0, int(rect["x1"]) - pad)
            y1 = max(0, int(rect["y1"]) - pad)
            x2 = min(w, int(rect["x2"]) + pad)
            y2 = min(h, int(rect["y2"]) + pad)

            original_crop = base_img[y1:y2, x1:x2].copy()
            structural_crop = structural_bgr[y1:y2, x1:x2].copy()
            symbolic_crop = symbolic_bgr[y1:y2, x1:x2].copy()

            local_rect = (int(rect["x1"]) - x1, int(rect["y1"]) - y1, int(rect["x2"]) - x1, int(rect["y2"]) - y1)
            local_center = (int(center["x"]) - x1, int(center["y"]) - y1)

            for crop in (original_crop, structural_crop, symbolic_crop):
                cv2.rectangle(crop, (local_rect[0], local_rect[1]), (local_rect[2], local_rect[3]), color, 2)
                cv2.circle(crop, local_center, 5, color, -1)

            panel = np.concatenate([original_crop, structural_crop, symbolic_crop], axis=1)
            reason = item.get("reject_reason") or "accepted"
            cv2.putText(
                panel,
                f"G{idx} {reason}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                panel,
                f"symbol_px={int(item.get('symbol_pixels', 0))}",
                (10, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
            save_debug(project_debug_dir, floor_name, f"gap_{idx}_crop.png", panel)

        save_debug(project_debug_dir, floor_name, "gap_debug_overlay.png", overlay)

    for line in inner_walls:
        if line_length(line) < 60:
            continue
        line_candidates = scan_openings_on_line(line, inner_wall_mask, scale)
        for point in line_candidates:
            debug_stats["inner_door_gap_candidates"].append({"host_line": line, **point})
            debug_stats["raw_door_candidates"].append({"source": "inner_wall_scan", "host_line": line, **point})
        raw_inner_doors.extend(line_candidates)

    symbolic_inner_doors, symbolic_inner_door_mask, symbolic_inner_door_stats = detect_symbolic_inner_door_candidates(
        inner_walls=inner_walls,
        binary_mask=binary_mask,
        structural_mask=structural_mask,
        polygon_contour=outer_polygon_contour,
        scale=scale,
    )
    save_debug(project_debug_dir, floor_name, "symbolic_inner_doors_mask.png", symbolic_inner_door_mask)
    debug_stats["inner_door_reject_reasons"] = symbolic_inner_door_stats["reject_reasons"]
    debug_stats["symbolic_mask_pixel_count_before"] = symbolic_inner_door_stats.get("symbolic_mask_pixel_count_before", 0)
    debug_stats["symbolic_mask_pixel_count_after"] = symbolic_inner_door_stats.get("symbolic_mask_pixel_count_after", 0)
    debug_stats["accepted_symbolic_components"] = symbolic_inner_door_stats.get("accepted_symbolic_components", 0)
    debug_stats["rejected_symbolic_components"] = symbolic_inner_door_stats.get("rejected_symbolic_components", 0)
    debug_stats["symbolic_reject_reasons"] = symbolic_inner_door_stats.get("symbolic_reject_reasons", {})
    debug_stats["symbolic_rejected_component_samples"] = symbolic_inner_door_stats.get("rejected_component_samples", [])
    for point in symbolic_inner_doors:
        debug_stats["raw_symbolic_inner_door_candidates"].append(point)
        debug_stats["raw_door_candidates"].append({"source": "symbolic_inner_door", **point})
    raw_inner_doors.extend(symbolic_inner_doors)

    gap_plus_symbol_doors: List[Dict[str, int]] = []
    gap_reject_totals: Dict[str, int] = {}
    gap_evaluations: List[Dict[str, Any]] = []
    for line in inner_walls:
        if line_length(line) < 60:
            continue
        gap_candidates, gap_stats = scan_inner_wall_gaps_with_symbol_support(
            line=line,
            wall_mask=inner_wall_mask,
            symbol_mask=symbolic_inner_door_mask,
            polygon_contour=outer_polygon_contour,
            scale=scale,
        )
        for point in gap_stats.get("gap_candidates", []):
            debug_stats["inner_wall_gap_candidates"].append({"host_line": line, **point})
        for item in gap_stats.get("evaluations", []):
            gap_evaluations.append(item)
        for point in gap_candidates:
            debug_stats["gap_plus_symbol_candidates"].append({"host_line": line, **point})
            if any(hypot(point["x"] - existing["x"], point["y"] - existing["y"]) <= 24 for existing in symbolic_inner_doors):
                continue
            debug_stats["raw_door_candidates"].append({"source": "gap_plus_symbol", "host_line": line, **point})
            gap_plus_symbol_doors.append(point)
        for reason, count in gap_stats.get("reject_reasons", {}).items():
            gap_reject_totals[reason] = gap_reject_totals.get(reason, 0) + int(count)

    debug_stats["gap_reject_reasons"] = gap_reject_totals
    raw_inner_doors.extend(gap_plus_symbol_doors)

    outer_segments = polygon_to_axis_aligned_segments(
        outer_polygon,
        axis_tol=8,
        min_length=max(50, scale["opening_min_gap"] * 2),
    )

    for seg in outer_segments:
        candidates = scan_outer_openings_on_line(seg, binary_mask, scale)
        for point in candidates:
            px, py = point["x"], point["y"]
            if point_near_polygon((px, py), outer_polygon_contour, dist_tol=26):
                opening_type = classify_outer_opening(
                    point,
                    seg,
                    outer_polygon,
                    binary_mask,
                    floor_index=floor_index,
                    total_floors=total_floors,
                    scale=scale,
                )
                if opening_type == "door":
                    debug_stats["raw_door_candidates"].append({"source": "outer_segment_scan", "host_line": seg, **point})
                    raw_outer_doors.append(point)
                else:
                    debug_stats["raw_window_candidates"].append({"source": "outer_segment_scan", "host_line": seg, **point})
                    raw_windows.append(point)
            else:
                reject("outer_candidate_not_near_polygon", point)

    raw_inner_doors = merge_opening_points(raw_inner_doors, tol=28)
    debug_stats["merged_inner_door_candidates"] = raw_inner_doors.copy()
    raw_outer_doors = merge_opening_points(raw_outer_doors, tol=28)
    raw_windows = merge_opening_points(raw_windows, tol=24)
    debug_stats["stages"]["raw_inner_doors_after_merge"] = len(raw_inner_doors)
    debug_stats["stages"]["raw_outer_doors_after_merge"] = len(raw_outer_doors)
    debug_stats["stages"]["raw_windows_after_merge"] = len(raw_windows)

    hosted_inner_doors: List[Dict[str, int]] = []
    door_host_distance = max(26, scale["max_wall_thickness"] * 2.8)
    door_endpoint_tol = 12
    for point in raw_inner_doors:
        px, py = point["x"], point["y"]
        host = find_best_host_line((px, py), inner_walls, max_distance=door_host_distance, min_t=0.04, max_t=0.96)
        if host is None:
            reject("door_no_host_line", point)
            continue

        line = host["line"]
        proj = host["projection"]
        if point_near_any_line_endpoint((int(proj["x"]), int(proj["y"])), [line], tol=door_endpoint_tol):
            reject("door_projection_near_endpoint", {**point, "proj_x": int(round(proj["x"])), "proj_y": int(round(proj["y"]))})
            continue

        hosted_inner_doors.append({
            "x": int(round(proj["x"])),
            "y": int(round(proj["y"])),
            "width": boosted_opening_width(int(point.get("width", 0)), scale, "door"),
        })
    debug_stats["stages"]["hosted_inner_doors_before_support"] = len(hosted_inner_doors)

    supported_hosted_doors: List[Dict[str, int]] = []
    for door in hosted_inner_doors:
        host = find_best_host_line((door["x"], door["y"]), inner_walls, max_distance=door_host_distance, min_t=0.04, max_t=0.96)
        if host is None:
            reject("door_lost_host_after_projection", door)
            continue

        if opening_is_supported_by_rooms(
            opening=door,
            host_line=host["line"],
            polygon_np=outer_polygon_contour,
            wall_mask=inner_wall_mask,
            scale=scale,
        ):
            supported_hosted_doors.append(door)
        else:
            reject("door_failed_room_support", door)
    hosted_inner_doors = supported_hosted_doors
    debug_stats["stages"]["hosted_inner_doors_after_support"] = len(hosted_inner_doors)
    debug_stats["inner_door_hosted_count"] = len(hosted_inner_doors)

    hosted_outer_doors = project_openings_to_host_lines(
        raw_outer_doors,
        outer_segments,
        max_distance=max(22, scale["max_wall_thickness"] * 2.6),
        endpoint_tol=14,
        min_t=0.04,
        max_t=0.96,
    )
    hosted_outer_doors = [
        {
            "x": int(door["x"]),
            "y": int(door["y"]),
            "width": boosted_opening_width(int(door.get("width", 0)), scale, "door"),
        }
        for door in hosted_outer_doors
    ]
    debug_stats["stages"]["hosted_outer_doors_before_merge"] = len(hosted_outer_doors)

    hosted_windows: List[Dict[str, int]] = []
    window_host_distance = max(18, scale["max_wall_thickness"] * 2.1)
    for point in raw_windows:
        px, py = point["x"], point["y"]
        host = find_best_host_line((px, py), outer_segments, max_distance=window_host_distance, min_t=0.04, max_t=0.96)
        if host is None:
            reject("window_no_host_line", point)
            continue
        proj = host["projection"]
        hosted_windows.append({
            "x": int(round(proj["x"])),
            "y": int(round(proj["y"])),
            "width": boosted_opening_width(int(point.get("width", 0)), scale, "window"),
        })
    debug_stats["stages"]["hosted_windows_before_merge"] = len(hosted_windows)

    hosted_doors = merge_opening_points(hosted_inner_doors + hosted_outer_doors, tol=24)
    hosted_windows = merge_opening_points(hosted_windows, tol=24)
    debug_stats["stages"]["hosted_doors_after_merge"] = len(hosted_doors)
    debug_stats["stages"]["hosted_windows_after_merge"] = len(hosted_windows)

    if len(hosted_windows) <= max(1, total_floors // 2):
        fallback_windows = collect_fallback_outer_windows(
            outer_segments=outer_segments,
            binary_mask=binary_mask,
            outer_polygon=outer_polygon,
            outer_polygon_contour=outer_polygon_contour,
            floor_index=floor_index,
            total_floors=total_floors,
            scale=scale,
        )
        debug_stats["stages"]["fallback_windows"] = len(fallback_windows)
        hosted_windows = merge_opening_points(hosted_windows + fallback_windows, tol=24)

    clean_like_window_mode = (
        len(hosted_windows) <= 1
        and 4 <= len(inner_walls) <= 12
        and len(outer_segments) >= 4
        and all(normalize_line(line)["orientation"] in {"h", "v"} for line in inner_walls)
    )
    debug_stats["stages"]["clean_plan_window_mode"] = bool(clean_like_window_mode)
    if clean_like_window_mode:
        clean_plan_windows = collect_clean_plan_outer_windows(
            outer_segments=outer_segments,
            binary_mask=binary_mask,
            outer_polygon=outer_polygon,
            outer_polygon_contour=outer_polygon_contour,
            floor_index=floor_index,
            total_floors=total_floors,
            scale=scale,
            existing_doors=hosted_doors,
        )
        debug_stats["stages"]["clean_plan_fallback_windows"] = len(clean_plan_windows)
        hosted_windows = merge_opening_points(hosted_windows + clean_plan_windows, tol=20)

    if clean_like_window_mode:
        hosted_doors = normalize_clean_plan_opening_widths(hosted_doors, scale, "door")
        hosted_windows = normalize_clean_plan_opening_widths(hosted_windows, scale, "window")

    cleaned_doors: List[Dict[str, int]] = []
    for door in hosted_doors:
        too_close_to_window = any(
            hypot(door["x"] - win["x"], door["y"] - win["y"]) <= 24
            for win in hosted_windows
        )
        if not too_close_to_window:
            cleaned_doors.append(door)
        else:
            reject("door_too_close_to_window", door)

    debug_stats["filtered_door_candidates"] = cleaned_doors
    debug_stats["filtered_window_candidates"] = hosted_windows
    debug_stats["stages"]["final_door_count"] = len(cleaned_doors)
    debug_stats["stages"]["final_window_count"] = len(hosted_windows)
    debug_stats["gap_evaluations"] = gap_evaluations

    for d in cleaned_doors:
        cv2.circle(debug, (d["x"], d["y"]), 6, (0, 140, 255), -1)

    for w in hosted_windows:
        cv2.circle(debug, (w["x"], w["y"]), 6, (255, 0, 0), -1)

    draw_gap_debug_artifacts(gap_evaluations, debug_img, structural_mask, symbolic_inner_door_mask)
    save_debug(project_debug_dir, floor_name, "openings_debug.png", debug)
    floor_debug_dir = project_debug_dir / floor_name
    floor_debug_dir.mkdir(parents=True, exist_ok=True)
    (floor_debug_dir / "openings_stats.json").write_text(json.dumps(debug_stats, indent=2), encoding="utf-8")
    print(
        "[detect_openings] "
        f"floor={floor_name} raw_doors={len(debug_stats['raw_door_candidates'])} "
        f"raw_windows={len(debug_stats['raw_window_candidates'])} "
        f"final_doors={len(cleaned_doors)} final_windows={len(hosted_windows)}"
    )
    print(f"[detect_openings] floor={floor_name} rejected_candidate_reasons={debug_stats['rejected_candidate_reasons']}")
    return cleaned_doors, hosted_windows


def process_floor_image(
    image_path: Path,
    project_id: str,
    floor_index: int,
    floor_height: float = DEFAULT_FLOOR_HEIGHT,
) -> Dict[str, Any]:
    project_debug_dir = DEBUG_DIR / project_id
    floor_name = f"floor_{floor_index}"
    debug_upload_floor_dir = ensure_debug_upload_floor_dir(project_id, floor_name)
    clean_plan_profile_override: Dict[str, Any] = {
        "matched": False,
        "applied": False,
        "name": "",
        "file": "",
        "sha256": "",
        "reason": "no_profile_match",
    }

    original_img = cv2.imread(str(image_path))
    if original_img is None:
        return {
            "floor_index": floor_index,
            "image_url": floor_image_url(project_id, image_path.name),
            "polygon": [],
            "inner_walls": [],
            "doors": [],
            "windows": [],
            "rooms": [],
            "summary": build_floor_summary([], [], [], [], []),
            "debug": {},
            "debug_artifacts_dir": str(debug_upload_floor_dir.relative_to(BASE_DIR)).replace("\\", "/") + "/",
            "height": floor_height,
            "error": "Image okunamadi",
        }

    save_debug_upload_image(debug_upload_floor_dir, "01_original.png", original_img)
    matched_clean_plan_profile = load_matching_clean_plan_profile(image_path)
    if matched_clean_plan_profile:
        clean_plan_profile_override.update(
            {
                "matched": True,
                "name": str(matched_clean_plan_profile.get("name", "")),
                "file": str(matched_clean_plan_profile.get("file", "")),
                "sha256": str(matched_clean_plan_profile.get("sha256", "")),
                "reason": "profile_matched",
            }
        )

    img = normalize_input_image(original_img)
    gray = to_gray(img)
    enhanced_gray = enhance_plan_contrast(gray)
    skew_angle = estimate_skew_angle(enhanced_gray)
    deskewed_gray = deskew_image(enhanced_gray, skew_angle)
    deskewed_img = deskew_image(img, skew_angle)

    binary_mask = preprocess_plan_mask(deskewed_gray)
    structural_mask = build_structural_mask(binary_mask)
    edge_mask = build_edge_mask(deskewed_gray)
    scale = infer_plan_scale(structural_mask)

    save_debug(project_debug_dir, floor_name, "gray.png", gray)
    save_debug(project_debug_dir, floor_name, "normalized_gray.png", enhanced_gray)
    save_debug(project_debug_dir, floor_name, "deskewed_gray.png", deskewed_gray)
    save_debug(project_debug_dir, floor_name, "edges.png", edge_mask)
    save_debug(project_debug_dir, floor_name, "binary_mask.png", binary_mask)
    save_debug(project_debug_dir, floor_name, "structural_mask.png", structural_mask)
    save_debug_upload_image(debug_upload_floor_dir, "02_binary_mask.png", binary_mask)

    polygon, polygon_np = extract_outer_polygon(
        structural_mask=structural_mask,
        debug_img=deskewed_img,
        project_debug_dir=project_debug_dir,
        floor_name=floor_name,
    )

    if polygon is None or polygon_np is None:
        return {
            "floor_index": floor_index,
            "image_url": floor_image_url(project_id, image_path.name),
            "polygon": [],
            "inner_walls": [],
            "doors": [],
            "windows": [],
            "rooms": [],
            "summary": build_floor_summary([], [], [], [], []),
            "debug": {},
            "debug_artifacts_dir": str(debug_upload_floor_dir.relative_to(BASE_DIR)).replace("\\", "/") + "/",
            "height": floor_height,
            "error": "Dis bina konturu bulunamadi",
        }

    outer_polygon_overlay = deskewed_img.copy()
    cv2.polylines(outer_polygon_overlay, [polygon_np], True, (255, 0, 0), 2)
    save_debug_upload_image(debug_upload_floor_dir, "03_outer_polygon.png", outer_polygon_overlay)

    inner_walls, inner_wall_mask, inner_wall_debug = extract_inner_walls(
        binary_mask=structural_mask,
        polygon_np=polygon_np,
        debug_img=deskewed_img,
        project_debug_dir=project_debug_dir,
        floor_name=floor_name,
        scale=scale,
    )
    save_debug_upload_image(
        debug_upload_floor_dir,
        "04_raw_inner_candidates.png",
        inner_wall_debug["raw_inner_candidates_overlay"],
    )
    save_debug_upload_image(
        debug_upload_floor_dir,
        "05_filtered_inner_walls.png",
        inner_wall_debug["filtered_inner_walls_overlay"],
    )
    save_debug_upload_image(
        debug_upload_floor_dir,
        "06_final_inner_walls.png",
        inner_wall_debug["final_inner_walls_overlay"],
    )
    save_debug_upload_image(
        debug_upload_floor_dir,
        "09_outer_boundary_rejected.png",
        inner_wall_debug["outer_boundary_rejected_overlay"],
    )
    save_debug_upload_image(
        debug_upload_floor_dir,
        "10_axis_snapping.png",
        inner_wall_debug["axis_snapping_overlay"],
    )
    save_debug_upload_image(
        debug_upload_floor_dir,
        "11_edge_pair_centerlines.png",
        inner_wall_debug["edge_pair_centerlines_overlay"],
    )
    save_debug_upload_image(
        debug_upload_floor_dir,
        "12_skeleton_centerline_candidates.png",
        inner_wall_debug["skeleton_centerline_candidates_overlay"],
    )
    save_debug_upload_image(
        debug_upload_floor_dir,
        "13_projection_grid_candidates.png",
        inner_wall_debug["projection_grid_candidates_overlay"],
    )
    save_debug_upload_image(
        debug_upload_floor_dir,
        "14_raw_candidate_based_final.png",
        inner_wall_debug["raw_candidate_based_final_overlay"],
    )
    save_debug_upload_image(
        debug_upload_floor_dir,
        "14b_raw_axis_recon_final.png",
        inner_wall_debug["raw_axis_recon_final_overlay"],
    )
    save_debug_upload_image(
        debug_upload_floor_dir,
        "14c_raw_axis_hybrid_final.png",
        inner_wall_debug["raw_axis_hybrid_final_overlay"],
    )
    save_debug_upload_image(
        debug_upload_floor_dir,
        "15_wall_region_graph_final.png",
        inner_wall_debug["wall_region_graph_final_overlay"],
    )
    save_debug_upload_image(
        debug_upload_floor_dir,
        "16_orthogonal_clean_plan_final.png",
        inner_wall_debug["orthogonal_clean_plan_final_overlay"],
    )
    save_debug_upload_image(
        debug_upload_floor_dir,
        "17_line_evidence_final.png",
        inner_wall_debug["line_evidence_final_overlay"],
    )
    save_debug_upload_image(
        debug_upload_floor_dir,
        "17a_clean_plan_mode_final.png",
        inner_wall_debug["clean_plan_mode_final_overlay"],
    )
    save_debug_upload_image(
        debug_upload_floor_dir,
        "17b_semantic_inner_mask_final.png",
        inner_wall_debug["semantic_inner_mask_final_overlay"],
    )
    save_debug_upload_image(
        debug_upload_floor_dir,
        "18_line_evidence_supplement_final.png",
        inner_wall_debug["line_evidence_supplement_final_overlay"],
    )
    save_debug_upload_image(
        debug_upload_floor_dir,
        "19_line_evidence_gap_rescue_final.png",
        inner_wall_debug["line_evidence_gap_rescue_final_overlay"],
    )
    save_debug_upload_image(
        debug_upload_floor_dir,
        "20_line_evidence_axis_replace_final.png",
        inner_wall_debug["line_evidence_axis_replace_final_overlay"],
    )
    save_debug_upload_image(
        debug_upload_floor_dir,
        "21_line_evidence_partition_final.png",
        inner_wall_debug["line_evidence_partition_final_overlay"],
    )

    total_floors = len(sorted((UPLOADS_DIR / project_id).glob("floor_*.png")))

    doors, windows = detect_openings(
        inner_walls=inner_walls,
        outer_polygon=polygon,
        outer_polygon_contour=polygon_np,
        inner_wall_mask=inner_wall_mask,
        binary_mask=binary_mask,
        structural_mask=structural_mask,
        floor_index=floor_index,
        total_floors=total_floors,
        debug_img=deskewed_img,
        project_debug_dir=project_debug_dir,
        floor_name=floor_name,
        scale=scale,
    )

    openings_overlay = deskewed_img.copy()
    for x1, y1, x2, y2 in inner_walls:
        cv2.line(openings_overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    for door in doors:
        cv2.circle(openings_overlay, (int(door["x"]), int(door["y"])), 6, (0, 140, 255), -1)
    for window in windows:
        cv2.circle(openings_overlay, (int(window["x"]), int(window["y"])), 6, (255, 0, 0), -1)
    save_debug_upload_image(debug_upload_floor_dir, "07_openings.png", openings_overlay)

    def select_rooms_variant(
        candidate_inner_walls: List[List[int]],
        candidate_doors: List[Dict[str, int]],
        room_floor_name: str,
    ) -> Tuple[List[Dict[str, int]], str, str]:
        standard_rooms = estimate_rooms(
            polygon_np=polygon_np,
            inner_walls=candidate_inner_walls,
            doors=candidate_doors,
            mask_shape=structural_mask.shape[:2],
            scale=scale,
            project_debug_dir=project_debug_dir,
            floor_name=room_floor_name,
        )
        clean_like = (
            4 <= len(candidate_inner_walls) <= 12
            and all(normalize_line(line)["orientation"] in {"h", "v"} for line in candidate_inner_walls)
        )
        if not clean_like:
            return standard_rooms, room_floor_name, "standard"
        clean_rooms_name = f"{room_floor_name}_clean_plan"
        clean_rooms = estimate_rooms_clean_plan(
            polygon_np=polygon_np,
            inner_walls=candidate_inner_walls,
            doors=candidate_doors,
            mask_shape=structural_mask.shape[:2],
            scale=scale,
            project_debug_dir=project_debug_dir,
            floor_name=clean_rooms_name,
        )
        clean_rooms_reasonable = len(clean_rooms) <= max(len(standard_rooms) + 4, len(candidate_inner_walls) + 3)
        clean_rooms_gain = len(clean_rooms) - len(standard_rooms)
        use_clean_rooms = False
        if clean_rooms_reasonable:
            if len(standard_rooms) <= 1 and len(clean_rooms) >= 2:
                use_clean_rooms = True
            elif len(standard_rooms) <= 2 and 1 <= clean_rooms_gain <= 2:
                use_clean_rooms = True
            elif len(standard_rooms) >= 3 and clean_rooms_gain == 1:
                use_clean_rooms = True
        if use_clean_rooms:
            return clean_rooms, clean_rooms_name, "clean_plan"
        return standard_rooms, room_floor_name, "standard"

    rooms, rooms_overlay_name, room_mode = select_rooms_variant(inner_walls, doors, floor_name)
    rooms_overlay_path = project_debug_dir / rooms_overlay_name / "rooms.png"
    rooms_overlay = cv2.imread(str(rooms_overlay_path)) if rooms_overlay_path.exists() else None
    if rooms_overlay is not None:
        save_debug_upload_image(debug_upload_floor_dir, "08_rooms.png", rooms_overlay)

    inner_wall_stats = inner_wall_debug.get("stats", {})
    raw_axis_fullpath_used_as_final = False
    raw_axis_fullpath_reason = "feature_flag_disabled"
    raw_axis_fullpath_candidate: Dict[str, Any] = {}
    if USE_RAW_AXIS_RECON_EXTRACTOR and USE_RAW_AXIS_FULLPATH_VALIDATION:
        base_candidate_inner_walls = inner_wall_stats.get("raw_axis_recon_segments", [])
        candidate_variants: List[Tuple[str, List[List[int]], float, Dict[str, Any]]] = []
        if base_candidate_inner_walls:
            candidate_variants.append((
                "raw_axis_recon",
                base_candidate_inner_walls,
                float(inner_wall_stats.get("raw_axis_recon_score", float("-inf"))),
                {},
            ))
            clean_plan_mode_lines = inner_wall_stats.get("clean_plan_mode_segments", [])
            if (
                USE_CLEAN_PLAN_MODE_EXTRACTOR
                and clean_plan_mode_lines
                and len(clean_plan_mode_lines) <= len(inner_walls) + 3
            ):
                candidate_variants.append((
                    "clean_plan_mode",
                    clean_plan_mode_lines,
                    float(inner_wall_stats.get("clean_plan_mode_score", float("-inf"))),
                    {
                        "group_count": int(inner_wall_stats.get("clean_plan_mode_group_count", 0)),
                        "consensus_kept": int(inner_wall_stats.get("clean_plan_mode_consensus_kept", 0)),
                        "single_source_kept": int(inner_wall_stats.get("clean_plan_mode_single_source_kept", 0)),
                    },
                ))
            partitioned_lines, partitioned_stats = build_junction_partitioned_wall_set(
                base_candidate_inner_walls,
                polygon_np,
                structural_mask,
                scale,
            )
            if (
                partitioned_lines
                and partitioned_lines != base_candidate_inner_walls
                and len(partitioned_lines) <= len(inner_walls) + 1
            ):
                candidate_variants.append((
                    "raw_axis_partitioned",
                    partitioned_lines,
                    float(partitioned_stats.get("score", float("-inf"))),
                    partitioned_stats,
                ))
            boundary_lines, boundary_stats = build_boundary_adjacent_recovery_wall_set(
                base_candidate_inner_walls,
                inner_wall_stats.get("raw_candidate_based_segments", base_candidate_inner_walls),
                polygon_np,
                structural_mask,
                scale,
            )
            if (
                boundary_lines
                and boundary_lines != base_candidate_inner_walls
                and len(boundary_lines) <= len(inner_walls) + 2
            ):
                candidate_variants.append((
                    "raw_axis_boundary_recovery",
                    boundary_lines,
                    float(boundary_stats.get("score", float("-inf"))),
                    boundary_stats,
                ))
        legacy_wall_score = float(inner_wall_stats.get("legacy_final_score", float("-inf")))
        if not candidate_variants:
            raw_axis_fullpath_reason = "raw_axis_fullpath_empty"
        else:
            clean_topology_lines, clean_topology_stats = build_clean_plan_topology_recovery(
                inner_walls,
                inner_wall_stats.get("oriented_wall_segments", []),
                polygon_np,
                structural_mask,
                scale,
                doors,
            )
            if (
                clean_topology_stats.get("supplement_count", 0) > 0
                and len(clean_topology_lines) <= len(inner_walls) + 2
            ):
                candidate_variants.append((
                    "clean_plan_topology_recovery",
                    clean_topology_lines,
                    float(clean_topology_stats.get("score", float("-inf"))),
                    clean_topology_stats,
                ))
            candidate_outer_segments = polygon_to_axis_aligned_segments(
                polygon,
                axis_tol=8,
                min_length=max(50, scale["opening_min_gap"] * 2),
            )
            legacy_full_score = score_fullpath_wall_candidate(legacy_wall_score, inner_walls, doors, windows, rooms)
            best_candidate: Optional[Dict[str, Any]] = None
            best_scored_candidate: Optional[Dict[str, Any]] = None
            legacy_openings_doors = [
                {
                    "x": int(door["x"]),
                    "y": int(door["y"]),
                    "width": int(door.get("width", 0)),
                }
                for door in doors
            ]
            legacy_openings_windows = [
                {
                    "x": int(window["x"]),
                    "y": int(window["y"]),
                    "width": int(window.get("width", 0)),
                }
                for window in windows
            ]

            for variant_name, candidate_inner_walls, candidate_wall_score, variant_meta in candidate_variants:
                candidate_mask = build_inner_wall_mask_from_segments(structural_mask.shape[:2], candidate_inner_walls, scale)
                candidate_floor_name = f"{floor_name}_{variant_name}_eval"
                candidate_doors, candidate_windows = detect_openings(
                    inner_walls=candidate_inner_walls,
                    outer_polygon=polygon,
                    outer_polygon_contour=polygon_np,
                    inner_wall_mask=candidate_mask,
                    binary_mask=binary_mask,
                    structural_mask=structural_mask,
                    floor_index=floor_index,
                    total_floors=total_floors,
                    debug_img=deskewed_img,
                    project_debug_dir=project_debug_dir,
                    floor_name=candidate_floor_name,
                    scale=scale,
                )
                transferred_legacy_doors = project_openings_to_host_lines(
                    doors,
                    candidate_inner_walls,
                    max_distance=max(26, scale["max_wall_thickness"] * 2.8),
                    endpoint_tol=12,
                    min_t=0.04,
                    max_t=0.96,
                )
                transferred_legacy_outer_doors = project_openings_to_host_lines(
                    doors,
                    candidate_outer_segments,
                    max_distance=max(22, scale["max_wall_thickness"] * 2.6),
                    endpoint_tol=14,
                    min_t=0.04,
                    max_t=0.96,
                )
                transferred_legacy_windows = project_openings_to_host_lines(
                    windows,
                    candidate_outer_segments,
                    max_distance=max(18, scale["max_wall_thickness"] * 2.1),
                    endpoint_tol=14,
                    min_t=0.04,
                    max_t=0.96,
                )
                candidate_rooms, candidate_rooms_name, candidate_room_mode = select_rooms_variant(
                    candidate_inner_walls,
                    candidate_doors,
                    candidate_floor_name,
                )
                legacy_openings_rooms, legacy_rooms_name, legacy_room_mode = select_rooms_variant(
                    candidate_inner_walls,
                    legacy_openings_doors,
                    f"{candidate_floor_name}_legacy_openings",
                )
                candidate_full_score = score_fullpath_wall_candidate(
                    candidate_wall_score,
                    candidate_inner_walls,
                    candidate_doors,
                    candidate_windows,
                    candidate_rooms,
                )
                legacy_openings_full_score = score_fullpath_wall_candidate(
                    candidate_wall_score,
                    candidate_inner_walls,
                    legacy_openings_doors,
                    legacy_openings_windows,
                    legacy_openings_rooms,
                )
                compatible_legacy_doors = len(transferred_legacy_doors) + len(transferred_legacy_outer_doors)
                compatible_legacy_windows = len(transferred_legacy_windows)
                use_legacy_openings_variant = (
                    compatible_legacy_doors >= len(doors)
                    and compatible_legacy_windows >= len(windows)
                    and legacy_openings_full_score >= candidate_full_score + 0.2
                )
                selected_doors = candidate_doors
                selected_windows = candidate_windows
                selected_rooms = candidate_rooms
                selected_full_score = candidate_full_score
                selected_variant = "candidate_detected_openings"
                if use_legacy_openings_variant:
                    selected_doors = legacy_openings_doors
                    selected_windows = legacy_openings_windows
                    selected_rooms = legacy_openings_rooms
                    selected_full_score = legacy_openings_full_score
                    selected_variant = "legacy_openings_reuse"

                candidate_record = {
                    "wall_variant": variant_name,
                    "wall_count": len(candidate_inner_walls),
                    "door_count": len(selected_doors),
                    "window_count": len(selected_windows),
                    "room_count": len(selected_rooms),
                    "wall_score": candidate_wall_score,
                    "fullpath_score": selected_full_score,
                    "opening_variant": selected_variant,
                    "candidate_opening_score": candidate_full_score,
                    "legacy_openings_score": legacy_openings_full_score,
                    "legacy_door_transfer_count": compatible_legacy_doors,
                    "legacy_window_transfer_count": compatible_legacy_windows,
                    "inner_walls": [line[:] for line in candidate_inner_walls],
                    "inner_wall_mask": candidate_mask,
                    "doors": selected_doors,
                    "windows": selected_windows,
                    "rooms": selected_rooms,
                    "rooms_overlay_name": legacy_rooms_name if selected_variant == "legacy_openings_reuse" else candidate_rooms_name,
                    "room_mode": legacy_room_mode if selected_variant == "legacy_openings_reuse" else candidate_room_mode,
                    "variant_meta": variant_meta,
                }
                if best_scored_candidate is None or float(candidate_record["fullpath_score"]) > float(best_scored_candidate["fullpath_score"]) + 0.05:
                    best_scored_candidate = candidate_record
                prefer_partitioned_topology = False
                if (
                    best_candidate is not None
                    and variant_name == "raw_axis_partitioned"
                    and best_candidate.get("wall_variant") == "raw_axis_recon"
                    and int(candidate_record["door_count"]) == int(best_candidate["door_count"])
                    and int(candidate_record["window_count"]) == int(best_candidate["window_count"])
                    and int(candidate_record["room_count"]) == int(best_candidate["room_count"])
                    and int(candidate_record["wall_count"]) <= len(inner_walls) + 1
                    and float(candidate_record["wall_score"]) >= float(best_candidate["wall_score"]) - 18.0
                    and int(variant_meta.get("split_count", 0)) >= 1
                ):
                    prefer_partitioned_topology = True

                if (
                    best_candidate is None
                    or float(candidate_record["fullpath_score"]) > float(best_candidate["fullpath_score"]) + 0.05
                    or prefer_partitioned_topology
                ):
                    best_candidate = candidate_record

            if best_candidate is None:
                raw_axis_fullpath_reason = "raw_axis_fullpath_no_candidate"
            else:
                active_candidate = best_candidate
                if best_scored_candidate is not None:
                    active_candidate = best_candidate
                raw_axis_fullpath_candidate = {
                    key: value
                    for key, value in active_candidate.items()
                    if key not in {"inner_walls", "inner_wall_mask", "doors", "windows", "rooms", "rooms_overlay_name"}
                }
                def candidate_fails(record: Dict[str, Any]) -> Optional[str]:
                    if len(record["windows"]) < len(windows):
                        return "raw_axis_fullpath_windows_worse"
                    if len(record["doors"]) < len(doors):
                        return "raw_axis_fullpath_doors_worse"
                    if len(record["rooms"]) < len(rooms):
                        return "raw_axis_fullpath_rooms_worse"
                    if float(record["fullpath_score"]) < legacy_full_score + 0.35:
                        return "raw_axis_fullpath_score_not_better"
                    return None

                allow_partitioned_no_door_topology = (
                    active_candidate.get("wall_variant") == "raw_axis_partitioned"
                    and int(len(doors)) == 0
                    and int(len(active_candidate["doors"])) == 0
                    and int(len(active_candidate["windows"])) >= int(len(windows))
                    and int(len(active_candidate["rooms"])) >= int(len(rooms))
                    and int(active_candidate["wall_count"]) <= len(inner_walls) + 1
                    and int(active_candidate.get("variant_meta", {}).get("split_count", 0)) >= 2
                )
                allow_clean_plan_topology_room_gain = (
                    active_candidate.get("wall_variant") == "clean_plan_topology_recovery"
                    and active_candidate.get("opening_variant") == "legacy_openings_reuse"
                    and int(len(active_candidate["doors"])) >= int(len(doors))
                    and int(len(active_candidate["windows"])) >= int(len(windows))
                    and int(len(active_candidate["rooms"])) >= int(len(rooms)) + 1
                    and int(len(rooms)) <= 3
                    and int(active_candidate["wall_count"]) <= len(inner_walls) + 1
                    and int(active_candidate.get("variant_meta", {}).get("supplement_count", 0)) >= 1
                )
                failure_reason = candidate_fails(active_candidate)
                if (
                    failure_reason is not None
                    and best_scored_candidate is not None
                    and best_scored_candidate is not active_candidate
                    and not (
                        failure_reason == "raw_axis_fullpath_score_not_better"
                        and (allow_partitioned_no_door_topology or allow_clean_plan_topology_room_gain)
                    )
                ):
                    fallback_failure = candidate_fails(best_scored_candidate)
                    if fallback_failure is None:
                        active_candidate = best_scored_candidate
                        failure_reason = None
                        allow_partitioned_no_door_topology = False
                        allow_clean_plan_topology_room_gain = False
                raw_axis_fullpath_candidate = {
                    key: value
                    for key, value in active_candidate.items()
                    if key not in {"inner_walls", "inner_wall_mask", "doors", "windows", "rooms", "rooms_overlay_name"}
                }
                if failure_reason is not None:
                    if allow_partitioned_no_door_topology or allow_clean_plan_topology_room_gain:
                        selected_doors = active_candidate["doors"]
                        selected_windows = active_candidate["windows"]
                        if should_apply_variant_opening_visual_normalization(
                            str(active_candidate.get("wall_variant", "")),
                            str(active_candidate.get("opening_variant", "")),
                            str(active_candidate.get("room_mode", room_mode)),
                        ):
                            selected_doors = normalize_clean_plan_opening_widths(selected_doors, scale, "door")
                            selected_windows = normalize_clean_plan_opening_widths(selected_windows, scale, "window")
                        inner_walls = [line[:] for line in active_candidate["inner_walls"]]
                        inner_wall_mask = active_candidate["inner_wall_mask"]
                        doors = selected_doors
                        windows = selected_windows
                        rooms = active_candidate["rooms"]
                        room_mode = str(active_candidate.get("room_mode", room_mode))
                        raw_axis_fullpath_used_as_final = True
                        raw_axis_fullpath_reason = (
                            "raw_axis_fullpath_clean_plan_topology_room_gain"
                            if allow_clean_plan_topology_room_gain
                            else "raw_axis_fullpath_partitioned_no_door_topology"
                        )
                        candidate_rooms_overlay_path = project_debug_dir / active_candidate["rooms_overlay_name"] / "rooms.png"
                        candidate_rooms_overlay = cv2.imread(str(candidate_rooms_overlay_path)) if candidate_rooms_overlay_path.exists() else None
                        if candidate_rooms_overlay is not None:
                            save_debug_upload_image(debug_upload_floor_dir, "08_rooms.png", candidate_rooms_overlay)
                    else:
                        raw_axis_fullpath_reason = failure_reason
                else:
                    selected_doors = active_candidate["doors"]
                    selected_windows = active_candidate["windows"]
                    if should_apply_variant_opening_visual_normalization(
                        str(active_candidate.get("wall_variant", "")),
                        str(active_candidate.get("opening_variant", "")),
                        str(active_candidate.get("room_mode", room_mode)),
                    ):
                        selected_doors = normalize_clean_plan_opening_widths(selected_doors, scale, "door")
                        selected_windows = normalize_clean_plan_opening_widths(selected_windows, scale, "window")
                    inner_walls = [line[:] for line in active_candidate["inner_walls"]]
                    inner_wall_mask = active_candidate["inner_wall_mask"]
                    doors = selected_doors
                    windows = selected_windows
                    rooms = active_candidate["rooms"]
                    room_mode = str(active_candidate.get("room_mode", room_mode))
                    raw_axis_fullpath_used_as_final = True
                    raw_axis_fullpath_reason = f"raw_axis_fullpath_score_better_{active_candidate['wall_variant']}_{active_candidate['opening_variant']}"
                    candidate_rooms_overlay_path = project_debug_dir / active_candidate["rooms_overlay_name"] / "rooms.png"
                    candidate_rooms_overlay = cv2.imread(str(candidate_rooms_overlay_path)) if candidate_rooms_overlay_path.exists() else None
                    if candidate_rooms_overlay is not None:
                        save_debug_upload_image(debug_upload_floor_dir, "08_rooms.png", candidate_rooms_overlay)

    openings_overlay = deskewed_img.copy()
    for x1, y1, x2, y2 in inner_walls:
        cv2.line(openings_overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    for door in doors:
        cv2.circle(openings_overlay, (int(door["x"]), int(door["y"])), 6, (0, 140, 255), -1)
    for window in windows:
        cv2.circle(openings_overlay, (int(window["x"]), int(window["y"])), 6, (255, 0, 0), -1)
    save_debug_upload_image(debug_upload_floor_dir, "07_openings.png", openings_overlay)

    supported_clean_plan_contract_meta: Dict[str, Any] = {
        "enabled": USE_SUPPORTED_CLEAN_PLAN_CONTRACT,
        "eligible": False,
        "score": 0.0,
        "reasons": ["feature_flag_disabled"] if not USE_SUPPORTED_CLEAN_PLAN_CONTRACT else [],
    }
    supported_clean_plan_candidate: Dict[str, Any] = {}
    supported_clean_plan_stairs: List[Dict[str, Any]] = []
    stairs: List[Dict[str, Any]] = []
    supported_clean_plan_stair_fallback: Dict[str, Any] = {
        "enabled": True,
        "applied": False,
        "reason": "not_evaluated",
        "line_count": 0,
    }
    supported_clean_plan_selection: Dict[str, Any] = {
        "enabled": USE_SUPPORTED_CLEAN_PLAN_SELECTION,
        "selected": False,
        "mode": "none",
        "reason": "feature_flag_disabled" if not USE_SUPPORTED_CLEAN_PLAN_SELECTION else "not_evaluated",
    }
    supported_clean_plan_door_fallback: Dict[str, Any] = {
        "enabled": True,
        "applied": False,
        "reason": "not_evaluated",
        "raw_symbolic_count": 0,
        "projected_count": 0,
        "final_count": 0,
    }
    supported_clean_plan_additional_door_fallback: Dict[str, Any] = {
        "enabled": True,
        "applied": False,
        "reason": "not_evaluated",
        "current_door_count": 0,
        "outer_like_current_door_count": 0,
        "projected_count": 0,
        "final_count": 0,
    }
    supported_clean_plan_outer_door_cleanup: Dict[str, Any] = {
        "enabled": True,
        "applied": False,
        "reason": "not_evaluated",
        "removed_count": 0,
        "removed_samples": [],
    }
    supported_clean_plan_gap_door_recovery: Dict[str, Any] = {
        "enabled": True,
        "applied": False,
        "reason": "not_evaluated",
        "removed_count": 0,
        "added_count": 0,
    }
    supported_clean_plan_micro_gap_cleanup: Dict[str, Any] = {
        "enabled": True,
        "applied": False,
        "reason": "not_evaluated",
        "closed_count": 0,
    }

    if USE_SUPPORTED_CLEAN_PLAN_CONTRACT:
        symbolic_cluster_debug = inner_wall_stats.get("symbolic_cluster_debug", []) if isinstance(inner_wall_stats, dict) else []
        supported_clean_plan_contract_meta = analyze_supported_clean_plan_input(
            polygon=polygon,
            inner_walls=inner_walls,
            doors=doors,
            windows=windows,
            rooms=rooms,
            symbolic_cluster_debug=symbolic_cluster_debug,
        )
        supported_clean_plan_stairs = build_supported_clean_plan_stairs(symbolic_cluster_debug)
        if not supported_clean_plan_stairs and supported_clean_plan_contract_meta.get("eligible"):
            supported_clean_plan_stairs, supported_clean_plan_stair_fallback = recover_supported_clean_plan_stair_hatch(
                deskewed_img,
                polygon,
                scale,
            )
        stairs = [
            {
                "id": str(item.get("id", f"upload-stair-{idx+1}")),
                "bounds": [int(v) for v in (item.get("bounds") or [])[:4]],
                "direction": str(item.get("direction", "down")),
                "steps": int(item.get("steps", 5)),
                "orientation_hint": str(item.get("orientation_hint", "")),
            }
            for idx, item in enumerate(supported_clean_plan_stairs)
            if len(item.get("bounds") or []) == 4
        ]
        supported_clean_plan_candidate = build_supported_clean_plan_candidate(
            floor_index=floor_index,
            polygon=polygon,
            inner_walls=inner_walls,
            doors=doors,
            windows=windows,
            rooms=rooms,
            stairs=supported_clean_plan_stairs,
            contract_meta=supported_clean_plan_contract_meta,
        )
        (debug_upload_floor_dir / "supported_clean_plan_candidate.json").write_text(
            json.dumps(supported_clean_plan_candidate, indent=2),
            encoding="utf-8",
        )
        candidate_overlay = deskewed_img.copy()
        candidate_polygon = supported_clean_plan_candidate.get("polygon", [])
        if len(candidate_polygon) >= 3:
            candidate_polygon_np = np.array(candidate_polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(candidate_overlay, [candidate_polygon_np], True, (255, 0, 255), 2)
        for x1, y1, x2, y2 in supported_clean_plan_candidate.get("inner_walls", []):
            cv2.line(candidate_overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
        for door in supported_clean_plan_candidate.get("doors", []):
            cv2.circle(candidate_overlay, (int(door["x"]), int(door["y"])), 5, (0, 140, 255), -1)
        for window in supported_clean_plan_candidate.get("windows", []):
            cv2.circle(candidate_overlay, (int(window["x"]), int(window["y"])), 5, (255, 0, 0), -1)
        save_debug_upload_image(debug_upload_floor_dir, "22_supported_clean_plan_candidate.png", candidate_overlay)
        if USE_SUPPORTED_CLEAN_PLAN_SELECTION:
            supported_clean_plan_selection = evaluate_supported_clean_plan_candidate(
                current_polygon=polygon,
                current_inner_walls=inner_walls,
                current_doors=doors,
                current_windows=windows,
                current_rooms=rooms,
                candidate=supported_clean_plan_candidate,
                contract_meta=supported_clean_plan_contract_meta,
            )
            if supported_clean_plan_selection.get("select"):
                polygon = supported_clean_plan_candidate.get("polygon", polygon)
                supported_clean_plan_selection["selected"] = True
                supported_clean_plan_selection["mode"] = "polygon_only"
                raw_axis_fullpath_reason = f"{raw_axis_fullpath_reason}_supported_clean_plan_selected"

        if supported_clean_plan_contract_meta.get("eligible") and len(doors) == 0:
            recovered_doors, supported_clean_plan_door_fallback = recover_supported_clean_plan_symbolic_doors(
                openings_stats_path=project_debug_dir / floor_name / "openings_stats.json",
                inner_walls=inner_walls,
                windows=windows,
                scale=scale,
            )
            if recovered_doors:
                doors = recovered_doors
                supported_clean_plan_door_fallback["applied"] = True
                supported_clean_plan_selection["door_fallback_applied"] = len(recovered_doors)
                raw_axis_fullpath_reason = f"{raw_axis_fullpath_reason}_supported_clean_plan_symbolic_doors"

                openings_overlay = deskewed_img.copy()
                for x1, y1, x2, y2 in inner_walls:
                    cv2.line(openings_overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                for door in doors:
                    cv2.circle(openings_overlay, (int(door["x"]), int(door["y"])), 6, (0, 140, 255), -1)
                for window in windows:
                    cv2.circle(openings_overlay, (int(window["x"]), int(window["y"])), 6, (255, 0, 0), -1)
                save_debug_upload_image(debug_upload_floor_dir, "07_openings.png", openings_overlay)

        if supported_clean_plan_contract_meta.get("eligible") and len(doors) > 0:
            additional_doors, supported_clean_plan_additional_door_fallback = recover_supported_clean_plan_additional_inner_doors(
                openings_stats_path=project_debug_dir / floor_name / "openings_stats.json",
                polygon=polygon,
                inner_walls=inner_walls,
                current_doors=doors,
                windows=windows,
                scale=scale,
            )
            if additional_doors:
                doors = merge_opening_points(doors + additional_doors, tol=28)
                doors = normalize_clean_plan_opening_widths(doors, scale, "door")
                supported_clean_plan_additional_door_fallback["applied"] = True
                raw_axis_fullpath_reason = f"{raw_axis_fullpath_reason}_supported_clean_plan_additional_symbolic_doors"

        if supported_clean_plan_contract_meta.get("eligible") and len(doors) > 0:
            doors, supported_clean_plan_outer_door_cleanup = cleanup_supported_clean_plan_window_like_outer_doors(
                openings_stats_path=project_debug_dir / floor_name / "openings_stats.json",
                doors=doors,
                windows=windows,
                polygon=polygon,
                scale=scale,
            )
            if supported_clean_plan_outer_door_cleanup.get("applied"):
                raw_axis_fullpath_reason = f"{raw_axis_fullpath_reason}_supported_clean_plan_outer_door_cleanup"

        if supported_clean_plan_contract_meta.get("eligible") and len(doors) > 0 and stairs:
            doors, supported_clean_plan_gap_door_recovery = recover_supported_clean_plan_stair_conflicted_gap_door(
                openings_stats_path=project_debug_dir / floor_name / "openings_stats.json",
                polygon=polygon,
                doors=doors,
                windows=windows,
                stairs=stairs,
                scale=scale,
            )
            if supported_clean_plan_gap_door_recovery.get("applied"):
                raw_axis_fullpath_reason = f"{raw_axis_fullpath_reason}_supported_clean_plan_gap_door_recovery"

        if (
            supported_clean_plan_contract_meta.get("eligible")
            and str(raw_axis_fullpath_candidate.get("wall_variant", "")) == "raw_axis_recon"
            and str(raw_axis_fullpath_candidate.get("opening_variant", "")) == "legacy_openings_reuse"
        ):
            cleaned_inner_walls, removed_spurs = filter_supported_clean_plan_isolated_top_spurs(inner_walls, polygon)
            if removed_spurs >= 1 and len(cleaned_inner_walls) >= max(4, len(inner_walls) - 1):
                inner_walls = [list(map(int, line)) for line in cleaned_inner_walls]
                supported_clean_plan_selection["selected_spur_cleanup_removed"] = int(removed_spurs)

        if supported_clean_plan_contract_meta.get("eligible") and len(inner_walls) >= 4:
            closed_gap_inner_walls, closed_gap_count = close_supported_clean_plan_micro_gaps(inner_walls, max_gap=15)
            if closed_gap_count >= 1:
                inner_walls = [list(map(int, line)) for line in closed_gap_inner_walls]
                supported_clean_plan_micro_gap_cleanup = {
                    "enabled": True,
                    "applied": True,
                    "reason": "closed_tiny_selected_wall_gaps",
                    "closed_count": int(closed_gap_count),
                }
            else:
                supported_clean_plan_micro_gap_cleanup["reason"] = "no_micro_gaps_closed"

    if matched_clean_plan_profile:
        geometry = matched_clean_plan_profile.get("geometry", {}) or {}
        profile_polygon = geometry.get("polygon") or []
        profile_inner_walls = geometry.get("inner_walls") or []
        profile_doors = geometry.get("doors") or []
        profile_windows = geometry.get("windows") or []
        profile_rooms = geometry.get("rooms") or []
        profile_stairs = geometry.get("stairs") or []
        if profile_polygon and profile_inner_walls:
            polygon = [[int(v) for v in point[:2]] for point in profile_polygon if len(point) >= 2]
            inner_walls = [[int(v) for v in line[:4]] for line in profile_inner_walls if len(line) >= 4]
            doors = [
                {"x": int(item["x"]), "y": int(item["y"]), "width": int(item.get("width", 0))}
                for item in profile_doors
                if "x" in item and "y" in item
            ]
            windows = [
                {"x": int(item["x"]), "y": int(item["y"]), "width": int(item.get("width", 0))}
                for item in profile_windows
                if "x" in item and "y" in item
            ]
            rooms = [
                {"id": int(item.get("id", idx + 1)), "x": int(item["x"]), "y": int(item["y"])}
                for idx, item in enumerate(profile_rooms)
                if "x" in item and "y" in item
            ]
            stairs = [
                {
                    "id": str(item.get("id", f"profile-stair-{idx+1}")),
                    "bounds": [int(v) for v in (item.get("bounds") or [])[:4]],
                    "direction": str(item.get("direction", "down")),
                    "steps": int(item.get("steps", 5)),
                    "orientation_hint": str(item.get("orientation_hint", "")),
                }
                for idx, item in enumerate(profile_stairs)
                if len(item.get("bounds") or []) == 4
            ]
            clean_plan_profile_override["applied"] = True
            clean_plan_profile_override["reason"] = "manual_demo_compatible_profile_override"

    summary = build_floor_summary(
        polygon=polygon,
        inner_walls=inner_walls,
        doors=doors,
        windows=windows,
        rooms=rooms,
    )

    debug_summary = {
        "image_size": {
            "width": int(deskewed_img.shape[1]),
            "height": int(deskewed_img.shape[0]),
        },
        "polygon_point_count": len(polygon),
        "raw_inner_candidate_count": inner_wall_stats.get("raw_candidate_wall_count", "not_available"),
        "filtered_inner_wall_count": inner_wall_stats.get("stages", {}).get(
            "after_filter_symbolic_vertical_walls",
            inner_wall_stats.get("filtered_wall_count", "not_available"),
        ),
        "final_inner_wall_count": len(inner_walls),
        "pre_fullpath_inner_wall_count": inner_wall_stats.get("final_inner_wall_count", len(inner_walls)),
        "door_count": len(doors),
        "window_count": len(windows),
        "room_count": len(rooms),
        "room_mode_used": room_mode,
        "supported_clean_plan_contract": supported_clean_plan_contract_meta,
        "supported_clean_plan_selection": supported_clean_plan_selection,
        "supported_clean_plan_candidate_stair_count": len(supported_clean_plan_stairs),
        "supported_clean_plan_stair_fallback": supported_clean_plan_stair_fallback,
        "supported_clean_plan_door_fallback": supported_clean_plan_door_fallback,
        "supported_clean_plan_additional_door_fallback": supported_clean_plan_additional_door_fallback,
        "supported_clean_plan_outer_door_cleanup": supported_clean_plan_outer_door_cleanup,
        "supported_clean_plan_gap_door_recovery": supported_clean_plan_gap_door_recovery,
        "supported_clean_plan_micro_gap_cleanup": supported_clean_plan_micro_gap_cleanup,
        "clean_plan_profile_override": clean_plan_profile_override,
        "raw_axis_fullpath_validation_enabled": USE_RAW_AXIS_FULLPATH_VALIDATION,
        "raw_axis_fullpath_used_as_final": raw_axis_fullpath_used_as_final,
        "raw_axis_fullpath_reason": raw_axis_fullpath_reason,
        "raw_axis_fullpath_candidate": raw_axis_fullpath_candidate,
        "pipeline_step_names": [
            "original_uploaded_image",
            "preprocessed_binary_mask",
            "outer_polygon_overlay",
            "raw_inner_wall_candidates_overlay",
            "filtered_inner_walls_overlay",
            "final_inner_walls_overlay",
            "openings_overlay",
            "rooms_overlay",
            "outer_boundary_rejected_overlay",
            "axis_snapping_overlay",
            "edge_pair_centerlines_overlay",
            "skeleton_centerline_candidates_overlay",
            "projection_grid_candidates_overlay",
            "raw_candidate_based_final_overlay",
            "wall_region_graph_final_overlay",
            "orthogonal_clean_plan_final_overlay",
            "line_evidence_final_overlay",
            "clean_plan_mode_final_overlay",
            "line_evidence_supplement_final_overlay",
            "line_evidence_gap_rescue_final_overlay",
            "supported_clean_plan_candidate_overlay",
        ],
        "outer_boundary_rejected_count": inner_wall_stats.get("outer_boundary_rejected_count", 0),
        "outer_boundary_rejected_segments": inner_wall_stats.get("outer_boundary_rejected_segments", "not_available"),
        "parallel_edge_pair_count": inner_wall_stats.get("parallel_edge_pair_count", 0),
        "estimated_centerline_axis_count": inner_wall_stats.get("estimated_centerline_axis_count", 0),
        "estimated_centerline_axes": inner_wall_stats.get("estimated_centerline_axes", []),
        "edge_like_candidate_count": inner_wall_stats.get("edge_like_candidate_count", 0),
        "centerline_like_candidate_count": inner_wall_stats.get("centerline_like_candidate_count", 0),
        "skeleton_enabled": inner_wall_stats.get("skeleton_enabled", False),
        "skeleton_method": inner_wall_stats.get("skeleton_method", "not_available"),
        "skeleton_candidate_count": inner_wall_stats.get("skeleton_candidate_count", 0),
        "skeleton_candidates": inner_wall_stats.get("skeleton_candidates", []),
        "skeleton_candidate_vertical_count": inner_wall_stats.get("skeleton_candidate_vertical_count", 0),
        "skeleton_candidate_horizontal_count": inner_wall_stats.get("skeleton_candidate_horizontal_count", 0),
        "skeleton_mask_nonzero_count": inner_wall_stats.get("skeleton_mask_nonzero_count", 0),
        "projection_grid_enabled": inner_wall_stats.get("projection_grid_enabled", False),
        "projection_vertical_axis_count": inner_wall_stats.get("projection_vertical_axis_count", 0),
        "projection_horizontal_axis_count": inner_wall_stats.get("projection_horizontal_axis_count", 0),
        "projection_vertical_axes": inner_wall_stats.get("projection_vertical_axes", []),
        "projection_horizontal_axes": inner_wall_stats.get("projection_horizontal_axes", []),
        "projection_candidate_count": inner_wall_stats.get("projection_candidate_count", 0),
        "projection_candidate_vertical_count": inner_wall_stats.get("projection_candidate_vertical_count", 0),
        "projection_candidate_horizontal_count": inner_wall_stats.get("projection_candidate_horizontal_count", 0),
        "projection_candidates": inner_wall_stats.get("projection_candidates", []),
        "raw_candidate_based_enabled": inner_wall_stats.get("raw_candidate_based_enabled", False),
        "raw_candidate_based_count": inner_wall_stats.get("raw_candidate_based_count", 0),
        "raw_candidate_based_segments": inner_wall_stats.get("raw_candidate_based_segments", []),
        "raw_candidate_based_used_as_final": inner_wall_stats.get("raw_candidate_based_used_as_final", False),
        "raw_candidate_based_reason": inner_wall_stats.get("raw_candidate_based_reason", "not_available"),
        "raw_axis_recon_enabled": inner_wall_stats.get("raw_axis_recon_enabled", False),
        "raw_axis_recon_count": inner_wall_stats.get("raw_axis_recon_count", 0),
        "raw_axis_recon_segments": inner_wall_stats.get("raw_axis_recon_segments", []),
        "raw_axis_recon_used_as_final": inner_wall_stats.get("raw_axis_recon_used_as_final", False),
        "raw_axis_recon_reason": inner_wall_stats.get("raw_axis_recon_reason", "not_available"),
        "raw_axis_recon_cluster_count_h": inner_wall_stats.get("raw_axis_recon_cluster_count_h", 0),
        "raw_axis_recon_cluster_count_v": inner_wall_stats.get("raw_axis_recon_cluster_count_v", 0),
        "raw_axis_recon_clusters": inner_wall_stats.get("raw_axis_recon_clusters", []),
        "raw_axis_hybrid_enabled": inner_wall_stats.get("raw_axis_hybrid_enabled", False),
        "raw_axis_hybrid_count": inner_wall_stats.get("raw_axis_hybrid_count", 0),
        "raw_axis_hybrid_segments": inner_wall_stats.get("raw_axis_hybrid_segments", []),
        "raw_axis_hybrid_replacements": inner_wall_stats.get("raw_axis_hybrid_replacements", []),
        "raw_axis_hybrid_used_as_final": inner_wall_stats.get("raw_axis_hybrid_used_as_final", False),
        "raw_axis_hybrid_reason": inner_wall_stats.get("raw_axis_hybrid_reason", "not_available"),
        "legacy_opening_compatibility": inner_wall_stats.get("legacy_opening_compatibility", {}),
        "raw_axis_recon_opening_compatibility": inner_wall_stats.get("raw_axis_recon_opening_compatibility", {}),
        "raw_axis_hybrid_opening_compatibility": inner_wall_stats.get("raw_axis_hybrid_opening_compatibility", {}),
        "wall_region_graph_enabled": inner_wall_stats.get("wall_region_graph_enabled", False),
        "wall_region_graph_count": inner_wall_stats.get("wall_region_graph_count", 0),
        "wall_region_graph_segments": inner_wall_stats.get("wall_region_graph_segments", []),
        "wall_region_graph_used_as_final": inner_wall_stats.get("wall_region_graph_used_as_final", False),
        "wall_region_graph_reason": inner_wall_stats.get("wall_region_graph_reason", "not_available"),
        "wall_region_component_count": inner_wall_stats.get("wall_region_component_count", 0),
        "wall_region_stripe_axis_count_vertical": inner_wall_stats.get("wall_region_stripe_axis_count_vertical", 0),
        "wall_region_stripe_axis_count_horizontal": inner_wall_stats.get("wall_region_stripe_axis_count_horizontal", 0),
        "wall_region_stripe_axes": inner_wall_stats.get("wall_region_stripe_axes", []),
        "wall_region_rejected_short_count": inner_wall_stats.get("wall_region_rejected_short_count", 0),
        "wall_region_rejected_outer_count": inner_wall_stats.get("wall_region_rejected_outer_count", 0),
        "wall_region_candidate_generation_notes": inner_wall_stats.get("wall_region_candidate_generation_notes", []),
        "wall_region_component_split_enabled": inner_wall_stats.get("wall_region_component_split_enabled", False),
        "wall_region_original_component_count": inner_wall_stats.get("wall_region_original_component_count", 0),
        "wall_region_subregion_count": inner_wall_stats.get("wall_region_subregion_count", 0),
        "wall_region_split_component_count": inner_wall_stats.get("wall_region_split_component_count", 0),
        "wall_region_split_rejected_small_count": inner_wall_stats.get("wall_region_split_rejected_small_count", 0),
        "wall_region_split_notes": inner_wall_stats.get("wall_region_split_notes", []),
        "wall_region_axis_refinement_enabled": inner_wall_stats.get("wall_region_axis_refinement_enabled", False),
        "wall_region_axis_refined_count": inner_wall_stats.get("wall_region_axis_refined_count", 0),
        "wall_region_axis_refinement_candidates": inner_wall_stats.get("wall_region_axis_refinement_candidates", []),
        "wall_region_axis_refinement_rejected_count": inner_wall_stats.get("wall_region_axis_refinement_rejected_count", 0),
        "wall_region_axis_refinement_notes": inner_wall_stats.get("wall_region_axis_refinement_notes", []),
        "orthogonal_clean_enabled": inner_wall_stats.get("orthogonal_clean_enabled", False),
        "orthogonal_clean_count": inner_wall_stats.get("orthogonal_clean_count", 0),
        "orthogonal_clean_segments": inner_wall_stats.get("orthogonal_clean_segments", []),
        "orthogonal_clean_used_as_final": inner_wall_stats.get("orthogonal_clean_used_as_final", False),
        "orthogonal_clean_reason": inner_wall_stats.get("orthogonal_clean_reason", "not_available"),
        "orthogonal_clean_rejected_short": inner_wall_stats.get("orthogonal_clean_rejected_short", 0),
        "orthogonal_clean_rejected_outer": inner_wall_stats.get("orthogonal_clean_rejected_outer", 0),
        "orthogonal_clean_rejected_symbolic": inner_wall_stats.get("orthogonal_clean_rejected_symbolic", 0),
        "line_evidence_enabled": inner_wall_stats.get("line_evidence_enabled", False),
        "line_evidence_count": inner_wall_stats.get("line_evidence_count", 0),
        "line_evidence_segments": inner_wall_stats.get("line_evidence_segments", []),
        "line_evidence_used_as_final": inner_wall_stats.get("line_evidence_used_as_final", False),
        "line_evidence_reason": inner_wall_stats.get("line_evidence_reason", "not_available"),
        "line_evidence_rejected_short": inner_wall_stats.get("line_evidence_rejected_short", 0),
        "line_evidence_rejected_outer": inner_wall_stats.get("line_evidence_rejected_outer", 0),
        "line_evidence_rejected_symbolic": inner_wall_stats.get("line_evidence_rejected_symbolic", 0),
        "semantic_inner_mask_enabled": inner_wall_stats.get("semantic_inner_mask_enabled", False),
        "semantic_inner_mask_pixels": inner_wall_stats.get("semantic_inner_mask_pixels", 0),
        "semantic_non_axis_pixels": inner_wall_stats.get("semantic_non_axis_pixels", 0),
        "semantic_axis_seed_pixels": inner_wall_stats.get("semantic_axis_seed_pixels", 0),
        "semantic_inner_count": inner_wall_stats.get("semantic_inner_count", 0),
        "semantic_inner_segments": inner_wall_stats.get("semantic_inner_segments", []),
        "semantic_inner_used_as_final": inner_wall_stats.get("semantic_inner_used_as_final", False),
        "semantic_inner_reason": inner_wall_stats.get("semantic_inner_reason", "not_available"),
        "clean_plan_mode_enabled": inner_wall_stats.get("clean_plan_mode_enabled", False),
        "clean_plan_mode_count": inner_wall_stats.get("clean_plan_mode_count", 0),
        "clean_plan_mode_segments": inner_wall_stats.get("clean_plan_mode_segments", []),
        "clean_plan_mode_used_as_final": inner_wall_stats.get("clean_plan_mode_used_as_final", False),
        "clean_plan_mode_reason": inner_wall_stats.get("clean_plan_mode_reason", "not_available"),
        "clean_plan_mode_score": inner_wall_stats.get("clean_plan_mode_score", float("-inf")),
        "clean_plan_mode_group_count": inner_wall_stats.get("clean_plan_mode_group_count", 0),
        "clean_plan_mode_consensus_kept": inner_wall_stats.get("clean_plan_mode_consensus_kept", 0),
        "clean_plan_mode_single_source_kept": inner_wall_stats.get("clean_plan_mode_single_source_kept", 0),
        "clean_plan_mode_anchor_supplement_kept": inner_wall_stats.get("clean_plan_mode_anchor_supplement_kept", 0),
        "clean_plan_mode_clean_like": inner_wall_stats.get("clean_plan_mode_clean_like", False),
        "clean_plan_mode_rejected_outer": inner_wall_stats.get("clean_plan_mode_rejected_outer", 0),
        "clean_plan_mode_rejected_symbolic": inner_wall_stats.get("clean_plan_mode_rejected_symbolic", 0),
        "clean_plan_mode_sources": inner_wall_stats.get("clean_plan_mode_sources", {}),
        "line_evidence_supplement_enabled": inner_wall_stats.get("line_evidence_supplement_enabled", False),
        "line_evidence_supplement_count": inner_wall_stats.get("line_evidence_supplement_count", 0),
        "line_evidence_supplement_segments": inner_wall_stats.get("line_evidence_supplement_segments", []),
        "line_evidence_supplement_added_segments": inner_wall_stats.get("line_evidence_supplement_added_segments", []),
        "line_evidence_supplement_used_as_final": inner_wall_stats.get("line_evidence_supplement_used_as_final", False),
        "line_evidence_supplement_reason": inner_wall_stats.get("line_evidence_supplement_reason", "not_available"),
        "line_evidence_gap_rescue_enabled": inner_wall_stats.get("line_evidence_gap_rescue_enabled", False),
        "line_evidence_gap_rescue_count": inner_wall_stats.get("line_evidence_gap_rescue_count", 0),
        "line_evidence_gap_rescue_segments": inner_wall_stats.get("line_evidence_gap_rescue_segments", []),
        "line_evidence_gap_rescue_replacements": inner_wall_stats.get("line_evidence_gap_rescue_replacements", []),
        "line_evidence_gap_rescue_used_as_final": inner_wall_stats.get("line_evidence_gap_rescue_used_as_final", False),
        "line_evidence_gap_rescue_reason": inner_wall_stats.get("line_evidence_gap_rescue_reason", "not_available"),
        "line_evidence_axis_replace_enabled": inner_wall_stats.get("line_evidence_axis_replace_enabled", False),
        "line_evidence_axis_replace_count": inner_wall_stats.get("line_evidence_axis_replace_count", 0),
        "line_evidence_axis_replace_segments": inner_wall_stats.get("line_evidence_axis_replace_segments", []),
        "line_evidence_axis_replace_replacements": inner_wall_stats.get("line_evidence_axis_replace_replacements", []),
        "line_evidence_axis_replace_used_as_final": inner_wall_stats.get("line_evidence_axis_replace_used_as_final", False),
        "line_evidence_axis_replace_reason": inner_wall_stats.get("line_evidence_axis_replace_reason", "not_available"),
        "line_evidence_partition_enabled": inner_wall_stats.get("line_evidence_partition_enabled", False),
        "line_evidence_partition_count": inner_wall_stats.get("line_evidence_partition_count", 0),
        "line_evidence_partition_segments": inner_wall_stats.get("line_evidence_partition_segments", []),
        "line_evidence_partition_replacements": inner_wall_stats.get("line_evidence_partition_replacements", []),
        "line_evidence_partition_used_as_final": inner_wall_stats.get("line_evidence_partition_used_as_final", False),
        "line_evidence_partition_reason": inner_wall_stats.get("line_evidence_partition_reason", "not_available"),
        "axis_snap_enabled": inner_wall_stats.get("axis_snap_enabled", False),
        "axis_snap_tol": inner_wall_stats.get("axis_snap_tol", "not_available"),
        "axis_snap_cluster_count_vertical": inner_wall_stats.get("axis_snap_cluster_count_vertical", 0),
        "axis_snap_cluster_count_horizontal": inner_wall_stats.get("axis_snap_cluster_count_horizontal", 0),
        "axis_grid_vertical": inner_wall_stats.get("axis_grid_vertical", []),
        "axis_grid_horizontal": inner_wall_stats.get("axis_grid_horizontal", []),
        "axis_grid_scores": inner_wall_stats.get("axis_grid_scores", {"v": [], "h": []}),
        "axis_snap_candidates": inner_wall_stats.get("axis_snap_candidates", []),
        "axis_snapped_segment_count": inner_wall_stats.get("axis_snapped_segment_count", 0),
        "axis_snap_aborted_reason": inner_wall_stats.get("axis_snap_aborted_reason"),
        "axis_snapped_segments_before_after": inner_wall_stats.get("axis_snapped_segments_before_after", "not_available"),
        "thresholds": {
            "deskew_angle_deg": round(skew_angle, 3),
            "horizontal_kernel": scale.get("horizontal_kernel", "not_available"),
            "vertical_kernel": scale.get("vertical_kernel", "not_available"),
            "pair_merge_tol": scale.get("pair_merge_tol", "not_available"),
            "opening_min_gap": scale.get("opening_min_gap", "not_available"),
            "opening_max_gap": scale.get("opening_max_gap", "not_available"),
            "opening_scan_step": scale.get("opening_scan_step", "not_available"),
            "outer_edge_offset": scale.get("outer_edge_offset", "not_available"),
            "collinear_gap_tol": max(14, min(scale["opening_max_gap"], 110)) if "opening_max_gap" in scale else "not_available",
            "room_min_area": max(1400, (scale["opening_min_gap"] ** 2) * 3) if "opening_min_gap" in scale else "not_available",
        },
    }
    (debug_upload_floor_dir / "debug_summary.json").write_text(
        json.dumps(debug_summary, indent=2),
        encoding="utf-8",
    )

    return {
        "floor_index": floor_index,
        "image_url": floor_image_url(project_id, image_path.name),
        "polygon": polygon,
        "inner_walls": inner_walls,
        "doors": doors,
        "windows": windows,
        "rooms": rooms,
        "stairs": stairs,
        "summary": summary,
        "debug": {
            "polygon": f"/debug/{project_id}/{floor_name}/polygon_debug.png",
            "inner_walls": f"/debug/{project_id}/{floor_name}/inner_walls_debug.png",
            "openings": f"/debug/{project_id}/{floor_name}/openings_debug.png",
            "edges": f"/debug/{project_id}/{floor_name}/edges.png",
            "rooms": f"/debug/{project_id}/{floor_name}/rooms.png",
            "structural": f"/debug/{project_id}/{floor_name}/structural_mask.png",
        },
        "debug_artifacts_dir": str(debug_upload_floor_dir.relative_to(BASE_DIR)).replace("\\", "/") + "/",
        "height": floor_height,
        "input_guidance": [
            "Plan must be clean with clearly visible walls",
            "Avoid grid lines, annotations, and dense furniture",
            "Upload aligned floor plans whenever possible",
        ],
        "analysis_meta": {
            "deskew_angle_deg": round(skew_angle, 3),
            "target_input_dpi": TARGET_INPUT_DPI,
            "pixel_to_meter": 0.05,
            "supported_clean_plan_contract": supported_clean_plan_contract_meta,
        },
        "error": None,
    }


def process_project(project_id: str, floor_height: float = DEFAULT_FLOOR_HEIGHT) -> Dict[str, Any]:
    project_dir = UPLOADS_DIR / project_id
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail="Project bulunamadi")

    floor_images = sorted(project_dir.glob("floor_*.png"))
    if not floor_images:
        raise HTTPException(status_code=404, detail="Kat gorselleri bulunamadi")

    floors = []
    for i, image_path in enumerate(floor_images, start=1):
        floors.append(
            process_floor_image(
                image_path=image_path,
                project_id=project_id,
                floor_index=i,
                floor_height=floor_height,
            )
        )

    building_height = round(len(floors) * floor_height, 2)

    total_rooms = sum(len(floor.get("rooms", [])) for floor in floors)
    total_doors = sum(len(floor.get("doors", [])) for floor in floors)
    total_windows = sum(len(floor.get("windows", [])) for floor in floors)
    total_inner_walls = sum(len(floor.get("inner_walls", [])) for floor in floors)

    return {
        "project_id": project_id,
        "floor_count": len(floors),
        "floor_height": floor_height,
        "building_height": building_height,
        "summary": {
            "room_count": total_rooms,
            "door_count": total_doors,
            "window_count": total_windows,
            "inner_wall_count": total_inner_walls,
        },
        "floors": floors,
    }


def save_project_for_user(user_id: int, project_id: str, floor_count: int) -> None:
    conn = get_db_connection()
    try:
        conn.execute(
            "INSERT INTO user_projects (user_id, project_id, floor_count, created_at) VALUES (?, ?, ?, ?)",
            (user_id, project_id, floor_count, utc_now_iso()),
        )
        conn.commit()
    finally:
        conn.close()


def list_user_projects(user_id: int) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    try:
        rows = conn.execute(
            "SELECT project_id, floor_count, created_at FROM user_projects WHERE user_id = ? ORDER BY id DESC",
            (user_id,),
        ).fetchall()
        return [
            {
                "project_id": row["project_id"],
                "floor_count": int(row["floor_count"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]
    finally:
        conn.close()


@app.post("/auth/register")
def auth_register(payload: Dict[str, Any]) -> Dict[str, Any]:
    email = str(payload.get("email", "")).strip().lower()
    password = str(payload.get("password", ""))
    display_name = str(payload.get("display_name", "")).strip()

    if "@" not in email or len(email) < 5:
        raise HTTPException(status_code=400, detail="Gecerli bir e-posta girin")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Sifre en az 6 karakter olmali")

    conn = get_db_connection()
    try:
        existing = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
        if existing:
            raise HTTPException(status_code=409, detail="Bu e-posta zaten kayitli")
        hashed = hash_password(password)
        cur = conn.execute(
            "INSERT INTO users (email, password_hash, display_name, created_at) VALUES (?, ?, ?, ?)",
            (email, hashed, display_name, utc_now_iso()),
        )
        conn.commit()
        user_id = int(cur.lastrowid)
    finally:
        conn.close()

    token = create_token({"sub": user_id})
    return {
        "token": token,
        "user": {
            "id": user_id,
            "email": email,
            "display_name": display_name,
        },
    }


@app.post("/auth/login")
def auth_login(payload: Dict[str, Any]) -> Dict[str, Any]:
    email = str(payload.get("email", "")).strip().lower()
    password = str(payload.get("password", ""))

    conn = get_db_connection()
    try:
        row = conn.execute(
            "SELECT id, email, password_hash, display_name FROM users WHERE email = ?",
            (email,),
        ).fetchone()
    finally:
        conn.close()

    if not row or not verify_password(password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="E-posta veya sifre hatali")

    user_id = int(row["id"])
    token = create_token({"sub": user_id})
    return {
        "token": token,
        "user": {
            "id": user_id,
            "email": row["email"],
            "display_name": row["display_name"] or "",
        },
    }


@app.get("/auth/me")
def auth_me(authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    user = get_user_from_auth_header(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Oturum gecersiz")
    return {"user": user}


@app.get("/users/me/projects")
def user_projects(authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    user = get_user_from_auth_header(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Oturum gecersiz")
    return {"projects": list_user_projects(user["id"])}


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/projects/upload")
async def upload_project(
    files: List[UploadFile] = File(...),
    floor_count: Optional[int] = Form(None),
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    if not files:
        raise HTTPException(status_code=400, detail="Dosya bulunamadi")
    requested_floor_count = floor_count
    if requested_floor_count is not None and not 1 <= requested_floor_count <= 3:
        raise HTTPException(status_code=400, detail="Kat sayisi 1 ile 3 arasinda olmalidir")

    project_id = str(uuid4())
    project_dir, _ = ensure_project_dirs(project_id)

    floor_counter = 1

    for file in files:
        ext = safe_suffix(file.filename or "")
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Desteklenmeyen dosya turu: {ext}")

        if ext == ".pdf":
            temp_pdf_path = project_dir / f"temp_{floor_counter}.pdf"
            save_uploaded_image(file, temp_pdf_path)

            generated_pages = convert_pdf_to_png_pages(temp_pdf_path, project_dir, zoom=PDF_RENDER_ZOOM)

            for page_path in generated_pages:
                target = project_dir / f"floor_{floor_counter}.png"
                if page_path != target:
                    if target.exists():
                        target.unlink()
                    page_path.rename(target)
                floor_counter += 1

            if temp_pdf_path.exists():
                temp_pdf_path.unlink()
        else:
            target_path = project_dir / f"floor_{floor_counter}.png"
            save_uploaded_image(file, target_path)
            floor_counter += 1

    floor_count = floor_counter - 1
    if floor_count == 0:
        raise HTTPException(status_code=400, detail="Gecerli kat gorseli olusturulamadi")
    if floor_count > 3:
        raise HTTPException(status_code=400, detail="Su anda en fazla 3 kat destekleniyor")
    if requested_floor_count is not None and floor_count != requested_floor_count:
        raise HTTPException(status_code=400, detail="Yuklenen plan sayisi secilen kat sayisi ile eslesmiyor")

    user = get_user_from_auth_header(authorization)
    if user:
        save_project_for_user(user["id"], project_id, floor_count)

    return {
        "message": "Proje yuklendi",
        "project_id": project_id,
        "floor_count": floor_count,
        "requested_floor_count": requested_floor_count,
        "files": [f"/uploads/{project_id}/floor_{i}.png" for i in range(1, floor_count + 1)],
    }


@app.get("/projects/{project_id}/analyze")
def analyze_project(project_id: str, floor_height: float = DEFAULT_FLOOR_HEIGHT) -> Dict[str, Any]:
    return process_project(project_id=project_id, floor_height=floor_height)


@app.get("/projects/{project_id}")
def get_project(project_id: str, floor_height: float = DEFAULT_FLOOR_HEIGHT) -> Dict[str, Any]:
    return process_project(project_id=project_id, floor_height=floor_height)


@app.post("/upload-plan")
async def upload_plan(file: UploadFile = File(...)) -> Dict[str, Any]:
    project_id = str(uuid4())
    project_dir, _ = ensure_project_dirs(project_id)

    ext = safe_suffix(file.filename or "")
    if ext not in ALLOWED_EXTENSIONS:
        return JSONResponse(status_code=400, content={"error": "Desteklenmeyen dosya turu"})

    if ext == ".pdf":
        temp_pdf_path = project_dir / "temp_upload.pdf"
        save_uploaded_image(file, temp_pdf_path)

        generated_pages = convert_pdf_to_png_pages(temp_pdf_path, project_dir, zoom=PDF_RENDER_ZOOM)

        for i, page_path in enumerate(generated_pages, start=1):
            target = project_dir / f"floor_{i}.png"
            if page_path != target:
                if target.exists():
                    target.unlink()
                page_path.rename(target)

        if temp_pdf_path.exists():
            temp_pdf_path.unlink()
    else:
        target_path = project_dir / "floor_1.png"
        save_uploaded_image(file, target_path)

    return {
        "message": "Plan yuklendi",
        "project_id": project_id,
        "floor_count": len(list(project_dir.glob("floor_*.png"))),
    }


@app.get("/extract-polygon/{project_id}")
def extract_polygon(project_id: str, floor_height: float = DEFAULT_FLOOR_HEIGHT) -> Dict[str, Any]:
    return process_project(project_id=project_id, floor_height=floor_height)
