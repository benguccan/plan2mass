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

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

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


def merge_collinear_lines(lines: List[List[int]], pos_tol: int = 10, gap_tol: int = 16) -> List[List[int]]:
    if not lines:
        return []

    normalized = [normalize_line(line) for line in lines]
    horiz = [x for x in normalized if x["orientation"] == "h"]
    vert = [x for x in normalized if x["orientation"] == "v"]

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

            if same_axis and overlap_or_close:
                last["fixed"] = int(round((last["fixed"] + item["fixed"]) / 2))
                last["start"] = min(last["start"], item["start"])
                last["end"] = max(last["end"], item["end"])
            else:
                merged.append(item.copy())

        return merged

    return [denormalize_line(x) for x in (merge_group(horiz) + merge_group(vert))]


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
) -> List[List[int]]:
    contours, _ = cv2.findContours(oriented_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segments: List[List[int]] = []

    max_wall_thickness = scale["max_wall_thickness"]
    min_wall_thickness = scale["min_wall_thickness"]

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)

        if area < 30:
            continue

        if orientation == "h":
            if w < MIN_WALL_LENGTH_PX or h > max_wall_thickness * 2 or h < min_wall_thickness:
                continue
            y_mid = int(round(y + h / 2))
            line = [x, y_mid, x + w, y_mid]
        else:
            if h < MIN_WALL_LENGTH_PX or w > max_wall_thickness * 2 or w < min_wall_thickness:
                continue
            x_mid = int(round(x + w / 2))
            line = [x_mid, y, x_mid, y + h]

        mx, my = midpoint_of_line(line)
        if not point_inside_polygon((mx, my), polygon_np):
            continue

        refined_segments = split_line_into_supported_runs(
            line,
            orientation=orientation,
            source_mask=source_mask,
            scale=scale,
        )

        for refined_line in refined_segments or [line]:
            rmx, rmy = midpoint_of_line(refined_line)
            if not point_inside_polygon((rmx, rmy), polygon_np):
                continue

            estimated_thickness = estimate_line_thickness(
                refined_line,
                orientation=orientation,
                source_mask=source_mask,
                max_radius=max_wall_thickness,
            )
            if estimated_thickness < min_wall_thickness:
                continue

            segments.append(refined_line)

    return segments


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
) -> List[List[int]]:
    filtered: List[List[int]] = []

    for line in lines:
        mx, my = midpoint_of_line(line)
        signed_dist = cv2.pointPolygonTest(polygon_np, (float(mx), float(my)), True)
        if signed_dist < outer_margin:
            continue
        filtered.append(line)

    return filtered


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
) -> List[List[int]]:
    if not lines:
        return []

    kept: List[List[int]] = []
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

        if length >= medium_len and connections >= 2 and signed_dist >= scale["outer_edge_offset"] * 1.5:
            kept.append(line)
            continue

        if (
            length >= short_len
            and connections >= 2
            and signed_dist >= max(scale["outer_edge_offset"] * 2.0, scale["opening_min_gap"] * 0.8)
        ):
            kept.append(line)
            continue

    return kept


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

    wall_mask = draw_lines_mask(
        mask_shape,
        inner_walls,
        thickness=max(8, scale["max_wall_thickness"]),
    )
    room_mask[wall_mask > 0] = 0

    # Temporarily close door openings so adjacent rooms are less likely to merge.
    if doors:
        door_close_mask = np.zeros(mask_shape, dtype=np.uint8)
        close_thickness = max(scale["max_wall_thickness"] + 6, scale["opening_min_gap"] // 2)
        for door in doors:
            cx, cy = int(door["x"]), int(door["y"])
            cv2.circle(door_close_mask, (cx, cy), close_thickness // 2, 255, -1)

        room_mask = cv2.bitwise_or(room_mask, door_close_mask)
        room_mask = cv2.bitwise_and(room_mask, cv2.fillPoly(np.zeros(mask_shape, dtype=np.uint8), [polygon_np], 255))

    room_mask = cv2.morphologyEx(
        room_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, scale["opening_min_gap"] // 3), max(5, scale["opening_min_gap"] // 3))),
        iterations=1,
    )
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


def extract_inner_walls(
    binary_mask: np.ndarray,
    polygon_np: np.ndarray,
    debug_img: np.ndarray,
    project_debug_dir: Path,
    floor_name: str,
    scale: Dict[str, int],
) -> Tuple[List[List[int]], np.ndarray]:
    horizontal_mask, vertical_mask, combined_mask, inner_candidates = extract_oriented_wall_masks(
        binary_mask,
        polygon_np,
        scale,
    )

    save_debug(project_debug_dir, floor_name, "inner_candidates.png", inner_candidates)
    save_debug(project_debug_dir, floor_name, "horizontal_walls_mask.png", horizontal_mask)
    save_debug(project_debug_dir, floor_name, "vertical_walls_mask.png", vertical_mask)
    save_debug(project_debug_dir, floor_name, "combined_walls_mask.png", combined_mask)

    horizontal_segments = segments_from_oriented_mask(horizontal_mask, "h", polygon_np, inner_candidates, scale)
    vertical_segments = segments_from_oriented_mask(vertical_mask, "v", polygon_np, inner_candidates, scale)

    wall_segments = horizontal_segments + vertical_segments
    wall_segments = collapse_parallel_double_lines(wall_segments, pair_tol=scale["pair_merge_tol"])
    wall_segments = merge_collinear_lines(wall_segments, pos_tol=8, gap_tol=14)
    wall_segments = remove_duplicate_lines(wall_segments, coord_tol=8, length_tol=12)
    wall_segments = filter_lines_inside_building(wall_segments, polygon_np, outer_margin=scale["outer_edge_offset"])
    wall_segments = prune_spurious_inner_walls(wall_segments, polygon_np, scale)
    wall_segments = merge_collinear_lines(wall_segments, pos_tol=8, gap_tol=12)
    wall_segments = remove_duplicate_lines(wall_segments, coord_tol=8, length_tol=12)

    debug = debug_img.copy()
    for x1, y1, x2, y2 in wall_segments:
        cv2.line(debug, (x1, y1), (x2, y2), (0, 0, 255), 2)

    save_debug(project_debug_dir, floor_name, "inner_walls_debug.png", debug)
    return wall_segments, combined_mask


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


def project_openings_to_host_lines(
    points: List[Dict[str, int]],
    lines: List[List[int]],
    max_distance: float,
    endpoint_tol: int = 22,
) -> List[Dict[str, int]]:
    hosted: List[Dict[str, int]] = []

    for point in points:
        px, py = point["x"], point["y"]

        host = find_best_host_line((px, py), lines, max_distance=max_distance)
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
    raw_doors: List[Dict[str, int]] = []
    raw_windows: List[Dict[str, int]] = []
    debug = debug_img.copy()

    for line in inner_walls:
        if line_length(line) < 60:
            continue
        raw_doors.extend(scan_openings_on_line(line, inner_wall_mask, scale))

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
                    raw_doors.append(point)
                else:
                    raw_windows.append(point)

    raw_doors = merge_opening_points(raw_doors, tol=28)
    raw_windows = merge_opening_points(raw_windows, tol=24)

    hosted_doors = project_openings_to_host_lines(
        raw_doors,
        inner_walls,
        max_distance=max(18, scale["max_wall_thickness"] * 2.0),
        endpoint_tol=22,
    )

    supported_hosted_doors: List[Dict[str, int]] = []
    for door in hosted_doors:
        host = find_best_host_line((door["x"], door["y"]), inner_walls, max_distance=max(18, scale["max_wall_thickness"] * 2.0))
        if host is None:
            continue

        if opening_is_supported_by_rooms(
            opening=door,
            host_line=host["line"],
            polygon_np=outer_polygon_contour,
            wall_mask=inner_wall_mask,
            scale=scale,
        ):
            supported_hosted_doors.append(door)
    hosted_doors = supported_hosted_doors

    hosted_windows = project_openings_to_host_lines(
        raw_windows,
        outer_segments,
        max_distance=max(18, scale["max_wall_thickness"] * 2.1),
        endpoint_tol=18,
    )

    hosted_doors = merge_opening_points(hosted_doors, tol=24)
    hosted_windows = merge_opening_points(hosted_windows, tol=24)

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
        hosted_windows = merge_opening_points(hosted_windows + fallback_windows, tol=24)

    cleaned_doors: List[Dict[str, int]] = []
    for door in hosted_doors:
        too_close_to_window = any(
            hypot(door["x"] - win["x"], door["y"] - win["y"]) <= 24
            for win in hosted_windows
        )
        if not too_close_to_window:
            cleaned_doors.append(door)

    for d in cleaned_doors:
        cv2.circle(debug, (d["x"], d["y"]), 6, (0, 140, 255), -1)

    for w in hosted_windows:
        cv2.circle(debug, (w["x"], w["y"]), 6, (255, 0, 0), -1)

    save_debug(project_debug_dir, floor_name, "openings_debug.png", debug)
    return cleaned_doors, hosted_windows


def process_floor_image(
    image_path: Path,
    project_id: str,
    floor_index: int,
    floor_height: float = DEFAULT_FLOOR_HEIGHT,
) -> Dict[str, Any]:
    project_debug_dir = DEBUG_DIR / project_id
    floor_name = f"floor_{floor_index}"

    img = cv2.imread(str(image_path))
    if img is None:
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
            "height": floor_height,
            "error": "Image okunamadi",
        }

    img = normalize_input_image(img)
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
            "height": floor_height,
            "error": "Dis bina konturu bulunamadi",
        }

    inner_walls, inner_wall_mask = extract_inner_walls(
        binary_mask=structural_mask,
        polygon_np=polygon_np,
        debug_img=deskewed_img,
        project_debug_dir=project_debug_dir,
        floor_name=floor_name,
        scale=scale,
    )

    doors, windows = detect_openings(
        inner_walls=inner_walls,
        outer_polygon=polygon,
        outer_polygon_contour=polygon_np,
        inner_wall_mask=inner_wall_mask,
        binary_mask=binary_mask,
        structural_mask=structural_mask,
        floor_index=floor_index,
        total_floors=len(sorted((UPLOADS_DIR / project_id).glob("floor_*.png"))),
        debug_img=deskewed_img,
        project_debug_dir=project_debug_dir,
        floor_name=floor_name,
        scale=scale,
    )

    rooms = estimate_rooms(
        polygon_np=polygon_np,
        inner_walls=inner_walls,
        doors=doors,
        mask_shape=structural_mask.shape[:2],
        scale=scale,
        project_debug_dir=project_debug_dir,
        floor_name=floor_name,
    )

    summary = build_floor_summary(
        polygon=polygon,
        inner_walls=inner_walls,
        doors=doors,
        windows=windows,
        rooms=rooms,
    )

    return {
        "floor_index": floor_index,
        "image_url": floor_image_url(project_id, image_path.name),
        "polygon": polygon,
        "inner_walls": inner_walls,
        "doors": doors,
        "windows": windows,
        "rooms": rooms,
        "summary": summary,
        "debug": {
            "polygon": f"/debug/{project_id}/{floor_name}/polygon_debug.png",
            "inner_walls": f"/debug/{project_id}/{floor_name}/inner_walls_debug.png",
            "openings": f"/debug/{project_id}/{floor_name}/openings_debug.png",
            "edges": f"/debug/{project_id}/{floor_name}/edges.png",
            "rooms": f"/debug/{project_id}/{floor_name}/rooms.png",
            "structural": f"/debug/{project_id}/{floor_name}/structural_mask.png",
        },
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
