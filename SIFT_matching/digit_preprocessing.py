"""
Crease-gated palm preprocessing for SIFT templates.

Same core path as palm preprocessing, with one RGB-only quality check after resize:
  1) optional decode-orientation rotate (iOS .mov)
  2) resize min-side 256 (keep aspect)
  3) MediaPipe hand landmarks
  4) finger crease detect on RGB  ← contrast/quality gate only ([1, 2, 2] top→bottom × 4)
  5) CLAHE
  6) emboss
  7) upright (wrist → middle finger up)
  8) palm polygon crop + finalize CLAHE

Creases are not cropped or saved — they only decide whether the frame’s
contrast is good enough. Saved templates are palm.png (same as palm pipeline).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

from hand_preprocessing import (
    DEFAULT_MODEL_PATH,
    PreprocessConfig,
    apply_filters,
    crop_palm_polygon,
    ensure_model,
    finalize_palm_crop,
    get_palm_polygon,
    landmarks_to_pixels,
    resize_frame,
    rotate_frame_bgr,
    upright_hand_image,
)

PIPELINE_VERSION = "digit_v3_crease_gate_palm"

# Top → bottom joint boxes: DIP–TIP, PIP–DIP, MCP–PIP
EXPECTED_CREASE_PATTERN = (1, 2, 2)
FINGER_NAMES = ("index", "middle", "ring", "pinky")
FINGER_LANDMARKS = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}


@dataclass
class CreaseConfig:
    blur: int = 3
    canny1: int = 1
    canny2: int = 5
    hough_threshold: int = 3
    max_line_gap: int = 50
    group_y_px: int = 8
    slope_tol: float = 0.2
    box_size: tuple[int, int] = (52, 45)


@dataclass
class DigitCrops:
    """Palm crop (BGR) after crease quality gate + shared palm pipeline."""

    palm: np.ndarray | None = None
    crease_counts: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    frame_passed_gate: bool = False

    def all_present(self) -> bool:
        return self.palm is not None

    def save(self, output_dir: Path) -> bool:
        if self.palm is None:
            return False
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_dir / "palm.png"), self.palm)
        return True


def biased_midpoint(lower: np.ndarray, upper: np.ndarray, bias: float = 0.5) -> tuple[int, int]:
    """Point between two landmarks, biased toward lower."""
    pt = lower.astype(np.float32) * bias + upper.astype(np.float32) * (1.0 - bias)
    return int(pt[0]), int(pt[1])


def detect_horizontal_creases_canny_hough(
    image_bgr: np.ndarray,
    box_center: tuple[int, int],
    box_size: tuple[int, int],
    angle: float,
    config: CreaseConfig,
) -> int:
    """Detect near-horizontal creases in a rotated joint box. Returns count."""
    h, w = image_bgr.shape[:2]
    center_x, center_y = int(box_center[0]), int(box_center[1])
    box_w, box_h = box_size
    half_w, half_h = box_w // 2, box_h // 2

    x1 = max(0, center_x - half_w)
    y1 = max(0, center_y - half_h)
    x2 = min(w, center_x + half_w)
    y2 = min(h, center_y + half_h)
    if x2 <= x1 or y2 <= y1:
        return 0

    roi = image_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return 0

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(roi_gray)
    k = config.blur if config.blur % 2 == 1 else config.blur + 1
    roi_blur = cv2.GaussianBlur(roi_eq, (k, k), 0) if k > 1 else roi_eq

    edges = cv2.Canny(roi_blur, threshold1=config.canny1, threshold2=config.canny2)

    roi_height, roi_width = y2 - y1, x2 - x1
    mask = np.zeros((roi_height, roi_width), dtype=np.uint8)
    roi_center = (half_w, half_h)
    roi_rot_rect = (roi_center, box_size, angle - 90.0)
    roi_box_points = cv2.boxPoints(roi_rot_rect).astype(np.int32)
    roi_box_points[:, 0] = np.clip(roi_box_points[:, 0], 0, roi_width - 1)
    roi_box_points[:, 1] = np.clip(roi_box_points[:, 1], 0, roi_height - 1)
    cv2.fillPoly(mask, [roi_box_points], 255)
    edges = cv2.bitwise_and(edges, mask)

    min_len = max(1, int(0.1 * box_w))
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=config.hough_threshold,
        minLineLength=min_len,
        maxLineGap=config.max_line_gap,
    )
    if lines is None:
        return 0

    horizontal_ys: list[int] = []
    for line in lines:
        x1_line, y1_line, x2_line, y2_line = line[0]
        dx = x2_line - x1_line
        dy = y2_line - y1_line
        if abs(dy) < abs(dx) * config.slope_tol:
            horizontal_ys.append(int((y1_line + y2_line) / 2))

    if not horizontal_ys:
        return 0

    horizontal_ys.sort()
    grouped: list[int] = []
    current = [horizontal_ys[0]]
    for y in horizontal_ys[1:]:
        if abs(y - current[-1]) < config.group_y_px:
            current.append(y)
        else:
            grouped.append(int(np.mean(current)))
            current = [y]
    grouped.append(int(np.mean(current)))
    return len(grouped)


def finger_joint_box_centers(
    pts: np.ndarray,
    mcp_idx: int,
    pip_idx: int,
    dip_idx: int,
    tip_idx: int,
) -> tuple[list[tuple[int, int]], float]:
    """Three joint-box centers top→bottom and finger angle (deg)."""
    mcp = pts[mcp_idx].astype(np.float32)
    pip = pts[pip_idx].astype(np.float32)
    dip = pts[dip_idx].astype(np.float32)
    tip = pts[tip_idx].astype(np.float32)

    dx, dy = tip - mcp
    angle = float(np.degrees(np.arctan2(dy, dx)))
    toward_base = mcp - tip
    base_norm = float(np.linalg.norm(toward_base))

    mcp_pip_lower_mid = biased_midpoint(pip, mcp, bias=0.35)
    pip_dip_mid = ((pip + dip) / 2.0).astype(np.float32)
    pip_dip_lower_mid = (
        int((pip_dip_mid[0] + pip[0]) / 2),
        int((pip_dip_mid[1] + pip[1]) / 2),
    )
    dip_tip_mid = ((dip + tip) / 2.0).astype(np.float32)
    dip_tip_lower_mid = (
        int((dip_tip_mid[0] + dip[0]) / 2),
        int((dip_tip_mid[1] + dip[1]) / 2),
    )

    if base_norm > 1e-6:
        tip_nudge = (toward_base / base_norm) * 10.0
        dip_tip_lower_mid = (
            int(dip_tip_lower_mid[0] + tip_nudge[0]),
            int(dip_tip_lower_mid[1] + tip_nudge[1]),
        )
        if mcp_idx in (9, 13):  # middle / ring: lower middle box toward base
            mid_nudge = (toward_base / base_norm) * 15.0
            pip_dip_lower_mid = (
                int(pip_dip_lower_mid[0] + mid_nudge[0]),
                int(pip_dip_lower_mid[1] + mid_nudge[1]),
            )

    # Top → bottom: tip box, middle box, base box
    centers = [dip_tip_lower_mid, pip_dip_lower_mid, mcp_pip_lower_mid]
    return centers, angle


def count_finger_creases(
    image_bgr: np.ndarray,
    pts: np.ndarray,
    finger_name: str,
    config: CreaseConfig | None = None,
) -> tuple[int, int, int]:
    """Return (top, mid, bottom) crease counts for one finger."""
    cfg = config or CreaseConfig()
    mcp, pip, dip, tip = FINGER_LANDMARKS[finger_name]
    centers, angle = finger_joint_box_centers(pts, mcp, pip, dip, tip)
    counts = [
        detect_horizontal_creases_canny_hough(image_bgr, c, cfg.box_size, angle, cfg)
        for c in centers
    ]
    return int(counts[0]), int(counts[1]), int(counts[2])


def count_all_finger_creases(
    image_bgr: np.ndarray,
    pts: np.ndarray,
    config: CreaseConfig | None = None,
) -> dict[str, tuple[int, int, int]]:
    cfg = config or CreaseConfig()
    return {
        name: count_finger_creases(image_bgr, pts, name, cfg)
        for name in FINGER_NAMES
    }


def passes_crease_gate(
    counts: dict[str, tuple[int, int, int]],
    pattern: tuple[int, int, int] = EXPECTED_CREASE_PATTERN,
) -> bool:
    """True only when every finger matches top→bottom pattern (20 creases total)."""
    if len(counts) != len(FINGER_NAMES):
        return False
    for name in FINGER_NAMES:
        if counts.get(name) != pattern:
            return False
    return True


def total_crease_count(counts: dict[str, tuple[int, int, int]]) -> int:
    return sum(sum(c) for c in counts.values())


def pattern_l1_distance(
    counts: dict[str, tuple[int, int, int]],
    pattern: tuple[int, int, int] = EXPECTED_CREASE_PATTERN,
) -> int:
    """L1 distance from observed per-box counts to the expected [1,2,2]×4 pattern."""
    dist = 0
    for name in FINGER_NAMES:
        observed = counts.get(name)
        if observed is None:
            return 10**9
        dist += sum(abs(int(a) - int(b)) for a, b in zip(observed, pattern))
    return dist


def passes_crease_gate_tolerant(
    counts: dict[str, tuple[int, int, int]],
    *,
    pattern: tuple[int, int, int] = EXPECTED_CREASE_PATTERN,
    max_l1_distance: int = 0,
) -> bool:
    """
    Gate for enrollment/query.

    max_l1_distance=0 → exact [1,2,2] on all four fingers (20 creases).
    Larger values allow small Canny/Hough jitter while still requiring all
    four fingers to be close to the anatomical pattern.
    """
    if len(counts) != len(FINGER_NAMES):
        return False
    return pattern_l1_distance(counts, pattern) <= int(max_l1_distance)


def draw_crease_debug(
    image_bgr: np.ndarray,
    pts: np.ndarray,
    counts: dict[str, tuple[int, int, int]],
    config: CreaseConfig | None = None,
) -> np.ndarray:
    """Overlay joint boxes and crease counts for debugging."""
    cfg = config or CreaseConfig()
    canvas = image_bgr.copy()
    for name in FINGER_NAMES:
        mcp, pip, dip, tip = FINGER_LANDMARKS[name]
        centers, angle = finger_joint_box_centers(pts, mcp, pip, dip, tip)
        finger_counts = counts.get(name, (0, 0, 0))
        for i, (center, count) in enumerate(zip(centers, finger_counts)):
            rot_rect = (center, cfg.box_size, angle - 90.0)
            box = cv2.boxPoints(rot_rect).astype(np.int32)
            ok = count == EXPECTED_CREASE_PATTERN[i]
            color = (0, 200, 0) if ok else (0, 0, 255)
            cv2.polylines(canvas, [box], isClosed=True, color=color, thickness=1)
            cv2.putText(
                canvas,
                str(count),
                (center[0] - 6, center[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )
    return canvas


def extract_digit_crops_with_steps(
    resized_rgb_bgr: np.ndarray,
    pts: np.ndarray,
    preprocess: PreprocessConfig,
    crease: CreaseConfig,
) -> tuple[DigitCrops, dict[str, np.ndarray]]:
    """
    Full path after resize→landmarks.
    Crease gate on RGB (quality/contrast only), then palm crop like hand_preprocessing.
    """
    steps: dict[str, np.ndarray] = {
        "step1_resized": resized_rgb_bgr.copy(),
    }

    counts = count_all_finger_creases(resized_rgb_bgr, pts, crease)
    steps["step2_crease_rgb"] = draw_crease_debug(resized_rgb_bgr, pts, counts, crease)
    gate_ok = passes_crease_gate(counts)

    crops = DigitCrops(crease_counts=counts, frame_passed_gate=gate_ok)

    filtered = apply_filters(resized_rgb_bgr, preprocess)
    steps["step3_clahe_emboss"] = filtered.copy()

    upright, pts_u = upright_hand_image(filtered, pts)
    steps["step4_upright"] = upright.copy()

    polygon = get_palm_polygon(pts_u)
    palm = crop_palm_polygon(upright, polygon)
    if palm is not None:
        palm = finalize_palm_crop(palm, preprocess)
        crops.palm = palm
        steps["step5_palm_crop"] = palm.copy()

    return crops, steps


@dataclass
class DigitPreprocessor:
    """Detect hand, crease quality-gate, return palm crop."""

    landmarker: HandLandmarker
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    crease: CreaseConfig = field(default_factory=CreaseConfig)
    mode: Literal["image", "video", "live_stream"] = "video"
    _frame_index: int = 0
    _timestamp_ms: int = 0

    @classmethod
    def create(
        cls,
        model_path: Path | None = None,
        mode: Literal["image", "video", "live_stream"] = "video",
        preprocess: PreprocessConfig | None = None,
        crease: CreaseConfig | None = None,
    ) -> DigitPreprocessor:
        path = ensure_model(model_path or DEFAULT_MODEL_PATH)
        running_mode = {
            "image": RunningMode.IMAGE,
            "video": RunningMode.VIDEO,
            "live_stream": RunningMode.LIVE_STREAM,
        }[mode]
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=running_mode,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return cls(
            landmarker=HandLandmarker.create_from_options(options),
            preprocess=preprocess or PreprocessConfig(),
            crease=crease or CreaseConfig(),
            mode=mode,
        )

    def close(self) -> None:
        self.landmarker.close()

    def __enter__(self) -> DigitPreprocessor:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def _detect(self, frame_bgr: np.ndarray, fps: float = 30.0) -> list | None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self.mode == "image":
            result = self.landmarker.detect(mp_image)
        elif self.mode == "video":
            timestamp_ms = int(self._frame_index * 1000 / fps)
            result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
            self._frame_index += 1
        else:
            self._timestamp_ms += int(1000 / fps)
            self.landmarker.detect_async(mp_image, self._timestamp_ms)
            return None

        if not result.hand_landmarks:
            return None
        return result.hand_landmarks

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        fps: float = 30.0,
        return_steps: bool = False,
        *,
        max_l1_distance: int | None = None,
    ) -> DigitCrops | tuple[DigitCrops, dict[str, np.ndarray]] | None:
        """
        Resize → MediaPipe → RGB crease quality gate → CLAHE/emboss → upright → palm crop.

        Returns None when no hand. Palm is built when a hand is present.
        frame_passed_gate reflects crease contrast quality ([1,2,2]×4 exact,
        or tolerant if max_l1_distance is set).
        """
        resized = resize_frame(frame_bgr, self.preprocess.frame_min_side)
        landmarks_list = self._detect(resized, fps=fps)
        if not landmarks_list:
            return (DigitCrops(), {}) if return_steps else None

        height, width = resized.shape[:2]
        pts = landmarks_to_pixels(landmarks_list[0], width, height)
        crops, steps = extract_digit_crops_with_steps(
            resized, pts, self.preprocess, self.crease,
        )
        if max_l1_distance is not None:
            crops.frame_passed_gate = passes_crease_gate_tolerant(
                crops.crease_counts, max_l1_distance=max_l1_distance,
            )
        if return_steps:
            return crops, steps
        return crops


def process_bgr_to_digit_crops(
    frame_bgr: np.ndarray,
    model_path: Path | None = None,
    *,
    rotate_deg: int = 90,
    crease: CreaseConfig | None = None,
    max_l1_distance: int = 0,
) -> DigitCrops | None:
    """
    Single entry for reference + query palm templates.

    Same pipeline as enrollment. Crease detection is quality-only;
    saved/returned crop is palm.png. max_l1_distance=0 requires exact
    [1,2,2]×4; use the same value as enrollment (e.g. 4) for query gating.
    """
    frame = rotate_frame_bgr(frame_bgr, rotate_deg)
    path = ensure_model(model_path or DEFAULT_MODEL_PATH)
    with DigitPreprocessor.create(path, mode="image", crease=crease) as preprocessor:
        return preprocessor.process_frame(
            frame, max_l1_distance=max_l1_distance,
        )


def save_pipeline_steps(steps_dir: Path, steps: dict[str, np.ndarray]) -> None:
    steps_dir.mkdir(parents=True, exist_ok=True)
    for name, image in steps.items():
        cv2.imwrite(str(steps_dir / f"{name}.png"), image)
