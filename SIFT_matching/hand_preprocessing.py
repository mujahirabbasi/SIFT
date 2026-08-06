"""
Shared palm preprocessing for signup, backfill, and sign-in (one pipeline).

Pipeline per frame (PIPELINE_VERSION):
  1) optional decode-orientation rotate (iOS .mov)
  2) resize min-side 256 (keep aspect)
  3) MediaPipe hand landmarks
  4) CLAHE contrast normalize (so flat vs punchy videos match)
  5) emboss
  6) upright (wrist → middle finger up)
  7) palm polygon crop
  8) CLAHE on crop (consistent saved palm.png for query + reference)

Import process_bgr_to_hand_crops() from app code — do not fork this path.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

# Bump when enroll/backfill must be re-run for matching compatibility
PIPELINE_VERSION = "palm_v3_clahe_emboss_upright"

CROP_NAMES = ("palm",)
FRAME_MIN_SIDE = 256

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"


@dataclass
class PreprocessConfig:
    frame_min_side: int = FRAME_MIN_SIDE
    clahe_clip_limit: float = 3.0
    clahe_tile_size: int = 8


@dataclass
class HandCrops:
    """Palm crop (BGR) after the shared pipeline above."""

    palm: np.ndarray | None = None

    def save(self, output_dir: Path) -> bool:
        if self.palm is None:
            return False
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_dir / "palm.png"), self.palm)
        return True


def ensure_model(model_path: Path = DEFAULT_MODEL_PATH) -> Path:
    if model_path.exists():
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hand landmarker model -> {model_path}")
    urllib.request.urlretrieve(MODEL_URL, model_path)
    return model_path


def resize_frame(image: np.ndarray, min_side: int) -> np.ndarray:
    """Resize so the shorter side equals min_side; aspect ratio is preserved."""
    h, w = image.shape[:2]
    if h <= w:
        new_h = min_side
        new_w = int(round(w * min_side / h))
    else:
        new_w = min_side
        new_h = int(round(h * min_side / w))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def rotate_frame_bgr(frame_bgr: np.ndarray, rotate_deg: int) -> np.ndarray:
    """Discrete rotation for OpenCV-decoded iOS videos (same for enroll + infer)."""
    deg = int(rotate_deg) % 360
    if deg == 0:
        return frame_bgr
    if deg == 90:
        return cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(frame_bgr, cv2.ROTATE_180)
    if deg == 270:
        return cv2.rotate(frame_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    h, w = frame_bgr.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -float(rotate_deg), 1.0)
    return cv2.warpAffine(
        frame_bgr,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def landmarks_to_pixels(landmarks, width: int, height: int) -> np.ndarray:
    pts = np.zeros((21, 2), dtype=np.float32)
    for i, lm in enumerate(landmarks):
        pts[i, 0] = lm.x * width
        pts[i, 1] = lm.y * height
    return pts


def get_palm_polygon(pts: np.ndarray) -> np.ndarray:
    """Palm polygon (same as process_frame.py)."""
    thumb_mid = (pts[1] + pts[2]) / 2.0
    return np.array(
        [thumb_mid, pts[5], pts[9], pts[13], pts[17], pts[0]],
        dtype=np.float32,
    )


def crop_palm_polygon(image_bgr: np.ndarray, polygon: np.ndarray) -> np.ndarray | None:
    h, w = image_bgr.shape[:2]
    poly_int = np.round(polygon).astype(np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly_int], 255)

    bgr = image_bgr.copy()
    bgr[mask == 0] = 0

    x1 = max(0, int(poly_int[:, 0].min()))
    y1 = max(0, int(poly_int[:, 1].min()))
    x2 = min(w, int(poly_int[:, 0].max()))
    y2 = min(h, int(poly_int[:, 1].max()))

    if x2 <= x1 or y2 <= y1:
        return None

    return bgr[y1:y2, x1:x2].copy()


def apply_clahe_bgr(
    image_bgr: np.ndarray,
    *,
    clip_limit: float = 3.0,
    tile_size: int = 8,
) -> np.ndarray:
    """Normalize contrast so flat registration frames match punchy sign-in clips."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    lightness, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(tile_size), int(tile_size)),
    )
    merged = cv2.merge([clahe.apply(lightness), a_ch, b_ch])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def apply_emboss(image_bgr: np.ndarray) -> np.ndarray:
    """Emboss with float filtering (no blur) so palm detail is preserved."""
    kernel = np.array(
        [[-2, -1, 0],
         [-1,  0, 1],
         [ 0,  1, 2]],
        dtype=np.float32,
    )

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    embossed = cv2.filter2D(
        gray.astype(np.float32),
        cv2.CV_32F,
        kernel,
    )

    embossed = np.clip(
        embossed + 128.0,
        0,
        255,
    ).astype(np.uint8)

    return cv2.cvtColor(embossed, cv2.COLOR_GRAY2BGR)


def apply_filters(image_bgr: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    """CLAHE then emboss — shared by enroll / backfill / sign-in."""
    normalized = apply_clahe_bgr(
        image_bgr,
        clip_limit=config.clahe_clip_limit,
        tile_size=config.clahe_tile_size,
    )
    return apply_emboss(normalized)


def finalize_palm_crop(crop_bgr: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    """Final CLAHE on the palm crop so saved query/reference PNGs match visually."""
    return apply_clahe_bgr(
        crop_bgr,
        clip_limit=config.clahe_clip_limit,
        tile_size=max(2, config.clahe_tile_size // 2),
    )


def draw_landmarks_and_palm(
    image_bgr: np.ndarray,
    pts: np.ndarray,
) -> np.ndarray:
    """Draw all 21 landmarks and the palm polygon on a copy of the image."""
    canvas = image_bgr.copy()
    polygon = get_palm_polygon(pts)
    poly_int = np.round(polygon).astype(np.int32)
    cv2.polylines(canvas, [poly_int], isClosed=True, color=(0, 255, 0), thickness=2)
    for i, (x, y) in enumerate(np.round(pts).astype(np.int32)):
        color = (0, 0, 255) if i in (0, 5, 9, 13, 17) else (255, 128, 0)
        cv2.circle(canvas, (int(x), int(y)), 3, color, -1, lineType=cv2.LINE_AA)
    return canvas


def upright_hand_image(
    image_bgr: np.ndarray,
    pts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rotate the frame so wrist → middle-MCP points upward.
    Keeps enroll and sign-in palm crops in a consistent orientation.
    """
    wrist = pts[0]
    middle = pts[9]
    vec = middle - wrist
    angle_deg = float(np.degrees(np.arctan2(vec[0], -vec[1])))
    h, w = image_bgr.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        image_bgr,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_h = np.hstack([pts.astype(np.float32), ones])
    pts_rot = (matrix @ pts_h.T).T.astype(np.float32)
    return rotated, pts_rot


def extract_palm_crop(
    resized: np.ndarray,
    all_pts: np.ndarray,
    config: PreprocessConfig,
) -> np.ndarray | None:
    """CLAHE → emboss → upright → polygon crop → finalize CLAHE."""
    filtered = apply_filters(resized, config)
    upright, pts = upright_hand_image(filtered, all_pts)
    polygon = get_palm_polygon(pts)
    crop = crop_palm_polygon(upright, polygon)
    if crop is None:
        return None
    return finalize_palm_crop(crop, config)


def extract_palm_crop_with_steps(
    resized: np.ndarray,
    all_pts: np.ndarray,
    config: PreprocessConfig,
) -> tuple[np.ndarray | None, dict[str, np.ndarray]]:
    """Debug steps for the shared pipeline after resize→MediaPipe."""
    steps: dict[str, np.ndarray] = {
        "step1_resized": resized.copy(),
        "step2_landmarks": draw_landmarks_and_palm(resized, all_pts),
    }

    normalized = apply_clahe_bgr(
        resized,
        clip_limit=config.clahe_clip_limit,
        tile_size=config.clahe_tile_size,
    )
    steps["step3_clahe"] = normalized.copy()

    filtered = apply_emboss(normalized)
    steps["step4_emboss"] = filtered.copy()

    upright, pts = upright_hand_image(filtered, all_pts)
    steps["step5_upright"] = upright.copy()

    polygon = get_palm_polygon(pts)
    crop = crop_palm_polygon(upright, polygon)
    if crop is None:
        return None, steps

    final = finalize_palm_crop(crop, config)
    steps["step6_palm_crop"] = final.copy()
    return final, steps


def save_pipeline_steps(steps_dir: Path, steps: dict[str, np.ndarray]) -> None:
    steps_dir.mkdir(parents=True, exist_ok=True)
    for name, image in steps.items():
        cv2.imwrite(str(steps_dir / f"{name}.png"), image)


def preprocess_hand_crops(
    resized: np.ndarray,
    hand_landmarks,
    config: PreprocessConfig | None = None,
) -> HandCrops:
    """Landmarks on resized frame → shared palm extract."""
    cfg = config or PreprocessConfig()
    height, width = resized.shape[:2]
    pts = landmarks_to_pixels(hand_landmarks, width, height)
    palm = extract_palm_crop(resized, pts, cfg)
    return HandCrops(palm=palm)


@dataclass
class HandPreprocessor:
    """Detect hand and return palm crop per frame (shared pipeline)."""

    landmarker: HandLandmarker
    config: PreprocessConfig = field(default_factory=PreprocessConfig)
    mode: Literal["image", "video", "live_stream"] = "video"
    _frame_index: int = 0
    _timestamp_ms: int = 0

    @classmethod
    def create(
        cls,
        model_path: Path | None = None,
        mode: Literal["image", "video", "live_stream"] = "video",
        config: PreprocessConfig | None = None,
    ) -> HandPreprocessor:
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
            config=config or PreprocessConfig(),
            mode=mode,
        )

    def close(self) -> None:
        self.landmarker.close()

    def __enter__(self) -> HandPreprocessor:
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
            result = self.landmarker.detect_async(mp_image, self._timestamp_ms)
            return None

        if not result.hand_landmarks:
            return None
        return result.hand_landmarks

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        fps: float = 30.0,
        return_steps: bool = False,
    ) -> HandCrops | tuple[HandCrops | None, dict[str, np.ndarray]] | None:
        """Resize → MediaPipe → CLAHE → emboss → upright → palm crop."""
        resized = resize_frame(frame_bgr, self.config.frame_min_side)
        landmarks_list = self._detect(resized, fps=fps)
        if not landmarks_list:
            return (None, {}) if return_steps else None

        if return_steps:
            height, width = resized.shape[:2]
            pts = landmarks_to_pixels(landmarks_list[0], width, height)
            palm, steps = extract_palm_crop_with_steps(
                resized, pts, self.config,
            )
            return HandCrops(palm=palm), steps
        return preprocess_hand_crops(resized, landmarks_list[0], self.config)

    def process_landmarks(
        self,
        frame_bgr: np.ndarray,
        hand_landmarks,
    ) -> HandCrops:
        resized = resize_frame(frame_bgr, self.config.frame_min_side)
        return preprocess_hand_crops(resized, hand_landmarks, self.config)


def process_bgr_to_hand_crops(
    frame_bgr: np.ndarray,
    model_path: Path | None = None,
    *,
    rotate_deg: int = 90,
) -> HandCrops | None:
    """
    Single entry point for signup, backfill, and sign-in.

    rotate → HandPreprocessor.process_frame (shared palm pipeline).
    """
    frame = rotate_frame_bgr(frame_bgr, rotate_deg)
    path = ensure_model(model_path or DEFAULT_MODEL_PATH)
    with HandPreprocessor.create(path, mode="image") as preprocessor:
        return preprocessor.process_frame(frame)
