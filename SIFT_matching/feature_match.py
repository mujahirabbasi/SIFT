"""
SIFT-match palm crops against stored reference images.

Query and reference use the same crease-gated palm pipeline from
digit_preprocessing (resize → MediaPipe → RGB crease quality gate →
CLAHE/emboss → upright → palm crop).

Videos are read from test_videos/ by default. References default to digit_output/.
Displays scores in real time and plots per-frame score + keypoint counts.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from digit_preprocessing import (
    DEFAULT_MODEL_PATH,
    DigitCrops,
    DigitPreprocessor,
    ensure_model,
    passes_crease_gate_tolerant,
)
from hand_preprocessing import (
    CROP_NAMES,
    FRAME_MIN_SIDE,
    rotate_frame_bgr,
)

VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv", ".webm"}
DEFAULT_VIDEOS_DIR = Path(__file__).resolve().parent.parent / "test_videos"
DEFAULT_REFERENCE_ROOT = Path(__file__).resolve().parent / "digit_output"
WINDOW_NAME = "SIFT Matching"
PRIORITY_REFERENCE_FRAMES = ("frame_40", "frame_41", "frame_42")
MIN_REFERENCE_FRAME = 1
MIN_QUERY_FRAME = 1
MIN_QUERY_SCORED_FRAMES = 3
MAX_QUERY_SCORED_FRAMES = 5
DEFAULT_ROTATE_DEG = 90
DEFAULT_MAX_PATTERN_DISTANCE = 4
DEFAULT_SIFT_NFEATURES = 0
DEFAULT_SIFT_CONTRAST_THRESHOLD = 0.04
DEFAULT_SIFT_EDGE_THRESHOLD = 10.0
DEFAULT_SIFT_N_OCTAVE_LAYERS = 3
DEFAULT_SIFT_SIGMA = 1.6
DEFAULT_MIN_GOOD_MATCHES = 8
DEFAULT_TARGET_GOOD_MATCHES = 15


@dataclass
class MatchResult:
    """SIFT + Lowe-ratio match outcome for one crop pair (no RANSAC)."""

    score: float
    kp_query: list
    kp_ref: list
    good_matches: list

    @property
    def num_good_matches(self) -> int:
        return len(self.good_matches)

    @property
    def num_kp_query(self) -> int:
        return len(self.kp_query)

    @property
    def num_kp_ref(self) -> int:
        return len(self.kp_ref)


@dataclass
class SIFTMatcher:
    """Reusable SIFT descriptor matcher for crop pairs (Lowe ratio only)."""

    ratio_threshold: float = 0.85
    nfeatures: int = DEFAULT_SIFT_NFEATURES
    contrast_threshold: float = DEFAULT_SIFT_CONTRAST_THRESHOLD
    edge_threshold: float = DEFAULT_SIFT_EDGE_THRESHOLD
    n_octave_layers: int = DEFAULT_SIFT_N_OCTAVE_LAYERS
    sigma: float = DEFAULT_SIFT_SIGMA
    min_good_matches: int = DEFAULT_MIN_GOOD_MATCHES
    target_good_matches: int = DEFAULT_TARGET_GOOD_MATCHES
    _sift: cv2.SIFT | None = field(default=None, init=False, repr=False)
    _bf: cv2.BFMatcher = field(default_factory=lambda: cv2.BFMatcher(cv2.NORM_L2))

    def __post_init__(self) -> None:
        self._sift = cv2.SIFT_create(
            nfeatures=self.nfeatures,
            nOctaveLayers=self.n_octave_layers,
            contrastThreshold=self.contrast_threshold,
            edgeThreshold=self.edge_threshold,
            sigma=self.sigma,
        )

    def score(self, image_a: np.ndarray, image_b: np.ndarray) -> float:
        """Compare two BGR crops with SIFT + Lowe ratio test. Score in [0, 1]."""
        return self.match_pair(image_a, image_b).score

    @staticmethod
    def palm_roi_mask(image_bgr: np.ndarray, *, min_value: int = 12) -> np.ndarray:
        """
        Binary mask for the palm region only (ignore black background outside
        the polygon crop so SIFT does not match on empty background).
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        mask = (gray > min_value).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return mask

    def compute_descriptors(
        self, image_bgr: np.ndarray,
    ) -> tuple[list, np.ndarray | None]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        mask = self.palm_roi_mask(image_bgr)
        if cv2.countNonZero(mask) < 16:
            return [], None
        kp, des = self._sift.detectAndCompute(gray, mask)
        return kp or [], des

    def good_matches(
        self,
        kp_a,
        des_a: np.ndarray | None,
        kp_b,
        des_b: np.ndarray | None,
    ) -> list:
        if (
            des_a is None
            or des_b is None
            or len(kp_a) < 2
            or len(kp_b) < 2
        ):
            return []

        pairs = self._bf.knnMatch(des_a, des_b, k=2)
        good = []
        for pair in pairs:
            if len(pair) < 2:
                continue
            match_a, match_b = pair
            if match_a.distance < self.ratio_threshold * match_b.distance:
                good.append(match_a)
        return good

    def score_from_matches(
        self,
        matched_count: int,
        num_kp_a: int,
        num_kp_b: int,
    ) -> float:
        """
        SIFT score = matched keypoints / min(keypoints in A, keypoints in B).
        Clamped to [0, 1].
        """
        denom = min(int(num_kp_a), int(num_kp_b))
        if denom <= 0 or matched_count <= 0:
            return 0.0
        return float(min(1.0, matched_count / denom))

    def match_pair(
        self,
        image_a: np.ndarray,
        image_b: np.ndarray,
    ) -> MatchResult:
        """Return SIFT + Lowe match details (no RANSAC)."""
        kp_a, des_a = self.compute_descriptors(image_a)
        kp_b, des_b = self.compute_descriptors(image_b)
        kp_a = kp_a or []
        kp_b = kp_b or []
        good = self.good_matches(kp_a, des_a, kp_b, des_b)
        return MatchResult(
            score=self.score_from_matches(len(good), len(kp_a), len(kp_b)),
            kp_query=kp_a,
            kp_ref=kp_b,
            good_matches=good,
        )

    def print_match_diagnostics(self, result: MatchResult) -> None:
        print(f"    Query keypoints: {result.num_kp_query}")
        print(f"    Reference keypoints: {result.num_kp_ref}")
        print(f"    Good SIFT matches: {result.num_good_matches}")
        print(f"    Final score: {result.score:.3f}")


@dataclass
class ReferenceFrame:
    name: str
    crops: dict[str, np.ndarray]


@dataclass
class MatchVisualization:
    """Separate query/reference palm images with labeled Lowe-match keypoints."""

    query_image: np.ndarray
    reference_image: np.ndarray
    num_kp_query: int
    num_kp_ref: int
    num_good_matches: int
    score: float
    query_label: str
    reference_label: str

    @property
    def num_matches(self) -> int:
        return self.num_good_matches

    @property
    def side_by_side(self) -> np.ndarray:
        """Clean preview: two labeled images side by side (no crossing lines)."""
        left = self.query_image
        right = self.reference_image
        target_h = max(left.shape[0], right.shape[0])

        def pad_to_height(img: np.ndarray) -> np.ndarray:
            if img.shape[0] == target_h:
                return img
            pad = target_h - img.shape[0]
            return cv2.copyMakeBorder(img, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        return np.hstack([pad_to_height(left), pad_to_height(right)])


@dataclass
class FrameMatchResult:
    frame_index: int
    scores: dict[str, float] = field(default_factory=dict)
    per_crop_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    best_reference: str | None = None
    best_score: float = 0.0
    match_viz: MatchVisualization | None = None

    @property
    def num_kp_query(self) -> int:
        return 0 if self.match_viz is None else self.match_viz.num_kp_query

    @property
    def num_kp_ref(self) -> int:
        return 0 if self.match_viz is None else self.match_viz.num_kp_ref

    @property
    def num_good_matches(self) -> int:
        return 0 if self.match_viz is None else self.match_viz.num_good_matches


@dataclass
class IdentityRunSummary:
    identity: str
    avg_best_score: float
    frame_count: int
    verdict: str
    frame_indices: list[int]
    frame_scores: list[float]
    frame_kp_query: list[int] = field(default_factory=list)
    frame_kp_ref: list[int] = field(default_factory=list)
    frame_good_matches: list[int] = field(default_factory=list)


def list_frame_folders(identity_dir: Path) -> list[Path]:
    def frame_number(path: Path) -> int:
        try:
            return int(path.name.split("_", 1)[1])
        except (IndexError, ValueError):
            return -1

    folders = sorted(
        p for p in identity_dir.iterdir()
        if p.is_dir()
        and p.name.startswith("frame_")
        and frame_number(p) >= MIN_REFERENCE_FRAME
    )
    if not folders:
        raise FileNotFoundError(
            f"No frame_* subfolders >= frame_{MIN_REFERENCE_FRAME} found in {identity_dir}"
        )
    return folders


def load_reference_crops(frame_dir: Path) -> dict[str, np.ndarray]:
    crops: dict[str, np.ndarray] = {}
    for name in CROP_NAMES:
        path = frame_dir / f"{name}.png"
        if not path.exists():
            continue
        image = cv2.imread(str(path))
        if image is not None:
            crops[name] = image
    if not crops:
        raise FileNotFoundError(f"No crop images found in {frame_dir}")
    return crops


def sample_reference_frames(
    identity_dir: Path,
    num_frames: int,
    seed: int | None = None,
) -> list[ReferenceFrame]:
    folders = list_frame_folders(identity_dir)
    folder_map = {folder.name: folder for folder in folders}
    priority = [folder_map[name] for name in PRIORITY_REFERENCE_FRAMES if name in folder_map]
    if len(priority) >= num_frames:
        picked = priority[:num_frames]
        return [ReferenceFrame(name=folder.name, crops=load_reference_crops(folder)) for folder in picked]

    rng = random.Random(seed)
    picked = rng.sample(folders, k=min(num_frames, len(folders)))
    return [
        ReferenceFrame(name=folder.name, crops=load_reference_crops(folder))
        for folder in picked
    ]


VIZ_PANEL_SIZE = 512  # min side of each saved keypoint panel (native crops are ~100px)


def _scale_keypoints(keypoints, scale: float):
    """Return OpenCV keypoints with coordinates scaled for an upscaled image."""
    scaled = []
    for kp in keypoints:
        scaled.append(
            cv2.KeyPoint(
                x=float(kp.pt[0] * scale),
                y=float(kp.pt[1] * scale),
                size=float(kp.size * scale),
                angle=kp.angle,
                response=kp.response,
                octave=kp.octave,
                class_id=kp.class_id,
            )
        )
    return scaled


def upscale_for_viz(image: np.ndarray, target_min_side: int = VIZ_PANEL_SIZE) -> tuple[np.ndarray, float]:
    """Upscale a tiny palm crop so keypoints/labels are readable. Returns (image, scale)."""
    h, w = image.shape[:2]
    min_side = min(h, w)
    if min_side <= 0:
        return image.copy(), 1.0
    if min_side >= target_min_side:
        return image.copy(), 1.0
    scale = target_min_side / min_side
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    up = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return up, scale


def make_labeled_panel(
    image: np.ndarray,
    title: str,
    subtitle: str,
    header_h: int = 78,
) -> np.ndarray:
    """Place image on a wide black canvas with a full-width readable header above it."""
    h, w = image.shape[:2]
    (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    (sw, _), _ = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    canvas_w = max(w + 32, tw + 40, sw + 40, 720)
    canvas = np.zeros((header_h + h + 24, canvas_w, 3), dtype=np.uint8)
    x0 = (canvas_w - w) // 2
    canvas[header_h:header_h + h, x0:x0 + w] = image
    cv2.putText(
        canvas, title, (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, subtitle, (16, 62),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA,
    )
    return canvas


def _draw_numbered_keypoints(
    image_bgr: np.ndarray,
    keypoints,
    match_indices: list[int],
    match_numbers: list[int],
    unmatched_color: tuple[int, int, int] = (255, 255, 255),
    matched_color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """
    Draw all SIFT keypoints (small white circles) and numbered Lowe matches (green),
    matching the saved QUERY/REFERENCE keypoint panel style.
    """
    out = image_bgr.copy()
    matched_set = set(match_indices)

    for i, kp in enumerate(keypoints):
        if i in matched_set:
            continue
        x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
        cv2.circle(out, (x, y), 5, unmatched_color, 1, cv2.LINE_AA)
        cv2.circle(out, (x, y), 1, unmatched_color, -1, cv2.LINE_AA)

    for kp_idx, match_no in zip(match_indices, match_numbers):
        if kp_idx < 0 or kp_idx >= len(keypoints):
            continue
        kp = keypoints[kp_idx]
        x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
        cv2.circle(out, (x, y), 12, matched_color, 2, cv2.LINE_AA)
        cv2.circle(out, (x, y), 2, (255, 255, 255), -1, cv2.LINE_AA)
        label = str(match_no)
        tx, ty = x + 14, y - 14
        cv2.putText(
            out, label, (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 3, cv2.LINE_AA,
        )
        cv2.putText(
            out, label, (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, matched_color, 2, cv2.LINE_AA,
        )
    return out


def build_match_visualization(
    query_bgr: np.ndarray,
    ref_bgr: np.ndarray,
    kp_query,
    kp_ref,
    good_matches: list,
    query_label: str,
    reference_label: str,
    score: float,
    max_matches: int | None = None,
) -> MatchVisualization:
    """
    Build emboss panels with all SIFT keypoints + numbered Lowe good matches.
    No RANSAC — green numbers are Lowe good matches only.
    """
    # Draw all good matches unless capped
    ordered = sorted(good_matches, key=lambda m: m.distance)
    matches_to_draw = ordered if max_matches is None else ordered[:max_matches]
    q_indices = [m.queryIdx for m in matches_to_draw]
    r_indices = [m.trainIdx for m in matches_to_draw]
    numbers = list(range(1, len(matches_to_draw) + 1))

    query_up, q_scale = upscale_for_viz(query_bgr)
    ref_up, r_scale = upscale_for_viz(ref_bgr)
    kp_q_scaled = _scale_keypoints(kp_query, q_scale)
    kp_r_scaled = _scale_keypoints(kp_ref, r_scale)

    query_drawn = _draw_numbered_keypoints(query_up, kp_q_scaled, q_indices, numbers)
    ref_drawn = _draw_numbered_keypoints(ref_up, kp_r_scaled, r_indices, numbers)

    subtitle = (
        f"SIFT keypoints={len(kp_query)}/{len(kp_ref)}   "
        f"Lowe good={len(good_matches)}   "
        f"score={score:.3f}"
    )
    query_image = make_labeled_panel(
        query_drawn,
        title=f"QUERY  |  {query_label}",
        subtitle=subtitle,
    )
    reference_image = make_labeled_panel(
        ref_drawn,
        title=f"REFERENCE  |  {reference_label}",
        subtitle=subtitle,
    )
    return MatchVisualization(
        query_image=query_image,
        reference_image=reference_image,
        num_kp_query=len(kp_query),
        num_kp_ref=len(kp_ref),
        num_good_matches=len(good_matches),
        score=score,
        query_label=query_label,
        reference_label=reference_label,
    )


def match_crops(
    query: DigitCrops,
    reference: dict[str, np.ndarray],
    matcher: SIFTMatcher,
    ref_name: str,
    identity_name: str,
    query_label: str = "query_palm",
) -> tuple[dict[str, float], MatchVisualization | None]:
    scores: dict[str, float] = {}
    viz: MatchVisualization | None = None
    if query.palm is not None and "palm" in reference:
        result = matcher.match_pair(query.palm, reference["palm"])
        scores["palm"] = result.score
        viz = build_match_visualization(
            query.palm,
            reference["palm"],
            result.kp_query,
            result.kp_ref,
            result.good_matches,
            query_label=query_label,
            reference_label=f"{identity_name}/{ref_name}/palm.png",
            score=result.score,
        )
    return scores, viz


def aggregate_score(per_crop_scores: dict[str, float]) -> float:
    if not per_crop_scores:
        return 0.0
    return float(np.mean(list(per_crop_scores.values())))


def match_frame_to_references(
    query: DigitCrops,
    references: list[ReferenceFrame],
    matcher: SIFTMatcher,
    identity_name: str,
    query_label: str = "query_palm",
) -> FrameMatchResult:
    result = FrameMatchResult(frame_index=-1)
    for ref in references:
        crop_scores, crop_viz = match_crops(
            query,
            ref.crops,
            matcher,
            ref.name,
            identity_name,
            query_label=query_label,
        )
        result.per_crop_scores[ref.name] = crop_scores
        result.scores[ref.name] = aggregate_score(crop_scores)
        if result.scores[ref.name] > result.best_score:
            result.best_score = result.scores[ref.name]
            result.best_reference = ref.name
            result.match_viz = crop_viz
    return result


def resize_for_display(image: np.ndarray, max_width: int = 960) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= max_width:
        return image
    scale = max_width / w
    return cv2.resize(image, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)


def resize_to_width(image: np.ndarray, target_width: int) -> np.ndarray:
    """Resize image to an exact display width (up or down)."""
    h, w = image.shape[:2]
    if w == target_width:
        return image
    scale = target_width / w
    return cv2.resize(image, (target_width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


def make_crop_panel(crops: DigitCrops, size: int = FRAME_MIN_SIDE) -> np.ndarray:
    """Show palm crop in the display panel."""
    if crops.palm is None:
        return np.zeros((size, size, 3), dtype=np.uint8)
    return cv2.resize(crops.palm, (size, size), interpolation=cv2.INTER_AREA)


def build_display(
    frame: np.ndarray,
    crops: DigitCrops | None,
    result: FrameMatchResult | None,
    identity: str,
    references: list[ReferenceFrame],
    match_threshold: float,
    frame_idx: int,
) -> np.ndarray:
    display = resize_for_display(frame, max_width=720)
    h, w = display.shape[:2]

    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (w, 130), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, display, 0.45, 0, display)

    ref_names = ", ".join(ref.name for ref in references)
    lines = [
        f"Identity: {identity}",
        f"Frame: {frame_idx + 1}   References: {ref_names}",
    ]

    if result is None or crops is None:
        lines.append("Hand: NOT DETECTED")
        color = (0, 165, 255)
    else:
        verdict = "MATCH" if result.best_score >= match_threshold else "NO MATCH"
        verdict_color = (0, 200, 0) if verdict == "MATCH" else (0, 0, 255)
        lines.append(f"Best ref: {result.best_reference}   Score: {result.best_score:.3f}")
        lines.append(f"Verdict: {verdict}   (threshold {match_threshold:.2f})")
        for i, line in enumerate(lines):
            cv2.putText(
                display, line, (12, 28 + i * 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA,
            )
        cv2.putText(
            display, lines[-1].split("   ")[0], (12, 28 + 3 * 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.62, verdict_color, 2, cv2.LINE_AA,
        )

        if result.best_reference and result.best_reference in result.per_crop_scores:
            crop_scores = result.per_crop_scores[result.best_reference]
            score_text = "  ".join(f"{n}:{crop_scores[n]:.2f}" for n in CROP_NAMES if n in crop_scores)
            cv2.putText(
                display, score_text, (12, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
            )

        panel = resize_to_width(make_crop_panel(crops), w)
        display = np.vstack([display, panel])
        if result.match_viz is not None:
            match_panel = resize_to_width(result.match_viz.side_by_side, w)
            display = np.vstack([display, match_panel])
        return display

    for i, line in enumerate(lines):
        cv2.putText(
            display, line, (12, 28 + i * 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA,
        )
    return display


def resolve_video_path(video_arg: Path, videos_dir: Path) -> Path:
    if video_arg.exists():
        return video_arg.resolve()
    candidate = videos_dir / video_arg.name
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(
        f"Video not found: {video_arg}\n"
        f"Looked in {videos_dir} for {video_arg.name}"
    )


def resolve_identity_dir(reference_root: Path, identity: str) -> Path:
    identity_dir = reference_root / identity
    if not identity_dir.is_dir():
        raise FileNotFoundError(
            f"Reference folder not found: {identity_dir}\n"
            f"Expected 1:1 folder for identity '{identity}' under {reference_root}"
        )
    return identity_dir


def list_identity_dirs(reference_root: Path) -> list[Path]:
    if not reference_root.is_dir():
        raise FileNotFoundError(f"Reference root not found: {reference_root}")
    identity_dirs = []
    for path in sorted(p for p in reference_root.iterdir() if p.is_dir()):
        if any(child.is_dir() and child.name.startswith("frame_") for child in path.iterdir()):
            identity_dirs.append(path)
    if not identity_dirs:
        raise FileNotFoundError(f"No identity folders with frame_* found in {reference_root}")
    return identity_dirs


def guess_identity_prefix_from_video(video_stem: str) -> str:
    """Eli_step1_1781296597 -> Eli, Mujahir_step1_... -> Mujahir."""
    return video_stem.split("_", 1)[0]


def resolve_sample_identity_dir(
    video_path: Path,
    identity_dirs: list[Path],
    explicit_identity: str | None = None,
) -> Path:
    """
    Pick which identity folder to use when saving sample keypoint images.

    Without --identity, samples were previously saved from the first folder
    alphabetically (often the wrong person). Match the video name prefix instead.
    """
    if explicit_identity:
        for identity_dir in identity_dirs:
            if identity_dir.name == explicit_identity:
                return identity_dir
        raise FileNotFoundError(
            f"Identity '{explicit_identity}' not found among available reference folders"
        )

    prefix = guess_identity_prefix_from_video(video_path.stem)
    prefix_matches = [
        d for d in identity_dirs
        if d.name.startswith(f"{prefix}_") or d.name == prefix
    ]
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    if len(prefix_matches) > 1:
        return sorted(prefix_matches, key=lambda p: p.name)[0]
    return identity_dirs[0]


def save_query_sample(
    crops: DigitCrops,
    sample_dir: Path,
    frame_no: int,
    match_viz: MatchVisualization | None = None,
) -> None:
    """Save query palm plus large labeled keypoint panels for query and reference."""
    if crops.palm is None:
        return
    frame_dir = sample_dir / f"frame_{frame_no}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(frame_dir / "palm.png"), crops.palm)
    if match_viz is not None:
        cv2.imwrite(str(frame_dir / "query_keypoints.png"), match_viz.query_image)
        cv2.imwrite(str(frame_dir / "reference_keypoints.png"), match_viz.reference_image)
        cv2.imwrite(str(frame_dir / "comparison.png"), match_viz.side_by_side)


def save_best_match_package(
    output_dir: Path,
    *,
    identity: str,
    query_video: str,
    query_frame_no: int,
    query_palm: np.ndarray,
    reference_name: str,
    reference_palm: np.ndarray,
    match_viz: MatchVisualization,
    score: float,
    match_threshold: float,
) -> Path:
    """
    Save graphs' companion evaluation images into output_dir (flat — no subfolders):
    final emboss palms + keypoint overlays + comparison + meta.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_dir / "query_emboss.png"), query_palm)
    cv2.imwrite(str(output_dir / "reference_emboss.png"), reference_palm)
    cv2.imwrite(str(output_dir / "query_keypoints.png"), match_viz.query_image)
    cv2.imwrite(str(output_dir / "reference_keypoints.png"), match_viz.reference_image)
    cv2.imwrite(str(output_dir / "comparison.png"), match_viz.side_by_side)

    meta = {
        "identity": identity,
        "query_video": query_video,
        "query_frame": query_frame_no,
        "reference_frame": reference_name,
        "score": round(float(score), 4),
        "threshold": float(match_threshold),
        "verdict": "MATCH" if score >= match_threshold else "NO MATCH",
        "keypoints_detected": {
            "query": int(match_viz.num_kp_query),
            "reference": int(match_viz.num_kp_ref),
        },
        "lowe_good_matches": int(match_viz.num_good_matches),
        "query_label": match_viz.query_label,
        "reference_label": match_viz.reference_label,
        "files": {
            "query_emboss": "query_emboss.png",
            "reference_emboss": "reference_emboss.png",
            "query_keypoints": "query_keypoints.png",
            "reference_keypoints": "reference_keypoints.png",
            "comparison": "comparison.png",
            "score_graph": "frame_scores.png",
            "keypoints_graph": "frame_keypoints.png",
        },
    }
    with open(output_dir / "match_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    summary_txt = (
        f"Identity: {identity}\n"
        f"Query video: {query_video}\n"
        f"Query frame: {query_frame_no}\n"
        f"Best reference: {reference_name}\n"
        f"Score: {score:.4f}  (threshold {match_threshold:.2f})  "
        f"{'MATCH' if score >= match_threshold else 'NO MATCH'}\n"
        f"SIFT keypoints detected — query: {match_viz.num_kp_query}, "
        f"reference: {match_viz.num_kp_ref}\n"
        f"Lowe good matches: {match_viz.num_good_matches}\n"
    )
    (output_dir / "match_summary.txt").write_text(summary_txt, encoding="utf-8")
    return output_dir


def run_realtime_matching(
    video_path: Path,
    identity_dir: Path,
    identity: str,
    model_path: Path,
    num_reference_frames: int = 2,
    seed: int | None = None,
    match_threshold: float = 0.25,
    sample_dir: Path | None = None,
    sift_nfeatures: int = DEFAULT_SIFT_NFEATURES,
    sift_contrast_threshold: float = DEFAULT_SIFT_CONTRAST_THRESHOLD,
    sift_edge_threshold: float = DEFAULT_SIFT_EDGE_THRESHOLD,
    sift_n_octave_layers: int = DEFAULT_SIFT_N_OCTAVE_LAYERS,
    sift_sigma: float = DEFAULT_SIFT_SIGMA,
    min_good_matches: int = DEFAULT_MIN_GOOD_MATCHES,
    target_good_matches: int = DEFAULT_TARGET_GOOD_MATCHES,
    rotate_deg: int = DEFAULT_ROTATE_DEG,
    max_pattern_distance: int = DEFAULT_MAX_PATTERN_DISTANCE,
    min_query_scored_frames: int = MIN_QUERY_SCORED_FRAMES,
    max_query_scored_frames: int = MAX_QUERY_SCORED_FRAMES,
    show_gui: bool = True,
) -> list[FrameMatchResult]:
    references = sample_reference_frames(identity_dir, num_reference_frames, seed=seed)
    matcher = SIFTMatcher(
        nfeatures=sift_nfeatures,
        contrast_threshold=sift_contrast_threshold,
        edge_threshold=sift_edge_threshold,
        n_octave_layers=sift_n_octave_layers,
        sigma=sift_sigma,
        min_good_matches=min_good_matches,
        target_good_matches=target_good_matches,
    )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    delay_ms = max(1, int(1000 / fps))
    results: list[FrameMatchResult] = []
    frame_idx = 0
    best_pack: dict | None = None

    print(f"Playing: {video_path.name}")
    print(f"Identity: {identity}")
    print(f"Reference frames: {', '.join(r.name for r in references)}")
    print(
        "Query preprocess: digit_preprocessing "
        f"(crease gate L1<={max_pattern_distance}, rotate={rotate_deg})"
    )
    print(
        f"SIFT settings: nfeatures={sift_nfeatures} "
        f"nOctaveLayers={sift_n_octave_layers} "
        f"contrastThreshold={sift_contrast_threshold} "
        f"edgeThreshold={sift_edge_threshold} "
        f"sigma={sift_sigma}"
    )
    print(
        f"Match scoring: Lowe good / min(kp_query, kp_ref) "
        f"(ratio={matcher.ratio_threshold}, palm ROI mask)"
    )
    print(
        f"Query frames scored: min {min_query_scored_frames}, "
        f"max {max_query_scored_frames}"
    )
    if sample_dir is not None:
        print(f"Saving evaluation images -> {sample_dir}")
    if show_gui:
        print("Press 'q' or ESC to stop.\n")
        try:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        except cv2.error:
            print("OpenCV GUI unavailable; continuing headless.\n")
            show_gui = False
    else:
        print("Headless mode (no preview window).\n")

    ref_by_name = {ref.name: ref for ref in references}

    with DigitPreprocessor.create(model_path, mode="video") as preprocessor:
        while True:
            if len(results) >= max_query_scored_frames:
                print(
                    f"Reached max scored query frames ({max_query_scored_frames}); stopping."
                )
                break

            ok, frame = cap.read()
            if not ok:
                break

            frame = rotate_frame_bgr(frame, rotate_deg)
            crops = preprocessor.process_frame(
                frame, fps=fps, max_l1_distance=max_pattern_distance,
            )
            result: FrameMatchResult | None = None
            current_frame_no = frame_idx + 1

            gate_ok = (
                crops is not None
                and crops.all_present()
                and passes_crease_gate_tolerant(
                    crops.crease_counts,
                    max_l1_distance=max_pattern_distance,
                )
            )

            if gate_ok and current_frame_no >= MIN_QUERY_FRAME:
                query_label = f"frame_{current_frame_no}/palm.png"
                print(f"frame {current_frame_no:>3}:")
                result = match_frame_to_references(
                    crops,
                    references,
                    matcher,
                    identity,
                    query_label=query_label,
                )
                result.frame_index = frame_idx
                results.append(result)

                if (
                    result.match_viz is not None
                    and result.best_reference is not None
                    and crops.palm is not None
                    and (
                        best_pack is None
                        or result.best_score > best_pack["score"]
                    )
                ):
                    ref_frame = ref_by_name.get(result.best_reference)
                    if ref_frame is not None and "palm" in ref_frame.crops:
                        best_pack = {
                            "score": result.best_score,
                            "query_frame_no": current_frame_no,
                            "query_palm": crops.palm.copy(),
                            "reference_name": result.best_reference,
                            "reference_palm": ref_frame.crops["palm"].copy(),
                            "match_viz": result.match_viz,
                        }

                if result.match_viz is not None:
                    print(f"  Query keypoints: {result.match_viz.num_kp_query}")
                    print(f"  Reference keypoints: {result.match_viz.num_kp_ref}")
                    print(f"  Good SIFT matches: {result.match_viz.num_good_matches}")
                    print(f"  Final score: {result.best_score:.3f}")
                print(
                    f"  best={result.best_reference} score={result.best_score:.3f} "
                    f"{'MATCH' if result.best_score >= match_threshold else 'NO MATCH'}"
                    f" kp={result.num_kp_query}/{result.num_kp_ref}"
                    f" lowe={result.num_good_matches}"
                )
            elif crops is None or not crops.all_present():
                print(f"frame {current_frame_no:>3}: no hand / no palm")
            else:
                print(
                    f"frame {current_frame_no:>3}: skipped "
                    f"(crease gate fail, L1 too high)"
                )

            if show_gui:
                display = build_display(
                    frame, crops if gate_ok else None, result, identity, references,
                    match_threshold, frame_idx,
                )
                cv2.imshow(WINDOW_NAME, display)

                key = cv2.waitKey(delay_ms) & 0xFF
                if key in (ord("q"), 27):
                    print("Stopped by user.")
                    break

            frame_idx += 1

    cap.release()
    if show_gui:
        cv2.destroyAllWindows()

    if sample_dir is not None and best_pack is not None:
        best_dir = save_best_match_package(
            sample_dir,
            identity=identity,
            query_video=video_path.name,
            query_frame_no=best_pack["query_frame_no"],
            query_palm=best_pack["query_palm"],
            reference_name=best_pack["reference_name"],
            reference_palm=best_pack["reference_palm"],
            match_viz=best_pack["match_viz"],
            score=best_pack["score"],
            match_threshold=match_threshold,
        )
        print(f"Saved best query/reference package -> {best_dir}")
        print(
            f"  query frame {best_pack['query_frame_no']} vs "
            f"{best_pack['reference_name']}  "
            f"score={best_pack['score']:.3f}  "
            f"kp={best_pack['match_viz'].num_kp_query}/"
            f"{best_pack['match_viz'].num_kp_ref}  "
            f"good={best_pack['match_viz'].num_good_matches}"
        )

    if len(results) < min_query_scored_frames:
        print(
            f"WARNING: only {len(results)} scored query frames "
            f"(wanted >= {min_query_scored_frames}). "
            "Plot will still be written with available frames."
        )
    return results


def print_summary(
    results: list[FrameMatchResult],
    match_threshold: float,
) -> tuple[float, str]:
    print("-" * 60)
    if not results:
        print("No hand detected in any processed frame.")
        return 0.0, "NO HAND"

    avg_best = float(np.mean([r.best_score for r in results]))
    verdict = "MATCH" if avg_best >= match_threshold else "NO MATCH"
    print(f"Frames matched: {len(results)}")
    print(f"Average best score: {avg_best:.3f}")
    print(f"Final verdict: {verdict}")
    return avg_best, verdict


def plot_identity_scores(
    video_name: str,
    summaries: list[IdentityRunSummary],
    threshold: float,
    score_output_path: Path,
    keypoints_output_path: Path,
) -> None:
    """Write two separate graphs: match score, and per-frame good-match counts."""
    if not summaries:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping score graphs.")
        return

    score_output_path.parent.mkdir(parents=True, exist_ok=True)
    keypoints_output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Graph 1: scores only ---
    fig_score, ax_score = plt.subplots(figsize=(11, 6))
    for summary in summaries:
        if not summary.frame_scores:
            continue
        label = f"{summary.identity} (avg {summary.avg_best_score:.3f})"
        ax_score.plot(
            summary.frame_indices, summary.frame_scores,
            linewidth=1.6, marker="o", markersize=3, label=label,
        )
    ax_score.axhline(
        y=threshold, color="#1f77b4", linestyle="--", label=f"Threshold {threshold:.2f}",
    )
    ax_score.set_ylim(0.0, 1.0)
    ax_score.set_ylabel("Best Score")
    ax_score.set_xlabel("Frame Index")
    ax_score.set_title(f"SIFT Match Score — {video_name}")
    ax_score.grid(True, alpha=0.25)
    ax_score.legend(loc="best", fontsize=8)
    fig_score.tight_layout()
    fig_score.savefig(score_output_path, dpi=160)
    print(f"Saved score plot: {score_output_path}")

    # --- Graph 2: good matches only (one line per identity, like the reference plot) ---
    fig_kp, ax_kp = plt.subplots(figsize=(10, 6))
    plotted = False
    for summary in summaries:
        if not summary.frame_good_matches:
            continue
        # Sequential sign-in indices (0..N-1), not absolute video frame numbers
        x = list(range(len(summary.frame_good_matches)))
        avg_good = float(np.mean(summary.frame_good_matches))
        short_name = summary.identity.split("_")[0] if "_" in summary.identity else summary.identity
        label = f"{short_name} (avg {avg_good:.1f})"
        ax_kp.plot(
            x,
            summary.frame_good_matches,
            linewidth=1.8,
            marker="o",
            markersize=5,
            label=label,
        )
        plotted = True

    if plotted:
        ax_kp.set_ylabel("Good SIFT matches (best of refs)")
        ax_kp.set_xlabel("Sign-in frame index")
        ax_kp.set_title("Per-frame good SIFT matches")
        ax_kp.grid(True, alpha=0.3)
        ax_kp.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=9,
            frameon=True,
        )
        fig_kp.tight_layout()
        fig_kp.savefig(keypoints_output_path, dpi=160, bbox_inches="tight")
        print(f"Saved keypoints plot: {keypoints_output_path}")
    else:
        print("No match-count data to plot; skipping keypoints graph.")
        plt.close(fig_kp)

    plt.show(block=False)
    plt.pause(0.5)
    plt.close("all")


def score_processed_images(
    image_a_path: Path,
    image_b_path: Path,
    sift_nfeatures: int = DEFAULT_SIFT_NFEATURES,
    sift_contrast_threshold: float = DEFAULT_SIFT_CONTRAST_THRESHOLD,
    sift_edge_threshold: float = DEFAULT_SIFT_EDGE_THRESHOLD,
    sift_n_octave_layers: int = DEFAULT_SIFT_N_OCTAVE_LAYERS,
    sift_sigma: float = DEFAULT_SIFT_SIGMA,
    min_good_matches: int = DEFAULT_MIN_GOOD_MATCHES,
    target_good_matches: int = DEFAULT_TARGET_GOOD_MATCHES,
) -> float:
    """Score two already-processed images with SIFT + Lowe (no RANSAC)."""
    image_a = cv2.imread(str(image_a_path))
    image_b = cv2.imread(str(image_b_path))
    if image_a is None:
        raise FileNotFoundError(f"Could not read image A: {image_a_path}")
    if image_b is None:
        raise FileNotFoundError(f"Could not read image B: {image_b_path}")

    matcher = SIFTMatcher(
        nfeatures=sift_nfeatures,
        contrast_threshold=sift_contrast_threshold,
        edge_threshold=sift_edge_threshold,
        n_octave_layers=sift_n_octave_layers,
        sigma=sift_sigma,
        min_good_matches=min_good_matches,
        target_good_matches=target_good_matches,
    )
    result = matcher.match_pair(image_a, image_b)
    print(f"Image A: {image_a_path}  shape={image_a.shape}")
    print(f"Image B: {image_b_path}  shape={image_b.shape}")
    print(
        f"SIFT settings: nfeatures={sift_nfeatures} "
        f"nOctaveLayers={sift_n_octave_layers} "
        f"contrastThreshold={sift_contrast_threshold} "
        f"edgeThreshold={sift_edge_threshold} "
        f"sigma={sift_sigma}"
    )
    matcher.print_match_diagnostics(result)
    return result.score


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-time SIFT matching for test videos, or score two processed images.",
        epilog=(
            "Examples:\n"
            "  python feature_match.py \\\n"
            "    --video Mujahir_step1_1782111905.mov\n\n"
            "  python feature_match.py \\\n"
            "    --video Mujahir_step1_1782111905.mov \\\n"
            "    --identity Mujahir_1772227836\n\n"
            "  python feature_match.py \\\n"
            "    --image-a output/Eli_1772247537/frame_40/palm.png \\\n"
            "    --image-b sample/Eli_step1_1781296597/frame_40/palm.png"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Video file name in test_videos/ (or full path)",
    )
    parser.add_argument(
        "--image-a",
        type=Path,
        default=None,
        help="Already-processed image A (use with --image-b; skips video preprocessing)",
    )
    parser.add_argument(
        "--image-b",
        type=Path,
        default=None,
        help="Already-processed image B (use with --image-a; skips video preprocessing)",
    )
    parser.add_argument(
        "--identity",
        type=str,
        default=None,
        help="Reference folder name under --reference-root (if omitted, compare all folders)",
    )
    parser.add_argument("--videos-dir", type=Path, default=DEFAULT_VIDEOS_DIR)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Folder for graphs + evaluation images. "
            "Default: evaluation/<video_stem>/."
        ),
    )
    parser.add_argument("--num-reference-frames", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.25, help="SIFT match threshold")
    parser.add_argument(
        "--rotate",
        type=int,
        default=DEFAULT_ROTATE_DEG,
        help="Decode orientation rotate (must match enrollment, default 90).",
    )
    parser.add_argument(
        "--max-distance",
        type=int,
        default=DEFAULT_MAX_PATTERN_DISTANCE,
        help="Crease-gate max L1 distance (must match enrollment, default 4).",
    )
    parser.add_argument(
        "--min-query-frames",
        type=int,
        default=MIN_QUERY_SCORED_FRAMES,
        help="Warn if fewer than this many crease-gated query frames are scored (default: 3).",
    )
    parser.add_argument(
        "--max-query-frames",
        type=int,
        default=MAX_QUERY_SCORED_FRAMES,
        help="Stop after scoring this many crease-gated query frames (default: 5).",
    )
    parser.add_argument(
        "--sift-nfeatures",
        type=int,
        default=DEFAULT_SIFT_NFEATURES,
        help="Max SIFT keypoints per image (0 = no limit)",
    )
    parser.add_argument(
        "--sift-contrast-threshold",
        type=float,
        default=DEFAULT_SIFT_CONTRAST_THRESHOLD,
        help="SIFT contrast threshold (higher = fewer keypoints)",
    )
    parser.add_argument(
        "--sift-edge-threshold",
        type=float,
        default=DEFAULT_SIFT_EDGE_THRESHOLD,
        help="SIFT edge threshold (higher = fewer edge-like keypoints)",
    )
    parser.add_argument(
        "--sift-n-octave-layers",
        type=int,
        default=DEFAULT_SIFT_N_OCTAVE_LAYERS,
        help="SIFT octave layers (more = more scale levels / keypoints)",
    )
    parser.add_argument(
        "--sift-sigma",
        type=float,
        default=DEFAULT_SIFT_SIGMA,
        help="SIFT Gaussian sigma at first octave",
    )
    parser.add_argument(
        "--min-good-matches",
        type=int,
        default=DEFAULT_MIN_GOOD_MATCHES,
        help="Minimum Lowe good matches required for a non-zero score",
    )
    parser.add_argument(
        "--target-good-matches",
        type=int,
        default=DEFAULT_TARGET_GOOD_MATCHES,
        help="Lowe good-match count that maps to score 1.0",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Skip OpenCV preview window (required when highgui is unavailable).",
    )
    args = parser.parse_args()

    if (args.image_a is None) ^ (args.image_b is None):
        parser.error("Provide both --image-a and --image-b together.")

    if args.image_a is not None and args.image_b is not None:
        score_processed_images(
            image_a_path=args.image_a.resolve(),
            image_b_path=args.image_b.resolve(),
            sift_nfeatures=args.sift_nfeatures,
            sift_contrast_threshold=args.sift_contrast_threshold,
            sift_edge_threshold=args.sift_edge_threshold,
            sift_n_octave_layers=args.sift_n_octave_layers,
            sift_sigma=args.sift_sigma,
            min_good_matches=args.min_good_matches,
            target_good_matches=args.target_good_matches,
        )
        return

    if args.video is None:
        parser.error("Provide --video, or both --image-a and --image-b.")

    videos_dir = args.videos_dir.resolve()
    video_path = resolve_video_path(args.video, videos_dir)
    reference_root = args.reference_root.resolve()
    model_path = ensure_model(args.model.resolve())
    identity_dirs = (
        [resolve_identity_dir(reference_root, args.identity)]
        if args.identity
        else list_identity_dirs(reference_root)
    )

    # One flat folder: evaluation/<video_stem>/ (graphs + eval images)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (Path.cwd() / "evaluation" / video_path.stem).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output folder (graphs + evaluation images): {output_dir}")

    # Save evaluation images only for the matched / requested identity
    eval_identity_dir = resolve_sample_identity_dir(
        video_path, identity_dirs, explicit_identity=args.identity,
    )

    summaries: list[IdentityRunSummary] = []
    for identity_dir in identity_dirs:
        identity_name = identity_dir.name
        save_eval = identity_dir.resolve() == eval_identity_dir.resolve()
        print("\n" + "=" * 60)
        print(f"Comparing against identity: {identity_name}")
        results = run_realtime_matching(
            video_path=video_path,
            identity_dir=identity_dir,
            identity=identity_name,
            model_path=model_path,
            num_reference_frames=args.num_reference_frames,
            seed=args.seed,
            match_threshold=args.threshold,
            sample_dir=output_dir if save_eval else None,
            sift_nfeatures=args.sift_nfeatures,
            sift_contrast_threshold=args.sift_contrast_threshold,
            sift_edge_threshold=args.sift_edge_threshold,
            sift_n_octave_layers=args.sift_n_octave_layers,
            sift_sigma=args.sift_sigma,
            min_good_matches=args.min_good_matches,
            target_good_matches=args.target_good_matches,
            rotate_deg=args.rotate,
            max_pattern_distance=args.max_distance,
            min_query_scored_frames=args.min_query_frames,
            max_query_scored_frames=args.max_query_frames,
            show_gui=not args.headless,
        )
        avg_best_score, verdict = print_summary(results, args.threshold)
        summaries.append(
            IdentityRunSummary(
                identity=identity_name,
                avg_best_score=avg_best_score,
                frame_count=len(results),
                verdict=verdict,
                frame_indices=[r.frame_index + 1 for r in results],
                frame_scores=[r.best_score for r in results],
                frame_kp_query=[r.num_kp_query for r in results],
                frame_kp_ref=[r.num_kp_ref for r in results],
                frame_good_matches=[r.num_good_matches for r in results],
            )
        )

    print("\n" + "#" * 60)
    print("Overall ranking by average best score")
    for i, row in enumerate(sorted(summaries, key=lambda x: x.avg_best_score, reverse=True), start=1):
        print(
            f"{i:>2}. {row.identity:<25} score={row.avg_best_score:.3f} "
            f"frames={row.frame_count:<4} verdict={row.verdict}"
        )
    score_plot_path = output_dir / "frame_scores.png"
    keypoints_plot_path = output_dir / "frame_keypoints.png"
    plot_identity_scores(
        video_path.name,
        summaries,
        args.threshold,
        score_plot_path,
        keypoints_plot_path,
    )
    print(f"\nAll outputs saved in: {output_dir}")
    print(
        "  frame_scores.png, frame_keypoints.png, "
        "query_emboss.png, reference_emboss.png, "
        "query_keypoints.png, reference_keypoints.png, comparison.png, "
        "match_meta.json, match_summary.txt"
    )


if __name__ == "__main__":
    main()
