"""
Extract crease-gated palm reference templates from hand videos.

Uses digit_preprocessing:
  resize → MediaPipe → RGB crease quality gate [1,2,2 top→bottom × 4] → CLAHE/emboss → upright → palm crop

Crease detection is only a contrast/quality check (not saved).
Saved templates are palm.png — same palm area as the shared palm pipeline.

By default checks every frame and saves the 2 closest to the crease pattern.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from digit_preprocessing import (
    DEFAULT_MODEL_PATH,
    EXPECTED_CREASE_PATTERN,
    DigitCrops,
    DigitPreprocessor,
    ensure_model,
    pattern_l1_distance,
    passes_crease_gate,
    passes_crease_gate_tolerant,
    save_pipeline_steps,
    total_crease_count,
)
from hand_preprocessing import rotate_frame_bgr

VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv", ".webm"}
DEFAULT_NUM_TEMPLATES = 2
DEFAULT_ROTATE_DEG = 90
# 0 = consider every frame (no early-frame skip).
DEFAULT_MIN_FRAME = 0
# Canny/Hough rarely hits exact [1,2,2]×4; allow small jitter unless --strict.
DEFAULT_MAX_PATTERN_DISTANCE = 4


@dataclass
class Candidate:
    frame_no: int
    crops: DigitCrops
    steps: dict[str, np.ndarray]
    distance: int
    total_creases: int


def process_video(
    video_path: Path,
    output_root: Path,
    model_path: Path,
    *,
    num_templates: int = DEFAULT_NUM_TEMPLATES,
    max_frames: int | None = None,
    min_frame: int = DEFAULT_MIN_FRAME,
    rotate_deg: int = DEFAULT_ROTATE_DEG,
    max_pattern_distance: int = DEFAULT_MAX_PATTERN_DISTANCE,
    save_steps: bool = False,
) -> list[dict]:
    """
    Scan video, rank frames by crease-pattern distance, save best num_templates.

    All frames are considered when min_frame=0 (default). If min_frame > 0,
    only frames with frame_no > min_frame are eligible; earlier frames are
    still fed to MediaPipe for tracking continuity.
    """
    video_stem = video_path.stem
    output_dir = output_root / video_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: Could not open {video_path.name}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0
    candidates: list[Candidate] = []
    num_candidates = 0

    with DigitPreprocessor.create(model_path, mode="video") as preprocessor:
        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break

            frame_no = frame_idx + 1
            frame = rotate_frame_bgr(frame, rotate_deg)
            crops, steps = preprocessor.process_frame(
                frame, fps=fps, return_steps=True,
            )
            frame_idx += 1

            # Optional early-frame skip (min_frame=0 → check every frame).
            if min_frame > 0 and frame_no <= min_frame:
                continue

            if crops is None or not crops.all_present():
                continue

            counts = crops.crease_counts
            if not passes_crease_gate_tolerant(
                counts, max_l1_distance=max_pattern_distance,
            ):
                continue

            num_candidates += 1
            candidates.append(
                Candidate(
                    frame_no=frame_no,
                    crops=crops,
                    steps=steps if save_steps else {},
                    distance=pattern_l1_distance(counts),
                    total_creases=total_crease_count(counts),
                )
            )
            # Prefer exact / closer pattern matches; keep only best so far.
            candidates.sort(key=lambda c: (c.distance, c.frame_no))
            del candidates[num_templates:]

    cap.release()
    selected = candidates

    saved: list[dict] = []
    for cand in selected:
        frame_folder = output_dir / f"frame_{cand.frame_no}"
        if not cand.crops.save(frame_folder):
            continue

        counts = cand.crops.crease_counts
        meta = {
            "frame": cand.frame_no,
            "crease_counts": {k: list(v) for k, v in counts.items()},
            "total_creases": cand.total_creases,
            "pattern_l1_distance": cand.distance,
            "exact_gate": passes_crease_gate(counts),
            "pattern": list(EXPECTED_CREASE_PATTERN),
            "max_pattern_distance": max_pattern_distance,
        }
        with open(frame_folder / "crease_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        if save_steps and cand.steps:
            steps_dir = frame_folder / "steps"
            save_pipeline_steps(steps_dir, cand.steps)
            print(f"  Saved pipeline steps -> {steps_dir}")

        saved.append(meta)
        print(
            f"  Saved template {len(saved)}/{num_templates}: "
            f"frame_{cand.frame_no} "
            f"(L1={cand.distance}, creases={cand.total_creases}, "
            f"exact={meta['exact_gate']}, counts={counts})"
        )

    summary = {
        "video": video_path.name,
        "num_requested": num_templates,
        "num_candidates": num_candidates,
        "num_saved": len(saved),
        "min_frame": min_frame,
        "max_pattern_distance": max_pattern_distance,
        "templates": saved,
        "pipeline": "digit_v4_crease_gate_palm",
        "expected_pattern_top_to_bottom": list(EXPECTED_CREASE_PATTERN),
    }
    with open(output_dir / "templates_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"  Done: {video_path.name} -> {output_dir} "
        f"({len(saved)}/{num_templates} templates from {num_candidates} candidates)"
    )
    return saved


def find_videos(videos_dir: Path) -> list[Path]:
    return sorted(
        p for p in videos_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract palm templates from videos. "
            "Crease detection on RGB after resize (quality/contrast gate only). "
            "Expected pattern per finger top→bottom: [1, 2, 2]. Saves palm.png."
        ),
        epilog=(
            "Examples:\n"
            "  python extract_digit_templates.py --num-templates 2\n"
            "  python extract_digit_templates.py --video Eli_1772247537.mov --steps\n"
            "  python extract_digit_templates.py --strict"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "videos",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "digit_output",
    )
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--min-frame",
        type=int,
        default=DEFAULT_MIN_FRAME,
        help=(
            "Only consider frames after this 1-based index "
            f"(default: {DEFAULT_MIN_FRAME} = every frame)."
        ),
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--num-templates",
        type=int,
        default=DEFAULT_NUM_TEMPLATES,
        help="Reference templates to save per video (default: 2).",
    )
    parser.add_argument(
        "--rotate",
        type=int,
        default=DEFAULT_ROTATE_DEG,
        help="Decode orientation rotate in degrees (default: 90 for iOS .mov).",
    )
    parser.add_argument(
        "--max-distance",
        type=int,
        default=DEFAULT_MAX_PATTERN_DISTANCE,
        help=(
            "Max L1 distance from [1,2,2]×4 to accept a frame "
            f"(default: {DEFAULT_MAX_PATTERN_DISTANCE}). 0 = exact only."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require exact [1,2,2] on all four fingers (same as --max-distance 0).",
    )
    parser.add_argument(
        "--steps",
        action="store_true",
        help="Also save intermediate pipeline images under each frame/steps/.",
    )
    args = parser.parse_args()

    max_distance = 0 if args.strict else args.max_distance
    videos_dir = args.videos_dir.resolve()
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    model_path = ensure_model(args.model.resolve())

    videos = [videos_dir / args.video] if args.video else find_videos(videos_dir)
    if not videos:
        print(f"No videos found in {videos_dir}")
        return

    print(
        f"Found {len(videos)} video(s). "
        f"Target pattern: {EXPECTED_CREASE_PATTERN} × 4 fingers. "
        f"Max L1 distance: {max_distance}. "
        f"Min frame: {'all' if args.min_frame <= 0 else f'>{args.min_frame}'}. "
        f"Templates/video: {args.num_templates}. Output -> {output_root}"
    )
    for video_path in videos:
        if not video_path.exists():
            print(f"  SKIP: {video_path.name} not found")
            continue
        print(f"Processing {video_path.name}...")
        process_video(
            video_path,
            output_root,
            model_path,
            num_templates=args.num_templates,
            max_frames=args.max_frames,
            min_frame=args.min_frame,
            rotate_deg=args.rotate,
            max_pattern_distance=max_distance,
            save_steps=args.steps,
        )


if __name__ == "__main__":
    main()
