"""
Extract palm reference templates from hand videos.

Uses digit_preprocessing:
  rotate → resize → MediaPipe → draw finger joint boxes (debug only) →
  CLAHE → emboss → upright → palm crop

No crease quality gate. By default picks 2 random frames with a palm after frame 30.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from digit_preprocessing import (
    DEFAULT_MODEL_PATH,
    DigitCrops,
    DigitPreprocessor,
    ensure_model,
    save_pipeline_steps,
)
from hand_preprocessing import rotate_frame_bgr

VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv", ".webm"}
DEFAULT_NUM_TEMPLATES = 2
DEFAULT_ROTATE_DEG = 90
# Only frames after this 1-based index are eligible for templates.
DEFAULT_MIN_FRAME = 30


@dataclass
class Candidate:
    frame_no: int
    crops: DigitCrops
    steps: dict[str, np.ndarray]


def process_video(
    video_path: Path,
    output_root: Path,
    model_path: Path,
    *,
    num_templates: int = DEFAULT_NUM_TEMPLATES,
    max_frames: int | None = None,
    min_frame: int = DEFAULT_MIN_FRAME,
    rotate_deg: int = DEFAULT_ROTATE_DEG,
    save_steps: bool = False,
    seed: int | None = None,
) -> list[dict]:
    """
    Scan video, collect palm frames after min_frame, randomly save num_templates.

    Earlier frames are still fed to MediaPipe for tracking continuity.
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

            if frame_no <= min_frame:
                continue
            if crops is None or not crops.all_present():
                continue

            candidates.append(
                Candidate(
                    frame_no=frame_no,
                    crops=crops,
                    steps=steps if save_steps else {},
                )
            )

    cap.release()

    rng = random.Random(seed)
    if len(candidates) <= num_templates:
        selected = candidates
    else:
        selected = rng.sample(candidates, k=num_templates)
    selected.sort(key=lambda c: c.frame_no)

    saved: list[dict] = []
    for cand in selected:
        frame_folder = output_dir / f"frame_{cand.frame_no}"
        if not cand.crops.save(frame_folder):
            continue

        meta = {
            "frame": cand.frame_no,
            "crease_counts": {
                k: list(v) for k, v in cand.crops.crease_counts.items()
            },
            "selection": "random_after_min_frame",
            "min_frame": min_frame,
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
            f"frame_{cand.frame_no} (counts={cand.crops.crease_counts})"
        )

    summary = {
        "video": video_path.name,
        "num_requested": num_templates,
        "num_candidates": len(candidates),
        "num_saved": len(saved),
        "min_frame": min_frame,
        "selection": "random",
        "seed": seed,
        "templates": saved,
        "pipeline": "digit_v5_boxes_no_gate",
    }
    with open(output_dir / "templates_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"  Done: {video_path.name} -> {output_dir} "
        f"({len(saved)}/{num_templates} templates from {len(candidates)} "
        f"eligible frames after {min_frame})"
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
            "No crease gate — picks random frames with a palm after --min-frame. "
            "Finger joint boxes are still drawn in --steps debug images. Saves palm.png."
        ),
        epilog=(
            "Examples:\n"
            "  python extract_digit_templates.py --num-templates 2\n"
            "  python extract_digit_templates.py --video Eli_1772247537.mov --steps\n"
            "  python extract_digit_templates.py --min-frame 30 --seed 0"
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
            f"(default: {DEFAULT_MIN_FRAME})."
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
        "--seed",
        type=int,
        default=None,
        help="RNG seed for random frame selection (default: nondeterministic).",
    )
    parser.add_argument(
        "--steps",
        action="store_true",
        help="Also save intermediate pipeline images under each frame/steps/.",
    )
    args = parser.parse_args()

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
        f"Selection: {args.num_templates} random palm frame(s) after frame "
        f"{args.min_frame}. Output -> {output_root}"
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
            save_steps=args.steps,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
