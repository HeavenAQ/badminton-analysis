from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd

from badminton_analysis.ml.handedness import (
    estimate_handedness,
    interpolated_keypoint,
)
from badminton_analysis.ml.skeleton_normalization import (
    landmark_dicts_to_array,
    normalize_skeleton_sequence,
    resample_phase_indices,
    resample_sequence,
)
from badminton_analysis.ml.skill_specs import get_skill_spec, supported_skill_choices
from badminton_analysis.models.types import COCOKeypoints, Handedness, Skill
from badminton_analysis.services.pose_detector import PoseDetector
from badminton_analysis.services.video_analyzer import VideoAnalyzer
from badminton_analysis.services.video_processor import VideoProcessor

VIDEO_EXTENSIONS = (".mp4", ".mov")
DEFAULT_BEGINNER_DIR = "scoring_videos/高遠球/初學者高遠球"
DEFAULT_EXPERT_DIR = "scoring_videos/高遠球/專家高遠球"

DEFAULT_SOURCE_DIRS = {
    "clear": (DEFAULT_BEGINNER_DIR, DEFAULT_EXPERT_DIR),
    "serve": ("scoring_videos/發球/無經驗同學", "scoring_videos/發球/羽球隊同學"),
    "lift": ("scoring_videos/挑球/初學者挑球", "scoring_videos/挑球/專家挑球"),
    "smash": ("scoring_videos/殺球/初學者殺球", "scoring_videos/殺球/專家殺球"),
}


def _video_fps(path: Path) -> float:
    capture = cv2.VideoCapture(str(path))
    try:
        value = float(capture.get(cv2.CAP_PROP_FPS))
        return value if np.isfinite(value) and value > 0 else 0.0
    finally:
        capture.release()


def _handedness(
    path: Path, overrides: Mapping[str, Handedness] | None = None
) -> Handedness:
    if overrides is not None and path.name in overrides:
        return overrides[path.name]
    return Handedness.LEFT if "left" in path.name.lower() else Handedness.RIGHT


def _load_handedness_overrides(path: Path | None) -> dict[str, Handedness]:
    if path is None or not path.exists():
        return {}
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("handedness overrides must be a JSON object")
    overrides: dict[str, Handedness] = {}
    for filename, value in values.items():
        if not isinstance(filename, str) or not isinstance(value, str):
            raise ValueError("handedness overrides must map filenames to strings")
        overrides[filename] = Handedness.convert_to_enum(value)
    return overrides


def extract_video_sequence(
    video_path: Path,
    label: str,
    output_root: Path,
    pose_detector: PoseDetector,
    target_frames: int,
    handedness_overrides: Mapping[str, Handedness] | None = None,
    skill: Skill = Skill.CLEAR,
) -> dict[str, Any]:
    fallback_handedness = _handedness(video_path, handedness_overrides)
    summary: dict[str, Any] = {
        "filename": video_path.name,
        "label": label,
        "skill": str(skill),
        "handedness": "",
        "handedness_source": "",
        "left_hand_motion_score": 0.0,
        "right_hand_motion_score": 0.0,
        "handedness_confidence_ratio": 1.0,
        "raw_tracked_frames": 0,
        "analysis_start": -1,
        "analysis_peak": -1,
        "analysis_end": -1,
        "resampled_frames": 0,
        "missing_joint_ratio": 1.0,
        "status": "error",
        "error": "",
    }
    try:
        processor = VideoProcessor(
            str(video_path), video_path.name, str(output_root), pose_detector=pose_detector
        )
        tracking = processor.process_frames(None)
        tracked_frames = len(tracking["original_landmarks"])
        summary["raw_tracked_frames"] = tracked_frames
        if tracked_frames < 5:
            raise ValueError("fewer than five tracked frames")

        body_2d = tracking.get("body_landmarks_2d")
        if body_2d is None or len(body_2d) != tracked_frames:
            raise ValueError("aligned 2D body landmarks are unavailable")
        full_skeleton_2d, full_confidence_2d = landmark_dicts_to_array(
            body_2d, 2
        )
        estimate = estimate_handedness(
            full_skeleton_2d, full_confidence_2d
        )
        handedness = estimate.handedness or fallback_handedness
        if estimate.handedness is not None:
            handedness_source = "wrist_motion"
        elif handedness_overrides and video_path.name in handedness_overrides:
            handedness_source = "metadata_override"
        else:
            handedness_source = "filename_fallback"
        summary.update(
            handedness=str(handedness),
            handedness_source=handedness_source,
            left_hand_motion_score=estimate.left_motion_score,
            right_hand_motion_score=estimate.right_motion_score,
            handedness_confidence_ratio=estimate.confidence_ratio,
        )
        wrist = (
            COCOKeypoints.LEFT_WRIST
            if handedness == Handedness.LEFT
            else COCOKeypoints.RIGHT_WRIST
        )
        elbow = (
            COCOKeypoints.LEFT_ELBOW
            if handedness == Handedness.LEFT
            else COCOKeypoints.RIGHT_ELBOW
        )
        hand_positions = interpolated_keypoint(
            full_skeleton_2d, full_confidence_2d, wrist
        )
        elbow_positions = interpolated_keypoint(
            full_skeleton_2d, full_confidence_2d, elbow
        )
        if len(hand_positions) != tracked_frames or len(elbow_positions) != tracked_frames:
            raise ValueError("dominant wrist or elbow tracking is insufficient")

        start, peak, end = VideoAnalyzer.find_analysis_window(
            skill=skill,
            hand_positions=list(hand_positions),
            elbow_positions=list(elbow_positions),
        )
        start = max(0, min(tracked_frames - 1, int(start)))
        peak = max(start, min(tracked_frames - 1, int(peak)))
        end = max(peak, min(tracked_frames - 1, int(end)))
        summary.update(
            analysis_start=start, analysis_peak=peak, analysis_end=end
        )
        if end - start < 4:
            raise ValueError(f"analysis window is too short: {(start, peak, end)}")

        skeleton_3d, confidence_3d = landmark_dicts_to_array(
            tracking["original_landmarks"][start : end + 1], 3
        )
        skeleton_2d = full_skeleton_2d[start : end + 1]
        confidence_2d = full_confidence_2d[start : end + 1]
        confidence = np.minimum(confidence_3d, confidence_2d)
        summary["missing_joint_ratio"] = float(1.0 - np.mean(confidence))
        for joint in (5, 6, 11, 12):
            if not np.any(confidence[:, joint] > 0):
                raise ValueError(f"critical torso joint {joint} is missing for the full window")

        normalized_3d, normalized_confidence = normalize_skeleton_sequence(
            skeleton_3d, confidence, handedness
        )
        normalized_2d, _ = normalize_skeleton_sequence(
            skeleton_2d, confidence, handedness
        )
        resampled_3d = resample_sequence(normalized_3d, target_frames)
        resampled_2d = resample_sequence(normalized_2d, target_frames)
        resampled_confidence = np.clip(
            resample_sequence(normalized_confidence, target_frames), 0.0, 1.0
        )
        phase_indices = resample_phase_indices((start, peak, end), target_frames)
        if np.any(np.diff(phase_indices) <= 0):
            raise ValueError(
                f"phase anchors are not strictly increasing: "
                f"{phase_indices.tolist()}"
            )
        tracked_source_indices = np.asarray(
            tracking.get("source_frame_indices", list(range(tracked_frames))),
            dtype=np.int64,
        )
        if tracked_source_indices.shape != (tracked_frames,):
            raise ValueError("source frame mapping is not aligned with pose frames")
        source_frame_indices = np.rint(
            resample_sequence(
                tracked_source_indices[start : end + 1].astype(np.float64),
                target_frames,
            )
        ).astype(np.int64)
        source_phase_indices = source_frame_indices[phase_indices]

        output_dir = output_root / label
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_dir / f"{video_path.stem}.npz",
            skeleton_3d=resampled_3d,
            skeleton_2d=resampled_2d,
            confidence=resampled_confidence,
            skill=np.asarray(str(skill)),
            handedness=np.asarray(str(handedness)),
            video_name=np.asarray(video_path.name),
            analysis_window=np.asarray((start, peak, end), dtype=np.int64),
            phase_indices=phase_indices,
            source_frame_indices=source_frame_indices,
            source_phase_indices=source_phase_indices,
            fps=np.asarray(_video_fps(video_path), dtype=np.float32),
        )
        summary.update(status="success", resampled_frames=target_frames)
    except Exception as exc:
        summary["error"] = str(exc)
        print(f"  ERROR: {exc}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract normalized skeleton sequences")
    parser.add_argument(
        "--skill", choices=supported_skill_choices(), default="clear"
    )
    parser.add_argument("--beginner-dir")
    parser.add_argument("--expert-dir")
    parser.add_argument("--output-root")
    parser.add_argument("--summary-path")
    parser.add_argument(
        "--frames",
        type=int,
        choices=(64,),
        default=64,
        help="Model sequence length; current checkpoints require exactly 64",
    )
    parser.add_argument(
        "--handedness-overrides",
        default=None,
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        help="Optional source filenames to extract, such as EG28.mp4 EG29.mp4",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=("beginners", "experts"),
        default=("beginners", "experts"),
        help="Dataset groups to extract",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = get_skill_spec(args.skill)
    output_root = Path(args.output_root) if args.output_root else spec.dataset_root
    output_root.mkdir(parents=True, exist_ok=True)
    detector = PoseDetector()
    overrides_path = (
        Path(args.handedness_overrides)
        if args.handedness_overrides
        else output_root / "handedness_overrides.json"
    )
    handedness_overrides = _load_handedness_overrides(overrides_path)
    rows: list[dict[str, Any]] = []
    default_beginner_dir, default_expert_dir = DEFAULT_SOURCE_DIRS[spec.slug]
    groups = (
        ("beginners", Path(args.beginner_dir or default_beginner_dir)),
        ("experts", Path(args.expert_dir or default_expert_dir)),
    )
    summary_path = (
        Path(args.summary_path)
        if args.summary_path
        else Path("stats/skeleton_correction")
        / f"{spec.slug}_dataset_summary.csv"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    for label, directory in groups:
        if label not in args.groups:
            continue
        videos = sorted(
            path
            for path in directory.iterdir()
            if path.is_file()
            and path.suffix.lower() in VIDEO_EXTENSIONS
            and (args.videos is None or path.name in args.videos)
        )
        for index, video_path in enumerate(videos, start=1):
            print(f"Processing {label} {index}/{len(videos)}: {video_path.name}")
            rows.append(
                extract_video_sequence(
                    video_path,
                    label,
                    output_root,
                    detector,
                    args.frames,
                    handedness_overrides,
                    spec.skill,
                )
            )
            pd.DataFrame(rows).to_csv(summary_path, index=False)

    pd.DataFrame(rows).to_csv(summary_path, index=False)
    success_count = sum(row["status"] == "success" for row in rows)
    print(f"Extracted {success_count}/{len(rows)} sequences; summary={summary_path}")
    return 0 if success_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
