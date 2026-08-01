from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from badminton_analysis.ml.infer_skeleton_corrector import (
    load_corrector,
    predict_correction,
)
from badminton_analysis.ml.clear_feedback import (
    load_feedback_display_score,
    load_feedback_problems,
)
from badminton_analysis.ml.models.skeleton_denoiser import SkeletonDenoiser
from badminton_analysis.ml.skeleton_dataset import load_sequence
from badminton_analysis.ml.skeleton_normalization import (
    landmark_dicts_to_array,
    resample_sequence,
)
from badminton_analysis.ml.skeleton_scoring import BONES
from badminton_analysis.ml.skill_specs import get_skill_spec
from badminton_analysis.models.types import Handedness
from badminton_analysis.services.pose_detector import PoseDetector
from badminton_analysis.services.video_processor import VideoProcessor

_LEFT_RIGHT_PAIRS = (
    (1, 2),
    (3, 4),
    (5, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
)

Corrector = tuple[SkeletonDenoiser, dict[str, Any]]


@lru_cache(maxsize=1)
def _header_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/System/Library/Fonts/PingFang.ttc",
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, 27)
    return ImageFont.load_default()


@lru_cache(maxsize=1)
def _feedback_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/System/Library/Fonts/PingFang.ttc",
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, 22)
    return ImageFont.load_default()


def _interpolate_missing(
    coordinates: NDArray[np.floating], confidence: NDArray[np.floating]
) -> NDArray[np.float32]:
    result = np.asarray(coordinates, dtype=np.float64).copy()
    observed = np.asarray(confidence, dtype=np.float64)
    timeline = np.arange(len(result), dtype=np.float64)
    for joint in range(result.shape[1]):
        for dimension in range(result.shape[2]):
            values = result[:, joint, dimension]
            valid = (observed[:, joint] > 0) & np.isfinite(values)
            if not np.any(valid):
                result[:, joint, dimension] = 0.0
            elif np.count_nonzero(valid) == 1:
                result[:, joint, dimension] = values[valid][0]
            else:
                result[:, joint, dimension] = np.interp(
                    timeline, timeline[valid], values[valid]
                )
    return result.astype(np.float32)


def _nearest_tracked_frame_indices(
    source_frame_indices: NDArray[np.integer],
    tracked_source_frame_indices: NDArray[np.integer],
) -> NDArray[np.int64]:
    targets = np.asarray(source_frame_indices, dtype=np.int64)
    tracked = np.asarray(tracked_source_frame_indices, dtype=np.int64)
    if targets.ndim != 1 or tracked.ndim != 1 or not len(tracked):
        raise ValueError("source frame mappings must be non-empty vectors")
    if np.any(np.diff(targets) < 0) or np.any(np.diff(tracked) < 0):
        raise ValueError("source frame mappings must be ordered")
    upper = np.clip(np.searchsorted(tracked, targets), 0, len(tracked) - 1)
    lower = np.maximum(upper - 1, 0)
    use_lower = np.abs(tracked[lower] - targets) <= np.abs(
        tracked[upper] - targets
    )
    return np.where(use_lower, lower, upper).astype(np.int64)


def _canonicalize_left_handed(
    coordinates: NDArray[np.float32], confidence: NDArray[np.float32]
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    canonical = coordinates.copy()
    canonical_confidence = confidence.copy()
    for left, right in _LEFT_RIGHT_PAIRS:
        canonical[:, [left, right]] = canonical[:, [right, left]]
        canonical_confidence[:, [left, right]] = canonical_confidence[:, [right, left]]
    return canonical, canonical_confidence


def _fit_affine(
    normalized: NDArray[np.float32],
    pixels: NDArray[np.float32],
    confidence: NDArray[np.float32],
) -> NDArray[np.float64]:
    valid = (
        (confidence > 0.05)
        & np.all(np.isfinite(normalized), axis=-1)
        & np.all(np.isfinite(pixels), axis=-1)
    )
    if np.count_nonzero(valid) < 3:
        raise ValueError("at least three visible joints are required for pixel alignment")
    count = int(np.count_nonzero(valid))
    source = np.concatenate(
        (
            normalized[valid].astype(np.float64),
            np.ones((count, 1), dtype=np.float64),
        ),
        axis=1,
    )
    transform, _, _, _ = np.linalg.lstsq(
        source, pixels[valid].astype(np.float64), rcond=None
    )
    return np.asarray(transform, dtype=np.float64)


def _map_to_pixels(
    normalized: NDArray[np.float32], transform: NDArray[np.float64]
) -> NDArray[np.float32]:
    homogeneous = np.concatenate(
        (
            normalized.astype(np.float64),
            np.ones((len(normalized), 1), dtype=np.float64),
        ),
        axis=1,
    )
    return np.asarray(homogeneous @ transform, dtype=np.float32)


def _draw_skeleton(
    frame: NDArray[np.uint8],
    coordinates: NDArray[np.float32],
    confidence: NDArray[np.float32],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    height, width = frame.shape[:2]
    points = np.rint(coordinates).astype(np.int32)
    for start, end in BONES:
        if confidence[start] <= 0.05 or confidence[end] <= 0.05:
            continue
        first = tuple(points[start])
        second = tuple(points[end])
        if not (
            -width <= first[0] < 2 * width
            and -height <= first[1] < 2 * height
            and -width <= second[0] < 2 * width
            and -height <= second[1] < 2 * height
        ):
            continue
        cv2.line(frame, first, second, (15, 15, 15), thickness + 4, cv2.LINE_AA)
        cv2.line(frame, first, second, color, thickness, cv2.LINE_AA)
    for joint, point in enumerate(points):
        if confidence[joint] <= 0.05:
            continue
        location = tuple(point)
        cv2.circle(frame, location, thickness + 3, (15, 15, 15), -1, cv2.LINE_AA)
        cv2.circle(frame, location, thickness + 1, color, -1, cv2.LINE_AA)


def _draw_header(
    frame: NDArray[np.uint8], filename: str, score: float | None
) -> None:
    name = Path(filename).stem
    label = name if score is None else f"{name}  總分 {score:.1f}"
    cv2.rectangle(frame, (18, 18), (540, 104), (20, 20, 20), -1)
    header = frame[18:62, 18:541]
    header_rgb = cv2.cvtColor(header, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(header_rgb)
    ImageDraw.Draw(image).text(
        (16, 3), label, font=_header_font(), fill=(245, 245, 245)
    )
    frame[18:62, 18:541] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    cv2.line(frame, (35, 78), (76, 78), (255, 210, 30), 5, cv2.LINE_AA)
    cv2.putText(
        frame,
        "detected",
        (88, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    cv2.line(frame, (220, 78), (261, 78), (55, 225, 75), 5, cv2.LINE_AA)
    cv2.putText(
        frame,
        "corrected",
        (273, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )


def _wrapped_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    maximum_width: int,
) -> list[str]:
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if draw.textlength(candidate, font=font) <= maximum_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _draw_feedback(
    frame: NDArray[np.uint8],
    detected_pixels: NDArray[np.float32],
    issues: list[dict[str, Any]],
    handedness: Handedness,
) -> None:
    height, width = frame.shape[:2]
    radius = max(22, round(min(height, width) * 0.025))
    dominant_shoulder_labels: list[tuple[tuple[int, int], str]] = []
    for issue in issues:
        for joint_id in issue["joint_ids"]:
            point = detected_pixels[int(joint_id)]
            location = (int(round(point[0])), int(round(point[1])))
            if 0 <= location[0] < width and 0 <= location[1] < height:
                cv2.circle(
                    frame, location, radius + 5, (15, 15, 15), 9, cv2.LINE_AA
                )
                cv2.circle(
                    frame, location, radius, (40, 40, 245), 7, cv2.LINE_AA
                )
                if int(joint_id) == 6:
                    side = "左肩" if handedness == Handedness.LEFT else "右肩"
                    dominant_shoulder_labels.append(
                        (location, f"慣用側（{side}）")
                    )

    font = _feedback_font()
    panel_height = min(height // 3, 112 + 76 * len(issues))
    panel_top = height - panel_height
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, panel_top), (width, height), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.90, frame, 0.10, 0.0, frame)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    for location, label in dominant_shoulder_labels:
        bounds = draw.textbbox((0, 0), label, font=font)
        label_width = bounds[2] - bounds[0]
        label_height = bounds[3] - bounds[1]
        x = min(width - label_width - 20, location[0] + radius + 14)
        x = max(12, x)
        y = max(12, location[1] - label_height // 2 - 8)
        draw.rectangle(
            (x - 8, y - 5, x + label_width + 8, y + label_height + 7),
            fill=(20, 20, 20),
        )
        draw.text((x, y), label, font=font, fill=(255, 125, 105))
    phase_labels = {
        "preparation": "準備動作",
        "rotation": "轉身",
        "contact": "擊球",
        "follow_through": "隨揮",
    }
    phase = phase_labels.get(str(issues[0].get("phase", "")), "動作分析")
    draw.text(
        (24, panel_top + 15),
        f"教練指導暫停 | {phase}",
        font=font,
        fill=(255, 100, 90),
    )
    y = panel_top + 51
    for issue_index, issue in enumerate(issues, start=1):
        score_label = ""
        if "criterion_score" in issue and "criterion_maximum" in issue:
            score_label = (
                f" {float(issue['criterion_score']):.1f}/"
                f"{float(issue['criterion_maximum']):.0f}分"
            )
        message = (
            f"{issue_index}. {issue['title']}{score_label}：{issue['feedback']}"
        )
        for line in _wrapped_lines(draw, message, font, width - 48)[:2]:
            draw.text((24, y), line, font=font, fill=(248, 248, 248))
            y += 29
        y += 9
    frame[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def render_video(
    *,
    video_path: Path,
    dataset_path: Path,
    model_path: Path,
    output_path: Path,
    results_path: Path | None,
    device: torch.device,
    pose_detector: PoseDetector | None = None,
    corrector: Corrector | None = None,
    results: pd.DataFrame | None = None,
    feedback: list[dict[str, Any]] | None = None,
    correction_score: float | None = None,
    pause_seconds: float = 2.0,
) -> int:
    if pause_seconds < 0:
        raise ValueError("pause seconds cannot be negative")
    sample = load_sequence(dataset_path)
    handedness = Handedness.convert_to_enum(str(sample["handedness"].item()))
    processor = VideoProcessor(
        str(video_path),
        video_path.name,
        str(output_path.parent),
        pose_detector=pose_detector or PoseDetector(),
    )
    tracking = processor.process_frames(handedness)
    start, _, end = (int(value) for value in sample["analysis_window"])
    target_frames = len(sample["skeleton_3d"])
    if "source_frame_indices" in sample:
        frame_indices = _nearest_tracked_frame_indices(
            sample["source_frame_indices"],
            np.asarray(tracking["source_frame_indices"], dtype=np.int64),
        )
        selected_landmarks = [
            tracking["body_landmarks_2d"][int(index)]
            for index in frame_indices
        ]
    else:
        if end >= len(tracking["frames"]):
            raise ValueError(
                f"stored analysis window ends at {end}, but tracker returned "
                f"{len(tracking['frames'])} frames"
            )
        frame_indices = np.rint(
            np.linspace(start, end, target_frames)
        ).astype(np.int64)
        selected_landmarks = tracking["body_landmarks_2d"][start : end + 1]

    raw_2d, raw_confidence = landmark_dicts_to_array(selected_landmarks, 2)
    raw_2d = _interpolate_missing(raw_2d, raw_confidence)
    if handedness == Handedness.LEFT:
        raw_2d, raw_confidence = _canonicalize_left_handed(raw_2d, raw_confidence)

    pixel_2d = resample_sequence(raw_2d, target_frames)
    pixel_confidence = np.clip(
        resample_sequence(raw_confidence, target_frames), 0.0, 1.0
    )
    original_3d = sample["skeleton_3d"].astype(np.float32)
    confidence = sample["confidence"].astype(np.float32)

    model, checkpoint = (
        corrector if corrector is not None else load_corrector(model_path, device)
    )
    dataset_skill = str(sample["skill"].item())
    checkpoint_skill = str(checkpoint.get("skill", "clear"))
    if checkpoint_skill != dataset_skill:
        raise ValueError(
            f"checkpoint skill is {checkpoint_skill}, but dataset skill is "
            f"{dataset_skill}"
        )
    corrected_3d = predict_correction(
        model,
        original_3d,
        confidence,
        device,
        sample["phase_indices"] if checkpoint.get("phase_aligned", False) else None,
        float(checkpoint.get("inference_strength", 1.0)),
        float(checkpoint.get("reference_guidance", 0.0)),
    )
    score: float | None = None
    rows = results
    if rows is None and results_path is not None and results_path.exists():
        rows = pd.read_csv(results_path)
    if rows is not None:
        match = rows[
            (rows["filename"] == video_path.name)
            & (rows["label"] == dataset_path.parent.name)
        ]
        if not match.empty:
            score = float(match.iloc[0]["total_grade"])
    if correction_score is not None:
        score = correction_score

    output_path.parent.mkdir(parents=True, exist_ok=True)
    first_frame = tracking["frames"][int(frame_indices[0])]
    height, width = first_frame.shape[:2]
    source_fps = float(sample["fps"].item())
    fps = source_fps if np.isfinite(source_fps) and source_fps > 0 else 30.0
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer: {output_path}")
    feedback_by_frame: dict[int, list[dict[str, Any]]] = {}
    for issue in feedback or []:
        feedback_by_frame.setdefault(int(issue["frame_index"]), []).append(issue)
    rendered_frames = 0
    try:
        for target_index, frame_index in enumerate(frame_indices):
            frame = tracking["frames"][int(frame_index)].copy()
            alignment_confidence = np.minimum(
                confidence[target_index], pixel_confidence[target_index]
            )
            transform = _fit_affine(
                original_3d[target_index],
                pixel_2d[target_index],
                alignment_confidence,
            )
            corrected_pixels = _map_to_pixels(
                corrected_3d[target_index], transform
            )
            _draw_skeleton(
                frame,
                pixel_2d[target_index],
                alignment_confidence,
                (255, 210, 30),
                4,
            )
            _draw_skeleton(
                frame,
                corrected_pixels,
                alignment_confidence,
                (55, 225, 75),
                3,
            )
            _draw_header(frame, video_path.name, score)
            issues = feedback_by_frame.get(target_index, [])
            if issues:
                _draw_feedback(
                    frame, pixel_2d[target_index], issues, handedness
                )
            repetitions = 1 + (round(fps * pause_seconds) if issues else 0)
            for _ in range(repetitions):
                writer.write(frame)
            rendered_frames += repetitions
    finally:
        writer.release()
    return rendered_frames


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Overlay detected and corrected skeletons on one video clip"
    )
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--model-path")
    parser.add_argument(
        "--results-path",
        default=None,
        help="Optional grading CSV; omitted by default because scores are unvalidated",
    )
    parser.add_argument("--output-path")
    parser.add_argument(
        "--feedback-path",
        help="Optional OpenAI feedback JSON with frame indices and joint IDs",
    )
    parser.add_argument("--pause-seconds", type=float, default=2.0)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_path = Path(args.dataset_path)
    sample = load_sequence(dataset_path)
    spec = get_skill_spec(str(sample["skill"].item()))
    model_path = Path(args.model_path) if args.model_path else spec.model_path
    output_path = (
        Path(args.output_path)
        if args.output_path
        else Path("stats/skeleton_correction")
        / f"{spec.slug}_debug_videos"
        / "corrected_skeleton.mp4"
    )
    feedback_path = Path(args.feedback_path) if args.feedback_path else None
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    frame_count = render_video(
        video_path=Path(args.video_path),
        dataset_path=dataset_path,
        model_path=model_path,
        output_path=output_path,
        results_path=Path(args.results_path) if args.results_path else None,
        device=device,
        feedback=(
            load_feedback_problems(feedback_path, spec)
            if feedback_path
            else None
        ),
        correction_score=(
            load_feedback_display_score(feedback_path) if feedback_path else None
        ),
        pause_seconds=args.pause_seconds,
    )
    print(f"Wrote {frame_count} frames to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
