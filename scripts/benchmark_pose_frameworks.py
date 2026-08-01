from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from badminton_analysis.ml.handedness import (
    estimate_handedness,
    interpolated_keypoint,
)
from badminton_analysis.ml.skeleton_normalization import (
    normalize_skeleton_sequence,
    resample_phase_indices,
    resample_sequence,
)
from badminton_analysis.models.types import COCOKeypoints, Handedness, Skill
from badminton_analysis.services.video_analyzer import VideoAnalyzer


SKILL_VIDEO_DIRS = {
    "serve": (
        "scoring_videos/發球/無經驗同學",
        "scoring_videos/發球/羽球隊同學",
    ),
    "lift": (
        "scoring_videos/挑球/初學者挑球",
        "scoring_videos/挑球/專家挑球",
    ),
    "clear": (
        "scoring_videos/高遠球/初學者高遠球",
        "scoring_videos/高遠球/專家高遠球",
    ),
    "smash": (
        "scoring_videos/殺球/初學者殺球",
        "scoring_videos/殺球/專家殺球",
    ),
}

BONES = np.asarray(
    (
        (5, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    ),
    dtype=np.int64,
)

ANGLE_TRIPLETS = np.asarray(
    (
        (5, 7, 9),
        (6, 8, 10),
        (7, 5, 11),
        (8, 6, 12),
        (5, 11, 13),
        (6, 12, 14),
        (11, 13, 15),
        (12, 14, 16),
    ),
    dtype=np.int64,
)

LEFT_RIGHT_PAIRS = (
    (1, 2),
    (3, 4),
    (5, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
)

H36M_TO_COCO = {
    0: 9,
    1: 10,
    2: 10,
    3: 10,
    4: 10,
    5: 11,
    6: 14,
    7: 12,
    8: 15,
    9: 13,
    10: 16,
    11: 4,
    12: 1,
    13: 5,
    14: 2,
    15: 6,
    16: 3,
}

MEDIAPIPE_TO_COCO = {
    0: 0,
    1: 2,
    2: 5,
    3: 7,
    4: 8,
    5: 11,
    6: 12,
    7: 13,
    8: 14,
    9: 15,
    10: 16,
    11: 23,
    12: 24,
    13: 25,
    14: 26,
    15: 27,
    16: 28,
}

KNOWN_LEFT_HANDED_SAMPLES = {
    ("clear", "beginners", "EG28"),
    ("clear", "beginners", "EG29"),
}


def _natural_key(path: Path) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.stem)
    )


def _evenly_spaced(values: Sequence[Path], count: int) -> list[Path]:
    indices = np.rint(np.linspace(0, len(values) - 1, count)).astype(int)
    return [values[index] for index in indices]


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def create_manifest(
    source_root: Path, dataset_root: Path, output_path: Path
) -> None:
    rows: list[dict[str, str]] = []
    for skill, (beginner_relative, expert_relative) in SKILL_VIDEO_DIRS.items():
        group_values = (
            ("beginners", source_root / beginner_relative),
            ("experts", source_root / expert_relative),
        )
        for group, video_dir in group_values:
            dataset_dir = dataset_root / skill / group
            available = sorted(dataset_dir.glob("*.npz"), key=_natural_key)
            count = 8 if group == "beginners" else 12
            selected = _evenly_spaced(available, count)
            if skill == "clear" and group == "beginners":
                required = [dataset_dir / "EG28.npz", dataset_dir / "EG29.npz"]
                selected = sorted(set((*selected, *required)), key=_natural_key)
            for index, dataset_path in enumerate(selected):
                video_path = video_dir / f"{dataset_path.stem}.mp4"
                if not video_path.exists():
                    alternatives = list(video_dir.glob(f"{dataset_path.stem}.*"))
                    if len(alternatives) != 1:
                        raise FileNotFoundError(video_path)
                    video_path = alternatives[0]
                handedness = (
                    "left"
                    if (skill, group, dataset_path.stem)
                    in KNOWN_LEFT_HANDED_SAMPLES
                    else "right"
                )
                role = (
                    "reference"
                    if group == "experts" and index % 3 != 2
                    else "evaluation"
                )
                rows.append(
                    {
                        "skill": skill,
                        "group": group,
                        "role": role,
                        "subject": dataset_path.stem,
                        "handedness": handedness,
                        "video_path": str(video_path),
                        "baseline_path": str(dataset_path),
                    }
                )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"manifest={output_path} rows={len(rows)}")


def _interpolate_missing(
    sequence: np.ndarray, confidence: np.ndarray
) -> np.ndarray:
    output = np.asarray(sequence, dtype=np.float64).copy()
    timeline = np.arange(len(output), dtype=np.float64)
    for joint in range(output.shape[1]):
        for dimension in range(output.shape[2]):
            values = output[:, joint, dimension]
            valid = (confidence[:, joint] > 0) & np.isfinite(values)
            if not np.any(valid):
                output[:, joint, dimension] = 0.0
            elif np.count_nonzero(valid) == 1:
                output[:, joint, dimension] = values[valid][0]
            else:
                output[:, joint, dimension] = np.interp(
                    timeline, timeline[valid], values[valid]
                )
    return output


def _normalize(
    sequence: np.ndarray, confidence: np.ndarray, handedness: str
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = _interpolate_missing(sequence, confidence)
    observed = np.asarray(confidence, dtype=np.float64).copy()
    if handedness == "left":
        for left, right in LEFT_RIGHT_PAIRS:
            coordinates[:, [left, right]] = coordinates[:, [right, left]]
            observed[:, [left, right]] = observed[:, [right, left]]

    left_shoulder = coordinates[0, 5]
    right_shoulder = coordinates[0, 6]
    root = (coordinates[0, 11] + coordinates[0, 12]) / 2.0
    shoulder_midpoint = (left_shoulder + right_shoulder) / 2.0
    x_axis = right_shoulder - left_shoulder
    x_axis /= max(float(np.linalg.norm(x_axis)), 1e-8)
    spine = shoulder_midpoint - root
    spine -= np.dot(spine, x_axis) * x_axis
    spine /= max(float(np.linalg.norm(spine)), 1e-8)
    if coordinates.shape[-1] == 3:
        z_axis = np.cross(x_axis, spine)
        z_axis /= max(float(np.linalg.norm(z_axis)), 1e-8)
        basis = np.vstack((x_axis, spine, z_axis))
    else:
        basis = np.vstack((x_axis, spine))

    starts, ends = BONES[:, 0], BONES[:, 1]
    lengths = np.linalg.norm(
        coordinates[:, starts] - coordinates[:, ends], axis=-1
    )
    valid = (
        (observed[:, starts] > 0)
        & (observed[:, ends] > 0)
        & np.isfinite(lengths)
        & (lengths > 1e-8)
    )
    scale = float(np.median(lengths[valid])) if np.any(valid) else 1.0
    normalized = np.empty_like(coordinates)
    for frame_index, frame in enumerate(coordinates):
        root = (frame[11] + frame[12]) / 2.0
        normalized[frame_index] = ((basis @ (frame - root).T).T) / scale
    if handedness == "left" and normalized.shape[-1] == 3:
        normalized[..., 2] *= -1.0
    return normalized.astype(np.float32), observed.astype(np.float32)


def _h36m_to_coco(values: np.ndarray) -> np.ndarray:
    output = np.zeros((17, values.shape[-1]), dtype=np.float64)
    for coco_index, h36m_index in H36M_TO_COCO.items():
        output[coco_index] = values[h36m_index]
    return output


def _flatten_predictions(value: Any) -> list[dict[str, Any]]:
    predictions = value if isinstance(value, list) else []
    if len(predictions) == 1 and isinstance(predictions[0], list):
        predictions = predictions[0]
    return [item for item in predictions if isinstance(item, dict)]


def _store_candidate(
    row: dict[str, str],
    output_root: Path,
    candidate: str,
    full_skeleton: np.ndarray,
    full_confidence: np.ndarray,
    full_skeleton_2d: np.ndarray,
    full_confidence_2d: np.ndarray,
    latency_seconds: float,
) -> None:
    metadata = _analysis_metadata(full_skeleton_2d, full_confidence_2d, row)
    start, _, end = metadata["analysis_window"]
    selected_3d = full_skeleton[start : end + 1]
    confidence = np.minimum(
        full_confidence[start : end + 1],
        full_confidence_2d[start : end + 1],
    )
    normalized, normalized_confidence = normalize_skeleton_sequence(
        selected_3d, confidence, metadata["handedness"]
    )
    normalized = resample_sequence(normalized, 64)
    normalized_confidence = np.clip(
        resample_sequence(normalized_confidence, 64), 0.0, 1.0
    )
    normalized_2d, _ = normalize_skeleton_sequence(
        full_skeleton_2d[start : end + 1],
        full_confidence_2d[start : end + 1],
        metadata["handedness"],
    )
    normalized_2d = resample_sequence(normalized_2d, 64)
    source_indices = np.rint(np.linspace(start, end, 64)).astype(np.int64)
    destination = (
        output_root
        / candidate
        / row["skill"]
        / row["group"]
        / f"{row['subject']}.npz"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        full_skeleton_2d=np.asarray(full_skeleton_2d, dtype=np.float32),
        full_confidence_2d=np.asarray(full_confidence_2d, dtype=np.float32),
        skeleton_3d=normalized,
        skeleton_2d=normalized_2d,
        confidence=normalized_confidence,
        phase_indices=metadata["phase_indices"],
        source_frame_indices=source_indices,
        source_phase_indices=metadata["source_phase_indices"],
        analysis_window=metadata["analysis_window"],
        handedness=np.asarray(str(metadata["handedness"])),
        handedness_source=np.asarray(metadata["handedness_source"]),
        skill=np.asarray(row["skill"]),
        video_name=np.asarray(Path(row["video_path"]).name),
        latency_seconds=np.asarray(latency_seconds, dtype=np.float64),
        source_frames=np.asarray(len(full_skeleton), dtype=np.int64),
    )


def _analysis_metadata(
    skeleton_2d: np.ndarray,
    confidence_2d: np.ndarray,
    row: dict[str, str],
    *,
    handedness_override: str | None = None,
) -> dict[str, Any]:
    coordinates = np.asarray(skeleton_2d, dtype=np.float64)
    confidence = np.asarray(confidence_2d, dtype=np.float64)
    if coordinates.ndim != 3 or coordinates.shape[1:] != (17, 2):
        raise ValueError("2D skeleton must have shape (T, 17, 2)")
    if confidence.shape != coordinates.shape[:2]:
        raise ValueError("2D confidence must have shape (T, 17)")
    estimate = estimate_handedness(coordinates, confidence)
    fallback = Handedness.convert_to_enum(row["handedness"])
    if handedness_override is not None:
        handedness = Handedness.convert_to_enum(handedness_override)
        handedness_source = "metadata_reference"
    else:
        handedness = estimate.handedness or fallback
        handedness_source = (
            "wrist_motion" if estimate.handedness is not None else "manifest_fallback"
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
    hand_positions = interpolated_keypoint(coordinates, confidence, wrist)
    elbow_positions = interpolated_keypoint(coordinates, confidence, elbow)
    if len(hand_positions) != len(coordinates) or len(elbow_positions) != len(
        coordinates
    ):
        raise ValueError("dominant wrist or elbow tracking is insufficient")
    start, peak, end = VideoAnalyzer.find_analysis_window(
        skill=Skill.convert_to_enum(row["skill"]),
        hand_positions=list(hand_positions),
        elbow_positions=list(elbow_positions),
    )
    start = max(0, min(len(coordinates) - 1, int(start)))
    peak = max(start, min(len(coordinates) - 1, int(peak)))
    end = max(peak, min(len(coordinates) - 1, int(end)))
    if end - start < 4:
        raise ValueError(f"analysis window is too short: {(start, peak, end)}")
    phase_indices = resample_phase_indices((start, peak, end), 64)
    source_phases = np.asarray(
        (start, (start + peak) // 2, peak, (peak + end) // 2, end),
        dtype=np.int64,
    )
    return {
        "analysis_window": np.asarray((start, peak, end), dtype=np.int64),
        "phase_indices": phase_indices,
        "source_phase_indices": source_phases,
        "handedness": handedness,
        "handedness_source": handedness_source,
        "left_motion_score": estimate.left_motion_score,
        "right_motion_score": estimate.right_motion_score,
        "handedness_confidence_ratio": estimate.confidence_ratio,
    }


class _LargestPersonPose2D:
    """Keep one athlete and retain the 2D evidence used by a pose lifter."""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.skeleton_rows: list[np.ndarray] = []
        self.confidence_rows: list[np.ndarray] = []

    def reset(self) -> None:
        self.skeleton_rows.clear()
        self.confidence_rows.clear()

    def __call__(self, *args: Any, **kwargs: Any) -> Iterable[dict[str, Any]]:
        for result in self.delegate(*args, **kwargs):
            predictions = result.get("predictions", [])
            if not predictions:
                self.skeleton_rows.append(np.full((17, 2), np.nan))
                self.confidence_rows.append(np.zeros(17))
                yield result
                continue

            def area(sample: Any) -> float:
                boxes = np.asarray(sample.pred_instances.bboxes)
                if not boxes.size:
                    return 0.0
                box = boxes.reshape(-1, 4)[0]
                return float(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))

            selected = max(predictions, key=area)
            instances = selected.pred_instances
            keypoints = np.asarray(instances.keypoints).reshape(-1, 17, 2)[0]
            scores = np.asarray(instances.keypoint_scores).reshape(-1, 17)[0]
            self.skeleton_rows.append(keypoints.astype(np.float64))
            self.confidence_rows.append(scores.astype(np.float64))
            filtered = dict(result)
            filtered["predictions"] = [selected]
            yield filtered


def extract_mmpose(
    manifest_path: Path,
    output_root: Path,
    candidate: str,
    device: str,
    limit: int | None = None,
) -> None:
    from mmpose.apis import MMPoseInferencer

    models = {
        "videopose3d": (
            "rtmpose-m_8xb256-420e_coco-256x192",
            "video-pose-lift_tcn-243frm-supv-cpn-ft_8xb128-200e_h36m",
        ),
        "motionbert": (
            "rtmpose-m_8xb256-420e_coco-256x192",
            "motionbert_dstformer-ft-243frm_8xb32-120e_h36m",
        ),
    }
    pose2d, pose3d = models[candidate]
    load_started = time.perf_counter()
    inferencer = MMPoseInferencer(
        pose2d=pose2d,
        pose3d=pose3d,
        device=device,
        show_progress=False,
    )
    pose2d_capture = _LargestPersonPose2D(inferencer.inferencer.pose2d_model)
    inferencer.inferencer.pose2d_model = pose2d_capture
    print(f"candidate={candidate} model_load_seconds={time.perf_counter() - load_started:.3f}")
    rows = _read_manifest(manifest_path)
    if limit is not None:
        rows = rows[:limit]
    for index, row in enumerate(rows, start=1):
        pose2d_capture.reset()
        started = time.perf_counter()
        skeleton_rows: list[np.ndarray] = []
        confidence_rows: list[np.ndarray] = []
        for result in inferencer(
            row["video_path"],
            return_datasamples=False,
            show=False,
            draw_bbox=False,
        ):
            predictions = _flatten_predictions(result.get("predictions", []))
            if not predictions:
                skeleton_rows.append(np.full((17, 3), np.nan))
                confidence_rows.append(np.zeros(17))
                continue
            prediction = predictions[0]
            keypoints = np.asarray(prediction["keypoints"], dtype=np.float64)
            scores = np.asarray(
                prediction.get("keypoint_scores", np.ones(17)), dtype=np.float64
            )
            keypoints = np.squeeze(keypoints)
            scores = np.squeeze(scores)
            skeleton_rows.append(_h36m_to_coco(keypoints))
            confidence_rows.append(_h36m_to_coco(scores[:, None])[:, 0])
        latency = time.perf_counter() - started
        full_skeleton = np.asarray(skeleton_rows, dtype=np.float64)
        full_skeleton_2d = np.asarray(pose2d_capture.skeleton_rows, dtype=np.float64)
        full_confidence_2d = np.asarray(
            pose2d_capture.confidence_rows, dtype=np.float64
        )
        if len(full_skeleton) != len(full_skeleton_2d):
            raise ValueError(
                f"3D/2D frame mismatch for {row['subject']}: "
                f"{len(full_skeleton)} != {len(full_skeleton_2d)}"
            )
        full_confidence = np.minimum(
            np.asarray(confidence_rows, dtype=np.float64), full_confidence_2d
        )
        _store_candidate(
            row,
            output_root,
            candidate,
            full_skeleton,
            full_confidence,
            full_skeleton_2d,
            full_confidence_2d,
            latency,
        )
        print(
            f"{candidate} {index:03d} {row['skill']}/{row['group']}/"
            f"{row['subject']} frames={len(full_skeleton)} latency={latency:.3f}s"
        )


def extract_mediapipe(
    manifest_path: Path,
    output_root: Path,
    model_path: Path,
    limit: int | None = None,
) -> None:
    import mediapipe as mp

    base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    rows = _read_manifest(manifest_path)
    if limit is not None:
        rows = rows[:limit]
    for index, row in enumerate(rows, start=1):
        skeleton_rows: list[np.ndarray] = []
        confidence_rows: list[np.ndarray] = []
        skeleton_2d_rows: list[np.ndarray] = []
        confidence_2d_rows: list[np.ndarray] = []
        capture = cv2.VideoCapture(row["video_path"])
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
        started = time.perf_counter()
        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
            frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(
                    image, round(frame_index * 1000.0 / fps)
                )
                skeleton = np.full((17, 3), np.nan, dtype=np.float64)
                confidence = np.zeros(17, dtype=np.float64)
                skeleton_2d = np.full((17, 2), np.nan, dtype=np.float64)
                confidence_2d = np.zeros(17, dtype=np.float64)
                if result.pose_world_landmarks:
                    world = result.pose_world_landmarks[0]
                    image_landmarks = result.pose_landmarks[0]
                    for coco_index, mediapipe_index in MEDIAPIPE_TO_COCO.items():
                        landmark = world[mediapipe_index]
                        skeleton[coco_index] = (landmark.x, -landmark.y, -landmark.z)
                        image_landmark = image_landmarks[mediapipe_index]
                        skeleton_2d[coco_index] = (
                            image_landmark.x * frame.shape[1],
                            image_landmark.y * frame.shape[0],
                        )
                        visibility = float(getattr(landmark, "visibility", 1.0))
                        presence = float(getattr(landmark, "presence", 1.0))
                        score = min(visibility, presence)
                        confidence[coco_index] = score
                        confidence_2d[coco_index] = score
                skeleton_rows.append(skeleton)
                confidence_rows.append(confidence)
                skeleton_2d_rows.append(skeleton_2d)
                confidence_2d_rows.append(confidence_2d)
                frame_index += 1
        capture.release()
        latency = time.perf_counter() - started
        _store_candidate(
            row,
            output_root,
            "mediapipe",
            np.asarray(skeleton_rows),
            np.asarray(confidence_rows),
            np.asarray(skeleton_2d_rows),
            np.asarray(confidence_2d_rows),
            latency,
        )
        print(
            f"mediapipe {index:03d} {row['skill']}/{row['group']}/"
            f"{row['subject']} frames={len(skeleton_rows)} latency={latency:.3f}s"
        )


def extract_yolo_reference(
    manifest_path: Path,
    output_root: Path,
    model_path: str,
    device: str,
    image_size: int,
    limit: int | None = None,
) -> None:
    from ultralytics import YOLO

    load_started = time.perf_counter()
    model = YOLO(model_path)
    print(
        f"pseudo_labeler=yolo26x model_load_seconds="
        f"{time.perf_counter() - load_started:.3f}"
    )
    rows = _read_manifest(manifest_path)
    if limit is not None:
        rows = rows[:limit]
    for index, row in enumerate(rows, start=1):
        skeleton_rows: list[np.ndarray] = []
        confidence_rows: list[np.ndarray] = []
        started = time.perf_counter()
        predictions = model.predict(
            source=row["video_path"],
            stream=True,
            device=device,
            imgsz=image_size,
            conf=0.1,
            max_det=4,
            verbose=False,
        )
        for result in predictions:
            skeleton = np.full((17, 2), np.nan, dtype=np.float64)
            confidence = np.zeros(17, dtype=np.float64)
            if result.boxes is not None and len(result.boxes):
                boxes = result.boxes.xyxy.detach().cpu().numpy()
                areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
                    0.0, boxes[:, 3] - boxes[:, 1]
                )
                selected = int(np.argmax(areas))
                keypoints = result.keypoints.data[selected].detach().cpu().numpy()
                skeleton = keypoints[:, :2].astype(np.float64)
                confidence = keypoints[:, 2].astype(np.float64)
            skeleton_rows.append(skeleton)
            confidence_rows.append(confidence)
        latency = time.perf_counter() - started
        full_skeleton_2d = np.asarray(skeleton_rows, dtype=np.float64)
        full_confidence_2d = np.asarray(confidence_rows, dtype=np.float64)
        metadata = _analysis_metadata(
            full_skeleton_2d,
            full_confidence_2d,
            row,
            handedness_override=row["handedness"],
        )
        start, _, end = metadata["analysis_window"]
        normalized_2d, normalized_confidence = normalize_skeleton_sequence(
            full_skeleton_2d[start : end + 1],
            full_confidence_2d[start : end + 1],
            metadata["handedness"],
        )
        normalized_2d = resample_sequence(normalized_2d, 64)
        normalized_confidence = np.clip(
            resample_sequence(normalized_confidence, 64), 0.0, 1.0
        )
        destination = (
            output_root
            / "pseudo_yolo26x"
            / row["skill"]
            / row["group"]
            / f"{row['subject']}.npz"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            full_skeleton_2d=full_skeleton_2d.astype(np.float32),
            full_confidence_2d=full_confidence_2d.astype(np.float32),
            skeleton_2d=normalized_2d.astype(np.float32),
            confidence=normalized_confidence.astype(np.float32),
            phase_indices=metadata["phase_indices"],
            source_phase_indices=metadata["source_phase_indices"],
            analysis_window=metadata["analysis_window"],
            handedness=np.asarray(str(metadata["handedness"])),
            handedness_source=np.asarray(metadata["handedness_source"]),
            latency_seconds=np.asarray(latency, dtype=np.float64),
            source_frames=np.asarray(len(full_skeleton_2d), dtype=np.int64),
            image_size=np.asarray(image_size, dtype=np.int64),
            model_path=np.asarray(model_path),
        )
        print(
            f"pseudo_yolo26x {index:03d} {row['skill']}/{row['group']}/"
            f"{row['subject']} frames={len(full_skeleton_2d)} "
            f"latency={latency:.3f}s window={metadata['analysis_window'].tolist()}"
        )


def _sample_path(
    row: dict[str, str], candidate: str, output_root: Path
) -> Path:
    if candidate == "rtmw3d":
        return Path(row["baseline_path"])
    return (
        output_root
        / candidate
        / row["skill"]
        / row["group"]
        / f"{row['subject']}.npz"
    )


def _pseudo_path(row: dict[str, str], output_root: Path) -> Path:
    return (
        output_root
        / "pseudo_yolo26x"
        / row["skill"]
        / row["group"]
        / f"{row['subject']}.npz"
    )


def _pseudo_metadata(sample: Any, row: dict[str, str]) -> dict[str, Any]:
    return _analysis_metadata(
        sample["full_skeleton_2d"].astype(np.float64),
        sample["full_confidence_2d"].astype(np.float64),
        row,
        handedness_override=row["handedness"],
    )


def _normalized_pseudo_sequence(
    sample: Any, row: dict[str, str]
) -> tuple[np.ndarray, np.ndarray]:
    raw = sample["full_skeleton_2d"].astype(np.float64)
    raw_confidence = sample["full_confidence_2d"].astype(np.float64)
    metadata = _pseudo_metadata(sample, row)
    start, _, end = metadata["analysis_window"]
    handedness = metadata["handedness"]
    normalized, confidence = normalize_skeleton_sequence(
        raw[start : end + 1],
        raw_confidence[start : end + 1],
        handedness,
    )
    return (
        resample_sequence(normalized, 64),
        np.clip(resample_sequence(confidence, 64), 0.0, 1.0),
    )


def evaluate_frame_selection(
    manifest_path: Path,
    output_root: Path,
    candidates: Sequence[str],
    details_path: Path,
    summary_path: Path,
) -> None:
    rows = _read_manifest(manifest_path)
    details: list[dict[str, Any]] = []
    for candidate in candidates:
        for row in rows:
            candidate_path = _sample_path(row, candidate, output_root)
            pseudo_path = _pseudo_path(row, output_root)
            if not candidate_path.exists() or not pseudo_path.exists():
                details.append(
                    {"candidate": candidate, **row, "status": "missing"}
                )
                continue
            with np.load(candidate_path, allow_pickle=False) as sample:
                candidate_2d = sample["skeleton_2d"].astype(np.float64)
                candidate_confidence = sample["confidence"].astype(np.float64)
                candidate_phases = sample["phase_indices"].astype(np.int64)
                candidate_source_phases = sample["source_phase_indices"].astype(
                    np.int64
                )
                candidate_window = sample["analysis_window"].astype(np.int64)
                candidate_handedness = str(sample["handedness"].item())
            with np.load(pseudo_path, allow_pickle=False) as pseudo:
                oracle_2d, oracle_confidence = _normalized_pseudo_sequence(
                    pseudo, row
                )
                oracle_metadata = _pseudo_metadata(pseudo, row)
                oracle_phases = oracle_metadata["phase_indices"]
                oracle_source_phases = oracle_metadata["source_phase_indices"]
                oracle_window = oracle_metadata["analysis_window"]
                oracle_handedness = str(oracle_metadata["handedness"])
            candidate_aligned = _phase_align(candidate_2d, candidate_phases)
            oracle_aligned = _phase_align(oracle_2d, oracle_phases)
            confidence = np.minimum(
                _phase_align(candidate_confidence, candidate_phases),
                _phase_align(oracle_confidence, oracle_phases),
            )
            distances = np.linalg.norm(candidate_aligned - oracle_aligned, axis=-1)
            observed = confidence > 0.05
            pose_distance = (
                float(np.sum(distances * confidence) / np.sum(confidence))
                if np.sum(confidence) > 1e-8
                else float("nan")
            )
            pck_02 = (
                float(np.mean(distances[observed] <= 0.2))
                if np.any(observed)
                else float("nan")
            )
            angle_mask = (
                confidence[:, ANGLE_TRIPLETS[:, 0]]
                * confidence[:, ANGLE_TRIPLETS[:, 1]]
                * confidence[:, ANGLE_TRIPLETS[:, 2]]
            )
            angle_error = np.rad2deg(
                np.abs(_angles(candidate_aligned) - _angles(oracle_aligned))
            )
            angle_mae = (
                float(np.sum(angle_error * angle_mask) / np.sum(angle_mask))
                if np.sum(angle_mask) > 1e-8
                else float("nan")
            )
            capture = cv2.VideoCapture(row["video_path"])
            fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
            capture.release()
            phase_error_frames = np.abs(
                candidate_source_phases - oracle_source_phases
            )
            start = max(int(candidate_window[0]), int(oracle_window[0]))
            end = min(int(candidate_window[2]), int(oracle_window[2]))
            intersection = max(0, end - start + 1)
            union = max(
                int(candidate_window[2]), int(oracle_window[2])
            ) - min(int(candidate_window[0]), int(oracle_window[0])) + 1
            details.append(
                {
                    "candidate": candidate,
                    **row,
                    "status": "success",
                    "handedness_agreement": float(
                        candidate_handedness == oracle_handedness
                    ),
                    "checkpoint_mae_frames": float(np.mean(phase_error_frames)),
                    "checkpoint_mae_milliseconds": float(
                        np.mean(phase_error_frames) * 1000.0 / fps
                    ),
                    "checkpoint_recall_100ms": float(
                        np.mean(phase_error_frames <= max(1, round(fps * 0.1)))
                    ),
                    "window_iou": float(intersection / max(union, 1)),
                    "pose_distance_2d": pose_distance,
                    "pck_02": pck_02,
                    "joint_angle_mae_degrees": angle_mae,
                }
            )
    details_path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in details for key in row})
    with details_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(details)

    metrics = (
        "handedness_agreement",
        "checkpoint_mae_frames",
        "checkpoint_mae_milliseconds",
        "checkpoint_recall_100ms",
        "window_iou",
        "pose_distance_2d",
        "pck_02",
        "joint_angle_mae_degrees",
    )
    summaries: list[dict[str, Any]] = []
    for candidate in candidates:
        successful = [
            row
            for row in details
            if row["candidate"] == candidate and row["status"] == "success"
        ]
        for skill in (*SKILL_VIDEO_DIRS, "all"):
            selected = (
                successful
                if skill == "all"
                else [row for row in successful if row["skill"] == skill]
            )
            summary: dict[str, Any] = {
                "candidate": candidate,
                "skill": skill,
                "samples": len(selected),
                "failures": sum(
                    row["candidate"] == candidate
                    and row["status"] != "success"
                    and (skill == "all" or row["skill"] == skill)
                    for row in details
                ),
            }
            for metric in metrics:
                values = np.asarray([row[metric] for row in selected], dtype=float)
                values = values[np.isfinite(values)]
                summary[f"{metric}_mean"] = (
                    float(np.mean(values)) if len(values) else float("nan")
                )
                summary[f"{metric}_p95"] = (
                    float(np.quantile(values, 0.95))
                    if len(values)
                    else float("nan")
                )
            summaries.append(summary)
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    for row in summaries:
        if row["skill"] == "all":
            print(json.dumps(row, sort_keys=True))


def _load_scoring_sequences(
    manifest: Sequence[dict[str, str]],
    output_root: Path,
    candidate: str,
) -> dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]]:
    loaded: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}
    for row in manifest:
        path = (
            _pseudo_path(row, output_root)
            if candidate == "pseudo_yolo26x"
            else _sample_path(row, candidate, output_root)
        )
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as sample:
            if candidate == "pseudo_yolo26x":
                values, confidence = _normalized_pseudo_sequence(sample, row)
                values = np.concatenate(
                    (values, np.zeros((*values.shape[:2], 1), dtype=np.float64)),
                    axis=-1,
                )
                phases = _pseudo_metadata(sample, row)["phase_indices"]
            else:
                values = sample["skeleton_3d"].astype(np.float64)
                confidence = sample["confidence"].astype(np.float64)
                phases = sample["phase_indices"].astype(np.int64)
        loaded[(row["skill"], row["group"], row["subject"])] = (
            _phase_align(values, phases),
            np.clip(_phase_align(confidence, phases), 0.0, 1.0),
        )
    return loaded


def _candidate_grades(
    manifest: Sequence[dict[str, str]],
    output_root: Path,
    candidate: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from badminton_analysis.ml.infer_skeleton_corrector import (
        phase_grading_details,
    )
    from badminton_analysis.ml.skeleton_scoring import (
        correction_distance,
        fit_score_calibration,
        select_bone_adapted_expert,
    )
    from badminton_analysis.ml.skill_specs import get_skill_spec

    loaded = _load_scoring_sequences(manifest, output_root, candidate)
    beginner_calibration: set[tuple[str, str, str]] = set()
    for skill in SKILL_VIDEO_DIRS:
        skill_beginners = sorted(
            (
                row
                for row in manifest
                if row["skill"] == skill and row["group"] == "beginners"
            ),
            key=lambda row: _natural_key(Path(row["subject"])),
        )
        beginner_calibration.update(
            (row["skill"], row["group"], row["subject"])
            for row in skill_beginners[:4]
        )

    raw_rows: list[dict[str, Any]] = []
    for skill in SKILL_VIDEO_DIRS:
        reference_rows = [
            row
            for row in manifest
            if row["skill"] == skill
            and row["group"] == "experts"
            and (skill, row["group"], row["subject"]) in loaded
        ]
        for row in (item for item in manifest if item["skill"] == skill):
            key = (skill, row["group"], row["subject"])
            if key not in loaded:
                continue
            source, source_confidence = loaded[key]
            eligible = [
                reference
                for reference in reference_rows
                if not (
                    row["group"] == "experts"
                    and reference["subject"] == row["subject"]
                )
            ]
            if not eligible:
                continue
            spec = get_skill_spec(skill)
            reference_values = np.stack(
                [loaded[(skill, "experts", item["subject"])][0] for item in eligible]
            )
            reference_confidence = np.stack(
                [loaded[(skill, "experts", item["subject"])][1] for item in eligible]
            )
            selected_index, corrected, score_confidence, total_distance = (
                select_bone_adapted_expert(
                    source,
                    reference_values,
                    source_confidence,
                    reference_confidence,
                    spec.joint_weights_array,
                )
            )
            _, components = correction_distance(
                source,
                corrected,
                score_confidence,
                joint_weights=spec.joint_weights_array,
            )
            split = (
                "calibration"
                if key in beginner_calibration
                else (
                    "expert_cross_validation"
                    if row["group"] == "experts"
                    else "test"
                )
            )
            raw_rows.append(
                {
                    "candidate": candidate,
                    **row,
                    "split": split,
                    "nearest_expert": eligible[selected_index]["subject"],
                    "nearest_expert_distance": total_distance,
                    "raw_score_distance": total_distance,
                    **components,
                    "_source": source,
                    "_corrected": corrected,
                    "_confidence": score_confidence,
                }
            )

    calibrations: dict[str, Any] = {}
    for skill in SKILL_VIDEO_DIRS:
        skill_rows = [row for row in raw_rows if row["skill"] == skill]
        expert_distances = np.asarray(
            [
                row["raw_score_distance"]
                for row in skill_rows
                if row["group"] == "experts"
            ]
        )
        beginner_distances = np.asarray(
            [
                row["raw_score_distance"]
                for row in skill_rows
                if row["group"] == "beginners" and row["split"] == "calibration"
            ]
        )
        calibration = fit_score_calibration(expert_distances, beginner_distances)
        calibrations[skill] = calibration
        spec = get_skill_spec(skill)
        for row in skill_rows:
            total_score = float(calibration.score(row["raw_score_distance"]))
            criteria = phase_grading_details(
                row.pop("_source"),
                row.pop("_corrected"),
                row.pop("_confidence"),
                calibration,
                total_grade=total_score,
                spec=spec,
            )
            row["total_score"] = total_score
            for criterion_index, (name, distance, grade) in enumerate(criteria):
                row[f"criterion_{criterion_index}_name"] = name
                row[f"criterion_{criterion_index}_distance"] = distance
                row[f"criterion_{criterion_index}_score"] = grade
    return raw_rows, calibrations


def evaluate_grading(
    manifest_path: Path,
    output_root: Path,
    candidates: Sequence[str],
    details_path: Path,
    summary_path: Path,
) -> None:
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    manifest = _read_manifest(manifest_path)
    evaluated_candidates = ("pseudo_yolo26x", *candidates)
    all_rows: list[dict[str, Any]] = []
    calibration_values: dict[tuple[str, str], Any] = {}
    for candidate in evaluated_candidates:
        rows, calibrations = _candidate_grades(manifest, output_root, candidate)
        all_rows.extend(rows)
        for skill, calibration in calibrations.items():
            calibration_values[(candidate, skill)] = calibration

    oracle = {
        (row["skill"], row["group"], row["subject"]): row
        for row in all_rows
        if row["candidate"] == "pseudo_yolo26x"
    }
    summaries: list[dict[str, Any]] = []
    public_rows: list[dict[str, Any]] = []
    for row in all_rows:
        public = dict(row)
        reference = oracle.get((row["skill"], row["group"], row["subject"]))
        if reference is not None:
            public["oracle_total_score"] = reference["total_score"]
            public["total_score_absolute_error"] = abs(
                row["total_score"] - reference["total_score"]
            )
            criterion_errors = []
            index = 0
            while f"criterion_{index}_score" in row:
                error = abs(
                    row[f"criterion_{index}_score"]
                    - reference[f"criterion_{index}_score"]
                )
                public[f"criterion_{index}_absolute_error"] = error
                criterion_errors.append(error)
                index += 1
            public["criterion_score_mae"] = (
                float(np.mean(criterion_errors)) if criterion_errors else float("nan")
            )
        public_rows.append(public)

    oracle_calibration_keys = {
        (row["skill"], row["group"], row["subject"])
        for row in public_rows
        if row["candidate"] == "pseudo_yolo26x"
        and row["split"] == "calibration"
    }

    for candidate in evaluated_candidates:
        for skill in (*SKILL_VIDEO_DIRS, "all"):
            expected_evaluation = sum(
                1
                for manifest_row in manifest
                if (skill == "all" or manifest_row["skill"] == skill)
                and (
                    manifest_row["group"] == "experts"
                    or (
                        manifest_row["skill"],
                        manifest_row["group"],
                        manifest_row["subject"],
                    )
                    not in oracle_calibration_keys
                )
            )
            selected = [
                row
                for row in public_rows
                if row["candidate"] == candidate
                and row["split"] in ("test", "expert_cross_validation")
                and (skill == "all" or row["skill"] == skill)
            ]
            experts = [row for row in selected if row["group"] == "experts"]
            beginners = [row for row in selected if row["group"] == "beginners"]
            calibration_beginners = [
                row
                for row in public_rows
                if row["candidate"] == candidate
                and row["split"] == "calibration"
                and (skill == "all" or row["skill"] == skill)
            ]
            oracle_scores = np.asarray(
                [row["oracle_total_score"] for row in selected], dtype=float
            )
            scores = np.asarray([row["total_score"] for row in selected], dtype=float)
            labels = np.asarray(
                [int(row["group"] == "beginners") for row in selected], dtype=int
            )
            correlation = (
                float(spearmanr(scores, oracle_scores).statistic)
                if len(scores) >= 3 and np.std(scores) > 0 and np.std(oracle_scores) > 0
                else float("nan")
            )
            auc = (
                float(roc_auc_score(labels, -scores))
                if len(set(labels.tolist())) == 2
                else float("nan")
            )
            summary: dict[str, Any] = {
                "candidate": candidate,
                "skill": skill,
                "test_samples": len(selected),
                "test_experts": len(experts),
                "test_beginners": len(beginners),
                "expected_evaluation_samples": expected_evaluation,
                "missing_evaluation_samples": expected_evaluation - len(selected),
                "total_score_mae": (
                    float(np.mean(np.abs(scores - oracle_scores)))
                    if len(scores)
                    else float("nan")
                ),
                "criterion_score_mae": (
                    float(np.mean([row["criterion_score_mae"] for row in selected]))
                    if selected
                    else float("nan")
                ),
                "score_spearman": correlation,
                "expert_beginner_auc": auc,
                "expert_score_mean": (
                    float(np.mean([row["total_score"] for row in experts]))
                    if experts
                    else float("nan")
                ),
                "beginner_score_mean": (
                    float(np.mean([row["total_score"] for row in beginners]))
                    if beginners
                    else float("nan")
                ),
                "calibration_beginner_score_mean": (
                    float(
                        np.mean(
                            [row["total_score"] for row in calibration_beginners]
                        )
                    )
                    if calibration_beginners
                    else float("nan")
                ),
            }
            if skill != "all":
                calibration = calibration_values[(candidate, skill)]
                summary.update(
                    calibration_distance_offset=calibration.distance_offset,
                    calibration_alpha=calibration.alpha,
                    calibration_target_reachable=calibration.target_reachable,
                )
            summaries.append(summary)

    details_path.parent.mkdir(parents=True, exist_ok=True)
    detail_fields = sorted({key for row in public_rows for key in row})
    with details_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=detail_fields)
        writer.writeheader()
        writer.writerows(public_rows)
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    for row in summaries:
        if row["skill"] == "all":
            print(json.dumps(row, sort_keys=True))


def _phase_align(values: np.ndarray, phases: np.ndarray) -> np.ndarray:
    target_phases = np.asarray((0, 16, 32, 48, 63), dtype=np.float64)
    positions = np.interp(np.arange(64), target_phases, phases.astype(float))
    timeline = np.arange(64, dtype=np.float64)
    flat = values.reshape(64, -1)
    output = np.empty_like(flat, dtype=np.float64)
    for column in range(flat.shape[1]):
        output[:, column] = np.interp(positions, timeline, flat[:, column])
    return output.reshape(values.shape)


def _angles(values: np.ndarray) -> np.ndarray:
    first = values[:, ANGLE_TRIPLETS[:, 0]]
    center = values[:, ANGLE_TRIPLETS[:, 1]]
    last = values[:, ANGLE_TRIPLETS[:, 2]]
    a = first - center
    b = last - center
    denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    cosine = np.divide(
        np.sum(a * b, axis=-1),
        denominator,
        out=np.ones_like(denominator),
        where=denominator > 1e-8,
    )
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def _draw_projected_skeleton(
    panel: np.ndarray, skeleton: np.ndarray, title: str
) -> None:
    height, width = panel.shape[:2]
    points = skeleton[:, :2].copy()
    low = np.min(points, axis=0)
    high = np.max(points, axis=0)
    span = np.maximum(high - low, 1e-6)
    scale = min((width - 36) / span[0], (height - 58) / span[1])
    points = (points - (low + high) / 2.0) * scale
    points[:, 0] += width / 2.0
    points[:, 1] = -points[:, 1] + (height + 22) / 2.0
    points = np.rint(points).astype(int)
    for start, end in BONES:
        cv2.line(
            panel,
            tuple(points[start]),
            tuple(points[end]),
            (75, 225, 95),
            3,
            cv2.LINE_AA,
        )
    for point in points:
        cv2.circle(panel, tuple(point), 4, (30, 210, 255), -1, cv2.LINE_AA)
    cv2.putText(
        panel,
        title,
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )


def _draw_pixel_skeleton(
    frame: np.ndarray,
    skeleton: np.ndarray,
    confidence: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    points = np.rint(skeleton).astype(np.int64)
    observed = np.asarray(confidence) > 0.1
    for start, end in BONES:
        if observed[start] and observed[end]:
            cv2.line(
                frame,
                tuple(points[start]),
                tuple(points[end]),
                color,
                4,
                cv2.LINE_AA,
            )
    for index, point in enumerate(points):
        if observed[index]:
            cv2.circle(frame, tuple(point), 5, color, -1, cv2.LINE_AA)


def render_frame_comparison(
    manifest_path: Path,
    output_root: Path,
    candidate: str,
    subject: str,
    skill: str,
    destination: Path,
) -> None:
    matches = [
        row
        for row in _read_manifest(manifest_path)
        if row["subject"] == subject and row["skill"] == skill
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one manifest row for {skill}/{subject}")
    row = matches[0]
    candidate_path = _sample_path(row, candidate, output_root)
    if not candidate_path.exists():
        raise FileNotFoundError(candidate_path)
    with np.load(candidate_path, allow_pickle=False) as sample:
        if "full_skeleton_2d" not in sample.files:
            raise ValueError(
                f"{candidate} does not retain raw 2D coordinates; rerun its extractor"
            )
        candidate_2d = sample["full_skeleton_2d"].astype(np.float64)
        candidate_confidence = sample["full_confidence_2d"].astype(np.float64)
        candidate_phases = sample["source_phase_indices"].astype(np.int64)
    with np.load(_pseudo_path(row, output_root), allow_pickle=False) as pseudo:
        oracle_2d = pseudo["full_skeleton_2d"].astype(np.float64)
        oracle_confidence = pseudo["full_confidence_2d"].astype(np.float64)
        oracle_phases = _pseudo_metadata(pseudo, row)["source_phase_indices"]

    capture = cv2.VideoCapture(row["video_path"])
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    frame_count = min(
        int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        len(candidate_2d),
        len(oracle_2d),
    )
    width, height = 1280, 720
    content_height = 620
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw_path = destination.with_name(destination.stem + ".raw.mp4")
    writer = cv2.VideoWriter(
        str(raw_path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (width, height)
    )
    frame_index = 0
    try:
        while frame_index < frame_count:
            ok, source = capture.read()
            if not ok:
                break
            annotated = source.copy()
            _draw_pixel_skeleton(
                annotated,
                oracle_2d[frame_index],
                oracle_confidence[frame_index],
                (60, 225, 90),
            )
            _draw_pixel_skeleton(
                annotated,
                candidate_2d[frame_index],
                candidate_confidence[frame_index],
                (255, 190, 45),
            )
            scale = min(width / annotated.shape[1], content_height / annotated.shape[0])
            resized = cv2.resize(
                annotated,
                (
                    max(1, round(annotated.shape[1] * scale)),
                    max(1, round(annotated.shape[0] * scale)),
                ),
            )
            canvas = np.full((height, width, 3), 18, dtype=np.uint8)
            left = (width - resized.shape[1]) // 2
            top = (content_height - resized.shape[0]) // 2
            canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
            cv2.putText(
                canvas,
                f"{skill} / {subject}  green: YOLO26x pseudo  cyan: {candidate}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.66,
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )
            line_start, line_end, line_y = 40, width - 40, 683
            cv2.line(canvas, (line_start, line_y), (line_end, line_y), (110, 110, 110), 2)
            for phases, color, tick_top in (
                (oracle_phases, (60, 225, 90), 651),
                (candidate_phases, (255, 190, 45), 666),
            ):
                for phase_index in phases:
                    x = line_start + round(
                        int(phase_index) * (line_end - line_start) / max(frame_count - 1, 1)
                    )
                    cv2.line(canvas, (x, tick_top), (x, line_y + 9), color, 3)
            cursor = line_start + round(
                frame_index * (line_end - line_start) / max(frame_count - 1, 1)
            )
            cv2.circle(canvas, (cursor, line_y), 6, (245, 245, 245), -1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                f"frame {frame_index:03d}/{frame_count - 1:03d}",
                (20, 645),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (225, 225, 225),
                1,
                cv2.LINE_AA,
            )
            writer.write(canvas)
            frame_index += 1
    finally:
        capture.release()
        writer.release()
    import subprocess

    subprocess.run(
        (
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(raw_path),
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ),
        check=True,
    )
    raw_path.unlink(missing_ok=True)
    print(destination)


def render(
    manifest_path: Path,
    output_root: Path,
    candidates: Sequence[str],
    subject: str,
    skill: str,
    destination: Path,
) -> None:
    rows = [
        row
        for row in _read_manifest(manifest_path)
        if row["subject"] == subject and row["skill"] == skill
    ]
    if len(rows) != 1:
        raise ValueError(f"expected one manifest row for {skill}/{subject}")
    row = rows[0]
    samples: list[tuple[str, np.ndarray]] = []
    source_indices: np.ndarray | None = None
    for candidate in candidates:
        path = _sample_path(row, candidate, output_root)
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as sample:
            samples.append((candidate, sample["skeleton_3d"].astype(np.float64)))
            if source_indices is None:
                source_indices = sample["source_frame_indices"].astype(np.int64)
    if source_indices is None:
        raise ValueError("no candidate outputs are available")
    capture = cv2.VideoCapture(row["video_path"])
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    source_frames: list[np.ndarray] = []
    index_set = set(int(value) for value in source_indices)
    frame_index = 0
    selected_by_index: dict[int, np.ndarray] = {}
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index in index_set:
            selected_by_index[frame_index] = frame
        frame_index += 1
    capture.release()
    source_frames = [selected_by_index[int(index)] for index in source_indices]

    width, height = 1280, 720
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw_path = destination.with_name(destination.stem + ".raw.mp4")
    writer = cv2.VideoWriter(
        str(raw_path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (width, height)
    )
    try:
        for frame_number, source in enumerate(source_frames):
            canvas = np.full((height, width, 3), 18, dtype=np.uint8)
            source_height = height
            source_width = 430
            scale = min(source_width / source.shape[1], source_height / source.shape[0])
            resized = cv2.resize(
                source,
                (round(source.shape[1] * scale), round(source.shape[0] * scale)),
            )
            x = (source_width - resized.shape[1]) // 2
            y = (source_height - resized.shape[0]) // 2
            canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
            panel_width = (width - source_width) // 2
            panel_height = height // 2
            for panel_index, (candidate, skeleton) in enumerate(samples[:4]):
                row_index, column_index = divmod(panel_index, 2)
                left = source_width + column_index * panel_width
                top = row_index * panel_height
                panel = canvas[
                    top : top + panel_height, left : left + panel_width
                ]
                cv2.rectangle(
                    panel,
                    (0, 0),
                    (panel_width - 1, panel_height - 1),
                    (65, 65, 65),
                    1,
                )
                _draw_projected_skeleton(
                    panel, skeleton[frame_number], candidate
                )
            writer.write(canvas)
    finally:
        writer.release()
    import subprocess

    subprocess.run(
        (
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(raw_path),
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ),
        check=True,
    )
    raw_path.unlink(missing_ok=True)
    print(destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproducible badminton 3D pose framework benchmark"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--source-root", type=Path, required=True)
    manifest.add_argument("--dataset-root", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)

    mmpose = subparsers.add_parser("extract-mmpose")
    mmpose.add_argument("--manifest", type=Path, required=True)
    mmpose.add_argument("--output-root", type=Path, required=True)
    mmpose.add_argument(
        "--candidate", choices=("videopose3d", "motionbert"), required=True
    )
    mmpose.add_argument("--device", default="cuda:0")
    mmpose.add_argument("--limit", type=int)

    pseudo = subparsers.add_parser("extract-yolo-reference")
    pseudo.add_argument("--manifest", type=Path, required=True)
    pseudo.add_argument("--output-root", type=Path, required=True)
    pseudo.add_argument("--model", default="yolo26x-pose.pt")
    pseudo.add_argument("--device", default="0")
    pseudo.add_argument("--image-size", type=int, default=1280)
    pseudo.add_argument("--limit", type=int)

    mediapipe = subparsers.add_parser("extract-mediapipe")
    mediapipe.add_argument("--manifest", type=Path, required=True)
    mediapipe.add_argument("--output-root", type=Path, required=True)
    mediapipe.add_argument("--model", type=Path, required=True)
    mediapipe.add_argument("--limit", type=int)

    frame_evaluation = subparsers.add_parser("evaluate-frame-selection")
    frame_evaluation.add_argument("--manifest", type=Path, required=True)
    frame_evaluation.add_argument("--output-root", type=Path, required=True)
    frame_evaluation.add_argument("--candidates", nargs="+", required=True)
    frame_evaluation.add_argument("--details", type=Path, required=True)
    frame_evaluation.add_argument("--summary", type=Path, required=True)

    grading = subparsers.add_parser("evaluate-grading")
    grading.add_argument("--manifest", type=Path, required=True)
    grading.add_argument("--output-root", type=Path, required=True)
    grading.add_argument("--candidates", nargs="+", required=True)
    grading.add_argument("--details", type=Path, required=True)
    grading.add_argument("--summary", type=Path, required=True)

    rendering = subparsers.add_parser("render")
    rendering.add_argument("--manifest", type=Path, required=True)
    rendering.add_argument("--output-root", type=Path, required=True)
    rendering.add_argument("--candidates", nargs="+", required=True)
    rendering.add_argument("--subject", required=True)
    rendering.add_argument("--skill", choices=tuple(SKILL_VIDEO_DIRS), required=True)
    rendering.add_argument("--output", type=Path, required=True)

    frame_rendering = subparsers.add_parser("render-frame-comparison")
    frame_rendering.add_argument("--manifest", type=Path, required=True)
    frame_rendering.add_argument("--output-root", type=Path, required=True)
    frame_rendering.add_argument("--candidate", required=True)
    frame_rendering.add_argument("--subject", required=True)
    frame_rendering.add_argument(
        "--skill", choices=tuple(SKILL_VIDEO_DIRS), required=True
    )
    frame_rendering.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "manifest":
        create_manifest(args.source_root, args.dataset_root, args.output)
    elif args.command == "extract-mmpose":
        extract_mmpose(
            args.manifest,
            args.output_root,
            args.candidate,
            args.device,
            args.limit,
        )
    elif args.command == "extract-yolo-reference":
        extract_yolo_reference(
            args.manifest,
            args.output_root,
            args.model,
            args.device,
            args.image_size,
            args.limit,
        )
    elif args.command == "extract-mediapipe":
        extract_mediapipe(
            args.manifest, args.output_root, args.model, args.limit
        )
    elif args.command == "evaluate-frame-selection":
        evaluate_frame_selection(
            args.manifest,
            args.output_root,
            args.candidates,
            args.details,
            args.summary,
        )
    elif args.command == "evaluate-grading":
        evaluate_grading(
            args.manifest,
            args.output_root,
            args.candidates,
            args.details,
            args.summary,
        )
    elif args.command == "render":
        render(
            args.manifest,
            args.output_root,
            args.candidates,
            args.subject,
            args.skill,
            args.output,
        )
    elif args.command == "render-frame-comparison":
        render_frame_comparison(
            args.manifest,
            args.output_root,
            args.candidate,
            args.subject,
            args.skill,
            args.output,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
