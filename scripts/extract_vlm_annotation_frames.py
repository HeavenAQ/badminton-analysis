"""Extract key-frame images from badminton videos for LLM/VLM annotation.

The script walks scoring videos and expert NSTC training videos, finds the same
five analysis key frames used by `descrptive_analysis.py`, draws pose
keypoints, and writes images plus annotation templates to
`llm-annotations/`.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from badminton_analysis.models.types import Handedness, Skill
from badminton_analysis.services.pose_detector import PoseDetector
from badminton_analysis.services.video_analyzer import VideoAnalyzer
from badminton_analysis.services.video_processor import VideoProcessor


SKILL_FOLDER_MAP: dict[str, Skill] = {
    "發球": Skill.SERVE,
    "高遠球": Skill.CLEAR,
    "殺球": Skill.SMASH,
    "挑球": Skill.LIFT,
    "serve": Skill.SERVE,
    "clear": Skill.CLEAR,
    "smash": Skill.SMASH,
    "lift": Skill.LIFT,
}

SKILL_DISPLAY_NAME: dict[Skill, str] = {
    Skill.SERVE: "serve",
    Skill.CLEAR: "clear",
    Skill.SMASH: "smash",
    Skill.LIFT: "lift",
}

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
KEY_FRAMES = [
    ("key_frame_0_start", "Start of analysis window"),
    ("key_frame_1_mid_start_peak", "Midpoint between start and peak"),
    ("key_frame_2_peak", "Peak/impact frame"),
    ("key_frame_3_mid_peak_end", "Midpoint between peak and end"),
    ("key_frame_4_end", "End of analysis window"),
]
ANNOTATION_FIELDS = [
    "sample_id",
    "source_dataset",
    "skill_zh",
    "skill",
    "handedness",
    "handedness_source",
    "cohort",
    "source_group",
    "video_file",
    "key_frame_index",
    "key_frame_name",
    "neighbor_offset",
    "frame_index",
    "image_path",
    "angles_json",
    "score",
    "feedback",
    "correction_suggestion",
    "usable_for_training",
    "annotator",
    "notes",
]


@dataclass(frozen=True)
class VideoJob:
    source_dataset: str
    skill_zh: str
    skill: Skill
    source_group: str
    cohort: str
    video_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract pose-overlay key frames from scoring_videos and create "
            "VLM annotation templates."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("scoring_videos"),
        help="Root containing Chinese skill folders.",
    )
    parser.add_argument(
        "--nstc-root",
        type=Path,
        default=Path("training_videos/nstc"),
        help="Optional NSTC expert-video root to include.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("llm-annotations"),
        help="Destination for extracted key-frame images and annotation files.",
    )
    parser.add_argument(
        "--handedness",
        type=lambda s: Handedness[s.upper()],
        default=Handedness.RIGHT,
        help=(
            "Fallback dominant hand used when the video filename does not end "
            f"with left/right. Choices: {[h.name for h in Handedness]}"
        ),
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove generated annotation CSV/JSONL/schema files before writing new ones.",
    )
    parser.add_argument(
        "--frame-radius",
        type=int,
        default=2,
        help="Number of neighboring frames to save before and after each detected key frame.",
    )
    return parser.parse_args()


def safe_name(value: str) -> str:
    """Keep Chinese/readable names, replacing path-hostile characters."""
    value = re.sub(r"[\\/:\*\?\"<>\|]+", "_", value.strip())
    value = re.sub(r"\s+", "_", value)
    return value or "unnamed"


def infer_cohort(folder_name: str) -> str:
    expert_markers = ("專家", "羽球隊", "expert", "advanced")
    beginner_markers = ("初學", "無經驗", "beginner", "novice")
    lower_name = folder_name.lower()

    if any(marker in lower_name for marker in expert_markers):
        return "expert"
    if any(marker in lower_name for marker in beginner_markers):
        return "beginner"
    return safe_name(folder_name)


def infer_handedness_from_filename(video_path: Path) -> Handedness | None:
    """Infer handedness from an explicit left/right filename suffix."""
    stem = video_path.stem.lower()
    parts = [part for part in re.split(r"[\s_\-]+", stem) if part]
    if not parts:
        return None

    suffix = parts[-1]
    if suffix == "left":
        return Handedness.LEFT
    if suffix == "right":
        return Handedness.RIGHT
    return None


def infer_handedness_from_path(video_path: Path) -> Handedness | None:
    for part in reversed(video_path.parts[:-1]):
        lower_part = part.lower()
        if lower_part == "left":
            return Handedness.LEFT
        if lower_part == "right":
            return Handedness.RIGHT
    return infer_handedness_from_filename(video_path)


def iter_scoring_video_jobs(input_root: Path) -> list[VideoJob]:
    jobs: list[VideoJob] = []
    for skill_dir in sorted(p for p in input_root.iterdir() if p.is_dir()):
        if skill_dir.name == "results":
            continue
        skill = SKILL_FOLDER_MAP.get(skill_dir.name)
        if skill is None:
            print(f"Skipping unsupported skill folder: {skill_dir}")
            continue

        for group_dir in sorted(p for p in skill_dir.iterdir() if p.is_dir()):
            cohort = infer_cohort(group_dir.name)
            for video_path in sorted(group_dir.iterdir()):
                if video_path.is_file() and video_path.suffix.lower() in VIDEO_EXTENSIONS:
                    jobs.append(
                        VideoJob(
                            source_dataset="scoring_videos",
                            skill_zh=skill_dir.name,
                            skill=skill,
                            source_group=group_dir.name,
                            cohort=cohort,
                            video_path=video_path,
                        )
                    )
    return jobs


def iter_nstc_video_jobs(nstc_root: Path) -> list[VideoJob]:
    if not nstc_root.exists():
        return []

    jobs: list[VideoJob] = []
    handedness_groups = {"left", "right"}
    for skill_dir in sorted(p for p in nstc_root.iterdir() if p.is_dir()):
        skill = SKILL_FOLDER_MAP.get(skill_dir.name.lower())
        if skill is None:
            print(f"Skipping unsupported NSTC skill folder: {skill_dir}")
            continue

        for group_dir in sorted(p for p in skill_dir.iterdir() if p.is_dir()):
            if group_dir.name.lower() not in handedness_groups:
                print(f"Skipping NSTC non-handedness folder: {group_dir}")
                continue

            for video_path in sorted(group_dir.iterdir()):
                if video_path.is_file() and video_path.suffix.lower() in VIDEO_EXTENSIONS:
                    jobs.append(
                        VideoJob(
                            source_dataset="training_videos/nstc",
                            skill_zh=skill_dir.name,
                            skill=skill,
                            source_group=group_dir.name,
                            cohort="expert",
                            video_path=video_path,
                        )
                    )
    return jobs


def get_target_indices(
    processor: VideoProcessor,
    analyzer: VideoAnalyzer,
    handedness: Handedness,
    skill: Skill,
) -> list[int]:
    processor.process_frames(handedness)
    if not processor.frames:
        raise RuntimeError("No pose-tracked frames were extracted")

    start, peak, end = analyzer.find_analysis_window(
        skill=skill,
        hand_positions=processor.hand_positions,
        elbow_positions=processor.elbow_positions,
    )
    indices = [start, (start + peak) // 2, peak, (peak + end) // 2, end]
    return [max(0, min(index, len(processor.frames) - 1)) for index in indices]


def write_key_frame(
    processor: VideoProcessor,
    frame_index: int,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = processor.frames[frame_index].copy()
    processor.pose_detector.show_pose(frame, processor.landmarks[frame_index])
    success = cv2.imwrite(str(output_path), frame)
    if not success:
        raise RuntimeError(f"Failed to write frame image to {output_path}")


def annotation_row(
    *,
    job: VideoJob,
    handedness: Handedness,
    handedness_source: str,
    key_frame_index: int,
    key_frame_dir: str,
    neighbor_offset: int,
    frame_index: int,
    image_path: Path,
    output_root: Path,
    angles: dict[str, float],
) -> dict[str, Any]:
    sample_id = (
        f"{safe_name(job.source_dataset)}__{safe_name(job.skill_zh)}__"
        f"{safe_name(job.source_group)}__{safe_name(job.video_path.stem)}__"
        f"kf{key_frame_index}__off{neighbor_offset:+d}__frame{frame_index}"
    )
    return {
        "sample_id": sample_id,
        "source_dataset": job.source_dataset,
        "skill_zh": job.skill_zh,
        "skill": str(job.skill),
        "handedness": str(handedness),
        "handedness_source": handedness_source,
        "cohort": job.cohort,
        "source_group": job.source_group,
        "video_file": job.video_path.name,
        "key_frame_index": key_frame_index,
        "key_frame_name": key_frame_dir,
        "neighbor_offset": neighbor_offset,
        "frame_index": frame_index,
        "image_path": image_path.relative_to(output_root).as_posix(),
        "angles_json": json.dumps(angles, ensure_ascii=False, sort_keys=True),
        "score": "",
        "feedback": "",
        "correction_suggestion": "",
        "usable_for_training": "yes",
        "annotator": "",
        "notes": "",
    }


def write_annotation_files(output_root: Path, rows: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    csv_path = output_root / "annotation_template.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=ANNOTATION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    jsonl_path = output_root / "annotation_template.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fp:
        for row in rows:
            record = {
                "sample_id": row["sample_id"],
                "image": row["image_path"],
                "metadata": {
                    "source_dataset": row["source_dataset"],
                    "skill_zh": row["skill_zh"],
                    "skill": row["skill"],
                    "handedness": row["handedness"],
                    "handedness_source": row["handedness_source"],
                    "cohort": row["cohort"],
                    "source_group": row["source_group"],
                    "video_file": row["video_file"],
                    "key_frame_index": row["key_frame_index"],
                    "key_frame_name": row["key_frame_name"],
                    "neighbor_offset": row["neighbor_offset"],
                    "frame_index": row["frame_index"],
                    "angles": json.loads(row["angles_json"]),
                },
                "expert_annotation": {
                    "score": "",
                    "feedback": "",
                    "correction_suggestion": "",
                    "usable_for_training": "yes",
                    "notes": "",
                },
                "vlm_sft_messages": [
                    {
                        "role": "user",
                        "content": (
                            "Evaluate this badminton key-frame image for the given "
                            "skill and checkpoint. Provide a score and concise "
                            "technical feedback."
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": {
                            "score": "",
                            "feedback": "",
                            "correction_suggestion": "",
                        },
                    },
                ],
            }
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    schema_path = output_root / "annotation_schema.json"
    schema = {
        "description": "Expert annotation schema for badminton VLM SFT key-frame images.",
        "score": {
            "type": "number",
            "recommended_range": "0-10",
            "meaning": "Technical quality score for this key frame and skill checkpoint.",
        },
        "feedback": {
            "type": "string",
            "meaning": "Short expert diagnosis of visible technique in the frame.",
        },
        "correction_suggestion": {
            "type": "string",
            "meaning": "Actionable correction the athlete should apply.",
        },
        "usable_for_training": {
            "type": "string",
            "allowed_values": ["yes", "no"],
            "meaning": "Whether this image/annotation pair should be used for VLM SFT.",
        },
        "notes": {
            "type": "string",
            "meaning": "Optional comments such as occlusion, bad pose tracking, or ambiguous checkpoint.",
        },
        "neighbor_offset": {
            "type": "integer",
            "meaning": "Frame offset from the detected key frame; 0 is the detected checkpoint.",
        },
        "angles_json": {
            "type": "object",
            "meaning": "Joint angles computed directly from detected landmarks for this exact frame.",
        },
        "key_frames": [
            {
                "index": index,
                "name": name,
                "description": description,
            }
            for index, (name, description) in enumerate(KEY_FRAMES)
        ],
    }
    schema_path.write_text(
        json.dumps(schema, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {csv_path}")
    print(f"Wrote {jsonl_path}")
    print(f"Wrote {schema_path}")


def clean_annotation_files(output_root: Path) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)


def process_job(
    job: VideoJob,
    output_root: Path,
    handedness: Handedness,
    analyzer: VideoAnalyzer,
    pose_detector: PoseDetector,
    frame_radius: int,
) -> list[dict[str, Any]]:
    print(f"Processing: {job.video_path}")
    inferred_handedness = infer_handedness_from_path(job.video_path)
    if inferred_handedness is None:
        video_handedness = handedness
        handedness_source = "fallback"
    else:
        video_handedness = inferred_handedness
        handedness_source = "path_or_filename"

    processor = VideoProcessor(
        str(job.video_path),
        job.video_path.name,
        str(output_root),
        pose_detector=pose_detector,
    )
    target_indices = get_target_indices(processor, analyzer, video_handedness, job.skill)
    rows: list[dict[str, Any]] = []

    for key_frame_index, center_frame_index in enumerate(target_indices):
        key_frame_dir, _ = KEY_FRAMES[key_frame_index]
        for offset in range(-frame_radius, frame_radius + 1):
            frame_index = max(
                0,
                min(center_frame_index + offset, len(processor.frames) - 1),
            )
            image_name = (
                f"{safe_name(job.source_dataset)}__{safe_name(job.source_group)}__"
                f"{safe_name(job.video_path.stem)}__kf{key_frame_index}__"
                f"off{offset:+d}__frame{frame_index}.jpg"
            )
            image_path = (
                output_root
                / SKILL_DISPLAY_NAME[job.skill]
                / key_frame_dir
                / image_name
            )
            write_key_frame(processor, frame_index, image_path)
            angles = analyzer.compute_angles(processor.landmarks[frame_index])
            row = annotation_row(
                job=job,
                handedness=video_handedness,
                handedness_source=handedness_source,
                key_frame_index=key_frame_index,
                key_frame_dir=key_frame_dir,
                neighbor_offset=offset,
                frame_index=frame_index,
                image_path=image_path,
                output_root=output_root,
                angles=angles,
            )
            rows.append(row)

    return rows


def main() -> None:
    args = parse_args()
    input_root = args.input_root
    nstc_root = args.nstc_root
    output_root = args.output_root

    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    if args.clean:
        clean_annotation_files(output_root)

    jobs = iter_scoring_video_jobs(input_root)
    jobs.extend(iter_nstc_video_jobs(nstc_root))
    print(f"Found {len(jobs)} videos")

    analyzer = VideoAnalyzer()
    pose_detector = PoseDetector()
    rows: list[dict[str, Any]] = []
    failures: list[tuple[Path, str]] = []
    for job in jobs:
        try:
            rows.extend(
                process_job(
                    job,
                    output_root,
                    args.handedness,
                    analyzer,
                    pose_detector,
                    max(0, args.frame_radius),
                )
            )
        except Exception as exc:
            failures.append((job.video_path, str(exc)))
            print(f"Failed: {job.video_path}: {exc}")

    write_annotation_files(output_root, rows)

    print(f"Extracted {len(rows)} key-frame images from {len(jobs) - len(failures)} videos")
    if failures:
        print(f"Failures: {len(failures)}")
        for path, reason in failures:
            print(f"- {path}: {reason}")


if __name__ == "__main__":
    main()
