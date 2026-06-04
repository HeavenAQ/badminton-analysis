"""Grade student badminton smash videos using GPT vision.

Produces a CSV with the same schema as grade_students.py so outputs can be
compared side-by-side. Reads OPENAI_API_KEY from the environment.

Usage:
    uv run python scripts/grade_students_gpt.py \\
        --skill smash --input-dir scoring_videos/ --output-dir output_gpt/
"""

import dotenv
import argparse
import base64
import json
import math
from pathlib import Path
from typing import Any, Sequence

import cv2
import pandas as pd
from openai import OpenAI

from badminton_analysis.models.types import (
    COCOKeypoints,
    CoordinateDict,
    GradingDetail,
    GradingOutcome,
    Handedness,
    Skill,
)
from badminton_analysis.services.video_analyzer import VideoAnalyzer
from badminton_analysis.services.video_processor import VideoProcessor

dotenv.load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GPT_MODEL = "gpt-5.5"
EXPERT_STATS_DIR = Path(__file__).parent.parent / "stats" / "smash"
MAX_EXPERT_IMAGE_WIDTH = 640
JPEG_QUALITY = 82
VIDEO_EXTENSIONS = (".mp4", ".mov")

# Joint angle keys used by the smash grader.
# After VideoAnalyzer.mirror_angles for left-handed players, the dominant arm
# always lives under the "Right" key — same as for right-handed players.
_DOM_SHOULDER = "Right Shoulder Angle"
_NDOM_SHOULDER = "Left Shoulder Angle"
_DOM_ELBOW = "Right Elbow Angle"


# ---------------------------------------------------------------------------
# Handedness-aware keypoint helpers (coordinate-based checkpoints)
# ---------------------------------------------------------------------------


def _dominant_hip_kp(handedness: Handedness) -> COCOKeypoints:
    return (
        COCOKeypoints.RIGHT_HIP
        if handedness == Handedness.RIGHT
        else COCOKeypoints.LEFT_HIP
    )


def _non_dominant_hip_kp(handedness: Handedness) -> COCOKeypoints:
    return (
        COCOKeypoints.LEFT_HIP
        if handedness == Handedness.RIGHT
        else COCOKeypoints.RIGHT_HIP
    )


def _dominant_foot_kp(handedness: Handedness) -> COCOKeypoints:
    return (
        COCOKeypoints.RIGHT_ANKLE
        if handedness == Handedness.RIGHT
        else COCOKeypoints.LEFT_ANKLE
    )


def _non_dominant_foot_kp(handedness: Handedness) -> COCOKeypoints:
    return (
        COCOKeypoints.LEFT_ANKLE
        if handedness == Handedness.RIGHT
        else COCOKeypoints.RIGHT_ANKLE
    )


def _dominant_shoulder_kp(handedness: Handedness) -> COCOKeypoints:
    return (
        COCOKeypoints.RIGHT_SHOULDER
        if handedness == Handedness.RIGHT
        else COCOKeypoints.LEFT_SHOULDER
    )


def _non_dominant_shoulder_kp(handedness: Handedness) -> COCOKeypoints:
    return (
        COCOKeypoints.LEFT_SHOULDER
        if handedness == Handedness.RIGHT
        else COCOKeypoints.RIGHT_SHOULDER
    )


# ---------------------------------------------------------------------------
# Expert stats helpers
# ---------------------------------------------------------------------------


def _load_expert_stats() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (mean, std) DataFrames indexed by feature name, columns 0–4."""
    mean = pd.read_csv(EXPERT_STATS_DIR / "mean.csv", index_col=0).set_index("feature")
    std = pd.read_csv(EXPERT_STATS_DIR / "std.csv", index_col=0).set_index("feature")
    mean.columns = [0, 1, 2, 3, 4]
    std.columns = [0, 1, 2, 3, 4]
    return mean, std


def _build_stats_block(mean: pd.DataFrame, std: pd.DataFrame) -> str:
    """Render expert angle stats as a human-readable block for the system prompt."""

    def line(label: str, feature: str, col: int) -> str:
        m, s = float(mean.loc[feature, col]), float(std.loc[feature, col])
        return f"  {label}: {m:.1f}° ± {s:.1f}°  (range [{m - s:.1f}°, {m + s:.1f}°])"

    return "\n".join(
        [
            "EXPERT REFERENCE STATISTICS (from professional players):",
            "",
            "Checkpoint 1 — Preparation (球拍舉至腰部預備), frame 0:",
            line("Dominant Shoulder    ", "Dominant Shoulder Angle", 0),
            line("Non-dominant Shoulder", "Non-dominant Shoulder Angle", 0),
            "",
            "Checkpoint 2 — Body Rotation (轉身), frames 0→1:",
            "  Based on hip-girdle line rotation (degrees, scale-invariant). Full 10 pts if",
            "  hip line rotates by > 10°.",
            "",
            "Checkpoint 3 — Hand Balance (雙手手肘平衡), frame 1:",
            line("Dominant Shoulder    ", "Dominant Shoulder Angle", 1),
            line("Non-dominant Shoulder", "Non-dominant Shoulder Angle", 1),
            "",
            "Checkpoint 4 — Elbow Forward (手肘往前轉至前方), frame 2:",
            line("Dominant Shoulder", "Dominant Shoulder Angle", 2),
            "",
            "Checkpoint 5 — Wrist Flick (手腕發力), frame 2:",
            line("Dominant Elbow", "Dominant Elbow Angle", 2),
            "",
            "Checkpoint 6 — Follow-through (慣用手肩膀往前轉), frames 0→3:",
            line("Dominant Shoulder", "Dominant Shoulder Angle", 3),
            "  + 10 pts if shoulder-girdle line rotates > 12° from frame 0 to frame 3.",
        ]
    )


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def _load_expert_image(frame_dir: Path) -> str | None:
    """Return base64 JPEG of the first image in frame_dir (resized), or None."""
    jpegs = sorted(frame_dir.glob("*.jpg"))
    if not jpegs:
        return None
    img = cv2.imread(str(jpegs[0]))
    if img is None:
        return None
    h, w = img.shape[:2]
    if w > MAX_EXPERT_IMAGE_WIDTH:
        img = cv2.resize(
            img, (MAX_EXPERT_IMAGE_WIDTH, int(h * MAX_EXPERT_IMAGE_WIDTH / w))
        )
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return base64.b64encode(buf).decode("utf-8")


def _encode_frame_with_pose(processor: VideoProcessor, frame_idx: int) -> str:
    """Draw skeleton + angle overlay on a frame and return base64 JPEG."""
    frame = processor.frames[frame_idx].copy()
    landmark = processor.landmarks[frame_idx]
    processor.pose_detector.show_pose(frame, landmark)
    processor.pose_detector.show_angles(frame, landmark)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return base64.b64encode(buf).decode("utf-8")


def _img_content(b64: str) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
    }


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _build_angles_block(
    angle_list: list[dict[str, float]],
    hip_rotation_deg: float,
    shoulder_rotation_deg: float,
) -> str:
    """Build the user-message text block with student angles and rotation metrics."""

    def v(frame_idx: int, key: str) -> str:
        return f"{angle_list[frame_idx].get(key, 0.0):.1f}°"

    return "\n".join(
        [
            "STUDENT DATA:",
            "",
            "Checkpoint 1 (frame 0 — Preparation):",
            f"  Dominant Shoulder:     {v(0, _DOM_SHOULDER)}",
            f"  Non-dominant Shoulder: {v(0, _NDOM_SHOULDER)}",
            "",
            "Checkpoint 2 (frames 0→1 — Body Rotation):",
            f"  Hip-girdle line rotation: {hip_rotation_deg:.1f}°",
            f"  (Rule: full 10 pts if rotation > 10°)",
            "",
            "Checkpoint 3 (frame 1 — Hand Balance):",
            f"  Dominant Shoulder:     {v(1, _DOM_SHOULDER)}",
            f"  Non-dominant Shoulder: {v(1, _NDOM_SHOULDER)}",
            "",
            "Checkpoint 4 (frame 2 — Elbow Forward):",
            f"  Dominant Shoulder: {v(2, _DOM_SHOULDER)}",
            "",
            "Checkpoint 5 (frame 2 — Wrist Flick):",
            f"  Dominant Elbow: {v(2, _DOM_ELBOW)}",
            "",
            "Checkpoint 6 (frame 3 — Follow-through):",
            f"  Dominant Shoulder: {v(3, _DOM_SHOULDER)}",
            f"  Shoulder-girdle line rotation frame 0→3: {shoulder_rotation_deg:.1f}°",
            f"  (Rule: +10 pts if rotation > 12°)",
        ]
    )


def _build_system_content(
    stats_block: str,
    expert_images: dict[str, str | None],
) -> list[dict[str, Any]]:
    system_text = f"""You are an expert badminton coach grading a student's overhead smash technique.

A smash is a powerful overhead shot hit steeply downward. There are six graded checkpoints totalling 100 points.

## Grading Criteria

### Checkpoint 1 — Preparation (球拍舉至腰部預備) — max 10 pts
Racket is raised to waist level in a ready stance.
- 5 pts for dominant shoulder angle
- 5 pts for non-dominant shoulder angle

### Checkpoint 2 — Body Rotation (轉身) — max 10 pts
The body rotates into the shot. Assessed by the rotation of the hip-girdle line between frame 0 and frame 1:
- Full 10 pts if hip-girdle line rotates > 10° (binary)
- 0 pts otherwise
Use the provided hip-girdle rotation value directly — do not estimate from images.

### Checkpoint 3 — Hand Balance (雙手手肘平衡) — max 20 pts
Both arms are raised and balanced, ready for the swing.
- 10 pts for dominant shoulder angle
- 10 pts for non-dominant shoulder angle

### Checkpoint 4 — Elbow Forward (手肘往前轉至前方) — max 20 pts
Dominant elbow drives forward toward the shuttlecock.
- 20 pts for dominant shoulder angle

### Checkpoint 5 — Wrist Flick (手腕發力) — max 20 pts
Wrist snaps through the shot to generate power.
- 20 pts for dominant elbow angle

### Checkpoint 6 — Follow-through (慣用手肩膀往前轉) — max 20 pts
Dominant shoulder rotates forward in the follow-through.
- 10 pts for dominant shoulder angle (angle-based)
- 10 pts for shoulder rotation: full credit if shoulder-girdle line rotates > 12° from frame 0 to frame 3 (binary)
Use the provided shoulder-girdle rotation value directly — do not estimate from images.

## {stats_block}

## Scoring Rule for Angle-Based Components
- Award FULL points if the student's angle is within [mean − std, mean + std].
- If outside the range:
    * Below range (angle < mean − std): score = max_pts × (angle ÷ lower_bound)
    * Above range (angle > mean + std): score = max_pts × (upper_bound ÷ angle)
- Angles are in degrees; a higher value means a more open/extended joint.
- Use the numerical data as the primary signal; images provide visual confirmation.

## Expert Reference Images
The following images show correct professional technique at each key frame."""

    content: list[dict[str, Any]] = [{"type": "text", "text": system_text}]

    cp_labels = [
        ("frame0", "Expert — Frame 0: Preparation"),
        ("frame1", "Expert — Frame 1: Mid-start / Body Rotation"),
        ("frame2", "Expert — Frame 2: Peak / Wrist Flick"),
        ("frame3", "Expert — Frame 3: Follow-through"),
    ]
    for key, label in cp_labels:
        b64 = expert_images.get(key)
        if b64:
            content.append({"type": "text", "text": label})
            content.append(_img_content(b64))

    content.append(
        {
            "type": "text",
            "text": """## Response Format
Return ONLY a valid JSON object — no markdown fences, no explanation:
{
  "total_grade": <float 0-100>,
  "grading_details": [
    {"description": "球拍舉至腰部預備",   "grade": <float 0-10>},
    {"description": "轉身",               "grade": <float 0-10>},
    {"description": "雙手手肘平衡",       "grade": <float 0-20>},
    {"description": "手肘往前轉至前方",   "grade": <float 0-20>},
    {"description": "手腕發力",           "grade": <float 0-20>},
    {"description": "慣用手肩膀往前轉",   "grade": <float 0-20>}
  ]
}""",
        }
    )

    return content


def _build_user_content(
    angles_block: str,
    student_frame_b64s: list[str],
) -> list[dict[str, Any]]:
    frame_labels = [
        "Student — Frame 0: Start / Preparation",
        "Student — Frame 1: Mid-start / Body Rotation",
        "Student — Frame 2: Peak / Wrist Flick",
        "Student — Frame 3: Follow-through",
        "Student — Frame 4: End",
    ]
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": f"Grade this student's smash technique.\n\n{angles_block}",
        },
        {"type": "text", "text": "## Student Key Frames (temporal order):"},
    ]
    for label, b64 in zip(frame_labels, student_frame_b64s):
        content.append({"type": "text", "text": label})
        content.append(_img_content(b64))
    return content


# ---------------------------------------------------------------------------
# GPT grading
# ---------------------------------------------------------------------------


def _grade_with_gpt(
    client: OpenAI,
    system_content: list[dict[str, Any]],
    user_content: list[dict[str, Any]],
) -> GradingOutcome:
    response = client.chat.completions.create(  # type: ignore[call-overload]
        model=GPT_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
    )
    data = json.loads(response.choices[0].message.content)
    if "total_grade" not in data or "grading_details" not in data:
        raise ValueError(f"Unexpected GPT response schema: {list(data.keys())}")
    details: list[GradingDetail] = [
        {"description": d["description"], "grade": float(d["grade"])}
        for d in data["grading_details"]
    ]
    return {"total_grade": float(data["total_grade"]), "grading_details": details}


# ---------------------------------------------------------------------------
# CSV row helper (mirrors grade_students.py)
# ---------------------------------------------------------------------------


def _flatten_details(details: list[GradingDetail]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for i, detail in enumerate(details, start=1):
        out[f"detail_{i}_desc"] = detail["description"]
        out[f"detail_{i}_grade"] = detail["grade"]
    return out


# ---------------------------------------------------------------------------
# Main grading loop
# ---------------------------------------------------------------------------


def grade_videos_in_dir(
    input_dir: str,
    output_dir: str,
    skill: Skill,
) -> tuple[list[dict[str, Any]], int]:
    source_dir = Path(input_dir)
    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(
        p
        for p in source_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        raise ValueError(f"No videos found in: {input_dir}")

    # Load once and reuse across all videos
    client = OpenAI()
    mean_df, std_df = _load_expert_stats()
    stats_block = _build_stats_block(mean_df, std_df)
    expert_images: dict[str, str | None] = {
        "frame0": _load_expert_image(EXPERT_STATS_DIR / "frame0"),
        "frame1": _load_expert_image(EXPERT_STATS_DIR / "frame1"),
        "frame2": _load_expert_image(EXPERT_STATS_DIR / "frame2"),
        "frame3": _load_expert_image(EXPERT_STATS_DIR / "frame3"),
    }
    system_content = _build_system_content(stats_block, expert_images)

    rows: list[dict[str, Any]] = []
    failures = 0

    for video_path in videos:
        print(f"Processing: {video_path.name}")
        handedness = (
            Handedness.LEFT if "left" in video_path.name.lower() else Handedness.RIGHT
        )

        try:
            processor = VideoProcessor(
                str(video_path), video_path.name, str(destination_dir)
            )
            tracking = processor.process_frames(handedness)

            if len(tracking["hand_positions"]) <= 2:
                raise ValueError("Too few frames with valid pose detected")

            start, peak, end = VideoAnalyzer.find_analysis_window(
                skill=skill,
                hand_positions=tracking["hand_positions"],
                elbow_positions=tracking["elbow_positions"],
            )
            frame_indices = [
                start,
                (start + peak) // 2,
                peak,
                (peak + end) // 2,
                end,
            ]

            landmark_list = [tracking["original_landmarks"][i] for i in frame_indices]
            angle_list = list(map(VideoAnalyzer.compute_angles, landmark_list))
            if handedness == Handedness.LEFT:
                angle_list = [VideoAnalyzer.mirror_angles(a) for a in angle_list]

            # Checkpoint 2: hip-girdle line rotation (degrees) from frame 0 → frame 1
            dom_hip = _dominant_hip_kp(handedness)
            ndom_hip = _non_dominant_hip_kp(handedness)

            def _line_angle(frame: CoordinateDict, kp_a: COCOKeypoints, kp_b: COCOKeypoints) -> float:
                dy = float(frame[kp_b][1] - frame[kp_a][1])
                dx = float(frame[kp_b][0] - frame[kp_a][0])
                return math.degrees(math.atan2(dy, dx))

            hip_angle_0 = _line_angle(landmark_list[0], dom_hip, ndom_hip)
            hip_angle_1 = _line_angle(landmark_list[1], dom_hip, ndom_hip)
            hip_rotation_deg = abs((hip_angle_1 - hip_angle_0 + 180) % 360 - 180)

            # Checkpoint 6: shoulder-girdle line rotation (degrees) from frame 0 → frame 3
            dom_sh = _dominant_shoulder_kp(handedness)
            ndom_sh = _non_dominant_shoulder_kp(handedness)
            shoulder_angle_0 = _line_angle(landmark_list[0], dom_sh, ndom_sh)
            shoulder_angle_3 = _line_angle(landmark_list[3], dom_sh, ndom_sh)
            shoulder_rotation_deg = abs((shoulder_angle_3 - shoulder_angle_0 + 180) % 360 - 180)

            student_frame_b64s = [
                _encode_frame_with_pose(processor, i) for i in frame_indices
            ]

            angles_block = _build_angles_block(
                angle_list, hip_rotation_deg, shoulder_rotation_deg
            )
            user_content = _build_user_content(angles_block, student_frame_b64s)
            grade = _grade_with_gpt(client, system_content, user_content)

            row: dict[str, Any] = {
                "filename": video_path.name,
                "skill": str(skill),
                "handedness": str(handedness),
                "status": "success",
                "error": "",
                "total_grade": grade["total_grade"],
                "start_frame": start,
                "peak_frame": peak,
                "end_frame": end,
            }
            row.update(_flatten_details(grade["grading_details"]))

        except Exception as exc:
            print(f"  ERROR: {exc}")
            failures += 1
            row = {
                "filename": video_path.name,
                "skill": str(skill),
                "handedness": str(handedness),
                "status": "error",
                "error": str(exc),
                "total_grade": 0,
                "start_frame": -1,
                "peak_frame": -1,
                "end_frame": -1,
            }

        rows.append(row)
        print(f"  Grade: {row['total_grade']:.1f}")

    return rows, failures


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Grade student badminton videos using GPT vision",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--skill",
        required=True,
        choices=[str(s) for s in Skill],
        help="Skill to grade",
    )
    parser.add_argument(
        "--input-dir", required=True, help="Directory containing student videos"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory to write grading results"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if not Path(args.input_dir).is_dir():
        print(f"Error: input directory not found: {args.input_dir}")
        return 1

    skill = Skill.convert_to_enum(args.skill)

    try:
        rows, failures = grade_videos_in_dir(args.input_dir, args.output_dir, skill)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    output_path = Path(args.output_dir) / "grading_results_gpt.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Completed: processed={len(rows)} failed={failures} csv={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
