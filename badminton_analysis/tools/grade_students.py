import argparse
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from badminton_analysis.models.types import GradingDetail, Handedness, Skill
from badminton_analysis.services.graders.player import PlayerGrader
from badminton_analysis.services.video_processor import VideoProcessor

VIDEO_EXTENSIONS = (".mp4", ".mov")


def _flatten_details(details: list[GradingDetail]) -> dict[str, Any]:
    columns: dict[str, Any] = {}
    for index, detail in enumerate(details, start=1):
        columns[f"detail_{index}_desc"] = detail["description"]
        columns[f"detail_{index}_grade"] = detail["grade"]
    return columns


def grade_videos_in_dir(
    input_dir: str,
    output_dir: str,
    skill: Skill,
    *,
    footwork_reference_path: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    source_dir = Path(input_dir)
    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        raise ValueError(f"No videos found in input directory: {input_dir}")

    player_grader = PlayerGrader()
    rows: list[dict[str, Any]] = []
    failures = 0

    for video_path in videos:
        print(f"Processing: {video_path}")
        handedness = Handedness.LEFT if "left" in video_path.name else Handedness.RIGHT

        try:
            processor = VideoProcessor(
                str(video_path),
                video_path.name,
                str(destination_dir),
            )
            tracking = processor.process_frames(handedness)
            grade, window = player_grader.grade(
                skill,
                handedness,
                tracking,
                footwork_reference_path=footwork_reference_path,
            )
            row: dict[str, Any] = {
                "filename": video_path.name,
                "skill": str(skill),
                "handedness": str(handedness),
                "status": "success",
                "error": "",
                "total_grade": grade["total_grade"],
                "start_frame": window[0],
                "peak_frame": window[1],
                "end_frame": window[2],
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

    return rows, failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Grade student badminton videos in a directory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--skill",
        required=True,
        choices=[str(skill) for skill in Skill],
        help="Skill to grade",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing student videos",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write grading results",
    )
    parser.add_argument(
        "--reference-data",
        help="JSON file containing precomputed footwork reference trajectories",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Error: input directory not found: {args.input_dir}")
        return 1

    skill = Skill.convert_to_enum(args.skill)

    if skill == Skill.FOOTWORK and not args.reference_data:
        print("Error: --reference-data is required when grading footwork")
        return 1

    try:
        rows, failures = grade_videos_in_dir(
            args.input_dir,
            args.output_dir,
            skill,
            footwork_reference_path=args.reference_data,
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    output_path = Path(args.output_dir) / "grading_results.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(
        f"Completed grading: processed={len(rows)} failed={failures} csv={output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
