from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from badminton_analysis.ml.skill_specs import get_skill_spec, supported_skill_choices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export calibrated skill-specific skeleton-correction scores"
    )
    parser.add_argument(
        "--skill", choices=supported_skill_choices(), default="clear"
    )
    parser.add_argument("--grading-results-path")
    parser.add_argument("--output-dir")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = get_skill_spec(args.skill)
    grading_results_path = (
        Path(args.grading_results_path)
        if args.grading_results_path
        else spec.grading_output_dir / "grading_results.csv"
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("stats")
        / f"openai_{spec.slug}_feedback"
        / "correction_scores"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    grading = pd.read_csv(grading_results_path)
    grading = grading[grading["status"] == "success"].copy()
    if "skill" in grading and set(grading["skill"].astype(str)) != {spec.slug}:
        raise ValueError(f"grading CSV contains rows outside skill {spec.slug}")
    criterion_columns = [rule.name_zh_tw for rule in spec.rules]
    detail_grade_columns = [
        f"detail_{index}_grade"
        for index in range(1, len(spec.rules) + 1)
    ]
    scores = grading[
        [
            "label",
            "filename",
            "handedness",
            *detail_grade_columns,
            "total_grade",
            "correction_distance",
            "position_distance",
            "angle_distance",
            "velocity_distance",
            "bone_length_distance",
        ]
    ].copy()
    scores.insert(0, "skill", spec.slug)
    scores["label"] = scores["label"].replace({"beginners": "students"})
    scores = scores.rename(
        columns={
            "label": "group",
            "total_grade": "total_score",
            **dict(zip(detail_grade_columns, criterion_columns, strict=True)),
        }
    )
    scores.to_csv(output_dir / "all_scores.csv", index=False)

    mean_columns = [
        *criterion_columns,
        "total_score",
        "correction_distance",
        "position_distance",
        "angle_distance",
        "velocity_distance",
        "bone_length_distance",
    ]
    means = scores.groupby("group", sort=False)[mean_columns].mean()
    means.insert(0, "clip_count", scores.groupby("group", sort=False).size())
    means.reset_index().to_csv(output_dir / "group_means.csv", index=False)

    students = scores[scores["group"] == "students"]
    if not students.empty:
        lowest_student = students.nsmallest(1, "total_score").iloc[0]
        print(
            f"Lowest student: {lowest_student['filename']} "
            f"({lowest_student['total_score']:.2f}/100)"
        )
    print(means[["clip_count", "total_score"]].to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
