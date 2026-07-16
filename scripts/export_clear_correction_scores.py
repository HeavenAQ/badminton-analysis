from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from badminton_analysis.ml.clear_feedback import CLEAR_RULES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export calibrated skeleton-correction clear scores"
    )
    parser.add_argument(
        "--grading-results-path",
        default=(
            "stats/skeleton_correction/clear_expert_guided_v3_grades/"
            "grading_results.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="stats/openai_clear_feedback/correction_scores",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grading = pd.read_csv(args.grading_results_path)
    grading = grading[grading["status"] == "success"].copy()
    criterion_columns = [str(rule["name_zh_tw"]) for rule in CLEAR_RULES]
    scores = grading[
        [
            "label",
            "filename",
            "handedness",
            *(f"detail_{index}_grade" for index in range(1, 7)),
            "total_grade",
            "correction_distance",
            "position_distance",
            "angle_distance",
            "velocity_distance",
            "bone_length_distance",
        ]
    ].copy()
    scores["label"] = scores["label"].replace({"beginners": "students"})
    scores = scores.rename(
        columns={
            "label": "group",
            "total_grade": "total_score",
            **{
                f"detail_{index}_grade": criterion
                for index, criterion in enumerate(criterion_columns, start=1)
            },
        }
    )
    scores.to_csv(output_dir / "all_scores.csv", index=False)

    means = scores.groupby("group", sort=False)[
        [
            *criterion_columns,
            "total_score",
            "correction_distance",
            "position_distance",
            "angle_distance",
            "velocity_distance",
            "bone_length_distance",
        ]
    ].mean()
    means.insert(0, "clip_count", scores.groupby("group", sort=False).size())
    means.reset_index().to_csv(output_dir / "group_means.csv", index=False)

    lowest_student = scores[scores["group"] == "students"].nsmallest(
        1, "total_score"
    ).iloc[0]
    print(
        f"Lowest student: {lowest_student['filename']} "
        f"({lowest_student['total_score']:.2f}/100)"
    )
    print(means[["clip_count", "total_score"]].to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
