from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from badminton_analysis.ml.clear_feedback import (
    ClearFeedbackAnalysis,
    SYSTEM_INSTRUCTIONS,
    build_response_input,
    coaching_target_joint_ids,
    load_advice_context,
    load_correction_grade_context,
    load_phase_indices,
    prompt_context,
    sample_video_frames,
    validate_analysis_frames,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a scored clear video with OpenAI and emit timestamped joint feedback"
        )
    )
    parser.add_argument(
        "--video-path",
        default=(
            "stats/skeleton_correction/clear_expert_guided_v3_videos/"
            "students/EG29.mp4"
        ),
    )
    parser.add_argument(
        "--dataset-path",
        default="datasets/skeleton_sequences/clear/beginners/EG29.npz",
        help="Sequence NPZ supplying the exact five grading checkpoint frames",
    )
    parser.add_argument(
        "--advice-path",
        default=(
            "stats/skeleton_correction/clear_expert_guided_v3_grades/"
            "advice_context.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir", default="stats/openai_clear_feedback/EG29"
    )
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument(
        "--grading-results-path",
        default=(
            "stats/skeleton_correction/clear_expert_guided_v3_grades/"
            "grading_results.csv"
        ),
    )
    parser.add_argument("--env-file", default=".env")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare frames and prompt context without calling OpenAI",
    )
    return parser


def _feedback_rows(
    payload: dict[str, Any], filename: str, model: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for problem in payload["analysis"]["problems"]:
        rows.append(
            {
                "filename": filename,
                "model": model,
                "correction_total_score": payload["correction_total_score"],
                "priority": problem["priority"],
                "frame_index": problem["frame_index"],
                "timestamp_seconds": problem["timestamp_seconds"],
                "phase": problem["phase"],
                "joint_ids": "|".join(str(value) for value in problem["joint_ids"]),
                "rule_reference": problem["rule_reference"],
                "confidence": problem["confidence"],
                "title": problem["title"],
                "criterion_score": problem["criterion_score"],
                "criterion_maximum": problem["criterion_maximum"],
                "feedback": problem["feedback"],
                "evidence": problem["evidence"],
            }
        )
    return rows


def _attach_rule_scores(
    analysis_data: dict[str, Any], context: dict[str, Any]
) -> None:
    evidence = context["correction_distance_grade"]["criteria"]
    scores = {criterion["name_zh_tw"]: criterion for criterion in evidence}
    for problem in analysis_data["problems"]:
        criterion = scores[problem["title"]]
        problem["criterion_score"] = float(criterion["score"])
        problem["criterion_maximum"] = float(criterion["maximum"])
        problem["joint_ids"] = coaching_target_joint_ids(
            problem["rule_reference"]
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    video_path = Path(args.video_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(args.dataset_path)
    phase_indices = load_phase_indices(dataset_path)
    samples = sample_video_frames(
        video_path, output_dir / "input_frames", phase_indices=phase_indices
    )
    advice = load_advice_context(Path(args.advice_path), video_path.name)
    correction_grade = load_correction_grade_context(
        Path(args.grading_results_path), video_path.name
    )
    context = prompt_context(
        advice,
        samples,
        phase_indices=phase_indices,
        correction_grade=correction_grade,
    )
    context_path = output_dir / "prompt_context.json"
    context_path.write_text(
        json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.dry_run:
        print(f"Prepared {len(samples)} frames and context at {output_dir}")
        return 0

    load_dotenv(args.env_file)
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set")
    client = OpenAI()
    response = client.responses.parse(
        model=args.model,
        instructions=SYSTEM_INSTRUCTIONS,
        input=build_response_input(context, samples),  # type: ignore[arg-type]
        text_format=ClearFeedbackAnalysis,
        reasoning={"effort": "medium"},
        max_output_tokens=2200,
        store=False,
    )
    analysis = response.output_parsed
    if analysis is None:
        raise ValueError("OpenAI response did not contain parsed feedback")
    validate_analysis_frames(analysis, samples, phase_indices)
    analysis_data = analysis.model_dump()
    _attach_rule_scores(analysis_data, context)
    timestamps = {
        sample.frame_index: sample.timestamp_seconds for sample in samples
    }
    for problem in analysis_data["problems"]:
        problem["timestamp_seconds"] = timestamps[problem["frame_index"]]
    payload = {
        "filename": video_path.name,
        "source_video": str(video_path),
        "model": args.model,
        "correction_total_score": float(correction_grade["total_score"]),
        "response_id": response.id,
        "sampled_frames": [sample.manifest() for sample in samples],
        "analysis": analysis_data,
    }
    feedback_path = output_dir / "feedback.json"
    feedback_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        _feedback_rows(payload, video_path.name, args.model)
    ).to_csv(output_dir / "feedback.csv", index=False)
    print(
        f"Received {len(analysis_data['problems'])} feedback items; "
        f"output={feedback_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
