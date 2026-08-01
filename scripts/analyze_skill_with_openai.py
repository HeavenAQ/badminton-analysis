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
    RawSkillFeedbackAnalysis,
    SkillFeedbackAnalysis,
    build_response_input,
    coaching_target_joint_ids,
    load_advice_context,
    load_correction_grade_context,
    load_phase_indices,
    load_source_frame_indices,
    prompt_context,
    sample_video_frames,
    system_instructions,
    validate_analysis_frames,
)
from badminton_analysis.ml.skeleton_dataset import load_sequence
from badminton_analysis.ml.skill_specs import get_skill_spec, supported_skill_choices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze one scored skill video with OpenAI and emit timestamped "
            "Traditional Chinese joint feedback"
        )
    )
    parser.add_argument(
        "--skill", choices=supported_skill_choices(), default="clear"
    )
    parser.add_argument("--video-path", required=True)
    parser.add_argument(
        "--video-frame-space",
        choices=("normalized", "source"),
        default="normalized",
        help=(
            "Use normalized for a rendered 64-frame correction video; use source "
            "for the original input video"
        ),
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Sequence NPZ supplying phases and source-frame provenance",
    )
    parser.add_argument("--advice-path")
    parser.add_argument("--grading-results-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--model", default="gpt-5.6-terra")
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
                "skill": payload["skill"],
                "model": model,
                "correction_total_score": payload["correction_total_score"],
                "priority": problem["priority"],
                "frame_index": problem["frame_index"],
                "timestamp_seconds": problem["timestamp_seconds"],
                "phase": problem["phase"],
                "joint_ids": "|".join(
                    str(value) for value in problem["joint_ids"]
                ),
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
    analysis_data: dict[str, Any],
    context: dict[str, Any],
    *,
    skill: str,
) -> None:
    spec = get_skill_spec(skill)
    evidence = context["correction_distance_grade"]["criteria"]
    scores = {criterion["name_zh_tw"]: criterion for criterion in evidence}
    for problem in analysis_data["problems"]:
        rule = spec.rule(problem["rule_reference"])
        problem["title"] = rule.name_zh_tw
        problem["phase"] = rule.phase
        allowed_frames = context["criterion_allowed_frames"][rule.name_zh_tw]
        problem["frame_index"] = min(
            allowed_frames,
            key=lambda frame: abs(frame - int(problem["frame_index"])),
        )
        criterion = scores[rule.name_zh_tw]
        problem["criterion_score"] = float(criterion["score"])
        problem["criterion_maximum"] = float(criterion["maximum"])
        problem["joint_ids"] = coaching_target_joint_ids(
            problem["rule_reference"], spec
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = get_skill_spec(args.skill)
    video_path = Path(args.video_path)
    dataset_path = Path(args.dataset_path)
    sample = load_sequence(dataset_path)
    sample_skill = str(sample["skill"].item())
    if sample_skill != spec.slug:
        raise ValueError(
            f"dataset skill is {sample_skill}, but --skill is {spec.slug}"
        )

    advice_path = (
        Path(args.advice_path)
        if args.advice_path
        else spec.grading_output_dir / "advice_context.jsonl"
    )
    grading_results_path = (
        Path(args.grading_results_path)
        if args.grading_results_path
        else spec.grading_output_dir / "grading_results.csv"
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("stats") / f"openai_{spec.slug}_feedback" / video_path.stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    phase_indices = load_phase_indices(dataset_path)
    source_mapping = (
        load_source_frame_indices(dataset_path)
        if args.video_frame_space == "source"
        else tuple(range(64))
    )
    samples = sample_video_frames(
        video_path,
        output_dir / "input_frames",
        phase_indices=phase_indices,
        source_frame_indices=source_mapping,
        spec=spec,
    )
    advice = load_advice_context(advice_path, video_path.name)
    correction_grade = load_correction_grade_context(
        grading_results_path, video_path.name, spec
    )
    context = prompt_context(
        advice,
        samples,
        phase_indices=phase_indices,
        correction_grade=correction_grade,
        spec=spec,
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
        instructions=system_instructions(spec),
        input=build_response_input(context, samples, spec),  # type: ignore[arg-type]
        text_format=RawSkillFeedbackAnalysis,
        reasoning={"effort": "medium"},
        max_output_tokens=2200,
        store=False,
    )
    analysis = response.output_parsed
    if analysis is None:
        raise ValueError("OpenAI response did not contain parsed feedback")
    analysis_data = analysis.model_dump()
    _attach_rule_scores(analysis_data, context, skill=spec.slug)
    validated_analysis = SkillFeedbackAnalysis.model_validate(analysis_data)
    validate_analysis_frames(validated_analysis, samples, phase_indices, spec)
    timestamps = {
        frame.frame_index: frame.timestamp_seconds for frame in samples
    }
    for problem in analysis_data["problems"]:
        problem["timestamp_seconds"] = timestamps[problem["frame_index"]]
    payload = {
        "filename": video_path.name,
        "skill": spec.slug,
        "source_video": str(video_path),
        "video_frame_space": args.video_frame_space,
        "model": args.model,
        "correction_total_score": float(correction_grade["total_score"]),
        "response_id": response.id,
        "sampled_frames": [frame.manifest() for frame in samples],
        "analysis": analysis_data,
    }
    feedback_path = output_dir / "feedback.json"
    feedback_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(_feedback_rows(payload, video_path.name, args.model)).to_csv(
        output_dir / "feedback.csv", index=False
    )
    print(
        f"Received {len(analysis_data['problems'])} feedback items; "
        f"output={feedback_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
