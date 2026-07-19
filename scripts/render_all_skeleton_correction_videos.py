from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import torch

from badminton_analysis.ml.infer_skeleton_corrector import load_corrector
from badminton_analysis.ml.skeleton_dataset import load_sequence
from badminton_analysis.services.pose_detector import PoseDetector
from render_skeleton_correction_video import render_video


def _transcode_h264(source: Path, destination: Path) -> None:
    subprocess.run(
        (
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(destination),
        ),
        check=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render scored skeleton-correction videos for a full dataset"
    )
    parser.add_argument(
        "--dataset-root", default="datasets/skeleton_sequences/clear"
    )
    parser.add_argument(
        "--student-video-dir", default="scoring_videos/高遠球/初學者高遠球"
    )
    parser.add_argument(
        "--expert-video-dir", default="scoring_videos/高遠球/專家高遠球"
    )
    parser.add_argument(
        "--model-path",
        default="models/skeleton_correction/clear_expert_guided_v3.pt",
    )
    parser.add_argument(
        "--results-path",
        default=(
            "stats/skeleton_correction/clear_expert_guided_v3_grades/"
            "grading_results.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="stats/skeleton_correction/clear_expert_guided_v3_videos",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=("beginners", "experts"),
        default=("beginners", "experts"),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    results_path = Path(args.results_path)
    results = pd.read_csv(results_path)
    required_columns = {"filename", "label", "total_grade"}
    missing_columns = required_columns - set(results.columns)
    if missing_columns:
        raise ValueError(
            f"grading CSV is missing columns: {sorted(missing_columns)}"
        )

    corrector = load_corrector(args.model_path, device)
    pose_detector = PoseDetector()
    groups = (
        (
            "beginners",
            "students",
            Path(args.student_video_dir),
        ),
        ("experts", "experts", Path(args.expert_video_dir)),
    )
    summary_path = output_root / "render_summary.csv"
    summary_rows: list[dict[str, Any]] = []
    failures = 0
    for label, output_group, video_dir in groups:
        if label not in args.groups:
            continue
        dataset_files = sorted((dataset_root / label).glob("*.npz"))
        if args.limit is not None:
            dataset_files = dataset_files[: args.limit]
        group_output = output_root / output_group
        group_output.mkdir(parents=True, exist_ok=True)
        for index, dataset_path in enumerate(dataset_files, start=1):
            sample = load_sequence(dataset_path)
            video_name = str(sample["video_name"].item())
            source_path = video_dir / video_name
            output_path = group_output / f"{Path(video_name).stem}.mp4"
            raw_path = group_output / f".{Path(video_name).stem}.raw.mp4"
            grade_rows = results[
                (results["filename"] == video_name)
                & (results["label"] == label)
            ]
            row: dict[str, Any] = {
                "filename": video_name,
                "label": label,
                "score": (
                    float(grade_rows.iloc[0]["total_grade"])
                    if len(grade_rows) == 1
                    else None
                ),
                "dataset_path": str(dataset_path),
                "source_video_path": str(source_path),
                "output_video_path": str(output_path),
                "frames": 0,
                "status": "error",
                "error": "",
            }
            try:
                if len(grade_rows) != 1:
                    raise ValueError(
                        f"expected one grade row, found {len(grade_rows)}"
                    )
                if not source_path.exists():
                    raise FileNotFoundError(source_path)
                if output_path.exists() and not args.overwrite:
                    row["status"] = "skipped"
                else:
                    frames = render_video(
                        video_path=source_path,
                        dataset_path=dataset_path,
                        model_path=Path(args.model_path),
                        output_path=raw_path,
                        results_path=results_path,
                        device=device,
                        pose_detector=pose_detector,
                        corrector=corrector,
                        results=results,
                    )
                    _transcode_h264(raw_path, output_path)
                    raw_path.unlink(missing_ok=True)
                    row["frames"] = frames
                    row["status"] = "success"
            except Exception as exc:
                raw_path.unlink(missing_ok=True)
                failures += 1
                row["error"] = str(exc)
            summary_rows.append(row)
            pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
            print(
                f"{label} {index:02d}/{len(dataset_files):02d} "
                f"{video_name}: {row['status']}"
            )

    successes = sum(row["status"] == "success" for row in summary_rows)
    skipped = sum(row["status"] == "skipped" for row in summary_rows)
    print(
        f"Rendered {successes}, skipped {skipped}, failed {failures}; "
        f"summary={summary_path}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
