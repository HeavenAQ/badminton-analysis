from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pandas as pd
import torch

from badminton_analysis.ml.infer_skeleton_corrector import (
    load_corrector,
    predict_correction,
)
from badminton_analysis.ml.skeleton_dataset import load_sequence
from badminton_analysis.ml.skeleton_scoring import correction_distance

BONES = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
)


def _safe_name(value: str) -> str:
    return re.sub(r"[^\w.-]+", "_", value, flags=re.UNICODE)


def _point(coordinate: np.ndarray, center: tuple[int, int], scale: float) -> tuple[int, int]:
    return (
        int(round(center[0] + float(coordinate[0]) * scale)),
        int(round(center[1] - float(coordinate[1]) * scale)),
    )


def _draw_pose(
    canvas: np.ndarray,
    pose: np.ndarray,
    confidence: np.ndarray,
    center: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    for start, end in BONES:
        if confidence[start] <= 0 or confidence[end] <= 0:
            continue
        cv2.line(
            canvas,
            _point(pose[start], center, scale),
            _point(pose[end], center, scale),
            color,
            thickness,
            cv2.LINE_AA,
        )


def render_overlay(
    output_path: Path,
    filename: str,
    total_grade: float,
    original: np.ndarray,
    corrected: np.ndarray,
    confidence: np.ndarray,
    phase_indices: np.ndarray,
) -> None:
    width, height = 1800, 720
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    display_name = filename if filename.isascii() else "expert sample"
    cv2.putText(
        canvas,
        f"{display_name} | score {total_grade:.1f}",
        (35, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (25, 25, 25),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "original", (35, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 95, 20), 2
    )
    cv2.putText(
        canvas,
        "corrected", (145, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (35, 145, 45), 2
    )
    cv2.putText(
        canvas,
        "vectors / joint magnitude", (270, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (40, 40, 200), 2,
    )
    panel_width = width // len(phase_indices)
    for phase_number, frame_index in enumerate(phase_indices):
        x_center = phase_number * panel_width + panel_width // 2
        center = (x_center, 405)
        source = original[int(frame_index)]
        target = corrected[int(frame_index)]
        mask = confidence[int(frame_index)]
        _draw_pose(canvas, source, mask, center, 130.0, (200, 95, 20), 4)
        _draw_pose(canvas, target, mask, center, 130.0, (35, 145, 45), 3)
        magnitudes = np.linalg.norm(target - source, axis=-1)
        maximum = max(1e-8, float(np.max(magnitudes)))
        for joint in range(len(source)):
            if mask[joint] <= 0:
                continue
            start = _point(source[joint], center, 130.0)
            end = _point(target[joint], center, 130.0)
            cv2.arrowedLine(canvas, start, end, (40, 40, 200), 2, cv2.LINE_AA, tipLength=0.25)
            intensity = int(round(255.0 * magnitudes[joint] / maximum))
            cv2.circle(canvas, end, 4 + intensity // 64, (0, 255 - intensity, intensity), -1)
        phase_distance, _ = correction_distance(
            original[max(0, int(frame_index) - 2) : int(frame_index) + 3],
            corrected[max(0, int(frame_index) - 2) : int(frame_index) + 3],
            confidence[max(0, int(frame_index) - 2) : int(frame_index) + 3],
        )
        cv2.putText(
            canvas,
            f"phase {phase_number + 1} | d={phase_distance:.3f}",
            (phase_number * panel_width + 40, 675),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (45, 45, 45),
            1,
            cv2.LINE_AA,
        )
        if phase_number:
            cv2.line(canvas, (phase_number * panel_width, 110), (phase_number * panel_width, 690), (215, 215, 215), 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"could not write overlay: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render skeleton correction debug overlays")
    parser.add_argument(
        "--results-path",
        default="stats/skeleton_correction/clear_feasibility/grading_results.csv",
    )
    parser.add_argument("--dataset-root", default="datasets/skeleton_sequences/clear")
    parser.add_argument(
        "--model-path",
        default="models/skeleton_correction/clear_expert_guided_v3.pt",
    )
    parser.add_argument("--output-dir", default="stats/skeleton_correction/clear_debug_overlays")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    model, checkpoint = load_corrector(args.model_path, device)
    phase_aligned = bool(checkpoint.get("phase_aligned", False))
    correction_strength = float(checkpoint.get("inference_strength", 1.0))
    results = pd.read_csv(args.results_path)
    beginners = results[results["label"] == "beginners"]
    experts = results[results["label"] == "experts"]
    random_count = min(args.count, len(results))
    selections = {
        "lowest_beginners": beginners.nsmallest(args.count, "total_grade"),
        "highest_beginners": beginners.nlargest(args.count, "total_grade"),
        "lowest_experts": experts.nsmallest(args.count, "total_grade"),
        "random": results.sample(random_count, random_state=2026),
    }
    output_dir = Path(args.output_dir)
    for group_name, rows in selections.items():
        group_dir = output_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        for old_overlay in group_dir.glob("*.png"):
            old_overlay.unlink()
        for rank, row in enumerate(rows.itertuples(index=False), start=1):
            dataset_path = Path(args.dataset_root) / row.label / f"{Path(row.filename).stem}.npz"
            sample = load_sequence(dataset_path)
            skeleton = sample["skeleton_3d"].astype(np.float32)
            confidence = sample["confidence"].astype(np.float32)
            corrected = predict_correction(
                model,
                skeleton,
                confidence,
                device,
                sample["phase_indices"] if phase_aligned else None,
                correction_strength,
            )
            render_overlay(
                group_dir / f"{rank:02d}_{_safe_name(row.filename)}.png",
                row.filename,
                float(row.total_grade),
                skeleton,
                corrected,
                confidence,
                sample["phase_indices"],
            )
    print(f"Wrote correction overlays to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
