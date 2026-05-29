import argparse
import json
from pathlib import Path
from typing import TypedDict

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from badminton_analysis.models.types import COCOKeypoints, Handedness
from badminton_analysis.services.video_processor import VideoProcessor

VIDEO_EXTENSIONS = (".mp4", ".mov")


class ExpertTrajectory(TypedDict):
    left_ankle: list[list[float]]
    right_ankle: list[list[float]]


class ExpertFrames(TypedDict):
    frames: list[NDArray[np.uint8]]


def collect_expert_trajectories(
    input_dir: str,
    output_dir: str,
    handedness: Handedness,
) -> tuple[dict[str, ExpertTrajectory], dict[str, ExpertFrames]]:
    source_dir = Path(input_dir) / f"{handedness}"
    destination_dir = Path(output_dir) / f"{handedness}"
    destination_dir.mkdir(parents=True, exist_ok=True)

    trajectories: dict[str, ExpertTrajectory] = {}
    expert_frames: dict[str, ExpertFrames] = {}

    for video_path in sorted(source_dir.iterdir()):
        if (
            not video_path.is_file()
            or video_path.suffix.lower() not in VIDEO_EXTENSIONS
        ):
            continue

        print(f"Processing: {video_path}")
        processor = VideoProcessor(
            str(video_path),
            video_path.name,
            str(destination_dir),
        )
        tracking = processor.process_frames(handedness)

        left_ankle: list[list[float]] = []
        right_ankle: list[list[float]] = []
        kept_frames: list[NDArray[np.uint8]] = []
        for frame, landmarks in zip(
            tracking["frames"],
            tracking["original_landmarks"],
            strict=False,
        ):
            left_coord = landmarks.get(COCOKeypoints.LEFT_ANKLE)
            right_coord = landmarks.get(COCOKeypoints.RIGHT_ANKLE)
            if left_coord is None or right_coord is None:
                continue
            left_ankle.append(left_coord.tolist())
            right_ankle.append(right_coord.tolist())
            kept_frames.append(frame)

        if len(left_ankle) < 2 or len(right_ankle) < 2:
            print(f"Skipping {video_path.name}: insufficient ankle trajectory data")
            continue

        trajectories[video_path.stem] = {
            "left_ankle": left_ankle,
            "right_ankle": right_ankle,
        }
        expert_frames[video_path.stem] = {"frames": kept_frames}

    return trajectories, expert_frames


def mean_curve(series_list: list[np.ndarray], max_length: int) -> np.ndarray:
    stacked = np.full((len(series_list), max_length), np.nan, dtype=np.float64)
    for row, series in enumerate(series_list):
        stacked[row, : len(series)] = series
    return np.nanmean(stacked, axis=0)


def find_flip_frames(
    trajectories: dict[str, ExpertTrajectory],
    frame_step: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    left_distance_series: list[np.ndarray] = []
    right_distance_series: list[np.ndarray] = []
    left_velocity_series: list[np.ndarray] = []
    right_velocity_series: list[np.ndarray] = []
    left_acceleration_series: list[np.ndarray] = []
    right_acceleration_series: list[np.ndarray] = []
    for trajectory in trajectories.values():
        left_points = np.asarray(trajectory["left_ankle"], dtype=np.float64)
        right_points = np.asarray(trajectory["right_ankle"], dtype=np.float64)
        left_distances = np.linalg.norm(left_points - left_points[0], axis=1)
        right_distances = np.linalg.norm(right_points - right_points[0], axis=1)
        left_velocity = np.linalg.norm(np.diff(left_points, axis=0), axis=1)
        right_velocity = np.linalg.norm(np.diff(right_points, axis=0), axis=1)
        left_distance_series.append(left_distances)
        right_distance_series.append(right_distances)
        left_velocity_series.append(left_velocity)
        right_velocity_series.append(right_velocity)
        left_acceleration_series.append(np.diff(left_velocity))
        right_acceleration_series.append(np.diff(right_velocity))

    max_length = max(
        max((len(series) for series in left_distance_series), default=0),
        max((len(series) for series in right_distance_series), default=0),
    )
    if max_length == 0:
        return (
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        )

    left_mean = mean_curve(left_distance_series, max_length)
    right_mean = mean_curve(right_distance_series, max_length)
    left_velocity_mean = mean_curve(left_velocity_series, max(max_length - 1, 0))
    right_velocity_mean = mean_curve(right_velocity_series, max(max_length - 1, 0))
    left_acceleration_mean = mean_curve(
        left_acceleration_series, max(max_length - 2, 0)
    )
    right_acceleration_mean = mean_curve(
        right_acceleration_series, max(max_length - 2, 0)
    )
    candidate_frames = np.arange(0, max_length, frame_step, dtype=int)
    if len(candidate_frames) < 2:
        return (
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.array([], dtype=int),
            left_mean,
            right_mean,
            left_velocity_mean,
            right_velocity_mean,
            left_acceleration_mean,
            right_acceleration_mean,
        )

    candidate_diff = left_mean[candidate_frames] - right_mean[candidate_frames]
    prior_diff = candidate_diff[:-1]
    next_diff = candidate_diff[1:]
    flip_mask = ((prior_diff < 0) & (next_diff > 0)) | (
        (prior_diff > 0) & (next_diff < 0)
    )
    distance_flip_frames = candidate_frames[1:][flip_mask]

    velocity_candidate_frames = candidate_frames[candidate_frames > 0]
    velocity_diff = (
        left_velocity_mean[velocity_candidate_frames - 1]
        - right_velocity_mean[velocity_candidate_frames - 1]
    )
    velocity_prior_diff = velocity_diff[:-1]
    velocity_next_diff = velocity_diff[1:]
    velocity_flip_mask = ((velocity_prior_diff < 0) & (velocity_next_diff > 0)) | (
        (velocity_prior_diff > 0) & (velocity_next_diff < 0)
    )
    velocity_flip_frames = velocity_candidate_frames[1:][velocity_flip_mask]

    acceleration_candidate_frames = candidate_frames[candidate_frames > 1]
    acceleration_diff = (
        left_acceleration_mean[acceleration_candidate_frames - 2]
        - right_acceleration_mean[acceleration_candidate_frames - 2]
    )
    acceleration_prior_diff = acceleration_diff[:-1]
    acceleration_next_diff = acceleration_diff[1:]
    acceleration_flip_mask = (
        (acceleration_prior_diff < 0) & (acceleration_next_diff > 0)
    ) | ((acceleration_prior_diff > 0) & (acceleration_next_diff < 0))
    acceleration_flip_frames = acceleration_candidate_frames[1:][acceleration_flip_mask]
    return (
        distance_flip_frames,
        velocity_flip_frames,
        acceleration_flip_frames,
        left_mean,
        right_mean,
        left_velocity_mean,
        right_velocity_mean,
        left_acceleration_mean,
        right_acceleration_mean,
    )


def save_highlighted_frames(
    trajectories: dict[str, ExpertTrajectory],
    expert_frames: dict[str, ExpertFrames],
    output_dir: str,
    handedness: Handedness,
    frame_indices: np.ndarray,
    subdir_name: str,
) -> Path:
    flip_dir = Path(output_dir) / f"{handedness}" / subdir_name
    flip_dir.mkdir(parents=True, exist_ok=True)

    for expert_name, payload in expert_frames.items():
        expert_dir = flip_dir / expert_name
        expert_dir.mkdir(parents=True, exist_ok=True)
        frames = payload["frames"]
        left_points = np.asarray(
            trajectories[expert_name]["left_ankle"], dtype=np.float64
        )
        right_points = np.asarray(
            trajectories[expert_name]["right_ankle"], dtype=np.float64
        )

        for frame_index in frame_indices:
            if frame_index >= len(frames):
                continue

            frame = frames[frame_index].copy()
            left_coord = left_points[frame_index]
            right_coord = right_points[frame_index]
            cv2.circle(
                frame,
                (int(left_coord[0]), int(left_coord[1])),
                8,
                (255, 0, 0),
                2,
            )
            cv2.circle(
                frame,
                (int(right_coord[0]), int(right_coord[1])),
                8,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                frame,
                f"frame={frame_index}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            output_path = expert_dir / f"frame_{frame_index:04d}.jpg"
            success = cv2.imwrite(str(output_path), frame)
            if not success:
                raise RuntimeError(f"Failed to write frame to {output_path}")

    return flip_dir


def plot_trajectories(
    trajectories: dict[str, ExpertTrajectory],
    output_dir: str,
    handedness: Handedness,
    distance_flip_frames: np.ndarray,
    velocity_flip_frames: np.ndarray,
    acceleration_flip_frames: np.ndarray,
    left_mean: np.ndarray,
    right_mean: np.ndarray,
    left_velocity_mean: np.ndarray,
    right_velocity_mean: np.ndarray,
    left_acceleration_mean: np.ndarray,
    right_acceleration_mean: np.ndarray,
) -> Path:
    if not trajectories:
        raise ValueError("No valid expert trajectories were extracted.")

    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    xy_specs = [
        (axes[0, 0], "left_ankle", "Left Ankle Trajectory", "X", "Y"),
        (axes[0, 1], "right_ankle", "Right Ankle Trajectory", "X", "Y"),
    ]
    distance_specs = [
        (
            axes[1, 0],
            "left_ankle",
            "Left Ankle Distance From Start",
            "Frame",
            "Distance from start",
        ),
        (
            axes[1, 1],
            "right_ankle",
            "Right Ankle Distance From Start",
            "Frame",
            "Distance from start",
        ),
    ]
    average_ax = axes[2, 0]
    velocity_ax = axes[2, 1]
    acceleration_ax = axes[3, 0]
    axes[3, 1].axis("off")
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(trajectories), 1)))

    for color, (expert_name, trajectory) in zip(
        colors, trajectories.items(), strict=False
    ):
        for ax, ankle_key, title, xlabel, ylabel in xy_specs:
            points = np.asarray(trajectory[ankle_key], dtype=np.float64)
            ax.plot(
                points[:, 0],
                points[:, 1],
                color=color,
                linewidth=2,
                alpha=0.8,
                label=expert_name,
            )
            ax.scatter(points[0, 0], points[0, 1], color=color, marker="o", s=40)
            ax.scatter(points[-1, 0], points[-1, 1], color=color, marker="X", s=55)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.set_aspect("equal", adjustable="box")

        for ax, ankle_key, title, xlabel, ylabel in distance_specs:
            points = np.asarray(trajectory[ankle_key], dtype=np.float64)
            distances = np.linalg.norm(points - points[0], axis=1)
            frames = np.arange(len(distances))
            ax.plot(
                frames,
                distances,
                color=color,
                linewidth=2,
                alpha=0.85,
                label=expert_name,
            )
            ax.scatter(frames[0], distances[0], color=color, marker="o", s=35)
            ax.scatter(frames[-1], distances[-1], color=color, marker="X", s=50)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)

        left_velocity = np.linalg.norm(
            np.diff(np.asarray(trajectory["left_ankle"], dtype=np.float64), axis=0),
            axis=1,
        )
        right_velocity = np.linalg.norm(
            np.diff(np.asarray(trajectory["right_ankle"], dtype=np.float64), axis=0),
            axis=1,
        )
        velocity_frames = np.arange(1, len(left_velocity) + 1)
        velocity_ax.plot(
            velocity_frames,
            left_velocity,
            color=color,
            linewidth=1.5,
            alpha=0.25,
        )
        velocity_ax.plot(
            velocity_frames,
            right_velocity,
            color=color,
            linewidth=1.5,
            alpha=0.25,
            linestyle="--",
        )
        left_acceleration = np.diff(left_velocity)
        right_acceleration = np.diff(right_velocity)
        acceleration_frames = np.arange(2, len(left_acceleration) + 2)
        acceleration_ax.plot(
            acceleration_frames,
            left_acceleration,
            color=color,
            linewidth=1.5,
            alpha=0.25,
        )
        acceleration_ax.plot(
            acceleration_frames,
            right_acceleration,
            color=color,
            linewidth=1.5,
            alpha=0.25,
            linestyle="--",
        )

    if len(left_mean) > 0 and len(right_mean) > 0:
        frame_axis = np.arange(len(left_mean))
        average_ax.plot(
            frame_axis,
            left_mean,
            color="#2E86AB",
            linewidth=3,
            label="Left ankle average",
        )
        average_ax.plot(
            frame_axis,
            right_mean,
            color="#D95D39",
            linewidth=3,
            label="Right ankle average",
        )
        if len(distance_flip_frames) > 0:
            average_ax.scatter(
                distance_flip_frames,
                left_mean[distance_flip_frames],
                color="#1B998B",
                marker="o",
                s=32,
                label="Left/Right distance flip frames",
                zorder=5,
            )
        average_ax.set_title("Average Distance From Start")
        average_ax.set_xlabel("Frame")
        average_ax.set_ylabel("Distance from start")
        average_ax.grid(True, alpha=0.3)
        average_ax.legend()

    if len(left_velocity_mean) > 0 and len(right_velocity_mean) > 0:
        velocity_frame_axis = np.arange(1, len(left_velocity_mean) + 1)
        velocity_ax.plot(
            velocity_frame_axis,
            left_velocity_mean,
            color="#2E86AB",
            linewidth=3,
            label="Left ankle average velocity",
        )
        velocity_ax.plot(
            velocity_frame_axis,
            right_velocity_mean,
            color="#D95D39",
            linewidth=3,
            linestyle="--",
            label="Right ankle average velocity",
        )
        if len(velocity_flip_frames) > 0:
            velocity_ax.scatter(
                velocity_flip_frames,
                left_velocity_mean[velocity_flip_frames - 1],
                color="#1B998B",
                marker="o",
                s=32,
                label="Left/Right velocity flip frames",
                zorder=5,
            )
        velocity_ax.set_title("Ankle Velocity Changes")
        velocity_ax.set_xlabel("Frame")
        velocity_ax.set_ylabel("Per-frame displacement")
        velocity_ax.grid(True, alpha=0.3)
        velocity_ax.legend()

    if len(left_acceleration_mean) > 0 and len(right_acceleration_mean) > 0:
        acceleration_frame_axis = np.arange(2, len(left_acceleration_mean) + 2)
        acceleration_ax.plot(
            acceleration_frame_axis,
            left_acceleration_mean,
            color="#2E86AB",
            linewidth=3,
            label="Left ankle average acceleration",
        )
        acceleration_ax.plot(
            acceleration_frame_axis,
            right_acceleration_mean,
            color="#D95D39",
            linewidth=3,
            linestyle="--",
            label="Right ankle average acceleration",
        )
        if len(acceleration_flip_frames) > 0:
            acceleration_ax.scatter(
                acceleration_flip_frames,
                left_acceleration_mean[acceleration_flip_frames - 2],
                color="#1B998B",
                marker="o",
                s=32,
                label="Left/Right acceleration flip frames",
                zorder=5,
            )
        acceleration_ax.set_title("Ankle Acceleration Changes")
        acceleration_ax.set_xlabel("Frame")
        acceleration_ax.set_ylabel("Per-frame velocity delta")
        acceleration_ax.grid(True, alpha=0.3)
        acceleration_ax.legend()

    fig.suptitle(f"Expert Footwork Trajectories ({handedness})", fontsize=14)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5))
    plt.tight_layout(rect=(0, 0, 0.85, 0.96))

    output_path = Path(output_dir) / f"{handedness}" / "footwork_trajectories.png"
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_trajectories_json(
    trajectories: dict[str, ExpertTrajectory],
    output_dir: str,
    handedness: Handedness,
) -> Path:
    output_path = Path(output_dir) / f"{handedness}" / "footwork_trajectories.json"
    payload = {
        "handedness": str(handedness),
        "experts": trajectories,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract and visualize expert footwork ankle trajectories",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir", required=True, help="Directory containing expert videos"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory to write outputs"
    )
    parser.add_argument(
        "--handedness",
        choices=[str(handedness) for handedness in Handedness],
        default=str(Handedness.RIGHT),
        help="Handedness of the experts in the videos",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=30,
        help="Evaluate left/right distance flips at this frame interval",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Error: input directory not found: {input_dir}")
        return 1

    handedness = Handedness.convert_to_enum(args.handedness)
    trajectories, expert_frames = collect_expert_trajectories(
        args.input_dir,
        args.output_dir,
        handedness,
    )
    if not trajectories:
        print("Error: no valid expert ankle trajectories were found.")
        return 1

    (
        distance_flip_frames,
        velocity_flip_frames,
        acceleration_flip_frames,
        left_mean,
        right_mean,
        left_velocity_mean,
        right_velocity_mean,
        left_acceleration_mean,
        right_acceleration_mean,
    ) = find_flip_frames(
        trajectories,
        args.frame_step,
    )
    plot_path = plot_trajectories(
        trajectories,
        args.output_dir,
        handedness,
        distance_flip_frames,
        velocity_flip_frames,
        acceleration_flip_frames,
        left_mean,
        right_mean,
        left_velocity_mean,
        right_velocity_mean,
        left_acceleration_mean,
        right_acceleration_mean,
    )
    json_path = save_trajectories_json(trajectories, args.output_dir, handedness)
    distance_flip_dir = save_highlighted_frames(
        trajectories,
        expert_frames,
        args.output_dir,
        handedness,
        distance_flip_frames,
        "flip_frames",
    )
    velocity_flip_dir = save_highlighted_frames(
        trajectories,
        expert_frames,
        args.output_dir,
        handedness,
        velocity_flip_frames,
        "velocity_flip_frames",
    )
    acceleration_flip_dir = save_highlighted_frames(
        trajectories,
        expert_frames,
        args.output_dir,
        handedness,
        acceleration_flip_frames,
        "acceleration_flip_frames",
    )
    print(f"Saved plot to {plot_path}")
    print(f"Saved trajectories to {json_path}")
    print(f"Saved left/right distance flip frames to {distance_flip_dir}")
    print(f"Saved left/right velocity flip frames to {velocity_flip_dir}")
    print(f"Saved left/right acceleration flip frames to {acceleration_flip_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
