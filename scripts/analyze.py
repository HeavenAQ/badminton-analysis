import os
from typing import Any
import cv2
import numpy as np
import pandas as pd
from badminton_analysis.models.joints import JOINTS
from badminton_analysis.models.types import Handedness, Skill
from badminton_analysis.services.video_analyzer import VideoAnalyzer
from badminton_analysis.services.video_processor import VideoProcessor
from badminton_analysis.services.body_normalizer import BodyCentricNormalizer
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def to_dominant_feature_name(feature: str) -> str:
    replacements = {
        "Left Elbow Angle": "Non-dominant Elbow Angle",
        "Right Elbow Angle": "Dominant Elbow Angle",
        "Left Knee Angle": "Non-dominant Knee Angle",
        "Right Knee Angle": "Dominant Knee Angle",
        "Left Shoulder Angle": "Non-dominant Shoulder Angle",
        "Right Shoulder Angle": "Dominant Shoulder Angle",
        "Left Crotch Angle": "Non-dominant Crotch Angle",
        "Right Crotch Angle": "Dominant Crotch Angle",
        "Nose Left Shoulder Elbow Angle": "Nose Non-dominant Shoulder Elbow Angle",
        "Nose Right Shoulder Elbow Angle": "Nose Dominant Shoulder Elbow Angle",
    }
    return replacements.get(feature, feature)


def create_dirs(output_dir: str) -> None:
    """Create the output directory tree for exported stats and key frames."""
    # create directories if not exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/video", exist_ok=True)
    os.makedirs(f"{output_dir}/frame0", exist_ok=True)
    os.makedirs(f"{output_dir}/frame1", exist_ok=True)
    os.makedirs(f"{output_dir}/frame2", exist_ok=True)
    os.makedirs(f"{output_dir}/frame3", exist_ok=True)
    os.makedirs(f"{output_dir}/frame4", exist_ok=True)


def get_target_indices(
    processor: VideoProcessor,
    analyzer: VideoAnalyzer,
    handedness: Handedness,
    skill: Skill,
) -> list[int]:
    _ = processor.process_frames(handedness)
    start, peak, end = analyzer.find_analysis_window(
        skill=skill,
        hand_positions=processor.hand_positions,
        elbow_positions=processor.elbow_positions,
    )
    return [
        0,
        (start + peak) // 2,
        peak,
        (peak + end) // 2,
        end,
    ]


def write_frames(
    processor: VideoProcessor, output_dir: str, filename: str, frames: list[int]
) -> None:
    frame_basename = f"{os.path.splitext(filename)[0]}.jpg"
    for i, frame in enumerate(frames):
        target_frame = processor.frames[frame].copy()
        processor.pose_detector.show_pose(
            target_frame,
            processor.landmarks[frame],
        )
        processor.pose_detector.show_angles(
            target_frame,
            processor.landmarks[frame],
        )
        output_path = f"{output_dir}/frame{i}/{frame_basename}"
        success = cv2.imwrite(output_path, target_frame)
        if not success:
            raise RuntimeError(f"Failed to write frame image to {output_path}")


data_avg = {
    "feature": [to_dominant_feature_name(feature) for feature in JOINTS.keys()],
    "check1": [],
    "check2": [],
    "check3": [],
    "check4": [],
    "check5": [],
}

data_std = {
    "feature": [to_dominant_feature_name(feature) for feature in JOINTS.keys()],
    "check1": [],
    "check2": [],
    "check3": [],
    "check4": [],
    "check5": [],
}


def process_videos_in_dir(
    input_dir: str,
    output_dir: str,
    handedness: Handedness,
    skill: Skill,
) -> list[list[Any]]:
    # setup
    create_dirs(output_dir)
    videos = os.listdir(input_dir)

    # for stats
    frame_angles: list[list[Any]] = [[] for _ in range(5)]

    analyzer = VideoAnalyzer()
    normalizer = BodyCentricNormalizer(handedness == Handedness.RIGHT)
    for video_file in videos:
        if video_file.lower().endswith((".mp4", ".mov")):
            video_path = os.path.join(input_dir, video_file)
            print(f"Processing: {video_path}")

            # get target indices
            processor = VideoProcessor(
                video_path,
                video_file,
                output_dir,
            )
            target_indices = get_target_indices(
                processor,
                analyzer,
                handedness,
                skill,
            )

            # write the target frames
            write_frames(
                processor,
                output_dir,
                video_file,
                target_indices,
            )

            # get the landmarks and angles
            landmark_list = [
                normalizer.normalize_pose(processor.landmarks[i])
                for i in target_indices
            ]
            angle_list = list(
                map(
                    analyzer.compute_angles,
                    landmark_list,
                )
            )
            if handedness == Handedness.LEFT:
                angle_list = [analyzer.mirror_angles(a) for a in angle_list]

            # save the angle stats
            for i, angles in enumerate(angle_list):
                frame_angles[i].append(angles)

            # save the output video
            output_path = processor.save_video_segment(
                target_indices[0],
                target_indices[-1],
                30,
                filename=f"video/{video_file}",
            )
            print(f"The processed videos is saved to {output_path}")
    return frame_angles


def compute_mean_std(frame_angles: list[list[Any]]) -> None:
    # doing stats
    for feature in JOINTS.keys():
        for i, angles in enumerate(frame_angles):
            target_feature = list(
                map(
                    lambda x: x[feature],
                    angles,
                )
            )
            data_avg[f"check{i + 1}"].append(
                np.mean(target_feature),
            )
            data_std[f"check{i + 1}"].append(
                np.std(target_feature),
            )


def save_stats(output_dir: str) -> None:
    pd.DataFrame(data_avg).to_csv(f"{output_dir}/mean.csv")
    pd.DataFrame(data_std).to_csv(f"{output_dir}/std.csv")


def sanity_check_handedness(
    raw_by_handedness: dict[Handedness, list[list[Any]]],
    skill: Skill,
    features: list[str] | None = None,
    checkpoint_labels: list[str] | None = None,
) -> None:
    """
    Overlay handedness distributions in the normalized dominant/non-dominant frame.
    """
    if checkpoint_labels is None:
        checkpoint_labels = ["start", "mid-start", "peak", "mid-end", "end"]
    if features is None:
        features = [to_dominant_feature_name(feature) for feature in JOINTS.keys()]

    source_features = [
        feature.replace("Dominant", "Right").replace("Non-dominant", "Left")
        for feature in features
    ]

    n_features = len(features)
    n_checkpoints = 5

    colors = {Handedness.RIGHT: "#378ADD", Handedness.LEFT: "#D85A30"}
    fig, axes = plt.subplots(
        n_features,
        n_checkpoints,
        figsize=(n_checkpoints * 3, n_features * 2.5),
        sharey="row",
    )
    fig.suptitle(
        f"Handedness sanity check in dominant/non-dominant frame — {skill}",
        fontsize=13,
        y=1.01,
    )

    for row, feature in enumerate(features):
        source_feature = source_features[row]
        for col, label in enumerate(checkpoint_labels):
            ax = axes[row, col] if n_features > 1 else axes[col]
            for handedness, frame_angles in raw_by_handedness.items():
                checkpoint_angles = frame_angles[col]
                values = [
                    a[source_feature]
                    for a in checkpoint_angles
                    if source_feature in a and a[source_feature] is not None
                ]

                if values:
                    ax.hist(
                        values,
                        bins=max(3, len(values) // 2),
                        alpha=0.45,
                        color=colors[handedness],
                        edgecolor="none",
                        label=f"{handedness.value}-handed"
                        if row == 0 and col == 0
                        else None,
                    )
                    ax.axvline(
                        np.mean(values),
                        color=colors[handedness],
                        linewidth=1.5,
                        linestyle="--",
                    )

            if row == 0:
                ax.set_title(label, fontsize=10)
            if col == 0:
                ax.set_ylabel(
                    feature,
                    fontsize=9,
                    rotation=0,
                    labelpad=60,
                    va="center",
                )

            ax.tick_params(labelsize=7)
            ax.set_xlabel("degrees", fontsize=7)

    handles, labels = (
        axes[0, 0].get_legend_handles_labels()
        if n_features > 1
        else axes[0].get_legend_handles_labels()
    )
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=9)

    plt.tight_layout()

    out_path = f"./stats/{skill}/sanity_check_handedness_overlay.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Sanity check saved to {out_path}")


if __name__ == "__main__":
    skill = Skill.LIFT

    # process selected data
    output_dir = f"./stats/{skill}/"
    raw_handedness: dict[Handedness, list[list[Any]]] = {}
    for handedness in [Handedness.LEFT, Handedness.RIGHT]:
        frame_angles: list[list[Any]] = [[] for _ in range(5)]
        input_dir = f"./training_videos/nstc/{skill}/{handedness}/"
        cur_frame_angles = process_videos_in_dir(
            input_dir,
            output_dir,
            handedness,
            skill,
        )

        for i, angles in enumerate(cur_frame_angles):
            frame_angles[i].extend(angles)

        raw_handedness[handedness] = frame_angles

    # sanity check — only plot angle features used by the serve grader.
    # Checkpoint 3 uses hip-axis rotation derived from landmark coordinates,
    # so it is not represented by a single angle feature in this plot.
    serve_features = [
        "Dominant Shoulder Angle",  # checkpoint 1 arms, checkpoint 5
        "Non-dominant Shoulder Angle",  # checkpoint 1 arms
        "Dominant Crotch Angle",  # checkpoint 1 legs, checkpoint 2 lower body
        "Non-dominant Crotch Angle",  # checkpoint 1 legs, checkpoint 2 lower body
        "Dominant Elbow Angle",  # checkpoint 4
        "Nose Dominant Shoulder Elbow Angle",  # checkpoint 5
    ]
    sanity_check_handedness(raw_handedness, skill, features=serve_features)
    # Then aggregate and save
    all_frame_angles: list[list[Any]] = [[] for _ in range(5)]
    for handedness, frame_angles in raw_handedness.items():
        for i, angles in enumerate(frame_angles):
            all_frame_angles[i].extend(angles)

    compute_mean_std(all_frame_angles)
    save_stats(output_dir)
    print("All videos processed successfully")
    print(f"Total processed videos: {len(all_frame_angles[0])}")
