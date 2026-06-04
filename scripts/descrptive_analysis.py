import math
import os
from typing import Any
import cv2
import numpy as np
import pandas as pd
from badminton_analysis.models.joints import JOINTS
from badminton_analysis.models.types import COCOKeypoints, CoordinateDict, Handedness, Skill
from badminton_analysis.services.video_analyzer import VideoAnalyzer
from badminton_analysis.services.video_processor import VideoProcessor
import matplotlib.pyplot as plt


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
        start,
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
        processor.pose_detector.show_pose(target_frame, processor.landmarks[frame])
        processor.pose_detector.show_angles(target_frame, processor.landmarks[frame])
        output_path = f"{output_dir}/frame{i}/{frame_basename}"
        success = cv2.imwrite(output_path, target_frame)
        if not success:
            raise RuntimeError(f"Failed to write frame image to {output_path}")


# ---------------------------------------------------------------------------
# Stats accumulators (populated by compute_stats)
# ---------------------------------------------------------------------------

data_avg: dict[str, list[Any]] = {
    "feature": [to_dominant_feature_name(feature) for feature in JOINTS.keys()],
    "check1": [],
    "check2": [],
    "check3": [],
    "check4": [],
    "check5": [],
}
data_std: dict[str, list[Any]] = {
    "feature": [to_dominant_feature_name(feature) for feature in JOINTS.keys()],
    "check1": [],
    "check2": [],
    "check3": [],
    "check4": [],
    "check5": [],
}
data_cv: dict[str, list[Any]] = {
    "feature": [to_dominant_feature_name(feature) for feature in JOINTS.keys()],
    "check1": [],
    "check2": [],
    "check3": [],
    "check4": [],
    "check5": [],
}
# mean absolute z-score across videos - a scalar measure of how "unusual" the
# average video is for each feature/checkpoint
data_z: dict[str, list[Any]] = {
    "feature": [to_dominant_feature_name(feature) for feature in JOINTS.keys()],
    "check1": [],
    "check2": [],
    "check3": [],
    "check4": [],
    "check5": [],
}


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------


# (feature_name, frame_a, frame_b, kp_from, kp_to) - always RIGHT->LEFT so abs() is handedness-invariant
_RotationEntry = tuple[str, int, int, COCOKeypoints, COCOKeypoints]

SKILL_ROTATION_CONFIG: dict[Skill, list[_RotationEntry]] = {
    Skill.SMASH: [
        ("Hip Line Rotation CP2", 0, 1, COCOKeypoints.RIGHT_HIP, COCOKeypoints.LEFT_HIP),
        ("Shoulder Line Rotation CP6", 0, 3, COCOKeypoints.RIGHT_SHOULDER, COCOKeypoints.LEFT_SHOULDER),
    ],
    Skill.CLEAR: [
        ("Hip Line Rotation CP2", 0, 1, COCOKeypoints.RIGHT_HIP, COCOKeypoints.LEFT_HIP),
        ("Shoulder Line Rotation CP6", 0, 3, COCOKeypoints.RIGHT_SHOULDER, COCOKeypoints.LEFT_SHOULDER),
    ],
    Skill.SERVE: [
        ("Hip Line Rotation CP3", 0, 4, COCOKeypoints.RIGHT_HIP, COCOKeypoints.LEFT_HIP),
    ],
}

_TrunkLeanEntry = tuple[str, int, int]  # (feature_name, frame_a, frame_b)

SKILL_TRUNK_LEAN_CONFIG: dict[Skill, list[_TrunkLeanEntry]] = {
    Skill.SMASH: [("Trunk Slope Delta CP6", 0, 4)],
    Skill.CLEAR: [("Trunk Slope Delta CP6", 0, 4)],
}


def _trunk_slope_delta(
    landmark_list: list[CoordinateDict],
    frame_a: int,
    frame_b: int,
) -> float:
    """Forward body lean: absolute change in trunk slope (shoulder_mid_x - hip_mid_x) / trunk_height."""
    def trunk_slope(frame: CoordinateDict) -> float:
        hip_x = (frame[COCOKeypoints.LEFT_HIP][0] + frame[COCOKeypoints.RIGHT_HIP][0]) / 2
        hip_y = (frame[COCOKeypoints.LEFT_HIP][1] + frame[COCOKeypoints.RIGHT_HIP][1]) / 2
        sho_x = (frame[COCOKeypoints.LEFT_SHOULDER][0] + frame[COCOKeypoints.RIGHT_SHOULDER][0]) / 2
        sho_y = (frame[COCOKeypoints.LEFT_SHOULDER][1] + frame[COCOKeypoints.RIGHT_SHOULDER][1]) / 2
        trunk_h = abs(hip_y - sho_y)
        return float((sho_x - hip_x) / trunk_h) if trunk_h > 1e-6 else 0.0
    return abs(trunk_slope(landmark_list[frame_b]) - trunk_slope(landmark_list[frame_a]))


def _line_rotation_deg(
    landmark_list: list[CoordinateDict],
    frame_a: int,
    frame_b: int,
    kp_from: COCOKeypoints,
    kp_to: COCOKeypoints,
) -> float:
    """Absolute rotation of the line kp_from->kp_to between two key frames (degrees)."""
    def angle(frame: CoordinateDict) -> float:
        dy = float(frame[kp_to][1] - frame[kp_from][1])
        dx = float(frame[kp_to][0] - frame[kp_from][0])
        return math.degrees(math.atan2(dy, dx))

    a, b = angle(landmark_list[frame_a]), angle(landmark_list[frame_b])
    return abs((b - a + 180) % 360 - 180)


def process_videos_in_dir(
    input_dir: str,
    output_dir: str,
    handedness: Handedness,
    skill: Skill,
) -> tuple[list[list[Any]], list[dict[str, float]]]:
    create_dirs(output_dir)
    videos = os.listdir(input_dir)

    frame_angles: list[list[Any]] = [[] for _ in range(5)]
    rotation_deltas: list[dict[str, float]] = []
    rotation_config = SKILL_ROTATION_CONFIG.get(skill, [])
    trunk_lean_config = SKILL_TRUNK_LEAN_CONFIG.get(skill, [])
    analyzer = VideoAnalyzer()

    for video_file in videos:
        if video_file.lower().endswith((".mp4", ".mov")):
            video_path = os.path.join(input_dir, video_file)
            print(f"Processing: {video_path}")

            processor = VideoProcessor(video_path, video_file, output_dir)
            target_indices = get_target_indices(processor, analyzer, handedness, skill)
            write_frames(processor, output_dir, video_file, target_indices)

            landmark_list = [processor.landmarks[i] for i in target_indices]

            angle_list = list(map(analyzer.compute_angles, landmark_list))
            # to_dominant_feature_name maps Right->Dominant, Left->Non-dominant.
            # For left-handed players, dominant is Left, so swap the labels
            # first so the mapping treats their dominant arm correctly.
            if handedness == Handedness.LEFT:
                angle_list = [analyzer.mirror_angles(a) for a in angle_list]

            for i, angles in enumerate(angle_list):
                frame_angles[i].append(angles)

            if rotation_config or trunk_lean_config:
                delta: dict[str, float] = {
                    name: _line_rotation_deg(landmark_list, fa, fb, kp_from, kp_to)
                    for name, fa, fb, kp_from, kp_to in rotation_config
                }
                for name, fa, fb in trunk_lean_config:
                    delta[name] = _trunk_slope_delta(landmark_list, fa, fb)
                rotation_deltas.append(delta)

            output_path = processor.save_video_segment(
                target_indices[0],
                target_indices[-1],
                30,
                filename=f"video/{video_file}",
            )
            print(f"Saved video to {output_path}")
    return frame_angles, rotation_deltas


def compute_stats(frame_angles: list[list[Any]]) -> None:
    for feature in JOINTS.keys():
        for i, angles in enumerate(frame_angles):
            values = np.asarray([a[feature] for a in angles], dtype=float)
            mean = float(np.mean(values))
            std = float(np.std(values))

            data_avg[f"check{i + 1}"].append(mean)
            data_std[f"check{i + 1}"].append(std)
            # CV: guard against near-zero mean to avoid division by zero
            data_cv[f"check{i + 1}"].append(
                std / mean if abs(mean) > 1e-6 else float("nan")
            )
            # mean absolute z-score across all videos for this feature/checkpoint
            if std > 1e-6:
                z_scores = np.abs((values - mean) / std)
                data_z[f"check{i + 1}"].append(float(np.mean(z_scores)))
            else:
                data_z[f"check{i + 1}"].append(0.0)


def save_stats(output_dir: str) -> None:
    pd.DataFrame(data_avg).to_csv(f"{output_dir}/mean.csv")
    pd.DataFrame(data_std).to_csv(f"{output_dir}/std.csv")
    pd.DataFrame(data_cv).to_csv(f"{output_dir}/coefficient_variety.csv")
    pd.DataFrame(data_z).to_csv(f"{output_dir}/mean_abs_z.csv")


def save_rotation_stats(
    rotation_deltas: list[dict[str, float]], output_dir: str
) -> None:
    if not rotation_deltas:
        return
    feature_names = list(rotation_deltas[0].keys())
    rows = []
    for name in feature_names:
        values = np.array([d[name] for d in rotation_deltas])
        rows.append({"feature": name, "mean": float(np.mean(values)), "std": float(np.std(values))})
        print(f"{name}: mean={np.mean(values):.1f}°  std={np.std(values):.1f}°")
    pd.DataFrame(rows).to_csv(f"{output_dir}/rotation_stats.csv", index=False)


def plot_rotation_distributions(
    rotation_deltas: list[dict[str, float]], output_dir: str, skill: Skill
) -> None:
    """Histograms of all collected rotation deltas for this skill."""
    if not rotation_deltas:
        return
    feature_names = list(rotation_deltas[0].keys())
    n = len(feature_names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    fig.suptitle(f"Expert Body Rotation Distributions — {skill}", fontsize=12)

    for ax, name in zip(axes, feature_names):
        values = np.array([d[name] for d in rotation_deltas])
        ax.hist(values, bins=max(5, len(values) // 2), color="#2196F3", edgecolor="white", alpha=0.8)
        ax.axvline(float(np.mean(values)), color="red", linewidth=1.5, linestyle="--", label=f"mean={np.mean(values):.1f}°")
        ax.axvspan(
            float(np.mean(values) - np.std(values)),
            float(np.mean(values) + np.std(values)),
            alpha=0.15, color="red", label=f"±1 std ({np.std(values):.1f}°)",
        )
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Rotation (degrees)", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = f"{output_dir}/rotation_distributions.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Rotation distributions saved to {out_path}")


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------


def plot_temporal_profiles(
    features: list[str],
    output_dir: str,
    skill: Skill,
) -> None:
    """Line plot: mean +/- 1 std for each feature across the 5 checkpoints.

    Shows the expected angle trajectory through the movement so you can
    quickly see how each joint should move from preparation to follow-through.
    """
    checkpoint_labels = ["Preparation", "Mid-start", "Peak", "Mid-end", "End"]
    x = np.arange(1, 6)
    n = len(features)

    fig, axes = plt.subplots(n, 1, figsize=(8, n * 2.2 + 1), sharex=True)
    if n == 1:
        axes = [axes]

    fig.suptitle(f"Expert Angle Trajectories — {skill}", fontsize=13, y=1.01)

    for ax, feature in zip(axes, features):
        idx = data_avg["feature"].index(feature)
        means = np.array([data_avg[f"check{c}"][idx] for c in range(1, 6)])
        stds = np.array([data_std[f"check{c}"][idx] for c in range(1, 6)])

        ax.plot(
            x, means, "o-", linewidth=2, markersize=5, color="#2196F3", label="mean"
        )
        ax.fill_between(
            x, means - stds, means + stds, alpha=0.2, color="#2196F3", label="±1 std"
        )
        ax.set_ylabel(f"{feature}\n(degrees)", fontsize=8)
        ax.set_ylim(bottom=max(0, (means - stds).min() - 10))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(checkpoint_labels, fontsize=9)
    axes[-1].set_xlabel("Checkpoint", fontsize=9)

    plt.tight_layout()
    out_path = f"{output_dir}/temporal_profiles.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Temporal profiles saved to {out_path}")


def plot_cv_heatmap(
    features: list[str],
    output_dir: str,
    skill: Skill,
) -> None:
    """Heatmap of coefficient of variation (feature x checkpoint).

    Green = low CV (consistent across experts), red = high CV (variable).
    Useful for understanding which joints/phases have the most consistent
    technique among professionals.
    """
    checkpoint_labels = ["Preparation", "Mid-start", "Peak", "Mid-end", "End"]
    n = len(features)

    matrix = np.zeros((n, 5))
    for r, feature in enumerate(features):
        idx = data_cv["feature"].index(feature)
        for c in range(5):
            val = data_cv[f"check{c + 1}"][idx]
            matrix[r, c] = val if not np.isnan(val) else 0.0

    fig, ax = plt.subplots(figsize=(9, n * 0.65 + 1.8))
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=0.5)

    ax.set_xticks(range(5))
    ax.set_xticklabels(checkpoint_labels, fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(features, fontsize=9)
    ax.set_title(
        f"Coefficient of Variation — {skill}\n(lower = more consistent among experts)",
        fontsize=11,
    )

    for r in range(n):
        for c in range(5):
            ax.text(
                c,
                r,
                f"{matrix[r, c]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )

    plt.colorbar(im, ax=ax, label="CV (std ÷ mean)")
    plt.tight_layout()
    out_path = f"{output_dir}/cv_heatmap.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"CV heatmap saved to {out_path}")


def sanity_check_handedness(
    raw_by_handedness: dict[Handedness, list[list[Any]]],
    skill: Skill,
    features: list[str] | None = None,
    checkpoint_labels: list[str] | None = None,
) -> None:
    """Overlay left/right-handed angle distributions after dominant/non-dominant
    normalization. Both groups should overlap if mirroring is working correctly.
    """
    if checkpoint_labels is None:
        checkpoint_labels = ["start", "mid-start", "peak", "mid-end", "end"]
    if features is None:
        features = [to_dominant_feature_name(feature) for feature in JOINTS.keys()]

    # After mirror_angles, dominant arm always lives under the "Right" key,
    # so we map Dominant->Right / Non-dominant->Left to look up the raw angle.
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
                        label=f"{handedness}-handed" if row == 0 and col == 0 else None,
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
                ax.set_ylabel(feature, fontsize=9, rotation=0, labelpad=60, va="center")
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


# ---------------------------------------------------------------------------
# Per-skill visualization features
# ---------------------------------------------------------------------------

SKILL_VIZ_FEATURES: dict[Skill, list[str]] = {
    Skill.SMASH: [
        "Dominant Shoulder Angle",
        "Non-dominant Shoulder Angle",
        "Dominant Elbow Angle",
        "Nose Dominant Shoulder Elbow Angle",
    ],
    Skill.CLEAR: [
        "Dominant Shoulder Angle",
        "Non-dominant Shoulder Angle",
        "Dominant Elbow Angle",
        "Nose Dominant Shoulder Elbow Angle",
    ],
    Skill.SERVE: [
        "Dominant Shoulder Angle",
        "Non-dominant Shoulder Angle",
        "Dominant Elbow Angle",
        "Dominant Crotch Angle",
        "Non-dominant Crotch Angle",
    ],
    Skill.LIFT: [
        "Dominant Shoulder Angle",
        "Dominant Elbow Angle",
    ],
    Skill.FOREHAND_DRIVE: [
        "Dominant Shoulder Angle",
        "Dominant Elbow Angle",
    ],
    Skill.BACKHAND_DRIVE: [
        "Dominant Shoulder Angle",
        "Dominant Elbow Angle",
    ],
    Skill.FOREHAND_NETKILL: [
        "Dominant Shoulder Angle",
        "Dominant Elbow Angle",
    ],
    Skill.BACKHAND_NETKILL: [
        "Dominant Shoulder Angle",
        "Dominant Elbow Angle",
    ],
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute expert stats for a skill.")
    parser.add_argument(
        "--skill",
        type=lambda s: Skill[s.upper()],
        default=Skill.SMASH,
        help=f"Skill to analyze. Choices: {[s.name for s in Skill]}",
    )
    args = parser.parse_args()
    skill: Skill = args.skill

    output_dir = f"./stats/{skill}/"
    raw_handedness: dict[Handedness, list[list[Any]]] = {}
    all_rotation_deltas: list[dict[str, float]] = []
    for handedness in [Handedness.LEFT, Handedness.RIGHT]:
        frame_angles: list[list[Any]] = [[] for _ in range(5)]
        input_dir = f"./training_videos/nstc/{skill}/{handedness}/"
        cur_frame_angles, cur_rotation_deltas = process_videos_in_dir(
            input_dir, output_dir, handedness, skill
        )

        for i, angles in enumerate(cur_frame_angles):
            frame_angles[i].extend(angles)

        all_rotation_deltas.extend(cur_rotation_deltas)
        raw_handedness[handedness] = frame_angles

    viz_features = SKILL_VIZ_FEATURES.get(
        skill,
        ["Dominant Shoulder Angle", "Non-dominant Shoulder Angle", "Dominant Elbow Angle"],
    )

    sanity_check_handedness(raw_handedness, skill, features=viz_features)

    all_frame_angles: list[list[Any]] = [[] for _ in range(5)]
    for handedness, frame_angles in raw_handedness.items():
        for i, angles in enumerate(frame_angles):
            all_frame_angles[i].extend(angles)

    compute_stats(all_frame_angles)
    save_stats(output_dir)
    save_rotation_stats(all_rotation_deltas, output_dir)

    plot_temporal_profiles(viz_features, output_dir, skill)
    plot_cv_heatmap(viz_features, output_dir, skill)
    plot_rotation_distributions(all_rotation_deltas, output_dir, skill)

    print("All videos processed successfully")
    print(f"Total processed videos: {len(all_frame_angles[0])}")
