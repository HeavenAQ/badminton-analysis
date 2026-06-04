import argparse
import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats
from typing_extensions import Sequence

from badminton_analysis.models.types import AngleDict, Handedness, Skill
from badminton_analysis.services import VideoProcessor
from badminton_analysis.services.video_analyzer import VideoAnalyzer


ROOT_DIR = Path("./training_videos/nstc/")


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate expert stability statistics for badminton skills",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--skill",
        required=True,
        choices=[str(s) for s in Skill],
        help="Skill to analyze",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write results",
    )
    return parser


LEFT_HANDED_EXPERTS = ["張恩嘉", "陳緯"]

EXPERT_NAMES = [
    "呂家弘",
    "張恩嘉",
    "徐紹文",
    "簡綵琳",
    "陳緯",
]


def safe_std(values: list[float]) -> float:
    if len(values) < 2:
        return np.nan
    return float(np.std(values, ddof=1))



def confidence_interval_95(values: list[float]) -> tuple[float, float, float]:
    """
    Returns ci_low, ci_high, ci_width.
    Uses t-distribution: mean ± t(0.975, df=n-1) * std / sqrt(n)
    """
    values = [v for v in values if not np.isnan(v)]
    n = len(values)

    if n < 2:
        return np.nan, np.nan, np.nan

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    se = std / np.sqrt(n)
    t_crit = float(scipy_stats.t.ppf(0.975, df=n - 1))

    ci_low = mean - t_crit * se
    ci_high = mean + t_crit * se
    ci_width = ci_high - ci_low

    return float(ci_low), float(ci_high), float(ci_width)


def variance_decomposition(rows: list[dict]) -> dict:
    """
    rows contains values for one checkpoint × one feature.

    Required keys:
      - expert
      - angle

    This computes:
      - within-expert variation
      - between-expert variation
      - eta squared
      - ANOVA-style F statistic
    """
    values = [float(r["angle"]) for r in rows if not np.isnan(float(r["angle"]))]

    if len(values) < 2:
        return {
            "n_total": len(values),
            "n_experts": 0,
            "overall_mean": np.nan,
            "overall_std": np.nan,
            "overall_cv": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "ci_width": np.nan,
            "ss_total": np.nan,
            "ss_between": np.nan,
            "ss_within": np.nan,
            "ms_between": np.nan,
            "ms_within": np.nan,
            "f_stat": np.nan,
            "eta_squared": np.nan,
            "judgment": "insufficient_data",
        }

    # Group angles by expert
    expert_values: dict[str, list[float]] = {}
    for row in rows:
        expert = row["expert"]
        angle = float(row["angle"])

        if np.isnan(angle):
            continue

        expert_values.setdefault(expert, []).append(angle)

    # Keep only experts with at least one value
    expert_values = {k: v for k, v in expert_values.items() if len(v) > 0}

    n_total = sum(len(v) for v in expert_values.values())
    n_experts = len(expert_values)

    if n_total < 2 or n_experts < 2:
        return {
            "n_total": n_total,
            "n_experts": n_experts,
            "overall_mean": np.nan,
            "overall_std": np.nan,
            "overall_cv": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "ci_width": np.nan,
            "ss_total": np.nan,
            "ss_between": np.nan,
            "ss_within": np.nan,
            "ms_between": np.nan,
            "ms_within": np.nan,
            "f_stat": np.nan,
            "eta_squared": np.nan,
            "judgment": "insufficient_data",
        }

    all_values = np.array(
        [angle for vals in expert_values.values() for angle in vals],
        dtype=float,
    )

    overall_mean = float(np.mean(all_values))
    overall_std = safe_std(all_values.tolist())
    overall_cv = (
        overall_std / abs(overall_mean)
        if overall_mean != 0 and not np.isnan(overall_std)
        else np.nan
    )

    ci_low, ci_high, ci_width = confidence_interval_95(all_values.tolist())

    # Between-expert SS:
    # sum_j n_j * (mean_j - overall_mean)^2
    ss_between = 0.0
    for vals in expert_values.values():
        vals_arr = np.array(vals, dtype=float)
        expert_mean = float(np.mean(vals_arr))
        ss_between += len(vals_arr) * (expert_mean - overall_mean) ** 2

    # Within-expert SS:
    # sum_j sum_i (x_ij - mean_j)^2
    ss_within = 0.0
    for vals in expert_values.values():
        vals_arr = np.array(vals, dtype=float)
        expert_mean = float(np.mean(vals_arr))
        ss_within += float(np.sum((vals_arr - expert_mean) ** 2))

    ss_total = ss_between + ss_within

    df_between = n_experts - 1
    df_within = n_total - n_experts

    ms_between = ss_between / df_between if df_between > 0 else np.nan
    ms_within = ss_within / df_within if df_within > 0 else np.nan

    f_stat = (
        ms_between / ms_within
        if not np.isnan(ms_within) and ms_within > 0
        else np.nan
    )

    eta_squared = ss_between / ss_total if ss_total > 0 else np.nan

    if np.isnan(eta_squared):
        judgment = "insufficient_data"
    elif eta_squared < 0.05:
        judgment = "good_standard"
    elif eta_squared < 0.14:
        judgment = "use_with_caution"
    else:
        judgment = "style_dependent"

    return {
        "n_total": n_total,
        "n_experts": n_experts,
        "overall_mean": overall_mean,
        "overall_std": overall_std,
        "overall_cv": overall_cv,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_width": ci_width,
        "ss_total": ss_total,
        "ss_between": ss_between,
        "ss_within": ss_within,
        "ms_between": ms_between,
        "ms_within": ms_within,
        "f_stat": f_stat,
        "eta_squared": eta_squared,
        "judgment": judgment,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        print(f"No rows to write: {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {path}")


CHECKPOINT_LABELS = ["Preparation", "Mid-start", "Peak", "Mid-end", "End"]
CHECKPOINT_KEYS = ["check1", "check2", "check3", "check4", "check5"]

EXPERT_COLORS = [
    "#2196F3",
    "#E91E63",
    "#4CAF50",
    "#FF9800",
    "#9C27B0",
    "#00BCD4",
    "#FF5722",
]

JUDGMENT_COLORS = {
    "good_standard": "#4CAF50",
    "use_with_caution": "#FFC107",
    "style_dependent": "#F44336",
    "insufficient_data": "#9E9E9E",
}


def _dominant_key(feature: str, handedness: Handedness) -> str:
    """Map a 'Dominant/Non-dominant' feature name to the raw 'Right/Left' angle dict key."""
    dom = "Right" if handedness == Handedness.RIGHT else "Left"
    non = "Left" if handedness == Handedness.RIGHT else "Right"
    return (
        feature.replace("Dominant", dom).replace("Non-dominant", non)
    )



def plot_stability_heatmap(
    stability_rows: list[dict],
    features: list[str],
    skill: Skill,
    output_dir: Path,
) -> None:
    """Heatmap of eta-squared by feature × checkpoint, colored by judgment category."""
    n_features = len(features)
    n_checkpoints = len(CHECKPOINT_KEYS)

    eta_matrix = np.full((n_features, n_checkpoints), np.nan)
    f_matrix = np.full((n_features, n_checkpoints), np.nan)
    judgment_matrix: list[list[str]] = [[""] * n_checkpoints for _ in range(n_features)]

    for row in stability_rows:
        if row["feature"] not in features:
            continue
        r = features.index(row["feature"])
        ck = row["checkpoint"]
        if ck not in CHECKPOINT_KEYS:
            continue
        c = CHECKPOINT_KEYS.index(ck)
        eta_matrix[r, c] = float(row["eta_squared"]) if not _isnan(row["eta_squared"]) else np.nan
        f_matrix[r, c] = float(row["f_stat"]) if not _isnan(row["f_stat"]) else np.nan
        judgment_matrix[r][c] = str(row["judgment"])

    fig, ax = plt.subplots(figsize=(10, n_features * 0.9 + 2.2))

    # Draw colored cells manually so we can use judgment colors
    color_matrix = np.zeros((n_features, n_checkpoints, 4))
    for r in range(n_features):
        for c in range(n_checkpoints):
            j = judgment_matrix[r][c]
            hex_color = JUDGMENT_COLORS.get(j, "#9E9E9E")
            rgba = _hex_to_rgba(hex_color, alpha=0.75)
            color_matrix[r, c] = rgba

    ax.imshow(color_matrix, aspect="auto")

    for r in range(n_features):
        for c in range(n_checkpoints):
            eta = eta_matrix[r, c]
            f = f_matrix[r, c]
            j = judgment_matrix[r][c]
            eta_str = f"{eta:.3f}" if not np.isnan(eta) else "n/a"
            f_str = f"F={f:.1f}" if not np.isnan(f) else ""
            label = f"η²={eta_str}\n{f_str}"
            ax.text(c, r, label, ha="center", va="center", fontsize=7.5, color="black")

    ax.set_xticks(range(n_checkpoints))
    ax.set_xticklabels(CHECKPOINT_LABELS, fontsize=9)
    ax.set_yticks(range(n_features))
    ax.set_yticklabels(features, fontsize=9)
    ax.set_title(
        f"Feature Stability (η²) — {skill}\n"
        "green=consistent  yellow=caution  red=style-dependent",
        fontsize=11,
    )

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=JUDGMENT_COLORS["good_standard"], label="good standard (η²<0.05)"),
        Patch(facecolor=JUDGMENT_COLORS["use_with_caution"], label="use with caution (η²<0.14)"),
        Patch(facecolor=JUDGMENT_COLORS["style_dependent"], label="style dependent (η²≥0.14)"),
        Patch(facecolor=JUDGMENT_COLORS["insufficient_data"], label="insufficient data"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1, -0.12),
              ncol=2, fontsize=8)

    plt.tight_layout()
    out = output_dir / "stability_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_per_expert_boxplots(
    angle_rows: list[dict],
    features: list[str],
    skill: Skill,
    output_dir: Path,
) -> None:
    """Box plots of angle distribution per expert, faceted by feature × checkpoint."""
    experts = sorted({r["expert"] for r in angle_rows})
    n_features = len(features)
    n_checkpoints = len(CHECKPOINT_KEYS)

    fig, axes = plt.subplots(
        n_features,
        n_checkpoints,
        figsize=(n_checkpoints * 2.4, n_features * 2.6 + 1),
        sharey="row",
    )
    if n_features == 1:
        axes = [axes]

    fig.suptitle(f"Per-expert Angle Distributions — {skill}", fontsize=13, y=1.01)

    for r, feature in enumerate(features):
        for c, (ck, ck_label) in enumerate(zip(CHECKPOINT_KEYS, CHECKPOINT_LABELS)):
            ax = axes[r][c] if n_checkpoints > 1 else axes[r]
            data = [
                [
                    row["angle"]
                    for row in angle_rows
                    if row["expert"] == expert
                    and row["checkpoint"] == ck
                    and row["feature"] == feature
                ]
                for expert in experts
            ]
            data_clean = [d if d else [float("nan")] for d in data]

            bp = ax.boxplot(
                data_clean,
                patch_artist=True,
                widths=0.55,
                medianprops={"color": "black", "linewidth": 1.5},
                whiskerprops={"linewidth": 1},
                capprops={"linewidth": 1},
                flierprops={"markersize": 3},
            )
            for patch, color in zip(bp["boxes"], EXPERT_COLORS):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_xticks(range(1, len(experts) + 1))
            ax.set_xticklabels(
                [e[:3] for e in experts], fontsize=6, rotation=45, ha="right"
            )
            if r == 0:
                ax.set_title(ck_label, fontsize=8)
            if c == 0:
                ax.set_ylabel(f"{feature}\n(degrees)", fontsize=7)
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(True, alpha=0.25, axis="y")

    plt.tight_layout()
    out = output_dir / "per_expert_boxplots.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def _isnan(val: object) -> bool:
    try:
        return bool(np.isnan(float(val)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return True


def _hex_to_rgba(hex_color: str, alpha: float = 1.0) -> tuple:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    return (r, g, b, alpha)


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    skill = Skill.convert_to_enum(args.skill)

    features = [
        "Dominant Shoulder Angle",
        "Non-dominant Shoulder Angle",
        "Dominant Elbow Angle",
    ]

    analyzer = VideoAnalyzer()

    # Long-format table:
    # expert, video, checkpoint, feature, angle
    angle_rows: list[dict] = []

    for expert in EXPERT_NAMES:
        expert_dir = ROOT_DIR / str(skill) / expert

        if not expert_dir.exists():
            print(f"Expert directory not found, skipping: {expert_dir}")
            continue

        videos = os.listdir(expert_dir)

        for video in videos:
            if not video.lower().endswith((".mp4", ".mov")):
                print(f"Skipping non-video file: {video}")
                continue

            video_path = expert_dir / video
            print(f"Processing: {video_path}")

            handedness = (
                Handedness.RIGHT
                if expert not in LEFT_HANDED_EXPERTS
                else Handedness.LEFT
            )

            try:
                processor = VideoProcessor(
                    str(video_path),
                    out_filename=video,
                    output_folder=str(output_dir),
                )

                target_indices = get_target_indices(
                    processor=processor,
                    analyzer=analyzer,
                    handedness=handedness,
                    skill=skill,
                )

                landmark_list = [processor.landmarks[i] for i in target_indices]
                angle_list: list[AngleDict] = list(
                    map(analyzer.compute_angles, landmark_list)
                )

                for check_idx, angles in enumerate(angle_list, start=1):
                    checkpoint = f"check{check_idx}"

                    for feature in features:
                        raw_key = _dominant_key(feature, handedness)
                        if raw_key not in angles:
                            print(
                                f"Feature missing: expert={expert}, "
                                f"video={video}, checkpoint={checkpoint}, "
                                f"feature={feature} (key={raw_key})"
                            )
                            continue

                        angle_rows.append(
                            {
                                "expert": expert,
                                "video": video,
                                "skill": str(skill),
                                "checkpoint": checkpoint,
                                "feature": feature,
                                "angle": float(angles[raw_key]),
                            }
                        )

            except Exception as e:
                print(f"Failed: {video_path} | Error: {e}")
                continue

    # Save raw long-format angle table.
    write_csv(output_dir / "expert_angle_long.csv", angle_rows)

    # Feature stability summary (ANOVA / variance decomposition):
    # one row per skill × checkpoint × feature
    stability_rows: list[dict] = []

    stability_keys = sorted(
        {(row["skill"], row["checkpoint"], row["feature"]) for row in angle_rows}
    )

    for skill_name, checkpoint, feature in stability_keys:
        rows = [
            row
            for row in angle_rows
            if row["skill"] == skill_name
            and row["checkpoint"] == checkpoint
            and row["feature"] == feature
        ]

        stats = variance_decomposition(rows)

        stability_rows.append(
            {
                "skill": skill_name,
                "checkpoint": checkpoint,
                "feature": feature,
                **stats,
            }
        )

    write_csv(output_dir / "feature_stability_summary.csv", stability_rows)

    plot_stability_heatmap(stability_rows, features, skill, output_dir)
    plot_per_expert_boxplots(angle_rows, features, skill, output_dir)


if __name__ == "__main__":
    main()
