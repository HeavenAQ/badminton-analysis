from pathlib import Path

import pandas as pd

from badminton_analysis.tools import grade_students


def _tracking() -> dict:
    return {
        "frames": [],
        "original_landmarks": [{} for _ in range(5)],
        "body_landmarks_2d": [{} for _ in range(5)],
        "hand_positions": [[float(index), 0.0] for index in range(5)],
        "elbow_positions": [[float(index), 1.0] for index in range(5)],
        "time_intervals": [],
    }


def test_grade_students_writes_correction_diagnostics(
    tmp_path: Path, monkeypatch
) -> None:
    input_dir = tmp_path / "videos"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "clear.mp4").write_bytes(b"")

    class FakeProcessor:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def process_frames(self, handedness):
            assert handedness is None
            return _tracking()

    class FakeBackend:
        def __init__(self, model_path, **kwargs) -> None:
            assert str(model_path).endswith("clear_expert_guided_v3.pt")

        def score(self, tracking, handedness, skill):
            return (
                {
                    "total_grade": 44.5,
                    "grading_details": [
                        {"description": "Preparation correction", "grade": 4.5}
                    ],
                },
                (3, 8, 12),
                {
                    "correction_distance": 0.42,
                    "position_distance": 0.3,
                    "angle_distance": 0.1,
                    "velocity_distance": 0.02,
                    "bone_length_distance": 0.01,
                    "model_path": "model.pt",
                    "scorer": "skeleton-correction",
                },
            )

    monkeypatch.setattr(grade_students, "PoseDetector", lambda: object())
    monkeypatch.setattr(grade_students, "VideoProcessor", FakeProcessor)
    monkeypatch.setattr(grade_students, "SkeletonCorrectionBackend", FakeBackend)

    exit_code = grade_students.main(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--handedness",
            "right",
        ]
    )

    assert exit_code == 0
    result = pd.read_csv(output_dir / "grading_results.csv")
    assert result.loc[0, "total_grade"] == 44.5
    assert result.loc[0, "scorer"] == "skeleton-correction"
    assert result.loc[0, "position_distance"] == 0.3


def test_grade_students_records_per_video_failure(
    tmp_path: Path, monkeypatch
) -> None:
    input_dir = tmp_path / "videos"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "bad.mov").write_bytes(b"")
    (input_dir / "good.mp4").write_bytes(b"")

    class FakeProcessor:
        def __init__(self, video_path, *args, **kwargs) -> None:
            self.video_path = str(video_path)

        def process_frames(self, handedness):
            if self.video_path.endswith("bad.mov"):
                raise RuntimeError("broken video")
            return _tracking()

    class FakeBackend:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def score(self, tracking, handedness, skill):
            return ({"total_grade": 80.0, "grading_details": []}, (0, 2, 4), {})

    monkeypatch.setattr(grade_students, "PoseDetector", lambda: object())
    monkeypatch.setattr(grade_students, "VideoProcessor", FakeProcessor)
    monkeypatch.setattr(grade_students, "SkeletonCorrectionBackend", FakeBackend)

    exit_code = grade_students.main(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--handedness",
            "left",
        ]
    )

    assert exit_code == 0
    result = pd.read_csv(output_dir / "grading_results.csv")
    assert list(result["status"]) == ["error", "success"]
    assert "broken video" in result.loc[result["filename"] == "bad.mov", "error"].iloc[0]


def test_grade_students_rejects_missing_input_directory(tmp_path: Path) -> None:
    assert (
        grade_students.main(
            [
                "--input-dir",
                str(tmp_path / "missing"),
                "--output-dir",
                str(tmp_path / "output"),
            ]
        )
        == 1
    )
