from pathlib import Path

import pandas as pd

from badminton_analysis.tools import grade_students


def test_grade_students_main_writes_csv(tmp_path: Path, monkeypatch) -> None:
    input_dir = tmp_path / "videos"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "a.mp4").write_bytes(b"")
    (input_dir / "b.mov").write_bytes(b"")
    (input_dir / "ignore.txt").write_text("x", encoding="utf-8")

    class FakeProcessor:
        def __init__(self, video_path: str, out_filename: str, output_folder: str) -> None:
            self.out_filename = out_filename

        def process_frames(self, handedness):
            return {
                "frames": [],
                "original_landmarks": [{} for _ in range(5)],
                "hand_positions": [],
                "elbow_positions": [],
                "time_intervals": [],
            }

    class FakePlayerGrader:
        def __init__(self) -> None:
            self.calls = 0

        def grade(self, skill, handedness, tracking, **kwargs):
            self.calls += 1
            return (
                {
                    "total_grade": 90 + self.calls,
                    "grading_details": [
                        {"description": "checkpoint", "grade": 45.0},
                    ],
                },
                (1, 2, 3),
            )

    monkeypatch.setattr(grade_students, "VideoProcessor", FakeProcessor)
    monkeypatch.setattr(grade_students, "PlayerGrader", FakePlayerGrader)

    exit_code = grade_students.main(
        [
            "--skill",
            "serve",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    result = pd.read_csv(output_dir / "grading_results.csv")
    assert list(result["filename"]) == ["a.mp4", "b.mov"]
    assert list(result["status"]) == ["success", "success"]


def test_grade_students_main_handles_per_video_failures(
    tmp_path: Path, monkeypatch
) -> None:
    input_dir = tmp_path / "videos"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "good.mp4").write_bytes(b"")
    (input_dir / "bad.mov").write_bytes(b"")

    class FakeProcessor:
        def __init__(self, video_path: str, out_filename: str, output_folder: str) -> None:
            self.out_filename = out_filename

        def process_frames(self, handedness):
            if self.out_filename == "bad.mov":
                raise RuntimeError("broken video")
            return {
                "frames": [],
                "original_landmarks": [{} for _ in range(5)],
                "hand_positions": [],
                "elbow_positions": [],
                "time_intervals": [],
            }

    class FakePlayerGrader:
        def grade(self, skill, handedness, tracking, **kwargs):
            return (
                {"total_grade": 88.0, "grading_details": []},
                (0, 1, 2),
            )

    monkeypatch.setattr(grade_students, "VideoProcessor", FakeProcessor)
    monkeypatch.setattr(grade_students, "PlayerGrader", FakePlayerGrader)

    exit_code = grade_students.main(
        [
            "--skill",
            "serve",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    result = pd.read_csv(output_dir / "grading_results.csv")
    assert list(result["status"]) == ["error", "success"]
    assert "broken video" in result.loc[result["filename"] == "bad.mov", "error"].iloc[0]


def test_grade_students_main_requires_footwork_reference(tmp_path: Path) -> None:
    input_dir = tmp_path / "videos"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "a.mp4").write_bytes(b"")

    exit_code = grade_students.main(
        [
            "--skill",
            "footwork",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 1
