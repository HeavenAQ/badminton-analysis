import json
from pathlib import Path

import numpy as np
import pytest

from badminton_analysis.models.types import COCOKeypoints, Handedness
from badminton_analysis.services.body_normalizer import BodyCentricNormalizer
from badminton_analysis.services.graders.footwork import BackCourtFootworkGrader


def _frame(
    left_ankle: tuple[float, float],
    right_ankle: tuple[float, float],
) -> dict[COCOKeypoints, np.ndarray]:
    return {
        COCOKeypoints.LEFT_SHOULDER: np.asarray([-1.0, 1.0], dtype=np.float64),
        COCOKeypoints.RIGHT_SHOULDER: np.asarray([1.0, 1.0], dtype=np.float64),
        COCOKeypoints.LEFT_HIP: np.asarray([-1.0, 0.0], dtype=np.float64),
        COCOKeypoints.RIGHT_HIP: np.asarray([1.0, 0.0], dtype=np.float64),
        COCOKeypoints.LEFT_ANKLE: np.asarray(left_ankle, dtype=np.float64),
        COCOKeypoints.RIGHT_ANKLE: np.asarray(right_ankle, dtype=np.float64),
    }


def _write_reference(tmp_path: Path, frames: list[dict[COCOKeypoints, np.ndarray]]) -> Path:
    normalizer = BodyCentricNormalizer(True)
    payload = {
        "right": {
            "left_ankle": [
                normalizer.normalize_pose(frame)[COCOKeypoints.LEFT_ANKLE].tolist()
                for frame in frames
            ],
            "right_ankle": [
                normalizer.normalize_pose(frame)[COCOKeypoints.RIGHT_ANKLE].tolist()
                for frame in frames
            ],
        }
    }
    path = tmp_path / "footwork_reference.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_footwork_grader_matches_reference(tmp_path: Path) -> None:
    frames = [
        _frame((-1.0, -1.0), (1.0, -1.0)),
        _frame((-1.1, -1.2), (1.1, -1.1)),
        _frame((-1.3, -1.5), (1.3, -1.4)),
    ]
    reference_path = _write_reference(tmp_path, frames)

    grader = BackCourtFootworkGrader(
        Handedness.RIGHT,
        reference_data_path=str(reference_path),
    )
    result = grader.grade([], frames)

    assert result["total_grade"] == pytest.approx(100.0)
    assert len(result["grading_details"]) == 2


def test_footwork_grader_penalizes_misalignment(tmp_path: Path) -> None:
    reference_frames = [
        _frame((-1.0, -1.0), (1.0, -1.0)),
        _frame((-1.1, -1.2), (1.1, -1.1)),
        _frame((-1.3, -1.5), (1.3, -1.4)),
    ]
    student_frames = [
        _frame((-0.2, -0.2), (0.2, -0.2)),
        _frame((-0.1, -0.1), (0.1, -0.1)),
        _frame((0.0, 0.0), (0.0, 0.0)),
    ]
    reference_path = _write_reference(tmp_path, reference_frames)

    grader = BackCourtFootworkGrader(
        Handedness.RIGHT,
        reference_data_path=str(reference_path),
    )
    result = grader.grade([], student_frames)

    assert result["total_grade"] < 100.0


def test_footwork_grader_requires_reference_data(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        BackCourtFootworkGrader(
            Handedness.RIGHT,
            reference_data_path=str(missing_path),
        )
