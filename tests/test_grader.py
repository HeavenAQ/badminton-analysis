import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from badminton_analysis.models.types import (
    COCOKeypoints,
    CoordinateDict,
    GradingOutcome,
    Handedness,
    Skill,
)
from badminton_analysis.services.graders.base import Grader
from badminton_analysis.services.graders.registry import GraderRegistry
from badminton_analysis.services.graders.serve import ServeGrader


def _mock_stats_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": [
                "Right Shoulder Angle",
                "Left Shoulder Angle",
                "Right Crotch Angle",
                "Left Crotch Angle",
                "Right Elbow Angle",
                "Nose Right Shoulder Elbow Angle",
                "Left Elbow Angle",
                "Nose Left Shoulder Elbow Angle",
                f"{COCOKeypoints.LEFT_HIP}_x",
                f"{COCOKeypoints.LEFT_HIP}_y",
                f"{COCOKeypoints.RIGHT_HIP}_x",
                f"{COCOKeypoints.RIGHT_HIP}_y",
            ],
            0: [0.0] * 12,
            1: [0.0] * 12,
            2: [0.0] * 12,
            3: [0.0] * 12,
            4: [0.0] * 12,
        }
    )


class TestGraderRegistry:
    def test_register_and_get_grader(self):
        class TestGrader(Grader):
            mean = MagicMock()
            std = MagicMock()

            def grade(self, angles, landmark_list) -> GradingOutcome:
                return {"total_grade": 100, "grading_details": []}

        GraderRegistry.register(Skill.SERVE, Handedness.RIGHT, TestGrader)
        grader = GraderRegistry.get(Skill.SERVE, Handedness.RIGHT)

        assert isinstance(grader, TestGrader)

    def test_get_unregistered_grader_raises_error(self):
        with patch.dict(GraderRegistry._registry, {}, clear=True):
            with pytest.raises(ValueError, match="No grader registered"):
                GraderRegistry.get(Skill.CLEAR, Handedness.LEFT)


class TestServeGrader:
    def setup_method(self):
        read_csv = patch(
            "badminton_analysis.services.graders.serve.pd.read_csv",
            return_value=_mock_stats_frame(),
        )
        self.addCleanup = getattr(self, "addCleanup", None)
        self.read_csv_patcher = read_csv
        self.mock_read_csv = read_csv.start()
        self.grader = ServeGrader(Handedness.RIGHT)

    def teardown_method(self):
        self.read_csv_patcher.stop()

    def test_grade_checkpoint_1_arms_with_empty_angles(self):
        result = self.grader.grade_checkpoint_1_arms([], 0)
        assert result == 0

    def test_grade_checkpoint_1_legs_with_valid_angles(self):
        angles = [{"Right Crotch Angle": 80, "Left Crotch Angle": 90}]
        result = self.grader.grade_checkpoint_1_legs(angles, 0)
        assert result == 10

    def test_grade_checkpoint_1_legs_with_invalid_angles(self):
        angles = [{"Right Crotch Angle": 100, "Left Crotch Angle": 90}]
        result = self.grader.grade_checkpoint_1_legs(angles, 0)
        assert result == 0

    @patch.object(ServeGrader, "angle_grader", return_value=15)
    def test_grade_checkpoint_4_calls_angle_grader(self, mock_angle_grader):
        angles = [{"Right Elbow Angle": 120}]

        result = self.grader.grade_checkpoint_4(angles, 0)

        mock_angle_grader.assert_called_once_with(20, "Right Elbow Angle", 0, angles)
        assert result == 15

    def test_grade_returns_grading_outcome(self):
        angles = [
            {
                "Right Shoulder Angle": 30,
                "Left Shoulder Angle": 30,
                "Right Crotch Angle": 80,
                "Left Crotch Angle": 90,
                "Right Elbow Angle": 120,
                "Nose Right Shoulder Elbow Angle": 45,
            }
            for _ in range(5)
        ]
        landmark_list: list[CoordinateDict] = [
            {
                COCOKeypoints.LEFT_HIP: [0.0, 0.0],
                COCOKeypoints.RIGHT_HIP: [1.0, 0.0],
                COCOKeypoints.RIGHT_EYE: [0.0, 0.0],
            }
            for _ in range(5)
        ]

        with patch.object(ServeGrader, "grade_checkpoint_2_upper_body", return_value=20), patch.object(
            ServeGrader, "grade_checkpoint_3", return_value=20
        ), patch.object(ServeGrader, "angle_grader", return_value=10), patch.object(
            ServeGrader, "disp_grader", return_value=10
        ):
            result = self.grader.grade(angles, landmark_list)

        assert isinstance(result, dict)
        assert "total_grade" in result
        assert "grading_details" in result
        assert isinstance(result["grading_details"], list)


class TestServeGraderProperties:
    @pytest.mark.parametrize(
        ("handedness", "expected"),
        [
            (
                Handedness.RIGHT,
                {
                    "dominant_shoulder_key": "Right Shoulder Angle",
                    "non_dominant_shoulder_key": "Left Shoulder Angle",
                    "dominant_crotch_key": "Right Crotch Angle",
                    "non_dominant_crotch_key": "Left Crotch Angle",
                    "dominant_elbow_key": "Right Elbow Angle",
                    "dominant_shoulder_elbow_key": "Nose Right Shoulder Elbow Angle",
                },
            ),
            (
                Handedness.LEFT,
                {
                    "dominant_shoulder_key": "Left Shoulder Angle",
                    "non_dominant_shoulder_key": "Right Shoulder Angle",
                    "dominant_crotch_key": "Left Crotch Angle",
                    "non_dominant_crotch_key": "Right Crotch Angle",
                    "dominant_elbow_key": "Left Elbow Angle",
                    "dominant_shoulder_elbow_key": "Nose Left Shoulder Elbow Angle",
                },
            ),
        ],
    )
    def test_handed_key_properties(self, handedness, expected):
        with patch(
            "badminton_analysis.services.graders.serve.pd.read_csv",
            return_value=_mock_stats_frame(),
        ):
            grader = ServeGrader(handedness)

        assert grader.dominant_shoulder_key == expected["dominant_shoulder_key"]
        assert (
            grader.non_dominant_shoulder_key
            == expected["non_dominant_shoulder_key"]
        )
        assert grader.dominant_crotch_key == expected["dominant_crotch_key"]
        assert grader.non_dominant_crotch_key == expected["non_dominant_crotch_key"]
        assert grader.dominant_elbow_key == expected["dominant_elbow_key"]
        assert (
            grader.dominant_shoulder_elbow_key
            == expected["dominant_shoulder_elbow_key"]
        )
