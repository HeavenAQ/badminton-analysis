import pytest
from unittest.mock import patch, MagicMock
from Grader import (
    Grader,
    GraderRegistry,
    ServeGrader,
    AngleDict,
    serve_angle_grader,
)
from Types import Skill, Handedness, GraderResult


class TestServeAngleGrader:
    @patch("Grader.serve_mean")
    @patch("Grader.serve_std")
    def test_serve_angle_grader_within_range(self, mock_std, mock_mean):
        mock_mean_loc = MagicMock()
        mock_mean_loc.__getitem__.return_value = 80  # Expected mean value
        mock_mean.loc = mock_mean_loc

        mock_std_loc = MagicMock()
        mock_std_loc.__getitem__.return_value = 10  # Expected std value
        mock_std.loc = mock_std_loc

        angle_dict: AngleDict = {"Right Shoulder": 85}
        result = serve_angle_grader(10, "Right Shoulder", "check1", angle_dict)

        assert result == 10

    @patch("Grader.serve_mean")
    @patch("Grader.serve_std")
    def test_serve_angle_grader_below_range(self, mock_std, mock_mean):
        mock_mean_loc = MagicMock()
        mock_mean_loc.__getitem__.return_value = 90  # Expected mean value
        mock_mean.loc = mock_mean_loc

        mock_std_loc = MagicMock()
        mock_std_loc.__getitem__.return_value = 10  # Expected std value
        mock_std.loc = mock_std_loc

        angle_dict: AngleDict = {"Right Shoulder": 70}
        result = serve_angle_grader(10, "Right Shoulder", "check1", angle_dict)

        assert result < 10


class TestGraderRegistry:
    def test_register_and_get_grader(self):
        class TestGrader(Grader):
            def grade(self, angles) -> GraderResult:
                return {"total_grade": 100, "grading_details": []}

        GraderRegistry.register(Skill.SERVE, Handedness.RIGHT, TestGrader)
        grader = GraderRegistry.get(Skill.SERVE, Handedness.RIGHT)

        assert isinstance(grader, TestGrader)

    def test_get_unregistered_grader_raises_error(self):
        with pytest.raises(ValueError, match="No grader registered"):
            GraderRegistry.get(Skill.CLEAR, Handedness.LEFT)


class TestServeRightHandedGrader:
    def setup_method(self):
        self.grader = ServeGrader(Handedness.RIGHT)

    def test_grade_checkpoint_1_arms_with_none_angle(self):
        result = self.grader.grade_checkpoint_1_arms(None)
        assert result == 0

    def test_grade_checkpoint_1_legs_with_valid_angles(self):
        angle_dict: AngleDict = {"Right Crotch": 80, "Left Crotch": 90}
        result = self.grader.grade_checkpoint_1_legs(angle_dict)
        assert result == 10

    def test_grade_checkpoint_1_legs_with_invalid_angles(self):
        angle_dict: AngleDict = {"Right Crotch": 100, "Left Crotch": 90}
        result = self.grader.grade_checkpoint_1_legs(angle_dict)
        assert result == 0

    def test_grade_checkpoint_2_with_valid_transfer(self):
        angle_dict1: AngleDict = {"Right Crotch": 80, "Left Crotch": 100}
        angle_dict2: AngleDict = {"Right Crotch": 90, "Left Crotch": 90}
        result = self.grader.grade_checkpoint_2(angle_dict1, angle_dict2)
        assert result == 20

    def test_grade_checkpoint_3_with_valid_rotation(self):
        angle_dict: AngleDict = {"Right Crotch": 100, "Left Crotch": 90}
        result = self.grader.grade_checkpoint_3(angle_dict)
        assert result == 20

    @patch("Grader.serve_angle_grader")
    def test_grade_checkpoint_4_calls_serve_angle_grader(self, mock_grader):
        mock_grader.return_value = 15
        angle_dict: AngleDict = {"Right Elbow": 120}

        result = self.grader.grade_checkpoint_4(angle_dict)

        mock_grader.assert_called_once_with(20, "Right Elbow", "check4", angle_dict)
        assert result == 15

    @patch("Grader.serve_angle_grader")
    def test_grade_returns_grading_outcome(self, mock_grader):
        mock_grader.return_value = 10

        angles: list[AngleDict] = [
            {
                "Right Shoulder": 90,
                "Left Shoulder": 90,
                "Right Crotch": 80,
                "Left Crotch": 90,
            },
            {"Right Crotch": 90, "Left Crotch": 80},
            {"Right Crotch": 100, "Left Crotch": 90},
            {"Right Elbow": 120},
            {"Right Shoulder": 90, "Nose Right Shoulder Elbow": 45},
        ]

        result = self.grader.grade(angles)

        assert isinstance(result, dict)
        assert "total_grade" in result
        assert "grading_details" in result
        assert isinstance(result["grading_details"], list)


class TestServeLeftHandedGrader:
    def setup_method(self):
        self.grader = ServeGrader(Handedness.LEFT)

    def test_grade_returns_empty_outcome(self):
        angles: list[AngleDict] = [{}]

        result = self.grader.grade(angles)

        assert result["total_grade"] == 0
        assert result["grading_details"] == []

    def test_dominant_shoulder_property(self):
        assert self.grader.dominant_shoulder == "Left Shoulder"

    def test_non_dominant_shoulder_property(self):
        assert self.grader.non_dominant_shoulder == "Right Shoulder"

    def test_dominant_crotch_property(self):
        assert self.grader.dominant_crotch == "Left Crotch"

    def test_non_dominant_crotch_property(self):
        assert self.grader.non_dominant_crotch == "Right Crotch"

    def test_dominant_elbow_property(self):
        assert self.grader.dominant_elbow == "Left Elbow"

    def test_dominant_shoulder_elbow_property(self):
        assert self.grader.dominant_shoulder_elbow == "Nose Left Shoulder Elbow"


class TestServeRightHandedGraderProperties:
    def setup_method(self):
        self.grader = ServeGrader(Handedness.RIGHT)

    def test_dominant_shoulder_property(self):
        assert self.grader.dominant_shoulder == "Right Shoulder"

    def test_non_dominant_shoulder_property(self):
        assert self.grader.non_dominant_shoulder == "Left Shoulder"

    def test_dominant_crotch_property(self):
        assert self.grader.dominant_crotch == "Right Crotch"

    def test_non_dominant_crotch_property(self):
        assert self.grader.non_dominant_crotch == "Left Crotch"

    def test_dominant_elbow_property(self):
        assert self.grader.dominant_elbow == "Right Elbow"

    def test_dominant_shoulder_elbow_property(self):
        assert self.grader.dominant_shoulder_elbow == "Nose Right Shoulder Elbow"
