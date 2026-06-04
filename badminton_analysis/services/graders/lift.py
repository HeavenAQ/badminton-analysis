import math

import pandas as pd
from typing import final, override

from badminton_analysis.models.types import (
    AngleDicts,
    COCOKeypoints,
    CoordinateDict,
    GradingDetail,
    GradingOutcome,
    Handedness,
)
from badminton_analysis.services.graders.base import Grader

_MIN_WRIST_LIFT_DELTA = 0.15
_FULL_WRIST_LIFT_DELTA = 0.85
_MIN_FINAL_WRIST_HEIGHT = -0.15
_FULL_FINAL_WRIST_HEIGHT = 0.30
_MIN_ACTION_SHOULDER_ANGLE = 55.0
_FULL_ACTION_SHOULDER_ANGLE = 90.0
_MIN_FINAL_SHOULDER_ANGLE = 65.0
_FULL_FINAL_SHOULDER_ANGLE = 100.0
_MIN_ELBOW_EXTENSION = 130.0
_FULL_ELBOW_EXTENSION = 165.0


@final
class LiftGrader(Grader):
    def __init__(self, handedness: Handedness):
        super().__init__(handedness)
        self.data_dir = self.data_dir / "lift"
        LiftGrader.mean = pd.read_csv(self.data_dir / "mean.csv", index_col=0).set_index("feature")
        LiftGrader.std = pd.read_csv(self.data_dir / "std.csv", index_col=0).set_index("feature")
        LiftGrader.mean.columns = [0, 1, 2, 3, 4]
        LiftGrader.std.columns = [0, 1, 2, 3, 4]

    def _dominant_feature_name(self, joint_name: str) -> str:
        dominant, non_dominant = "Right", "Left"
        if self.handedness == Handedness.LEFT:
            dominant, non_dominant = "Left", "Right"

        feature_name = joint_name.replace(dominant, "Dominant")
        return feature_name.replace(non_dominant, "Non-dominant")

    def _lift_angle_grader(
        self,
        max_grade: float,
        joint_name: str,
        check_idx: int,
        angles: AngleDicts,
        *,
        falloff: float = 1.0,
    ) -> float:
        """
        Lift-specific angle scoring.

        The base ratio penalty is too forgiving for lift follow-through angles:
        a low arm can still receive meaningful credit against a high expert
        shoulder angle. Keep full credit inside the expert band, then decay by
        z-distance outside it.
        """
        feature_name = self._dominant_feature_name(joint_name)
        mean = float(self.mean.loc[(feature_name, check_idx)])
        std = float(self.std.loc[(feature_name, check_idx)])
        current_angle = float(angles[check_idx][joint_name])

        if std < 1e-6:
            return max_grade if abs(current_angle - mean) < 1e-6 else 0.0

        z = abs((current_angle - mean) / std)
        if z <= 1.0:
            return float(max_grade)

        excess_z = z - 1.0
        return float(max_grade * math.exp(-0.5 * (excess_z / falloff) ** 2))

    @staticmethod
    def _linear_grade(max_grade: float, value: float, low: float, high: float) -> float:
        if value <= low:
            return 0.0
        if value >= high:
            return float(max_grade)
        return float(max_grade * (value - low) / (high - low))

    @staticmethod
    def _one_sided_min_grade(
        max_grade: float,
        value: float,
        minimum: float,
        full_credit: float,
    ) -> float:
        return LiftGrader._linear_grade(max_grade, value, minimum, full_credit)

    def _relative_wrist_height(self, frame: CoordinateDict) -> float:
        wrist = (
            COCOKeypoints.RIGHT_WRIST
            if self.handedness == Handedness.RIGHT
            else COCOKeypoints.LEFT_WRIST
        )
        shoulder = self.dominant_shoulder_keypoint
        hip = self.dominant_hip_keypoint

        shoulder_y = float(frame[shoulder][1])
        wrist_y = float(frame[wrist][1])
        torso = abs(float(frame[hip][1]) - shoulder_y)
        if torso < 1e-6:
            return 0.0
        return (shoulder_y - wrist_y) / torso

    def grade_checkpoint_1(self, angles: AngleDicts, frame_idx: int) -> float:
        """
        Relaxed preparation. Full score: 10
        """
        grade = self.angle_grader(5, self.dominant_shoulder_key, frame_idx, angles)
        grade += self.angle_grader(5, self.non_dominant_shoulder_key, frame_idx, angles)
        return grade

    def grade_checkpoint_2(self, angles: AngleDicts, frame_idx: int) -> float:
        """
        Draw wrist back for backswing. Full score: 25
        """
        grade = self.angle_grader(12.5, self.dominant_shoulder_key, frame_idx, angles)
        grade += self.angle_grader(12.5, self.dominant_elbow_key, frame_idx, angles)
        return grade

    def grade_checkpoint_3(self, angles: AngleDicts, frame_idx: int) -> float:
        """
        Press wrist forward into the lift. Full score: 35
        """
        frame_indices = [frame_idx]
        if frame_idx + 1 < len(angles):
            frame_indices.append(frame_idx + 1)
        shoulder = max(angles[i][self.dominant_shoulder_key] for i in frame_indices)
        elbow = max(angles[i][self.dominant_elbow_key] for i in frame_indices)
        grade = self._one_sided_min_grade(
            20,
            shoulder,
            _MIN_ACTION_SHOULDER_ANGLE,
            _FULL_ACTION_SHOULDER_ANGLE,
        )
        grade += self._one_sided_min_grade(
            15,
            elbow,
            _MIN_ELBOW_EXTENSION,
            _FULL_ELBOW_EXTENSION,
        )
        return grade

    def grade_checkpoint_4(
        self,
        angles: AngleDicts,
        landmark_list: list[CoordinateDict],
        frame_idx: int,
    ) -> float:
        """
        Finish with the racket arm lifted through the shuttle. Full score: 30
        """
        shoulder = angles[frame_idx][self.dominant_shoulder_key]
        elbow = angles[frame_idx][self.dominant_elbow_key]
        grade = self._one_sided_min_grade(
            12,
            shoulder,
            _MIN_FINAL_SHOULDER_ANGLE,
            _FULL_FINAL_SHOULDER_ANGLE,
        )
        grade += self._one_sided_min_grade(
            8,
            elbow,
            _MIN_ELBOW_EXTENSION,
            _FULL_ELBOW_EXTENSION,
        )

        if len(landmark_list) > frame_idx:
            start_height = self._relative_wrist_height(landmark_list[2])
            final_height = self._relative_wrist_height(landmark_list[frame_idx])
            lift_delta = final_height - start_height
            grade += self._linear_grade(
                6,
                lift_delta,
                _MIN_WRIST_LIFT_DELTA,
                _FULL_WRIST_LIFT_DELTA,
            )
            grade += self._linear_grade(
                4,
                final_height,
                _MIN_FINAL_WRIST_HEIGHT,
                _FULL_FINAL_WRIST_HEIGHT,
            )
        return grade

    @override
    def grade(
        self, angles: AngleDicts, landmark_list: list[CoordinateDict]
    ) -> GradingOutcome:
        # full score for this: 100
        check1 = self.grade_checkpoint_1(angles, 0)
        check2 = self.grade_checkpoint_2(angles, 2)
        check3 = self.grade_checkpoint_3(angles, 3)
        check4 = self.grade_checkpoint_4(angles, landmark_list, 4)

        total = check1 + check2 + check3 + check4
        print(f"Total grade: {total}")
        grading_details: list[GradingDetail] = [
            GradingDetail(description="手腕放置腰部放鬆預備", grade=check1),
            GradingDetail(description="手腕往後引拍", grade=check2),
            GradingDetail(description="手腕往前壓並向上延伸", grade=check3),
            GradingDetail(description="手腕向上延伸完成挑球", grade=check4),
        ]

        return GradingOutcome(
            grading_details=grading_details,
            total_grade=total,
        )
