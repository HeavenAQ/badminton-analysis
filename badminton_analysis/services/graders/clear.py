import math

import numpy as np
import pandas as pd
from typing import final, override

from badminton_analysis.models.types import (
    AngleDicts,
    CoordinateDict,
    GradingDetail,
    GradingOutcome,
    Handedness,
)
from badminton_analysis.services.graders.base import Grader

# Trunk slope change threshold: abs(slope_end - slope_start) > this -> body leaned forward.
# Slope = (shoulder_mid_x - hip_mid_x) / trunk_height, so 0.08 is approx 4.6 deg of tilt.
_TRUNK_LEAN_THRESHOLD = 0.08


@final
class ClearGrader(Grader):
    mean: pd.DataFrame
    std: pd.DataFrame
    hip_rotation_mean: float | None
    hip_rotation_std: float | None
    shoulder_rotation_mean: float | None
    shoulder_rotation_std: float | None

    def __init__(self, handedness: Handedness):
        super().__init__(handedness)
        self.data_dir = self.data_dir / "clear"
        ClearGrader.mean = pd.read_csv(self.data_dir / "mean.csv", index_col=0).set_index("feature")
        ClearGrader.std = pd.read_csv(self.data_dir / "std.csv", index_col=0).set_index("feature")
        ClearGrader.mean.columns = [0, 1, 2, 3, 4]
        ClearGrader.std.columns = [0, 1, 2, 3, 4]

        rot_path = self.data_dir / "rotation_stats.csv"
        if rot_path.exists():
            rot = pd.read_csv(rot_path).set_index("feature")
            hip_mean = float(rot.loc["Hip Line Rotation CP2", "mean"])
            hip_std = float(rot.loc["Hip Line Rotation CP2", "std"])
            # A mean > 45 deg between adjacent frames is physically implausible and
            # indicates corrupted stats (e.g., YOLO left/right hip keypoint swaps).
            if hip_mean <= 45.0:
                ClearGrader.hip_rotation_mean = hip_mean
                ClearGrader.hip_rotation_std = hip_std
            else:
                ClearGrader.hip_rotation_mean = None
                ClearGrader.hip_rotation_std = None
            ClearGrader.shoulder_rotation_mean = float(rot.loc["Shoulder Line Rotation CP6", "mean"])
            ClearGrader.shoulder_rotation_std = float(rot.loc["Shoulder Line Rotation CP6", "std"])
        else:
            ClearGrader.hip_rotation_mean = None
            ClearGrader.hip_rotation_std = None
            ClearGrader.shoulder_rotation_mean = None
            ClearGrader.shoulder_rotation_std = None

    def _rotation_grader(self, max_grade: float, rotation_deg: float, mean: float, std: float) -> float:
        """One-tailed score: full credit when rotation >= expert mean, Gaussian decay below."""
        z = (rotation_deg - mean) / std if std > 1e-6 else 0.0
        if z >= 0:
            return float(max_grade)
        return float(max_grade * np.exp(-0.5 * (z / 0.8) ** 2))

    def grade_checkpoint_1(self, angles: AngleDicts, frame_idx: int) -> float:
        """Preparation phase. Full score: 10"""
        grade = self.angle_grader(5, self.dominant_shoulder_key, frame_idx, angles)
        grade += self.angle_grader(5, self.non_dominant_shoulder_key, frame_idx, angles)
        return grade

    def grade_checkpoint_2(
        self, landmark_list: list[CoordinateDict], start_idx: int, end_idx: int
    ) -> float:
        """Body rotation measured as hip-girdle line rotation (degrees). Full score: 10"""
        start_frame = landmark_list[start_idx]
        end_frame = landmark_list[end_idx]

        def hip_line_angle(frame: CoordinateDict) -> float:
            dy = float(frame[self.non_dominant_hip_keypoint][1] - frame[self.dominant_hip_keypoint][1])
            dx = float(frame[self.non_dominant_hip_keypoint][0] - frame[self.dominant_hip_keypoint][0])
            return math.degrees(math.atan2(dy, dx))

        start_angle = hip_line_angle(start_frame)
        end_angle = hip_line_angle(end_frame)
        rotation_deg = abs((end_angle - start_angle + 180) % 360 - 180)

        hip_mean = ClearGrader.hip_rotation_mean
        hip_std = ClearGrader.hip_rotation_std
        if hip_mean is not None and hip_std is not None:
            return self._rotation_grader(10, rotation_deg, hip_mean, hip_std)
        return 10.0 if rotation_deg > 10 else 0.0

    def grade_checkpoint_3(self, angles: AngleDicts, frame_idx: int) -> float:
        """Both arms raised and balanced. Full score: 20"""
        grade = self.angle_grader(10, self.dominant_shoulder_key, frame_idx, angles)
        grade += self.angle_grader(10, self.non_dominant_shoulder_key, frame_idx, angles)
        return grade

    def grade_checkpoint_4(self, angles: AngleDicts, frame_idx: int) -> float:
        """
        Elbow driving forward at contact. Full score: 20
        Arm elevation (dom shoulder, 10 pts) + arm geometry at contact (nose-dom-shoulder-elbow, 10 pts)
        """
        grade = self.angle_grader(10, self.dominant_shoulder_key, frame_idx, angles)
        grade += self.angle_grader(10, self.dominant_shoulder_elbow_key, frame_idx, angles)
        return grade

    def grade_checkpoint_5(self, angles: AngleDicts, frame_idx: int) -> float:
        """Wrist flick at contact. Full score: 20"""
        return self.angle_grader(20, self.dominant_elbow_key, frame_idx, angles)

    def grade_checkpoint_6(
        self, angles: AngleDicts, landmark_list: list[CoordinateDict], frame_idx: int
    ) -> float:
        """Follow-through. Full score: 20.
        Trunk lean gates the score: no lean -> 0. Lean detected -> grade shoulder rotation.
        """
        delta = self._trunk_slope_delta(landmark_list, 0, len(landmark_list) - 1)
        if delta <= _TRUNK_LEAN_THRESHOLD:
            return 0.0

        # Trunk lean confirmed; grade shoulder rotation (10 pts angle + 10 pts rotation).
        grade = self.angle_grader(10, self.dominant_shoulder_key, frame_idx, angles)

        def shoulder_line_angle(frame: CoordinateDict) -> float:
            dy = float(frame[self.non_dominant_shoulder_keypoint][1] - frame[self.dominant_shoulder_keypoint][1])
            dx = float(frame[self.non_dominant_shoulder_keypoint][0] - frame[self.dominant_shoulder_keypoint][0])
            return math.degrees(math.atan2(dy, dx))

        start_angle = shoulder_line_angle(landmark_list[0])
        end_angle = shoulder_line_angle(landmark_list[frame_idx])
        rotation_deg = abs((end_angle - start_angle + 180) % 360 - 180)
        shoulder_mean = ClearGrader.shoulder_rotation_mean
        shoulder_std = ClearGrader.shoulder_rotation_std
        if shoulder_mean is not None and shoulder_std is not None:
            grade += self._rotation_grader(10, rotation_deg, shoulder_mean, shoulder_std)
        elif rotation_deg > 12:
            grade += 10

        return grade

    @override
    def grade(
        self, angles: AngleDicts, landmark_list: list[CoordinateDict]
    ) -> GradingOutcome:
        check1 = self.grade_checkpoint_1(angles, 0)
        check2 = self.grade_checkpoint_2(landmark_list, 0, 1)
        check3 = self.grade_checkpoint_3(angles, 1)
        check4 = self.grade_checkpoint_4(angles, 2)
        check5 = self.grade_checkpoint_5(angles, 2)
        check6 = self.grade_checkpoint_6(angles, landmark_list, frame_idx=3)

        total = check1 + check2 + check3 + check4 + check5 + check6
        print(f"Total grade: {total}")
        grading_details: list[GradingDetail] = [
            GradingDetail(description="球拍舉至腰部預備", grade=check1),
            GradingDetail(description="轉身", grade=check2),
            GradingDetail(description="雙手手肘平衡", grade=check3),
            GradingDetail(description="手肘往前轉至前方", grade=check4),
            GradingDetail(description="手腕發力", grade=check5),
            GradingDetail(description="慣用手肩膀往前轉", grade=check6),
        ]

        return GradingOutcome(
            grading_details=grading_details,
            total_grade=total,
        )
