import math
from typing import override

import numpy as np
import pandas as pd

from .base import Grader
from badminton_analysis.models.types import (
    AngleDicts,
    COCOKeypoints,
    CoordinateDict,
    GradingDetail,
    GradingOutcome,
    Handedness,
)


class ServeGrader(Grader):
    def __init__(self, handedness: Handedness):
        super().__init__(handedness)
        self.data_dir = self.data_dir / "serve"
        self.mean = pd.read_csv(self.data_dir / "mean.csv", index_col=0).set_index("feature")
        self.std = pd.read_csv(self.data_dir / "std.csv", index_col=0).set_index("feature")
        self.mean.columns = [0, 1, 2, 3, 4]
        self.std.columns = [0, 1, 2, 3, 4]

        self.hip_rotation_mean: float | None = None
        self.hip_rotation_std: float | None = None
        rot_path = self.data_dir / "rotation_stats.csv"
        if rot_path.exists():
            rot = pd.read_csv(rot_path).set_index("feature")
            try:
                self.hip_rotation_mean = float(rot.loc["Hip Line Rotation CP3", "mean"])
                self.hip_rotation_std = float(rot.loc["Hip Line Rotation CP3", "std"])
            except KeyError:
                pass

    def _rotation_grader(self, max_grade: float, rotation_deg: float, mean: float, std: float) -> float:
        """One-tailed: full credit when rotation >= expert mean, Gaussian decay below."""
        z = (rotation_deg - mean) / std if std > 1e-6 else 0.0
        if z >= 0:
            return float(max_grade)
        return float(max_grade * np.exp(-0.5 * (z / 0.8) ** 2))

    def grade_checkpoint_1_arms(self, angles: AngleDicts, frame_idx: int) -> float:
        """Both arms raised, preparation. Full score: 10"""
        if not angles:
            return 0
        grade = 0.0
        if angles[frame_idx][self.dominant_shoulder_key] >= 25:
            grade += self.angle_grader(5, self.dominant_shoulder_key, frame_idx, angles)
        if angles[frame_idx][self.non_dominant_shoulder_key] >= 25:
            grade += self.angle_grader(5, self.non_dominant_shoulder_key, frame_idx, angles)
        return float(grade)

    def grade_checkpoint_1_legs(self, angles: AngleDicts, frame_idx: int) -> float:
        """Weight on racket foot. Full score: 10"""
        if not angles:
            return 0
        if (
            angles[frame_idx][self.dominant_crotch_key]
            <= angles[frame_idx][self.non_dominant_crotch_key]
        ):
            return 10
        return 0

    def grade_checkpoint_2_lower_body(
        self, angles: AngleDicts, frame_idx: int
    ) -> float:
        """
        Lower body in weight-transfer position at frame 1. Full score: 20
        Both crotch angles are checked against expert stats — camera-invariant
        because crotch angle is a joint angle, not a pixel coordinate.
        """
        if not angles:
            return 0
        grade = self.angle_grader(10, self.dominant_crotch_key, frame_idx, angles)
        grade += self.angle_grader(10, self.non_dominant_crotch_key, frame_idx, angles)
        return float(grade)

    def grade_checkpoint_2_upper_body(
        self, angles: AngleDicts, frame_idx: int
    ) -> float:
        """
        Upper body continues weight transfer at frame 3. Full score: 20
        Crotch angles at frame 3 confirm the transfer has completed and
        the legs are in the correct driving/receiving positions.
        """
        if not angles:
            return 0
        grade = self.angle_grader(10, self.dominant_crotch_key, frame_idx, angles)
        grade += self.angle_grader(10, self.non_dominant_crotch_key, frame_idx, angles)
        return float(grade)

    def grade_checkpoint_3(
        self,
        landmark_list: list[CoordinateDict],
        start_frame: int,
        end_frame: int,
    ) -> float:
        """
        Hip-girdle line rotation over the full swing window (frames 0→4). Full score: 20
        Uses atan2 of the RIGHT→LEFT hip vector — scale-invariant and
        camera-direction-invariant.
        """
        def hip_line_angle(frame: CoordinateDict) -> float:
            dy = float(frame[COCOKeypoints.LEFT_HIP][1] - frame[COCOKeypoints.RIGHT_HIP][1])
            dx = float(frame[COCOKeypoints.LEFT_HIP][0] - frame[COCOKeypoints.RIGHT_HIP][0])
            return math.degrees(math.atan2(dy, dx))

        start_angle = hip_line_angle(landmark_list[start_frame])
        end_angle = hip_line_angle(landmark_list[end_frame])
        rotation_deg = abs((end_angle - start_angle + 180) % 360 - 180)

        hip_mean = self.hip_rotation_mean
        hip_std = self.hip_rotation_std
        if hip_mean is not None and hip_std is not None:
            return self._rotation_grader(20, rotation_deg, hip_mean, hip_std)
        return 20.0 if rotation_deg > 10 else 0.0

    def grade_checkpoint_4(self, angles: AngleDicts, frame_idx: int) -> float:
        """Wrist flick. Full score: 20"""
        return self.angle_grader(20, self.dominant_elbow_key, frame_idx, angles)

    def grade_checkpoint_5(self, angles: AngleDicts, frame_idx: int) -> float:
        """Shoulder rotation. Full score: 20"""
        grade = self.angle_grader(10, self.dominant_shoulder_key, frame_idx, angles)
        grade += self.angle_grader(10, self.dominant_shoulder_elbow_key, frame_idx, angles)
        return float(grade)

    @override
    def grade(
        self, angles: AngleDicts, landmark_list: list[CoordinateDict]
    ) -> GradingOutcome:
        check1_arms = self.grade_checkpoint_1_arms(angles, 1)
        check1_legs = self.grade_checkpoint_1_legs(angles, 1)

        check2_lower = self.grade_checkpoint_2_lower_body(angles, 1)
        check2_upper = self.grade_checkpoint_2_upper_body(angles, 3)
        # Both lower and upper body must participate in the weight transfer
        check2 = min(check2_lower, check2_upper)

        check3 = self.grade_checkpoint_3(landmark_list, 1, 4)
        check4 = self.grade_checkpoint_4(angles, 3)
        check5 = self.grade_checkpoint_5(angles, 4)

        total = check1_arms + check1_legs + check2 + check3 + check4 + check5
        print(f"Total grade: {total}")
        grading_details: list[GradingDetail] = [
            GradingDetail(description="雙手平舉", grade=check1_arms),
            GradingDetail(description="將重心放至持拍腳", grade=check1_legs),
            GradingDetail(description="重心轉移至非持拍腳", grade=check2),
            GradingDetail(description="髖關節前旋", grade=check3),
            GradingDetail(description="持拍手手腕發力", grade=check4),
            GradingDetail(description="肩膀旋轉朝前", grade=check5),
        ]

        return GradingOutcome(
            grading_details=grading_details,
            total_grade=total,
        )
