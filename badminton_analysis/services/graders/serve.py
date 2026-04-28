from typing import Any, override
import pandas as pd
import numpy as np
from pathlib import Path
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
        self.mean = pd.read_csv(self.data_dir / "mean.csv").set_index("feature")
        self.std = pd.read_csv(self.data_dir / "std.csv").set_index("feature")
        self.mean.columns = [0, 1, 2, 3, 4]
        self.std.columns = [0, 1, 2, 3, 4]

    def grade_checkpoint_1_arms(self, angles: AngleDicts, frame_idx: int) -> float:
        """
        The preparation phase of the serve. Full score for this checkpoint: 10
        """
        if not angles:
            return 0
        grade = 0.0
        if angles[frame_idx][self.dominant_shoulder_key] >= 25:
            grade += self.angle_grader(5, self.dominant_shoulder_key, frame_idx, angles)
        if angles[frame_idx][self.non_dominant_shoulder_key] >= 25:
            grade += self.angle_grader(
                5, self.non_dominant_shoulder_key, frame_idx, angles
            )
        return float(grade)

    def grade_checkpoint_1_legs(self, angles: AngleDicts, frame_idx: int) -> float:
        """
        The preparation phase of the serve. Full score for this checkpoint: 10
        """
        if not angles:
            return 0
        if (
            angles[frame_idx][self.dominant_crotch_key]
            <= angles[frame_idx][self.non_dominant_crotch_key]
        ):
            return 10
        return 0

    def grade_checkpoint_2_lower_body(
        self,
        angles: AngleDicts,
        start_frame: int,
        end_frame: int,
    ) -> float:
        """
        Lower Body weight transfer. Full score for this checkpoint: 20
        """
        if not angles:
            return 0

        dom_crotch_diff = (
            angles[end_frame][self.dominant_crotch_key]
            - angles[start_frame][self.dominant_crotch_key]
        )
        non_dom_crotch_diff = (
            angles[end_frame][self.non_dominant_crotch_key]
            - angles[start_frame][self.non_dominant_crotch_key]
        )

        grade = 0.0
        grade += self.disp_grader(
            10,
            dom_crotch_diff,
            (self.dominant_crotch_key, start_frame),
            (self.dominant_crotch_key, end_frame),
        )
        grade += self.disp_grader(
            10,
            non_dom_crotch_diff,
            (self.non_dominant_crotch_key, start_frame),
            (self.non_dominant_crotch_key, end_frame),
        )

        return float(grade)

    def grade_checkpoint_2_upper_body(
        self, landmark_list: list[CoordinateDict], start_frame: int, end_frame: int
    ) -> float:
        """
        Upper Body weight transfer. Full score for this checkpoint: 20
        """
        # extract the coordinates needed for analysis
        part = (
            COCOKeypoints.RIGHT_EYE
            if self.handedness == Handedness.RIGHT
            else COCOKeypoints.LEFT_EYE
        )
        try:
            start_eye = landmark_list[start_frame][part][0]
            end_eye = landmark_list[end_frame][part][0]
        except KeyError:
            try:
                part = (
                    COCOKeypoints.RIGHT_EAR
                    if self.handedness == Handedness.RIGHT
                    else COCOKeypoints.LEFT_EAR
                )
                start_eye = landmark_list[start_frame][part][0]
                end_eye = landmark_list[end_frame][part][0]
            except KeyError:
                part = (
                    COCOKeypoints.LEFT_EYE
                    if self.handedness == Handedness.RIGHT
                    else COCOKeypoints.RIGHT_EYE
                )
                start_eye = landmark_list[start_frame][part][0]
                end_eye = landmark_list[end_frame][part][0]

        # calculate the displacement between coordinates
        learner_disp = end_eye - start_eye
        return self.disp_grader(
            20,
            learner_disp,
            (f"{part}_x", start_frame),
            (f"{part}_x", end_frame),
        )

    def _hip_angle(self, frame: CoordinateDict) -> float:
        lp = COCOKeypoints.LEFT_HIP
        rp = COCOKeypoints.RIGHT_HIP

        lx, ly = frame[lp]
        rx, ry = frame[rp]

        return float(np.degrees(np.arctan2(ry - ly, rx - lx)))

    def grade_checkpoint_3(
        self,
        landmark_list: list[CoordinateDict],
        start_frame: int,
        end_frame: int,
    ) -> float:
        """
        Bottom rotation using hip-axis angle: Full 20
        """

        # Learner rotation
        start_angle = self._hip_angle(landmark_list[start_frame])
        end_angle = self._hip_angle(landmark_list[end_frame])

        learner_rot = end_angle - start_angle
        learner_rot = (learner_rot + 180) % 360 - 180  # normalize to [-180, 180]

        # Expert mean rotation
        exp_start_lp = self.mean.loc[f"{COCOKeypoints.LEFT_HIP}_x", start_frame]
        exp_start_lp_y = self.mean.loc[f"{COCOKeypoints.LEFT_HIP}_y", start_frame]
        exp_start_rp = self.mean.loc[f"{COCOKeypoints.RIGHT_HIP}_x", start_frame]
        exp_start_rp_y = self.mean.loc[f"{COCOKeypoints.RIGHT_HIP}_y", start_frame]

        exp_end_lp = self.mean.loc[f"{COCOKeypoints.LEFT_HIP}_x", end_frame]
        exp_end_lp_y = self.mean.loc[f"{COCOKeypoints.LEFT_HIP}_y", end_frame]
        exp_end_rp = self.mean.loc[f"{COCOKeypoints.RIGHT_HIP}_x", end_frame]
        exp_end_rp_y = self.mean.loc[f"{COCOKeypoints.RIGHT_HIP}_y", end_frame]

        exp_start_angle = np.degrees(
            np.arctan2(exp_start_rp_y - exp_start_lp_y, exp_start_rp - exp_start_lp)
        )
        exp_end_angle = np.degrees(
            np.arctan2(exp_end_rp_y - exp_end_lp_y, exp_end_rp - exp_end_lp)
        )

        expert_mean_rot = exp_end_angle - exp_start_angle
        expert_mean_rot = (expert_mean_rot + 180) % 360 - 180

        # Expert std (angle domain)
        stds = [
            self.std.loc[f"{COCOKeypoints.LEFT_HIP}_x", start_frame],
            self.std.loc[f"{COCOKeypoints.LEFT_HIP}_y", start_frame],
            self.std.loc[f"{COCOKeypoints.RIGHT_HIP}_x", start_frame],
            self.std.loc[f"{COCOKeypoints.RIGHT_HIP}_y", start_frame],
            self.std.loc[f"{COCOKeypoints.LEFT_HIP}_x", end_frame],
            self.std.loc[f"{COCOKeypoints.LEFT_HIP}_y", end_frame],
            self.std.loc[f"{COCOKeypoints.RIGHT_HIP}_x", end_frame],
            self.std.loc[f"{COCOKeypoints.RIGHT_HIP}_y", end_frame],
        ]

        # Soft pooled std (much smaller than full propagation)
        expert_std_rot = np.sqrt(sum(s**2 for s in stds) / len(stds))

        if expert_std_rot < 1e-6:
            return 0.0

        z = (learner_rot - expert_mean_rot) / expert_std_rot

        max_grade = 20
        if z >= -0.1:
            return max_grade

        return 0

    def grade_checkpoint_4(self, angles: AngleDicts, frame_idx: int) -> float:
        """
        Wrist flick. Full score for this checkpoint: 20
        """
        return self.angle_grader(20, self.dominant_elbow_key, frame_idx, angles)

    def grade_checkpoint_5(self, angles: AngleDicts, frame_idx: int) -> float:
        """
        Shoulder rotation. Full score for this checkpoint: 20
        """
        grade = 0.0
        grade += self.angle_grader(10, self.dominant_shoulder_key, frame_idx, angles)
        grade += self.angle_grader(
            10, self.dominant_shoulder_elbow_key, frame_idx, angles
        )
        return float(grade)

    @override
    def grade(
        self, angles: AngleDicts, landmark_list: list[CoordinateDict]
    ) -> GradingOutcome:
        # full score for this: 100
        check1_arms = self.grade_checkpoint_1_arms(angles, 0)
        check1_legs = self.grade_checkpoint_1_legs(angles, 0)

        check2_lower = self.grade_checkpoint_2_lower_body(
            angles,
            0,
            1,
        )
        check2_upper = self.grade_checkpoint_2_upper_body(
            landmark_list,
            1,
            3,
        )
        # if one fails, body weight transfer fails
        check2 = min(check2_lower, check2_upper)

        check3 = self.grade_checkpoint_3(
            landmark_list,
            0,
            4,
        )
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
