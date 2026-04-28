import pandas as pd
from typing import final, override

from badminton_analysis.core.types import (
    AngleDicts,
    CoordinateDict,
    GradingDetail,
    GradingOutcome,
    Handedness,
)
from badminton_analysis.graders.base import Grader


@final
class SmashGrader(Grader):
    def __init__(self, handedness: Handedness):
        super().__init__(handedness)
        self.data_dir = self.data_dir / "smash"
        SmashGrader.mean = pd.read_csv(self.data_dir / "mean.csv").set_index("feature")
        SmashGrader.std = pd.read_csv(self.data_dir / "std.csv").set_index("feature")
        SmashGrader.mean.columns = [0, 1, 2, 3, 4, 5]
        SmashGrader.std.columns = [0, 1, 2, 3, 4, 5]

    def grade_checkpoint_1(self, angles: AngleDicts, frame_idx: int) -> float:
        """
        The preparation phase of the smash. Full score: 10
        """
        grade = self.angle_grader(5, self.dominant_shoulder_key, frame_idx, angles)
        grade += self.angle_grader(5, self.non_dominant_shoulder_key, frame_idx, angles)
        return grade

    def grade_checkpoint_2(
        self, landmark_list: list[CoordinateDict], start_idx: int, end_idx: int
    ) -> float:
        """
        Body rotation: Full score: 10
        """
        start_frame = landmark_list[start_idx]
        end_frame = landmark_list[end_idx]

        start_dist = (
            start_frame[self.non_dominant_foot_key][0]
            - start_frame[self.dominant_foot_key][0]
        )
        end_dist = (
            end_frame[self.non_dominant_foot_key][0]
            - end_frame[self.dominant_foot_key][0]
        )

        if end_dist - start_dist > 10:
            return 10
        return 0

    def grade_checkpoint_3(self, angles: AngleDicts, frame_idx: int) -> float:
        """
        Hand balancing: Full score: 20
        """
        grade = self.angle_grader(10, self.dominant_shoulder_key, frame_idx, angles)
        grade += self.angle_grader(
            10, self.non_dominant_shoulder_key, frame_idx, angles
        )
        return grade

    def grade_checkpoint_4(self, angles: AngleDicts, frame_idx: int) -> float:
        """
        Elbow facing forward: Full score: 20
        """
        return self.angle_grader(20, self.dominant_shoulder_key, frame_idx, angles)

    def grade_checkpoint_5(self, angles: AngleDicts, frame_idx: int) -> float:
        """
        Wrist Flick: Full score: 20
        """
        return self.angle_grader(20, self.dominant_elbow_key, frame_idx, angles)

    def grade_checkpoint_6(
        self, angles: AngleDicts, landmark_list: list[CoordinateDict], frame_idx: int
    ):
        """
        Ending Pose. Full score: 20
        """

        # if hand is down
        grade = self.angle_grader(10, self.dominant_shoulder_key, frame_idx, angles)

        # if body is leaning forward
        landmark = landmark_list[frame_idx]
        if landmark[self.dominant_shoulder_keypoint][0] - landmark[self.non_dominant_shoulder_keypoint][0] > 5:
            grade += 10

        return grade

    @override
    def grade(
        self, angles: AngleDicts, landmark_list: list[CoordinateDict]
    ) -> GradingOutcome:
        # full score for this: 100
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
