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


class DriveGrader(Grader):
    def grade_checkpoint_1(self, angles: AngleDicts, frame_idx: int) -> float:
        """
        The preparation phase of the drive. Full score: 10
        """
        grade = self.angle_grader(5, self.dominant_shoulder_key, frame_idx, angles)
        grade += self.angle_grader(5, self.non_dominant_shoulder_key, frame_idx, angles)
        return grade

    def grade_checkpoint_2(self, angles: AngleDicts, frame_idx: int) -> float:
        """
        Draw wrist back for backswing. Full score: 45
        """
        grade = self.angle_grader(22.5, self.dominant_shoulder_key, frame_idx, angles)
        grade += self.angle_grader(22.5, self.dominant_elbow_key, frame_idx, angles)
        return grade

    def grade_checkpoint_3(self, angles: AngleDicts, frame_idx: int) -> float:
        """
        Press wrist forward (snap forward). Full score: 45
        """
        grade = self.angle_grader(22.5, self.dominant_shoulder_key, frame_idx, angles)
        grade += self.angle_grader(22.5, self.dominant_elbow_key, frame_idx, angles)
        return grade

    @override
    def grade(
        self, angles: AngleDicts, landmark_list: list[CoordinateDict]
    ) -> GradingOutcome:
        # full score for this: 100
        check1 = self.grade_checkpoint_1(angles, 0)
        check2 = self.grade_checkpoint_2(angles, 2)
        check3 = self.grade_checkpoint_3(angles, 3)

        total = check1 + check2 + check3
        print(f"Total grade: {total}")
        grading_details: list[GradingDetail] = [
            GradingDetail(description="手腕放置腰部放鬆預備", grade=check1),
            GradingDetail(description="手腕往後引拍", grade=check2),
            GradingDetail(description="手腕往前壓", grade=check3),
        ]

        return GradingOutcome(
            grading_details=grading_details,
            total_grade=total,
        )


@final
class BackhandDriveGrader(DriveGrader):
    def __init__(self, handedness: Handedness):
        super().__init__(handedness)
        dir = "右中" if self.handedness == Handedness.LEFT else "左中"
        self.data_dir = self.data_dir / dir
        self.mean = pd.read_csv(self.data_dir / "mean.csv", index_col=0).set_index("feature")
        self.std = pd.read_csv(self.data_dir / "std.csv", index_col=0).set_index("feature")
        self.mean.columns = [0, 1, 2, 3, 4]
        self.std.columns = [0, 1, 2, 3, 4]


@final
class ForehandDriveGrader(DriveGrader):
    def __init__(self, handedness: Handedness):
        super().__init__(handedness)
        dir = "左中" if self.handedness == Handedness.LEFT else "右中"
        self.data_dir = self.data_dir / dir
        self.mean = pd.read_csv(self.data_dir / "mean.csv", index_col=0).set_index("feature")
        self.std = pd.read_csv(self.data_dir / "std.csv", index_col=0).set_index("feature")
        self.mean.columns = [0, 1, 2, 3, 4]
        self.std.columns = [0, 1, 2, 3, 4]
