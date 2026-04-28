from typing import Any, override
import pandas as pd
from .base import Grader, EMPTY_GRADER_RESULT
from core.logger import Logger
from core.types import (
    AngleDicts,
    GraderInput,
    GradingDetail,
    GraderResult,
    Handedness,
    AngleDict,
)

serve_mean: Any = None
serve_std: Any = None


def serve_angle_grader(
    angle_max_grade: float,
    joint_name: str,
    frame_idx: str,
    angle_dict: dict[str, float],
) -> float:
    logger = Logger("serve_angle_grader")
    logger.debug(f"Grading angle for joint: {joint_name}, frame: {frame_idx}")
    global serve_mean, serve_std
    if serve_mean is None or serve_std is None:
        serve_mean = pd.read_excel(
            "./stats/serve/expert angle stats.xlsx", sheet_name="mean"
        ).set_index("Unnamed: 0")
        serve_std = pd.read_excel(
            "./stats/serve/expert angle stats.xlsx", sheet_name="std"
        ).set_index("Unnamed: 0")
    idx = joint_name, frame_idx
    mean = float(serve_mean.loc[idx])
    std = float(serve_std.loc[idx])
    min_angle = mean - std
    max_angle = mean + std
    current_angle = angle_dict[joint_name]
    if min_angle <= current_angle <= max_angle:
        return angle_max_grade
    else:
        if min_angle > current_angle:
            return float(angle_max_grade) * (float(current_angle) / float(min_angle))
        else:
            return float(angle_max_grade) * (float(max_angle) / float(current_angle))


class ServeGrader(Grader):
    def __init__(self, handedness: Handedness):
        super().__init__(handedness)

    @property
    def dominant_shoulder(self) -> str:
        return f"{str(self.handedness).capitalize()} Shoulder"

    @property
    def non_dominant_shoulder(self) -> str:
        non_dominant = (
            Handedness.LEFT if self.handedness == Handedness.RIGHT else Handedness.RIGHT
        )
        return f"{str(non_dominant).capitalize()} Shoulder"

    @property
    def dominant_crotch(self) -> str:
        return f"{str(self.handedness).capitalize()} Crotch"

    @property
    def non_dominant_crotch(self) -> str:
        non_dominant = (
            Handedness.LEFT if self.handedness == Handedness.RIGHT else Handedness.RIGHT
        )
        return f"{str(non_dominant).capitalize()} Crotch"

    @property
    def dominant_elbow(self) -> str:
        return f"{str(self.handedness).capitalize()} Elbow"

    @property
    def dominant_shoulder_elbow(self) -> str:
        return f"Nose {str(self.handedness).capitalize()} Shoulder Elbow"

    def grade_checkpoint_1_arms(self, angle_dict: AngleDict) -> float:
        if not angle_dict:
            return 0
        grade: float = 0.0
        grade += serve_angle_grader(5, self.dominant_shoulder, "check1", angle_dict)
        grade += serve_angle_grader(5, self.non_dominant_shoulder, "check1", angle_dict)
        return grade

    def grade_checkpoint_1_legs(self, angle_dict: AngleDict) -> float:
        if not angle_dict:
            return 0
        if angle_dict[self.dominant_crotch] <= angle_dict[self.non_dominant_crotch]:
            return 10
        return 0

    def grade_checkpoint_2(
        self, angle_dict1: AngleDict, angle_dict2: AngleDict
    ) -> float:
        if not angle_dict1 or not angle_dict2:
            return 0
        grade: float = 0.0
        if angle_dict1[self.dominant_crotch] < angle_dict2[self.dominant_crotch]:
            grade += 10
        if (
            angle_dict1[self.non_dominant_crotch]
            > angle_dict2[self.non_dominant_crotch]
        ):
            grade += 10
        return grade

    def grade_checkpoint_3(self, angle_dict: AngleDict) -> float:
        grade: float = 0.0
        if not angle_dict:
            return grade
        if angle_dict[self.dominant_crotch] > angle_dict[self.non_dominant_crotch]:
            grade += 20
        return grade

    def grade_checkpoint_4(self, angle_dict: AngleDict) -> float:
        grade: float = 0.0
        if not angle_dict:
            return grade
        grade += serve_angle_grader(20, self.dominant_elbow, "check4", angle_dict)
        return grade

    def grade_checkpoint_5(self, angle: AngleDict) -> float:
        grade: float = 0.0
        if not angle:
            return grade
        grade += serve_angle_grader(10, self.dominant_shoulder, "check5", angle)
        grade += serve_angle_grader(10, self.dominant_shoulder_elbow, "check5", angle)
        return grade

    @override
    def grade(self, grader_input: GraderInput) -> GraderResult:
        if not isinstance(grader_input, list) or len(grader_input) < 5:
            return EMPTY_GRADER_RESULT
        angle_list: list[AngleDict] = grader_input
        check1_arms = self.grade_checkpoint_1_arms(angle_list[0])
        check1_legs = self.grade_checkpoint_1_legs(angle_list[0])
        check2 = self.grade_checkpoint_2(angle_list[0], angle_list[1])
        check3 = self.grade_checkpoint_3(angle_list[2])
        check4 = self.grade_checkpoint_4(angle_list[3])
        check5 = self.grade_checkpoint_5(angle_list[4])
        total = check1_arms + check1_legs + check2 + check3 + check4 + check5
        grading_details: list[GradingDetail] = [
            {"description": "雙手平舉", "grade": check1_arms},
            {"description": "將重心放至持拍腳", "grade": check1_legs},
            {"description": "身體重心轉移至非持拍腳", "grade": check2},
            {"description": "髖關節前旋", "grade": check3},
            {"description": "持拍手手腕發力", "grade": check4},
            {"description": "肩膀旋轉朝前", "grade": check5},
        ]
        return {
            "grading_details": grading_details,
            "total_grade": total,
        }


__all__ = ["ServeGrader", "serve_angle_grader", "serve_mean", "serve_std"]
