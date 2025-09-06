from typing import cast
import pandas as pd
from Grader import Grader, EMPTY_GRADER_RESULT
from Logger import Logger
from Types import (
    AngleDicts,
    GraderInput,
    GradingDetail,
    GraderResult,
    Handedness,
    AngleDict,
)

# Expert data
serve_mean = pd.read_excel(
    "./stats/serve/expert angle stats.xlsx", sheet_name="mean"
).set_index("Unnamed: 0")
serve_std = pd.read_excel(
    "./stats/serve/expert angle stats.xlsx", sheet_name="std"
).set_index("Unnamed: 0")


def serve_angle_grader(
    angle_max_grade: float,
    joint_name: str,
    frame_idx: str,
    angle_dict: dict[str, float],
) -> float:
    logger = Logger("serve_angle_grader")
    logger.debug(f"Grading angle for joint: {joint_name}, frame: {frame_idx}")

    # Use joint name and frame index to get the mean and std from the expert data
    idx = joint_name, frame_idx
    mean = serve_mean.loc[idx]
    std = serve_std.loc[idx]
    logger.debug(f"Expert data - mean: {mean}, std: {std}")

    # Calculate the min and max angle based on the mean and std
    min_angle = mean - std
    max_angle = mean + std

    # get current angle
    current_angle = angle_dict[joint_name]
    logger.debug(f"Current angle: {current_angle}, range: [{min_angle}, {max_angle}]")

    if min_angle <= current_angle <= max_angle:
        logger.info(f"Angle within range, full score: {angle_max_grade}")
        return angle_max_grade
    else:
        if min_angle > current_angle:
            score = angle_max_grade * (current_angle / min_angle)
            logger.warning(f"Angle below range, reduced score: {score}")
            return score
        else:
            score = angle_max_grade * (max_angle / current_angle)
            logger.warning(f"Angle above range, reduced score: {score}")
            return score


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
        """
        The preparation phase of the serve. Full score for this checkpoint: 20
        """
        if not angle_dict:
            return 0
        grade = 0
        grade += serve_angle_grader(5, self.dominant_shoulder, "check1", angle_dict)
        grade += serve_angle_grader(5, self.non_dominant_shoulder, "check1", angle_dict)
        return grade

    def grade_checkpoint_1_legs(self, angle_dict: AngleDict) -> float:
        """
        The preparation phase of the serve. Full score for this checkpoint: 20
        """
        if not angle_dict:
            return 0
        if angle_dict[self.dominant_crotch] <= angle_dict[self.non_dominant_crotch]:
            return 10
        return 0

    def grade_checkpoint_2(self, angle_dict1: AngleDict, angle_dict2) -> float:
        """
        Body weight transfer. Full score for this checkpoint: 20
        """
        if not angle_dict1 or not angle_dict2:
            return 0
        grade = 0
        if angle_dict1[self.dominant_crotch] < angle_dict2[self.dominant_crotch]:
            grade += 10
        if (
            angle_dict1[self.non_dominant_crotch]
            > angle_dict2[self.non_dominant_crotch]
        ):
            grade += 10
        return grade

    def grade_checkpoint_3(self, angle_dict: AngleDict) -> float:
        """
        Bottom rotation. Full score for this checkpoint: 20
        """
        grade = 0
        if not angle_dict:
            return grade
        if angle_dict[self.dominant_crotch] > angle_dict[self.non_dominant_crotch]:
            grade += 20
        return grade

    def grade_checkpoint_4(self, angle_dict: AngleDict) -> float:
        """
        Wrist flick. Full score for this checkpoint: 20
        """
        grade = 0
        if not angle_dict:
            return grade
        grade += serve_angle_grader(20, self.dominant_elbow, "check4", angle_dict)
        return grade

    def grade_checkpoint_5(self, angle: AngleDict) -> float:
        """
        Shoulder rotation. Full score for this checkpoint: 20
        """
        grade = 0
        if not angle:
            return grade
        grade += serve_angle_grader(10, self.dominant_shoulder, "check5", angle)
        grade += serve_angle_grader(10, self.dominant_shoulder_elbow, "check5", angle)
        return grade

        # full score for this frame: 20

    def grade(self, grader_input: GraderInput) -> GraderResult:
        self.logger.debug("Starting serve grading")
        if not isinstance(grader_input, list) or len(grader_input) < 5:
            self.logger.error(
                f"Invalid grader input: expected list with 5 elements, got {type(grader_input)} with length {len(grader_input) if isinstance(grader_input, list) else 'N/A'}"
            )
            return EMPTY_GRADER_RESULT

        self.logger.info(f"Grading serve for {self.handedness} handed player")

        # full score for this: 100
        angle_list = cast(AngleDicts, grader_input)

        self.logger.debug("Evaluating checkpoint 1 - arms position")
        check1_arms = self.grade_checkpoint_1_arms(angle_list[0])
        self.logger.debug("Evaluating checkpoint 1 - leg position")
        check1_legs = self.grade_checkpoint_1_legs(angle_list[0])
        self.logger.debug("Evaluating checkpoint 2 - weight transfer")
        check2 = self.grade_checkpoint_2(angle_list[0], angle_list[1])
        self.logger.debug("Evaluating checkpoint 3 - hip rotation")
        check3 = self.grade_checkpoint_3(angle_list[2])
        self.logger.debug("Evaluating checkpoint 4 - wrist flick")
        check4 = self.grade_checkpoint_4(angle_list[3])
        self.logger.debug("Evaluating checkpoint 5 - shoulder rotation")
        check5 = self.grade_checkpoint_5(angle_list[4])

        total = check1_arms + check1_legs + check2 + check3 + check4 + check5
        self.logger.info(f"Serve grading completed. Total score: {total}/100")
        self.logger.debug(
            f"Individual scores - Arms: {check1_arms}, Legs: {check1_legs}, Transfer: {check2}, Hip: {check3}, Wrist: {check4}, Shoulder: {check5}"
        )

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
