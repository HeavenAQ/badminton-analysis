from pathlib import Path
import numpy as np
from abc import ABC, abstractmethod
from pandas import DataFrame

from badminton_analysis.core.logger import Logger
from badminton_analysis.models.types import (
    AngleDicts,
    COCOKeypoints,
    CoordinateDict,
    GradingOutcome,
    Handedness,
)

EMPTY_GRADER_RESULT: GradingOutcome = {
    "grading_details": [],
    "total_grade": 0,
}


class Grader(ABC):
    data_dir: Path
    mean: DataFrame
    std: DataFrame

    def __init__(self, handedness: Handedness):
        self.handedness: Handedness = handedness
        self.logger: Logger = Logger(self.__class__.__name__)
        self.data_dir = Path(__file__).resolve().parent / "stats" / f"{handedness}"

    @classmethod
    def angle_grader(
        cls,
        max_grade: float,
        joint_name: str,
        frame_idx: int,
        angles: AngleDicts,
    ) -> float:
        # expert stats
        idx = joint_name, frame_idx
        mean = cls.mean.loc[idx]
        std = cls.std.loc[idx]

        # Calculate the min and max angle based on the mean and std
        min_angle = mean - std
        max_angle = mean + std

        # get current angle
        current_angle = angles[frame_idx][joint_name]

        if min_angle <= current_angle <= max_angle:
            return max_grade
        else:
            if min_angle > current_angle:
                return float(max_grade * (current_angle / min_angle))
            else:
                return float(max_grade * (max_angle / current_angle))

    @classmethod
    def disp_grader(
        cls,
        max_grade: int,
        learner_disp: float,
        start_index: tuple[str, int],
        end_index: tuple[str, int],
    ) -> float:
        expert_mean_disp = cls.mean.loc[end_index] - cls.mean.loc[start_index]
        expert_std_disp = np.sqrt(
            cls.std.loc[start_index] ** 2 + cls.std.loc[end_index] ** 2
        )

        z = cls.z_score(
            learner_disp,
            expert_mean_disp,
            expert_std_disp,
        )

        if z >= 0:
            return float(max_grade)
        return float(max_grade * np.exp(-0.5 * (z / 0.8) ** 2))

    # --- Handedness-aware key helpers ---
    @property
    def _dom_side(self) -> str:
        return "Right" if self.handedness == Handedness.RIGHT else "Left"

    @property
    def _non_dom_side(self) -> str:
        return "Left" if self.handedness == Handedness.RIGHT else "Right"

    @property
    def dominant_shoulder_key(self) -> str:
        return f"{self._dom_side} Shoulder Angle"

    @property
    def non_dominant_shoulder_key(self) -> str:
        return f"{self._non_dom_side} Shoulder Angle"

    @property
    def dominant_crotch_key(self) -> str:
        return f"{self._dom_side} Crotch Angle"

    @property
    def non_dominant_crotch_key(self) -> str:
        return f"{self._non_dom_side} Crotch Angle"

    @property
    def dominant_elbow_key(self) -> str:
        return f"{self._dom_side} Elbow Angle"

    @property
    def dominant_shoulder_elbow_key(self) -> str:
        return f"Nose {self._dom_side} Shoulder Elbow Angle"

    @property
    def dominant_shoulder_keypoint(self) -> COCOKeypoints:
        return COCOKeypoints.RIGHT_SHOULDER if self.handedness == Handedness.RIGHT else COCOKeypoints.LEFT_SHOULDER

    @property
    def non_dominant_shoulder_keypoint(self) -> COCOKeypoints:
        return COCOKeypoints.LEFT_SHOULDER if self.handedness == Handedness.RIGHT else COCOKeypoints.RIGHT_SHOULDER

    @property
    def dominant_foot_key(self) -> COCOKeypoints:
        if self.handedness == Handedness.RIGHT:
            return COCOKeypoints.RIGHT_ANKLE
        return COCOKeypoints.LEFT_ANKLE

    @property
    def non_dominant_foot_key(self) -> COCOKeypoints:
        if self.handedness == Handedness.RIGHT:
            return COCOKeypoints.LEFT_ANKLE
        return COCOKeypoints.RIGHT_ANKLE

    @classmethod
    def z_score(cls, value: float, mean: float, std: float) -> float:
        if std < 1e-6:
            return 0.0
        return float((value - mean) / std)

    @abstractmethod
    def grade(
        self, angles: AngleDicts, landmark_list: list[CoordinateDict]
    ) -> GradingOutcome:
        raise NotImplementedError
