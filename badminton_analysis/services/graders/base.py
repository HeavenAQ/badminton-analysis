import math
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

        # find the project root
        cur_file_path = Path(__file__).resolve()
        project_root = None
        for p in [cur_file_path] + list(cur_file_path.parents):
            if (p / "pyproject.toml").exists():
                project_root = p
        else:
            if project_root is None:
                raise RuntimeError("The project root was not found")

        self.data_dir = project_root / "stats"

    def angle_grader(
        self,
        max_grade: float,
        joint_name: str,
        check_idx: int,
        angles: AngleDicts,
    ) -> float:
        # convert handedness info to dominant or non-dominant
        df_joint_name = ""
        if "Right" in joint_name or "Left" in joint_name:
            dominant, non_dominant = "Right", "Left"
            if self.handedness == Handedness.LEFT:
                dominant, non_dominant = "Left", "Right"

            df_joint_name = joint_name.replace(dominant, "Dominant")
            df_joint_name = df_joint_name.replace(non_dominant, "Non-dominant")

        # expert stats
        idx = df_joint_name, check_idx
        mean = self.mean.loc[idx]
        std = self.std.loc[idx]

        tolerance = 1.0
        min_angle = mean - tolerance * std
        max_angle = mean + tolerance * std

        # get current angle
        current_angle = angles[check_idx][joint_name]

        if min_angle <= current_angle <= max_angle:
            return max_grade
        else:
            if min_angle > current_angle:
                return float(max_grade * (current_angle / min_angle)) if min_angle > 1e-6 else 0.0
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
        return (
            COCOKeypoints.RIGHT_SHOULDER
            if self.handedness == Handedness.RIGHT
            else COCOKeypoints.LEFT_SHOULDER
        )

    @property
    def non_dominant_shoulder_keypoint(self) -> COCOKeypoints:
        return (
            COCOKeypoints.LEFT_SHOULDER
            if self.handedness == Handedness.RIGHT
            else COCOKeypoints.RIGHT_SHOULDER
        )

    @property
    def dominant_hip_keypoint(self) -> COCOKeypoints:
        return (
            COCOKeypoints.RIGHT_HIP
            if self.handedness == Handedness.RIGHT
            else COCOKeypoints.LEFT_HIP
        )

    @property
    def non_dominant_hip_keypoint(self) -> COCOKeypoints:
        return (
            COCOKeypoints.LEFT_HIP
            if self.handedness == Handedness.RIGHT
            else COCOKeypoints.RIGHT_HIP
        )

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

    @staticmethod
    def _trunk_slope_delta(landmark_list: list[CoordinateDict], frame_a: int, frame_b: int) -> float:
        """Forward body lean: absolute change in trunk slope (shoulder_mid horizontal offset / trunk height).
        Scale-invariant. A value of 0.08 is approx 4.6 deg of forward tilt."""
        def trunk_slope(frame: CoordinateDict) -> float:
            hip_x = (frame[COCOKeypoints.LEFT_HIP][0] + frame[COCOKeypoints.RIGHT_HIP][0]) / 2
            hip_y = (frame[COCOKeypoints.LEFT_HIP][1] + frame[COCOKeypoints.RIGHT_HIP][1]) / 2
            sho_x = (frame[COCOKeypoints.LEFT_SHOULDER][0] + frame[COCOKeypoints.RIGHT_SHOULDER][0]) / 2
            sho_y = (frame[COCOKeypoints.LEFT_SHOULDER][1] + frame[COCOKeypoints.RIGHT_SHOULDER][1]) / 2
            trunk_h = abs(hip_y - sho_y)
            return float((sho_x - hip_x) / trunk_h) if trunk_h > 1e-6 else 0.0
        return abs(trunk_slope(landmark_list[frame_b]) - trunk_slope(landmark_list[frame_a]))

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
