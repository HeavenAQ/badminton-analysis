from enum import IntEnum, auto
from typing import Literal, TypedDict, TypeAlias, override

import numpy as np
from numpy.typing import NDArray


class COCOKeypoints(IntEnum):
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


class Skill(IntEnum):
    SERVE = auto()
    CLEAR = auto()
    SMASH = auto()
    LIFT = auto()
    BACKHAND_DRIVE = auto()
    FOREHAND_DRIVE = auto()
    BACKHAND_NETKILL = auto()
    FOREHAND_NETKILL = auto()
    FOOTWORK = auto()

    @classmethod
    def convert_to_enum(cls, skill: str) -> "Skill":
        return Skill[skill.upper()]

    @override
    def __str__(self) -> str:
        return self.name.lower()


class Handedness(IntEnum):
    RIGHT = 0
    LEFT = 1

    @classmethod
    def convert_to_enum(cls, handedness: str) -> "Handedness":
        return Handedness[handedness.upper()]

    @override
    def __str__(self) -> str:
        return self.name.lower()


class BodyCoordinateSystem(TypedDict):
    origin: NDArray[np.float64]
    x_axis: NDArray[np.float64]
    y_axis: NDArray[np.float64]


Coordinate: TypeAlias = NDArray[np.float64]  # shape (2,)
Coordinates: TypeAlias = NDArray[np.float64]  # shape (N, 2)
CoordinateDict: TypeAlias = dict[COCOKeypoints, Coordinate]
CoordinatesDict: TypeAlias = dict[COCOKeypoints, Coordinates]
AngleDict: TypeAlias = dict[str, float]
AngleDicts: TypeAlias = list[AngleDict]


class GradingDetail(TypedDict):
    description: str
    grade: float


class GradingOutcome(TypedDict):
    total_grade: float
    grading_details: list[GradingDetail]


class VideoAnalysisResponse(TypedDict):
    grade: GradingOutcome
    used_angles_data: list[dict[str, float] | None]
    processed_video: str


class TrackingData(TypedDict):
    frames: list[NDArray[np.uint8]]
    original_landmarks: list[CoordinateDict]
    hand_positions: list[Coordinate]
    elbow_positions: list[Coordinate]
    time_intervals: list[float]


StepSequence: TypeAlias = list[Literal["L", "R"]]
