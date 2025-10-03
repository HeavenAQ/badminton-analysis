from enum import IntEnum
from typing import Any, Dict, List, Tuple, TypedDict, TypeAlias

import numpy as np
from numpy import floating
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
    SERVE = 0
    CLEAR = 1
    FOOTWORK = 2

    @classmethod
    def convert_to_enum(cls, skill: str) -> "Skill":
        return Skill[skill.upper()]

    def __str__(self) -> str:
        return self.name.lower()


class Handedness(IntEnum):
    RIGHT = 0
    LEFT = 1

    @classmethod
    def convert_to_enum(cls, handedness: str) -> "Handedness":
        return Handedness[handedness.upper()]

    def __str__(self) -> str:
        return self.name.lower()


# Body Coordinate System
class BodyCoordinateSystem(TypedDict):
    origin: NDArray[np.float64]
    x_axis: NDArray[np.float64]
    y_axis: NDArray[np.float64]


# body coordinates and angles (standardized to NumPy arrays)
# Note: Shapes are documented but not enforced at type level.
Coordinate: TypeAlias = NDArray[np.float64]  # shape (2,)
Coordinates: TypeAlias = NDArray[np.float64]  # shape (N, 2)
CoordinateDict = Dict[COCOKeypoints, Coordinate]
CoordinatesDict = Dict[COCOKeypoints, Coordinates]
AngleDict = Dict[str, float] | None
AngleDicts = List[AngleDict]


# Types reated to Graders
GraderInput = AngleDicts | CoordinatesDict


class GradingDetail(TypedDict):
    description: str
    grade: float


class GraderResult(TypedDict):
    total_grade: float
    grading_details: list[GradingDetail]


class VideoAnalysisResponse(TypedDict):
    grade: GraderResult
    used_angles_data: list[dict[str, float] | None]
    processed_video: str
