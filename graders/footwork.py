from dataclasses import dataclass
from collections import deque
from typing import Any, Dict, Optional, Tuple, Type, TypedDict
import numpy as np
from numpy.typing import NDArray
from .base import EMPTY_GRADER_RESULT, Grader
from core.types import (
    COCOKeypoints,
    Coordinate,
    CoordinateDict,
    CoordinatesDict,
    GraderInput,
    GraderResult,
    Handedness,
)


_EPS = 1e-10


class ToleranceDict(TypedDict):
    contact: float
    behind_margin: float
    back_move: float


@dataclass
class FootworkGrade:
    check1: float = 0
    check2: float = 0
    check3: float = 0
    check4: float = 0
    check5: float = 0
    check6: float = 0


class BackCourtFootworkGrader(Grader):
    ORIGIN_TOLERANCE_RATE = 0.35
    ORIGIN_FRAME = 5

    def __init__(self, handedness: Handedness):
        super().__init__(handedness)
        self.lead, self.trail = self.feet
        self.pelvis_hist = deque(maxlen=6)
        self.stage = 0
        self.last_trail = None
        self.last_lead = None
        self.contact_at_cp2 = None
        self.before_jump = np.array([])
        self.req = [
            COCOKeypoints.RIGHT_HIP,
            COCOKeypoints.LEFT_SHOULDER,
            COCOKeypoints.RIGHT_SHOULDER,
            COCOKeypoints.LEFT_ANKLE,
            COCOKeypoints.RIGHT_ANKLE,
        ]
        self.grades = FootworkGrade()

    @staticmethod
    def _unit(v: Coordinate) -> Coordinate:
        mag = np.linalg.norm(v)
        return v / mag if mag > _EPS else v

    @staticmethod
    def _angle_deg(u: Coordinate, v: Coordinate) -> float:
        nv = BackCourtFootworkGrader._unit(v)
        nu = BackCourtFootworkGrader._unit(u)
        return float(np.degrees(np.arccos(np.clip(nv @ nu, -1, 1))))

    @staticmethod
    def _proj_scalar(p: Coordinate, origin: Coordinate, dir_unit: Coordinate) -> float:
        return float((p - origin) @ dir_unit)

    @staticmethod
    def _body_scale(LHIP: Coordinate, RHIP: Coordinate) -> float:
        return float(np.linalg.norm(RHIP - LHIP)) + _EPS

    @staticmethod
    def _coord_dist(a: Coordinate, b: Coordinate) -> float:
        return float(np.linalg.norm(a - b))

    def _tols(self, LHIP: Coordinate, RHIP: Coordinate) -> ToleranceDict:
        s = self._body_scale(LHIP, RHIP)
        return {
            "contact": 0.35 * s,
            "behind_margin": 0.25 * s,
            "back_move": 0.30 * s,
        }

    @property
    def feet(self) -> Tuple[COCOKeypoints, COCOKeypoints]:
        keypoint_map = {
            Handedness.RIGHT: (COCOKeypoints.RIGHT_ANKLE, COCOKeypoints.LEFT_ANKLE),
            Handedness.LEFT: (COCOKeypoints.LEFT_ANKLE, COCOKeypoints.RIGHT_ANKLE),
        }
        return keypoint_map[self.handedness]

    def grade(self, grader_input: GraderInput) -> GraderResult:
        # Placeholder
        return EMPTY_GRADER_RESULT
