import json
import os
from pathlib import Path
from typing import Any, TypedDict, final, override

import numpy as np

from badminton_analysis.models.types import (
    AngleDicts,
    COCOKeypoints,
    Coordinate,
    CoordinateDict,
    GradingDetail,
    GradingOutcome,
    Handedness,
)
from badminton_analysis.services.body_normalizer import BodyCentricNormalizer
from badminton_analysis.services.video_analyzer import VideoAnalyzer

from .base import EMPTY_GRADER_RESULT, Grader


class FootworkReference(TypedDict):
    left_ankle: list[list[float]]
    right_ankle: list[list[float]]


@final
class BackCourtFootworkGrader(Grader):
    def __init__(
        self,
        handedness: Handedness,
        reference_data_path: str | os.PathLike[str] | None = None,
    ) -> None:
        super().__init__(handedness)
        self.reference_data_path = self._resolve_reference_data_path(reference_data_path)
        self.normalizer = BodyCentricNormalizer(handedness == Handedness.RIGHT)
        self.reference = self._load_reference(self.reference_data_path, handedness)

    @staticmethod
    def _resolve_reference_data_path(
        reference_data_path: str | os.PathLike[str] | None,
    ) -> Path:
        raw_path = reference_data_path or os.environ.get(
            "BADMINTON_FOOTWORK_REFERENCE"
        )
        if not raw_path:
            raise ValueError(
                "Footwork grading requires a reference JSON file. "
                "Pass reference_data_path or set BADMINTON_FOOTWORK_REFERENCE."
            )
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"Footwork reference data not found: {path}")
        return path

    @staticmethod
    def _normalize_reference_payload(payload: Any) -> list[Coordinate]:
        if not isinstance(payload, list):
            raise ValueError("Footwork reference trajectory must be a list of [x, y].")

        normalized: list[Coordinate] = []
        for point in payload:
            if not isinstance(point, list | tuple) or len(point) != 2:
                raise ValueError(
                    "Footwork reference trajectory points must be [x, y] pairs."
                )
            normalized.append(np.asarray(point, dtype=np.float64))
        return normalized

    @classmethod
    def _load_reference(
        cls,
        path: Path,
        handedness: Handedness,
    ) -> dict[COCOKeypoints, list[Coordinate]]:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if not isinstance(payload, dict):
            raise ValueError("Footwork reference JSON must be an object.")

        handedness_key = str(handedness)
        if "left_ankle" in payload and "right_ankle" in payload:
            selected = payload
        else:
            candidate = payload.get(handedness_key)
            if not isinstance(candidate, dict):
                raise ValueError(
                    f"Footwork reference data missing handedness entry: {handedness_key}"
                )
            selected = candidate

        if "left_ankle" not in selected or "right_ankle" not in selected:
            raise ValueError(
                "Footwork reference data must contain left_ankle and right_ankle trajectories."
            )

        return {
            COCOKeypoints.LEFT_ANKLE: cls._normalize_reference_payload(
                selected["left_ankle"]
            ),
            COCOKeypoints.RIGHT_ANKLE: cls._normalize_reference_payload(
                selected["right_ankle"]
            ),
        }

    def _extract_trajectory(
        self,
        landmark_list: list[CoordinateDict],
        keypoint: COCOKeypoints,
    ) -> list[Coordinate]:
        trajectory: list[Coordinate] = []
        for landmark in landmark_list:
            normalized = self.normalizer.normalize_pose(landmark)
            coord = normalized.get(keypoint)
            if coord is None:
                continue
            trajectory.append(np.asarray(coord, dtype=np.float64))
        return trajectory

    @staticmethod
    def _normalized_dtw_cost(
        student_signal: list[Coordinate],
        expert_signal: list[Coordinate],
    ) -> float:
        path, cost = VideoAnalyzer.dynamic_time_warping(student_signal, expert_signal)
        return float(cost / max(len(path), 1))

    @staticmethod
    def _cost_to_grade(max_grade: float, cost: float) -> float:
        return float(max_grade * np.exp(-cost))

    @override
    def grade(
        self, angles: AngleDicts, landmark_list: list[CoordinateDict]
    ) -> GradingOutcome:
        del angles

        left_student = self._extract_trajectory(landmark_list, COCOKeypoints.LEFT_ANKLE)
        right_student = self._extract_trajectory(
            landmark_list, COCOKeypoints.RIGHT_ANKLE
        )

        if len(left_student) < 2 or len(right_student) < 2:
            self.logger.warning("Insufficient ankle trajectory data for footwork grading")
            return EMPTY_GRADER_RESULT

        left_cost = self._normalized_dtw_cost(
            left_student,
            self.reference[COCOKeypoints.LEFT_ANKLE],
        )
        right_cost = self._normalized_dtw_cost(
            right_student,
            self.reference[COCOKeypoints.RIGHT_ANKLE],
        )

        left_grade = self._cost_to_grade(50.0, left_cost)
        right_grade = self._cost_to_grade(50.0, right_cost)
        total = left_grade + right_grade
        grading_details: list[GradingDetail] = [
            GradingDetail(description="左腳步伐軌跡對齊", grade=left_grade),
            GradingDetail(description="右腳步伐軌跡對齊", grade=right_grade),
        ]
        return GradingOutcome(grading_details=grading_details, total_grade=total)
