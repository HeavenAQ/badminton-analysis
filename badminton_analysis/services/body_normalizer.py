from typing import Optional
import numpy as np
from core.logger import Logger
from core.types import BodyCoordinateSystem, COCOKeypoints, CoordinateDict

_EPS = 1e-8


class BodyCentricNormalizer:
    def __init__(self) -> None:
        self.logger = Logger(self.__class__.__name__)

    def __create_body_coordinate_system(
        self, landmarks: CoordinateDict
    ) -> Optional[BodyCoordinateSystem]:
        left_shoulder = np.array(landmarks[COCOKeypoints.LEFT_SHOULDER])
        right_shoulder = np.array(landmarks[COCOKeypoints.RIGHT_SHOULDER])
        left_hip = np.array(landmarks[COCOKeypoints.LEFT_HIP])
        right_hip = np.array(landmarks[COCOKeypoints.RIGHT_HIP])

        mid_hip = (right_hip + left_hip) / 2
        mid_shoulder = (right_shoulder + left_shoulder) / 2
        mid_body = (mid_hip + mid_shoulder) / 2

        shoulder = right_shoulder - left_shoulder
        x_norm = np.linalg.norm(shoulder)
        if x_norm < _EPS:
            return None
        x_axis = shoulder / x_norm

        spine = mid_shoulder - mid_hip
        spine_ortho = spine - np.dot(spine, x_axis) * x_axis
        y_norm = np.linalg.norm(spine_ortho)
        if y_norm < _EPS:
            return None
        y_axis = spine_ortho / y_norm

        return {
            "origin": mid_body,
            "x_axis": x_axis,
            "y_axis": y_axis,
        }

    def __apply_matrix_transformation(
        self, landmarks: CoordinateDict, body_system: BodyCoordinateSystem
    ) -> CoordinateDict:
        origin = body_system["origin"]
        x_axis = body_system["x_axis"]
        y_axis = body_system["y_axis"]
        r = np.column_stack((x_axis, y_axis))
        rt = r.T
        out: CoordinateDict = {}
        for joint, coordinate in landmarks.items():
            coord = np.asarray(coordinate, dtype=np.float64)
            if coord.ndim != 1 or coord.shape[0] != 2:
                continue
            translated = coord - origin
            x_y = rt @ translated
            out[joint] = np.asarray(x_y, dtype=np.float64)
        return out

    def __normalize_scale(self, landmarks: CoordinateDict) -> CoordinateDict:
        if not (
            COCOKeypoints.LEFT_SHOULDER in landmarks
            and COCOKeypoints.RIGHT_SHOULDER in landmarks
        ):
            return {}
        left_shoulder = np.asarray(landmarks[COCOKeypoints.LEFT_SHOULDER])
        right_shoulder = np.asarray(landmarks[COCOKeypoints.RIGHT_SHOULDER])
        scale_base = np.linalg.norm(left_shoulder - right_shoulder)
        if scale_base < _EPS:
            return {}
        return {j: np.asarray(landmarks[j], dtype=float) / scale_base for j in landmarks}

    def normalize_pose(self, landmarks: CoordinateDict) -> CoordinateDict:
        if not landmarks:
            return {}
        body_system = self.__create_body_coordinate_system(landmarks)
        if not body_system:
            return {}
        translated_landmarks = self.__apply_matrix_transformation(landmarks, body_system)
        return self.__normalize_scale(translated_landmarks)

