from typing import Optional
import numpy as np
from Logger import Logger
from Types import BodyCoordinateSystem, COCOKeypoints, CoordinateDict

_EPS = 1e-8


class BodyCentricNormalizer:
    def __init__(self) -> None:
        self.logger = Logger(self.__class__.__name__)

    def __create_body_coordinate_system(
        self, landmarks: CoordinateDict
    ) -> Optional[BodyCoordinateSystem]:
        self.logger.debug("Creating body-centric coordinate system")

        # Get the coordinates needed
        left_shoulder = np.array(landmarks[COCOKeypoints.LEFT_SHOULDER])
        right_shoulder = np.array(landmarks[COCOKeypoints.RIGHT_SHOULDER])
        left_hip = np.array(landmarks[COCOKeypoints.LEFT_HIP])
        right_hip = np.array(landmarks[COCOKeypoints.RIGHT_HIP])

        self.logger.debug(
            f"Key landmarks - Left shoulder: {left_shoulder}, Right shoulder: {right_shoulder}"
        )
        self.logger.debug(
            f"Hip landmarks - Left hip: {left_hip}, Right hip: {right_hip}"
        )

        # calculate center parts and the origin
        mid_hip = (right_hip + left_hip) / 2
        mid_shoulder = (right_shoulder + left_shoulder) / 2
        mid_body = (mid_hip + mid_shoulder) / 2
        self.logger.debug(f"Calculated origin at mid-body: {mid_body}")

        # x-axis (left -> right)
        shoulder = right_shoulder - left_shoulder
        x_norm = np.linalg.norm(shoulder)
        if x_norm < _EPS:
            return None
        x_axis = shoulder / x_norm
        self.logger.debug(f"X-axis (shoulder direction): {x_axis}")

        # Gram-Schmidt: Remove x-component from spine
        spine = mid_shoulder - mid_hip
        spine_ortho = spine - np.dot(spine, x_axis) * x_axis

        # y-axis (down -> up)
        y_norm = np.linalg.norm(spine_ortho)
        if y_norm < _EPS:
            return None
        y_axis = spine_ortho / y_norm
        self.logger.debug(f"Y-axis (spine direction): {y_axis}")

        return {
            "origin": mid_body,
            "x_axis": x_axis,
            "y_axis": y_axis,
        }

    def __apply_matrix_transformation(
        self,
        landmarks: CoordinateDict,
        body_system: BodyCoordinateSystem,
    ) -> CoordinateDict:
        self.logger.debug("Applying matrix transformation to landmarks")
        origin = body_system["origin"]
        x_axis = body_system["x_axis"]
        y_axis = body_system["y_axis"]

        # Orthonomal basis
        r = np.column_stack((x_axis, y_axis))  # (2, 2)
        rt = r.T

        out: CoordinateDict = {}
        for joint, coordinate in landmarks.items():
            coord = np.asarray(coordinate, dtype=np.float64)
            if coord.ndim != 1 or coord.shape[0] != 2:
                # skip invalid coordinate
                continue
            translated = coord - origin
            x_y = rt @ translated
            out[joint] = np.asarray(x_y, dtype=np.float64)

        self.logger.debug(f"Transformed {len(out)} landmarks to body coordinate system")
        return out

    def __normalize_scale(self, landmarks: CoordinateDict) -> CoordinateDict:
        """
        Normalize the scale to avoid the difference between of body length caused by the distance between camera and body
        """
        self.logger.debug("Normalizing landmarks by shoulder width")

        if not (
            COCOKeypoints.LEFT_SHOULDER in landmarks
            and COCOKeypoints.RIGHT_SHOULDER in landmarks
        ):
            self.logger.error("Missing shoulder landmarks for normalization")
            return {}

        # use shoulder width as the base for scaling
        left_shoulder = np.asarray(landmarks[COCOKeypoints.LEFT_SHOULDER])
        right_shoulder = np.asarray(landmarks[COCOKeypoints.RIGHT_SHOULDER])
        scale_base = np.linalg.norm(left_shoulder - right_shoulder)

        if scale_base < _EPS:
            self.logger.error("Zero shoulder width detected, cannot normalize")
            return {}

        self.logger.debug(
            f"Shoulder width: {scale_base}, normalizing {len(landmarks)} landmarks"
        )

        # normalize every coordinate and return as numpy arrays
        normalized_landmarks = {
            j: np.asarray(landmarks[j], dtype=float) / scale_base for j in landmarks
        }
        self.logger.info(
            f"Successfully normalized {len(normalized_landmarks)} landmarks"
        )
        return normalized_landmarks

    def normalize_pose(self, landmarks: CoordinateDict) -> CoordinateDict:
        self.logger.info("Starting pose normalization process")

        if not landmarks:
            self.logger.warning("Empty landmarks provided for normalization")
            return {}

        self.logger.debug(f"Normalizing pose with {len(landmarks)} landmarks")

        # Create the body-centric coordinate system
        body_system = self.__create_body_coordinate_system(landmarks)
        if not body_system:
            return {}

        # Project coordinates onto body-centric coordinate system
        translated_landmarks = self.__apply_matrix_transformation(
            landmarks, body_system
        )

        # Normalize by shoulder width
        normalized_result = self.__normalize_scale(translated_landmarks)

        self.logger.info("Pose normalization completed successfully")
        return normalized_result
