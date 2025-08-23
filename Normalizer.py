import numpy as np
from Types import BodyCoordinateSystem, COCOKeypoints, CoordinateDict


class BodyCentricNormalizer:
    def __init__(self):
        pass

    def __create_body_coordinate_system(
        self, landmarks: CoordinateDict
    ) -> BodyCoordinateSystem:
        # Get the coordinates needed
        left_shoulder = np.array(landmarks[COCOKeypoints.LEFT_SHOULDER])
        right_shoulder = np.array(landmarks[COCOKeypoints.RIGHT_SHOULDER])
        left_hip = np.array(landmarks[COCOKeypoints.LEFT_HIP])
        right_hip = np.array(landmarks[COCOKeypoints.RIGHT_HIP])

        # calculate center parts and the origin
        mid_hip = (right_hip + left_hip) / 2
        mid_shoulder = (right_shoulder + left_shoulder) / 2
        mid_body = (mid_hip + mid_shoulder) / 2

        # x-axis (left -> right)
        shoulder_vector = right_shoulder - left_shoulder
        x_axis = shoulder_vector / np.linalg.norm(shoulder_vector)

        # y-axis (down -> up)
        spine_vector = mid_shoulder - mid_hip
        y_axis = spine_vector / np.linalg.norm(spine_vector)

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
        translated_landmarks = {}
        origin = body_system["origin"]
        x_axis = body_system["x_axis"]
        y_axis = body_system["y_axis"]

        for joint, coordinate in landmarks.items():
            translated = coordinate - origin
            # project on to body axes
            x_coord = np.dot(translated, x_axis)
            y_coord = np.dot(translated, y_axis)
            translated_landmarks[joint] = (x_coord, y_coord)
        return translated_landmarks

    def __normalize_by_shoulder_width(
        self, landmarks: CoordinateDict
    ) -> CoordinateDict:
        """
        Normalize the scale to avoid the difference between of body length caused by the distance between camera and body
        """
        if not (
            COCOKeypoints.LEFT_SHOULDER in landmarks
            and COCOKeypoints.RIGHT_SHOULDER in landmarks
        ):
            return {}

        # use shoulder width as the base for scaling
        left_shoulder = np.array(landmarks[COCOKeypoints.LEFT_SHOULDER])
        right_shoulder = np.array(landmarks[COCOKeypoints.RIGHT_SHOULDER])
        scale_base = np.linalg.norm(left_shoulder - right_shoulder)

        # normalize every coordniate
        normalized_landmarks = {}
        for joint, (x, y) in landmarks.items():
            normalized_landmarks[joint] = (
                x / scale_base,
                y / scale_base,
            )
        return normalized_landmarks

    def normalize_pose(self, landmarks: CoordinateDict) -> CoordinateDict:
        body_system = self.__create_body_coordinate_system(landmarks)
        translated_landmarks = self.__apply_matrix_transformation(
            landmarks, body_system
        )
        return self.__normalize_by_shoulder_width(translated_landmarks)
