import numpy as np
from Logger import Logger
from Types import BodyCoordinateSystem, COCOKeypoints, CoordinateDict


class BodyCentricNormalizer:
    def __init__(self):
        self.logger = Logger(self.__class__.__name__)

    def __create_body_coordinate_system(
        self, landmarks: CoordinateDict
    ) -> BodyCoordinateSystem:
        self.logger.debug("Creating body-centric coordinate system")
        
        # Get the coordinates needed
        left_shoulder = np.array(landmarks[COCOKeypoints.LEFT_SHOULDER])
        right_shoulder = np.array(landmarks[COCOKeypoints.RIGHT_SHOULDER])
        left_hip = np.array(landmarks[COCOKeypoints.LEFT_HIP])
        right_hip = np.array(landmarks[COCOKeypoints.RIGHT_HIP])
        
        self.logger.debug(f"Key landmarks - Left shoulder: {left_shoulder}, Right shoulder: {right_shoulder}")
        self.logger.debug(f"Hip landmarks - Left hip: {left_hip}, Right hip: {right_hip}")

        # calculate center parts and the origin
        mid_hip = (right_hip + left_hip) / 2
        mid_shoulder = (right_shoulder + left_shoulder) / 2
        mid_body = (mid_hip + mid_shoulder) / 2
        self.logger.debug(f"Calculated origin at mid-body: {mid_body}")

        # x-axis (left -> right)
        shoulder_vector = right_shoulder - left_shoulder
        x_axis = shoulder_vector / np.linalg.norm(shoulder_vector)
        self.logger.debug(f"X-axis (shoulder direction): {x_axis}")

        # y-axis (down -> up)
        spine_vector = mid_shoulder - mid_hip
        y_axis = spine_vector / np.linalg.norm(spine_vector)
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
        
        self.logger.debug(f"Transformed {len(translated_landmarks)} landmarks to body coordinate system")
        return translated_landmarks

    def __normalize_by_shoulder_width(
        self, landmarks: CoordinateDict
    ) -> CoordinateDict:
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
        left_shoulder = np.array(landmarks[COCOKeypoints.LEFT_SHOULDER])
        right_shoulder = np.array(landmarks[COCOKeypoints.RIGHT_SHOULDER])
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
        
        if shoulder_width == 0:
            self.logger.error("Zero shoulder width detected, cannot normalize")
            return {}
        
        self.logger.debug(f"Shoulder width: {shoulder_width}, normalizing {len(landmarks)} landmarks")

        # normalize every coordniate
        normalized_landmarks = {}
        for joint, (x, y) in landmarks.items():
            normalized_landmarks[joint] = (
                x / shoulder_width,
                y / shoulder_width,
            )
        
        self.logger.info(f"Successfully normalized {len(normalized_landmarks)} landmarks")
        return normalized_landmarks

    def normalize_pose(self, landmarks: CoordinateDict) -> CoordinateDict:
        self.logger.info("Starting pose normalization process")
        
        if not landmarks:
            self.logger.warning("Empty landmarks provided for normalization")
            return {}
        
        self.logger.debug(f"Normalizing pose with {len(landmarks)} landmarks")
        
        body_system = self.__create_body_coordinate_system(landmarks)
        translated_landmarks = self.__apply_matrix_transformation(
            landmarks, body_system
        )
        normalized_result = self.__normalize_by_shoulder_width(translated_landmarks)
        
        self.logger.info("Pose normalization completed successfully")
        return normalized_result
