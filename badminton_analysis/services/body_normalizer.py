from typing import Any, Optional, final

import numpy as np

from badminton_analysis.core.logger import Logger
from badminton_analysis.models.types import BodyCoordinateSystem, COCOKeypoints, CoordinateDict

_EPS = 1e-8


@final
class BodyCentricNormalizer:
    """Normalizes 2D pose landmarks into a body-centric coordinate system.

    Handedness is a first-class concern: the coordinate system is always defined
    so that positive x points toward the dominant side. This means left-handed

    Coordinate system convention:
        - Origin:  torso center (average of mid-hip and mid-shoulder)
        - X-axis:  non-dominant shoulder → dominant shoulder (positive = dominant side)
        - Y-axis:  spine direction (inferosuperior), orthogonalized against x-axis
        - Scale:   normalized by shoulder width in the original frame

    Note: 2D only — operates on (x, y) keypoint coordinates.

    Args:
        is_right_handed: True for right-handed players (default), False for left-handed.
                         Controls the x-axis direction so dominant-side geometry is always
                         positive without any manual negation by the caller.
    """

    def __init__(self, is_right_handed: bool = True) -> None:
        self.logger = Logger(self.__class__.__name__)
        self._is_right_handed = is_right_handed

    @property
    def dominant_shoulder(self) -> COCOKeypoints:
        return (
            COCOKeypoints.RIGHT_SHOULDER
            if self._is_right_handed
            else COCOKeypoints.LEFT_SHOULDER
        )

    @property
    def non_dominant_shoulder(self) -> COCOKeypoints:
        return (
            COCOKeypoints.LEFT_SHOULDER
            if self._is_right_handed
            else COCOKeypoints.RIGHT_SHOULDER
        )

    @property
    def dominant_hip(self) -> COCOKeypoints:
        return (
            COCOKeypoints.RIGHT_HIP if self._is_right_handed else COCOKeypoints.LEFT_HIP
        )

    @property
    def non_dominant_hip(self) -> COCOKeypoints:
        return (
            COCOKeypoints.LEFT_HIP if self._is_right_handed else COCOKeypoints.RIGHT_HIP
        )

    @property
    def _critical_keypoints(self) -> list[COCOKeypoints]:
        return [
            self.dominant_shoulder,
            self.non_dominant_shoulder,
            self.dominant_hip,
            self.non_dominant_hip,
        ]

    def _create_body_coordinate_system(
        self, landmarks: CoordinateDict
    ) -> BodyCoordinateSystem | None:
        """Build the torso-anchored body coordinate frame.

        X-axis always points non-dominant → dominant (positive = dominant side),

        Returns None if the frame cannot be constructed (degenerate pose).
        """
        dom_shoulder = np.asarray(landmarks[self.dominant_shoulder], dtype=np.float64)
        non_dom_shoulder = np.asarray(
            landmarks[self.non_dominant_shoulder], dtype=np.float64
        )
        dom_hip = np.asarray(landmarks[self.dominant_hip], dtype=np.float64)
        non_dom_hip = np.asarray(landmarks[self.non_dominant_hip], dtype=np.float64)

        mid_hip = (dom_hip + non_dom_hip) / 2
        mid_shoulder = (dom_shoulder + non_dom_shoulder) / 2
        torso_center = (mid_hip + mid_shoulder) / 2

        shoulder_vec = dom_shoulder - non_dom_shoulder
        x_norm = np.linalg.norm(shoulder_vec)
        if x_norm < _EPS:
            self.logger.warning("Degenerate x-axis: shoulders are too close together.")
            return None
        x_axis = shoulder_vec / x_norm

        spine_vec = mid_shoulder - mid_hip
        spine_ortho = spine_vec - np.dot(spine_vec, x_axis) * x_axis
        y_norm = np.linalg.norm(spine_ortho)
        if y_norm < _EPS:
            self.logger.warning(
                "Degenerate y-axis: spine vector is parallel to shoulder axis."
            )
            return None
        y_axis = spine_ortho / y_norm

        return {
            "origin": torso_center,
            "x_axis": x_axis,
            "y_axis": y_axis,
        }

    def _apply_matrix_transformation(
        self, landmarks: CoordinateDict, body_system: BodyCoordinateSystem
    ) -> CoordinateDict:
        """Project all landmarks into the body coordinate frame via rigid body transform."""
        origin = body_system["origin"]
        x_axis = body_system["x_axis"]
        y_axis = body_system["y_axis"]

        # R columns are basis vectors; R.T projects global → local
        Rt = np.vstack((x_axis, y_axis))

        out: CoordinateDict = {}
        for joint, coordinate in landmarks.items():
            coord = np.asarray(coordinate, dtype=np.float64)
            if coord.ndim != 1 or coord.shape[0] != 2:
                self.logger.warning(
                    f"Skipping keypoint {joint}: expected shape (2,), got {coord.shape}."
                )
                continue
            out[joint] = Rt @ (coord - origin)

        return out

    def normalize_pose(
        self,
        landmarks: CoordinateDict,
        confidences: dict[Any, Any] | None = None,
        conf_threshold: float = 0.5,
    ) -> CoordinateDict:
        """Normalize a single pose frame into the body-centric coordinate system.

        Handedness is handled internally — dominant-side x-coordinates are always
        positive. No negation is needed by the caller.

        Args:
            landmarks:      Dict mapping COCOKeypoints → (x, y) array.
            confidences:    Optional dict mapping COCOKeypoints → float confidence
                            score. Keypoints below conf_threshold are excluded.
            conf_threshold: Minimum confidence score to retain a keypoint (default 0.5).

        Returns:
            Normalized CoordinateDict in body-centric coordinates, scaled by shoulder
            width. Returns an empty dict if the frame cannot be reliably normalized.
        """
        if not landmarks:
            return {}

        # Filter by confidence
        if confidences:
            landmarks = {
                k: v
                for k, v in landmarks.items()
                if confidences.get(k, 0.0) >= conf_threshold
            }

        # Verify critical keypoints are present
        missing = [k for k in self._critical_keypoints if k not in landmarks]
        if missing:
            self.logger.warning(
                f"Cannot normalize frame: missing or low-confidence critical keypoints: {missing}"
            )
            return {}

        # Compute scale BEFORE transformation
        # intent explicit and is robust to future axis definition changes.
        dom_shoulder = np.asarray(landmarks[self.dominant_shoulder], dtype=np.float64)
        non_dom_shoulder = np.asarray(
            landmarks[self.non_dominant_shoulder], dtype=np.float64
        )
        scale_base = np.linalg.norm(dom_shoulder - non_dom_shoulder)
        if scale_base < _EPS:
            self.logger.warning(
                "Cannot normalize frame: shoulder width is effectively zero."
            )
            return {}

        # Build coordinate frame
        body_system = self._create_body_coordinate_system(landmarks)
        if not body_system:
            return {}

        # Apply rigid body transformation
        transformed = self._apply_matrix_transformation(landmarks, body_system)
        if not transformed:
            return {}

        # Scale by shoulder width
        return {
            joint: np.asarray(coord, dtype=np.float64) / scale_base
            for joint, coord in transformed.items()
        }
