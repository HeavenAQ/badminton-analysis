from typing import Optional, Tuple, Literal

import numpy as np

from GraderRegistry import GraderRegistry
from Logger import Logger
from PoseModule import PoseDetector
from Joints import JOINTS
from Types import (
    Coordinate,
    CoordinateDict,
    GraderResult,
    Handedness,
    Skill,
)

# Local constants (duplicated from VideoProcessor to avoid circular imports)
SMOOTHING_WINDOW_SIZE = 5
PEAK_ACCELERATION_OFFSET = 2
IMPACT_FRAME_SEARCH_WINDOW_BEFORE = 60
IMPACT_FRAME_SEARCH_WINDOW_AFTER = 60
ANALYSIS_WINDOW_PADDING_BEFORE = 30


class VideoAnalyzer:
    def __init__(self) -> None:
        self.logger = Logger(self.__class__.__name__)

    @staticmethod
    def moving_average(
        positions: np.ndarray | list[Coordinate],
        window_size: int = 5,
        pad_mode: Literal["edge"] | Literal["reflect"] = "edge",
    ) -> np.ndarray:
        pos = np.asarray(positions, dtype=np.float64)
        if pos.ndim != 2 or pos.shape[1] != 2:
            raise ValueError("positions must have shape (N, 2)")

        k = np.ones(window_size) / window_size
        pad = window_size // 2
        x = np.pad(pos[:, 0], (pad, pad), mode=pad_mode)
        y = np.pad(pos[:, 1], (pad, pad), mode=pad_mode)

        # apply box filter
        xs = np.convolve(x, k, mode="valid")
        ys = np.convolve(y, k, mode="valid")
        return np.column_stack((xs, ys))

    @staticmethod
    def calculate_velocity(
        positions: np.ndarray | list[Coordinate],
        time_intervals: list[float],
    ) -> np.ndarray:
        pos = np.asarray(positions, dtype=np.float64)
        if pos.ndim != 2 or pos.shape[1] != 2:
            raise ValueError("positions must have shape (N, 2)")
        displacement = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        if len(time_intervals) < 2:
            return np.array([])
        return np.array(displacement / np.asarray(time_intervals[1:], dtype=np.float64))

    @staticmethod
    def calculate_acceleration(
        velocities: np.ndarray, time_intervals: list[float]
    ) -> np.ndarray:
        if len(time_intervals) < 2:
            return np.array([])
        diffs = np.diff(velocities, axis=0)
        # Prefer element-wise division when lengths align; otherwise fall back to scalar
        den = np.asarray(time_intervals[1:], dtype=np.float64)
        if den.shape[0] == diffs.shape[0]:
            return diffs / den
        # Use the last available interval as a scalar to avoid shape mismatch
        scalar = float(den[-1]) if den.size > 0 else 1.0
        return diffs / scalar

    # ------- Window selection -------
    def find_analysis_window(
        self,
        hand_positions: list[Coordinate],
        elbow_positions: list[Coordinate] | None = None,
    ) -> Tuple[int, int, int]:
        self.logger.debug("Finding analysis window using kinematic analysis")

        # smoothed_positions = self.moving_average(
        #     hand_positions, window_size=SMOOTHING_WINDOW_SIZE
        # )
        synthetic_intervals = [1.0] * max(1, len(hand_positions))
        velocities = self.calculate_velocity(hand_positions, synthetic_intervals)
        accelerations = self.calculate_acceleration(velocities, synthetic_intervals)

        peak_frame = int(np.argmax(accelerations)) if accelerations.size > 0 else 0
        start_frame = max(0, peak_frame - IMPACT_FRAME_SEARCH_WINDOW_BEFORE)
        end_frame = min(
            len(hand_positions),
            peak_frame + IMPACT_FRAME_SEARCH_WINDOW_AFTER,
        )
        return start_frame, peak_frame, end_frame

    def find_serve_analysis_window(
        self,
        hand_positions: list[Coordinate],
        elbow_positions: list[Coordinate],
        start_frame: int,
        peak_frame: int,
        end_frame: int,
    ) -> Tuple[int, int, int]:
        sub_range_positions = hand_positions[int(start_frame) : int(end_frame)]
        arr = np.asarray(sub_range_positions, dtype=np.float64)
        if arr.size > 0:
            y_values = arr[:, 1]
            lowest_hand_relative_index = int(np.argmax(y_values))
            peak_frame = start_frame + lowest_hand_relative_index

        subset_elbow_pos = elbow_positions[peak_frame:]
        arr_elbow = np.asarray(subset_elbow_pos, dtype=np.float64)
        composite_metric = (
            arr_elbow[:, 0] - arr_elbow[:, 1] if arr_elbow.size > 0 else np.array([])
        )
        relative_end_index = (
            int(np.argmax(composite_metric)) if composite_metric.size > 0 else 0
        )
        end_frame = int(peak_frame) + int(relative_end_index)

        start_frame = max(0, peak_frame - ANALYSIS_WINDOW_PADDING_BEFORE)
        final_end_frame = min(len(hand_positions), end_frame)

        self.logger.info(
            f"Analysis window determined: start={start_frame}, peak={peak_frame}, end={final_end_frame}"
        )
        return int(start_frame), int(peak_frame), int(final_end_frame)

    # ------- Angle/grade helpers -------
    def compute_angles(
        self,
        frame_index: int,
        normalized_landmarks: list[CoordinateDict | None],
        pose_detector: PoseDetector | None = None,
    ) -> Optional[dict[str, float]]:
        if (
            frame_index >= len(normalized_landmarks)
            or not normalized_landmarks[frame_index]
        ):
            self.logger.warning(
                f"No normalized landmarks available for frame {frame_index}"
            )
            return None

        landmarks = normalized_landmarks[frame_index]
        assert landmarks is not None

        angles: dict[str, float] = {key: 0.0 for key in JOINTS.keys()}
        for joint_name, (point_a_id, point_b_id, point_c_id) in JOINTS.items():
            if all(kp in landmarks for kp in (point_a_id, point_b_id, point_c_id)):
                point_a = landmarks[point_a_id]
                point_b = landmarks[point_b_id]
                point_c = landmarks[point_c_id]
                # Use static compute_angle from PoseDetector; instance not required
                angle = PoseDetector.compute_angle(point_a, point_b, point_c)
                if angle is not None and isinstance(angle, float):
                    angles[joint_name] = angle
        return angles

    def calculate_grade(
        self,
        skill: Skill,
        handedness: Handedness,
        window: Tuple[int, int, int],
        normalized_landmarks: list[CoordinateDict | None],
        pose_detector: PoseDetector | None = None,
    ) -> GraderResult:
        start, peak, end = window
        key_frames_indices = (start, (start + peak) // 2, peak, (peak + end) // 2, end)
        angle_lists = [
            self.compute_angles(i, normalized_landmarks, pose_detector)
            for i in key_frames_indices
        ]
        grader = GraderRegistry.get(skill, handedness)
        return grader.grade(angle_lists)

    # Note: Video orchestration and clip creation live in VideoProcessor
