from typing import Any, Optional, Tuple, Literal

import numpy as np
from numpy.typing import NDArray

from core.logger import Logger
from core.joints import JOINTS
from core.types import (
    Coordinate,
    CoordinateDict,
)
from pose import PoseDetector
from video.constants import (
    SMOOTHING_WINDOW_SIZE,
    IMPACT_FRAME_SEARCH_WINDOW_BEFORE,
    IMPACT_FRAME_SEARCH_WINDOW_AFTER,
    ANALYSIS_WINDOW_PADDING_BEFORE,
)


class VideoAnalyzer:
    logger = Logger("VideoAnalyzer")

    @staticmethod
    def moving_average(
        positions: NDArray[np.floating[Any]] | list[Coordinate],
        window_size: int = 5,
        pad_mode: Literal["edge"] | Literal["reflect"] = "edge",
    ) -> NDArray[np.floating[Any]]:
        pos = np.asarray(positions, dtype=np.float64)
        if pos.ndim != 2 or pos.shape[1] != 2:
            raise ValueError("positions must have shape (N, 2)")
        k = np.ones(window_size) / window_size
        pad = window_size // 2
        x = np.pad(pos[:, 0], (pad, pad), mode=pad_mode)
        y = np.pad(pos[:, 1], (pad, pad), mode=pad_mode)
        xs = np.convolve(x, k, mode="valid")
        ys = np.convolve(y, k, mode="valid")
        return np.column_stack((xs, ys))

    @staticmethod
    def calculate_velocity(
        positions: NDArray[np.floating[Any]] | list[Coordinate],
        dt: float,
        n: int = 1,
    ) -> NDArray[np.floating[Any]]:
        positions = np.asarray(positions, dtype=np.float64)
        pos_shift = positions[n:] - positions[:-n]
        return 10.0 * (np.linalg.norm(pos_shift, axis=1) / (n * dt))

    @staticmethod
    def calculate_acceleration(
        velocities: NDArray[np.floating[Any]],
        dt: float,
        n: int = 1,
    ) -> NDArray[np.floating[Any]]:
        return 10.0 * (np.diff(velocities) / (n * dt))

    @staticmethod
    def find_acc_analysis_window(
        hand_positions: list[Coordinate],
    ) -> Tuple[int, int, int]:
        smoothed_positions = VideoAnalyzer.moving_average(
            hand_positions, window_size=SMOOTHING_WINDOW_SIZE
        )
        velocities = VideoAnalyzer.calculate_velocity(smoothed_positions, 1, 1)
        accelerations = VideoAnalyzer.calculate_acceleration(velocities, 1, 1)
        peak_frame = int(np.argmax(accelerations)) + 2 if accelerations.size > 0 else 0
        start_frame = max(0, peak_frame - IMPACT_FRAME_SEARCH_WINDOW_BEFORE)
        end_frame = min(
            len(hand_positions) - 1, peak_frame + IMPACT_FRAME_SEARCH_WINDOW_AFTER
        )
        return start_frame, peak_frame, end_frame

    @staticmethod
    def find_smash_analysis_window(
        hand_positions: list[Coordinate],
    ) -> Tuple[int, int, int]:
        start_frame, _, end_frame = VideoAnalyzer.find_acc_analysis_window(
            hand_positions
        )
        idx = np.argmin(np.asarray(hand_positions)[start_frame:end_frame, 1])
        new_peak = int(idx + start_frame)
        new_start = max(0, new_peak - IMPACT_FRAME_SEARCH_WINDOW_BEFORE)
        new_end = min(
            len(hand_positions) - 1, new_peak + IMPACT_FRAME_SEARCH_WINDOW_AFTER
        )
        return new_start, new_peak, new_end

    @staticmethod
    def find_serve_analysis_window(
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
        return int(start_frame), int(peak_frame), int(final_end_frame)

    @staticmethod
    def compute_angles(
        frame_index: int,
        normalized_landmarks: list[CoordinateDict | None],
        pose_detector: PoseDetector | None = None,
    ) -> Optional[dict[str, float]]:
        if (
            frame_index >= len(normalized_landmarks)
            or not normalized_landmarks[frame_index]
        ):
            return None
        landmarks = normalized_landmarks[frame_index]
        assert landmarks is not None
        angles: dict[str, float] = {key: 0.0 for key in JOINTS.keys()}
        for joint_name, (point_a_id, point_b_id, point_c_id) in JOINTS.items():
            if all(kp in landmarks for kp in (point_a_id, point_b_id, point_c_id)):
                point_a = landmarks[point_a_id]
                point_b = landmarks[point_b_id]
                point_c = landmarks[point_c_id]
                angle = PoseDetector.compute_angle(point_a, point_b, point_c)
                if angle is not None and isinstance(angle, float):
                    angles[joint_name] = angle
        return angles
