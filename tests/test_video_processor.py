from unittest.mock import MagicMock

import numpy as np
import pytest

from badminton_analysis.services.video_analyzer import VideoAnalyzer
from badminton_analysis.services.video_processor import VideoProcessor


def test_video_processor_accepts_shared_pose_detector() -> None:
    detector = MagicMock()
    processor = VideoProcessor("test.mp4", "output.mp4", "/tmp", detector)

    assert processor.video_path == "test.mp4"
    assert processor.out_filename == "output.mp4"
    assert processor.output_folder == "/tmp"
    assert processor.pose_detector is detector


def test_moving_average_preserves_shape() -> None:
    positions = np.asarray(
        [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)], dtype=float
    )
    smoothed = VideoAnalyzer.moving_average(positions, window_size=3)

    assert smoothed.shape == positions.shape


def test_moving_average_uses_edge_padding() -> None:
    positions = np.asarray([(0, 0), (10, 10)], dtype=float)
    smoothed = VideoAnalyzer.moving_average(
        positions, window_size=3, pad_mode="edge"
    )

    np.testing.assert_allclose(smoothed[0], np.array([3.33, 3.33]), atol=0.1)


def test_velocity_uses_coordinate_distance() -> None:
    positions = np.asarray([(0, 0), (3, 4), (6, 8)], dtype=float)
    velocities = VideoAnalyzer.calc_velocity(positions, 1, 1)

    assert velocities[0] == pytest.approx(50.0, rel=1e-2)


def test_acceleration_uses_velocity_delta() -> None:
    accelerations = VideoAnalyzer.calc_acceleration(
        np.asarray([10, 20, 30], dtype=float), 1, 1
    )

    assert accelerations[0] == pytest.approx(100.0, rel=1e-2)


def test_compute_angles_handles_missing_landmarks() -> None:
    result = VideoAnalyzer().compute_angles({})

    assert all(angle == 0.0 for angle in result.values())
