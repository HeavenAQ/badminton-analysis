from typing import List
import pytest
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
from VideoProcessor import VideoProcessor
from Types import CoordinateList, Skill, Handedness


class TestVideoProcessor:
    @patch("VideoProcessor.PoseDetector")
    def setup_method(self, mock_pose_detector):
        mock_pose_detector.return_value = MagicMock()
        self.processor = VideoProcessor("test.mp4", "output.mp4", "/tmp")

    def test_video_processor_initialization(self):
        assert self.processor.video_path == "test.mp4"
        assert self.processor.out_filename == "output.mp4"
        assert self.processor.output_folder == "/tmp"
        assert hasattr(self.processor, "pose_detector")

    def test_moving_average_basic(self):
        positions: CoordinateList = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
        smoothed = self.processor.moving_average(positions, window_size=3)

        assert len(smoothed) == len(positions)
        assert isinstance(smoothed, list)
        assert all(isinstance(pos, tuple) for pos in smoothed)

    def test_moving_average_edge_padding(self):
        positions: CoordinateList = [(0, 0), (10, 10)]
        smoothed = self.processor.moving_average(
            positions, window_size=3, pad_mode="edge"
        )

        assert len(smoothed) == 2
        assert smoothed[0] == pytest.approx((0, 0), abs=1e-6)

    def test_calculate_velocity_dynamic(self):
        positions: CoordinateList = [(0, 0), (3, 4), (6, 8)]
        time_intervals = [0.1, 0.1, 0.1]

        velocities = self.processor.calculate_velocity_dynamic(
            positions, time_intervals
        )

        assert len(velocities) == 2
        assert velocities[0] == pytest.approx(50.0, rel=1e-2)

    def test_calculate_acceleration_dynamic(self):
        velocities: List[float] = [10, 20, 30]
        time_intervals = [0.1, 0.1, 0.1]

        accelerations = self.processor.calculate_acceleration_dynamic(
            velocities, time_intervals
        )

        assert len(accelerations) == 1
        assert accelerations[0] == pytest.approx(100.0, rel=1e-2)

    @patch("VideoProcessor.cv2.VideoCapture")
    @patch("VideoProcessor.threading.Thread")
    def test_process_frames_no_frames(self, mock_thread, mock_cap):
        mock_cap_instance = MagicMock()
        mock_cap_instance.isOpened.return_value = False
        mock_cap.return_value = mock_cap_instance

        result = self.processor.process_frames(Skill.SERVE, Handedness.RIGHT)

        assert result["grade"]["total_grade"] == 0

    def test_compute_angles_no_landmarks(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch.object(self.processor.pose_detector, "get_pose", return_value=None):
            with patch.object(
                self.processor.pose_detector, "get_2d_landmarks", return_value=None
            ):
                result = self.processor.compute_angles(frame)

        assert result is None

    @patch("VideoProcessor.cv2.VideoWriter")
    @patch("VideoProcessor.os.path.join")
    def test_save_video_segment(self, mock_join, mock_writer):
        mock_join.return_value = "/tmp/segment.mp4"
        mock_writer_instance = MagicMock()
        mock_writer.return_value = mock_writer_instance

        self.processor.frames = [
            np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)
        ]
        self.processor.landmarks = [None for _ in range(5)]

        result = self.processor.save_video_segment(0, 4, 30.0)

        assert result == "/tmp/segment.mp4"
        mock_writer.assert_called_once()

    @patch("builtins.open", new_callable=mock_open, read_data=b"fake video data")
    @patch("VideoProcessor.base64.b64encode")
    def test_process_metrics_with_positions(self, mock_b64encode, mock_file):
        mock_b64encode.return_value = b"encoded_data"

        self.processor.right_hand_positions = [(i, i) for i in range(50)]
        self.processor.right_elbow_positions = [(i, i) for i in range(50)]
        self.processor.time_intervals = [0.033 for _ in range(50)]
        self.processor.frames = [
            np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(50)
        ]

        with patch.object(
            self.processor, "compute_angles", return_value={"Right Elbow": 90}
        ):
            with patch.object(
                self.processor, "save_video_segment", return_value="/tmp/test.mp4"
            ):
                with patch("VideoProcessor.GraderRegistry.get") as mock_grader_get:
                    mock_grader = MagicMock()
                    mock_grader.grade.return_value = {
                        "total_grade": 85,
                        "grading_details": [],
                    }
                    mock_grader_get.return_value = mock_grader

                    result = self.processor.process_metrics(
                        30.0, Skill.SERVE, Handedness.RIGHT
                    )

        assert result["grade"]["total_grade"] == 85
        assert "processed_video" in result

