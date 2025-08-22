from typing import List
import pytest
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
from VideoProcessor import VideoProcessor
from Types import CoordinateList, Skill, Handedness


class TestVideoProcessor:
    def setup_method(self, mock_pose_detector):
        with patch("VideoProcessor.PoseDetector") as mock_pose_detector:
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
        # With edge padding, first point is average of [(0,0), (0,0), (10,10)] = (3.33, 3.33)
        assert smoothed[0] == pytest.approx((3.33, 3.33), abs=0.1)

    def test_calculate_velocity_dynamic(self):
        positions: CoordinateList = [(0, 0), (3, 4), (6, 8)]

        # Ensure time_intervals matches expected size
        self.processor.time_intervals = [0.1, 0.1]  # 2 intervals for 3 positions

        velocities = self.processor.calculate_velocity(positions)

        assert len(velocities) == 2
        assert velocities[0] == pytest.approx(50.0, rel=1e-2)

    def test_calculate_acceleration_dynamic(self):
        velocities = np.asarray([10, 20, 30])

        # Ensure time_intervals matches expected size for acceleration
        self.processor.time_intervals = [0.1, 0.1]  # 2 intervals for 3 velocities

        accelerations = self.processor.calculate_acceleration(velocities)

        assert len(accelerations) == 2  # Should be len(velocities) - 1
        assert accelerations[0] == pytest.approx(100.0, rel=1e-2)

    # @patch("VideoProcessor.cv2.VideoCapture")
    # @patch("VideoProcessor.threading.Thread")
    # def test_process_frames_no_frames(self, mock_thread, mock_cap):
    #     import signal
    #
    #     def timeout_handler(signum, frame):
    #         raise TimeoutError("Test hung - likely in process_frames method")
    #
    #     # Set 10 second timeout
    #     signal.signal(signal.SIGALRM, timeout_handler)
    #     signal.alarm(10)
    #
    #     try:
    #         print("Setting up VideoCapture mock...")
    #         mock_cap_instance = MagicMock()
    #         mock_cap_instance.isOpened.return_value = False
    #         mock_cap_instance.read.return_value = (False, None)
    #         mock_cap_instance.release = MagicMock()
    #         mock_cap.return_value = mock_cap_instance
    #
    #         print("Setting up Thread mock...")
    #         mock_thread_instance = MagicMock()
    #         mock_thread.return_value = mock_thread_instance
    #         mock_thread_instance.start = MagicMock()
    #         mock_thread_instance.join = MagicMock()
    #
    #         print("Calling process_frames...")
    #         result = self.processor.process_frames(Skill.SERVE, Handedness.RIGHT)
    #         print("Got result!")
    #
    #         assert result["grade"]["total_grade"] == 0
    #
    #     except TimeoutError as e:
    #         print(f"Test timed out: {e}")
    #         raise
    #     finally:
    #         signal.alarm(0)  # Cancel timeout

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

    # @patch("builtins.open", new_callable=mock_open, read_data=b"fake video data")
    # @patch("VideoProcessor.base64.b64encode")
    # def test_process_metrics_with_positions(self, mock_b64encode, mock_file):
    #     mock_b64encode.return_value = b"encoded_data"
    #
    #     # Ensure all arrays have consistent sizes
    #     num_frames = 50
    #     self.processor.right_hand_positions = [(i, i) for i in range(num_frames)]
    #     self.processor.right_elbow_positions = [(i, i) for i in range(num_frames)]
    #
    #     # Time intervals should be num_frames - 1
    #     self.processor.time_intervals = [0.033 for _ in range(num_frames - 1)]
    #
    #     self.processor.frames = [
    #         np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(num_frames)
    #     ]
    #
    #     with patch.object(
    #         self.processor, "compute_angles", return_value={"Right Elbow": 90}
    #     ):
    #         with patch.object(
    #             self.processor, "save_video_segment", return_value="/tmp/test.mp4"
    #         ):
    #             with patch("VideoProcessor.GraderRegistry.get") as mock_grader_get:
    #                 mock_grader = MagicMock()
    #                 mock_grader.grade.return_value = {
    #                     "total_grade": 85,
    #                     "grading_details": [],
    #                 }
    #                 mock_grader_get.return_value = mock_grader
    #
    #                 result = self.processor.process_metrics(
    #                     30.0, Skill.SERVE, Handedness.RIGHT
    #                 )
    #
    #     assert result["grade"]["total_grade"] == 85
    #     assert "processed_video" in result
