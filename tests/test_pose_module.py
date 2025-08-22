import pytest
import numpy as np
from unittest.mock import patch, MagicMock, PropertyMock
from PoseModule import PoseDetector
from Types import COCOKeypoints


class TestPoseDetector:
    def setup_method(self, method):
        with patch("PoseModule.YOLO") as mock_yolo:
            mock_yolo.return_value = MagicMock()
            with patch("PoseModule.Logger") as mock_logger:  # Mock the Logger
                mock_logger.return_value.info = MagicMock()  # Mock the info method
                self.detector = PoseDetector()

    def test_pose_detector_initialization(self):
        assert self.detector.min_detection_confidence == 0.5
        assert hasattr(self.detector, "model")
        assert hasattr(self.detector, "logger")

    # def test_fps_calculation(self):
    #     # Reset any existing time tracking
    #     if hasattr(self.detector, "_last_time"):
    #         self.detector._last_time = None
    #
    #     with patch("time.time", side_effect=[1.0, 1.1]):
    #         fps1 = self.detector.fps  # First call to establish baseline
    #         fps2 = self.detector.fps  # Second call to calculate FPS
    #         assert fps2 == pytest.approx(10.0, rel=1e-2)

    def test_fps_calculation_zero_diff(self):
        with patch("time.time", return_value=1.0):
            fps1 = self.detector.fps
            fps2 = self.detector.fps
            assert fps2 > 0

    def test_compute_angle_valid_points(self):
        point_a = (0, 1)
        point_b = (0, 0)
        point_c = (1, 0)
        angle = self.detector.compute_angle(point_a, point_b, point_c)
        assert angle == pytest.approx(90.0, rel=1e-2)

    def test_compute_angle_straight_line(self):
        point_a = (0, 0)
        point_b = (1, 0)
        point_c = (2, 0)
        angle = self.detector.compute_angle(point_a, point_b, point_c)
        assert angle == pytest.approx(180.0, rel=1e-2)

    def test_compute_angle_zero_vector(self):
        point_a = (0, 0)
        point_b = (0, 0)
        point_c = (1, 0)
        angle = self.detector.compute_angle(point_a, point_b, point_c)
        assert angle is None

    @patch("PoseModule.cv2.line")
    @patch("PoseModule.cv2.circle")
    def test_show_pose_with_valid_landmarks(self, mock_circle, mock_line):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        landmarks = {
            COCOKeypoints.LEFT_SHOULDER: (100, 200),
            COCOKeypoints.LEFT_ELBOW: (150, 250),
        }
        self.detector.show_pose(img, landmarks)
        assert mock_circle.call_count >= 1

    def test_show_pose_with_none_landmarks(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        self.detector.show_pose(img, None)

    # @patch("PoseModule.cv2.putText")
    # def test_show_fps(self, mock_put_text):
    #     img = np.zeros((480, 640, 3), dtype=np.uint8)
    #
    #     # Mock time to create predictable FPS calculation
    #     with patch("time.time", side_effect=[1.0, 1.0333]):  # 30 FPS difference
    #         _ = self.detector.fps  # Initialize timing
    #         _ = self.detector.fps  # Calculate FPS
    #         self.detector.show_fps(img)
    #
    #     mock_put_text.assert_called_once()

    def test_get_2d_landmarks_no_results(self):
        results = None
        landmarks = self.detector.get_2d_landmarks(results)
        assert landmarks is None

    def test_get_2d_landmarks_no_keypoints(self):
        results = [MagicMock()]
        results[0].keypoints = None
        landmarks = self.detector.get_2d_landmarks(results)
        assert landmarks is None

    def test_get_2d_landmarks_with_keypoints(self):
        mock_results = MagicMock()
        mock_keypoints = MagicMock()

        # Create mock tensor-like objects with .cpu() method
        mock_xy_tensor = MagicMock()
        mock_xy_tensor.cpu.return_value.numpy.return_value = np.array(
            [[100, 200], [150, 250]]
        )

        mock_conf_tensor = MagicMock()
        mock_conf_tensor.cpu.return_value.numpy.return_value = np.array([0.9, 0.8])

        mock_keypoints.xy = [mock_xy_tensor]
        mock_keypoints.conf = [mock_conf_tensor]
        mock_results.keypoints = mock_keypoints

        results = [mock_results]
        landmarks = self.detector.get_2d_landmarks(results)

        assert isinstance(landmarks, dict)
        assert len(landmarks) == 2
