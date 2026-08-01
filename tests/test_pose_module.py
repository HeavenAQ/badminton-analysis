import sys
from types import SimpleNamespace
from unittest.mock import patch, MagicMock, PropertyMock

import numpy as np
import pytest
from badminton_analysis.services.pose_detector import PoseDetector
from badminton_analysis.models.types import COCOKeypoints


class TestPoseDetector:
    def setup_method(self, method):
        with patch("badminton_analysis.core.logger.Logger") as mock_logger:
            mock_logger.return_value.info = MagicMock()
            self.detector = PoseDetector()

    def test_pose_detector_initialization(self):
        assert self.detector.min_detection_confidence == 0.5
        assert self.detector.model is None
        assert hasattr(self.detector, "logger")

    def test_onnx_gpu_model_bootstraps_on_cpu_before_provider_assignment(
        self, monkeypatch
    ):
        calls = {}

        def wholebody3d(**kwargs):
            calls.update(kwargs)
            return object()

        monkeypatch.setitem(
            sys.modules,
            "rtmlib",
            SimpleNamespace(Wholebody3d=wholebody3d),
        )
        self.detector.device = "cuda"
        self.detector.backend = "onnxruntime"
        self.detector._configure_execution_providers = MagicMock()

        model = self.detector._load_inferencer()

        assert calls["device"] == "cpu"
        self.detector._configure_execution_providers.assert_called_once_with(model)

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

    @patch("badminton_analysis.services.pose_detector.cv2.line")
    @patch("badminton_analysis.services.pose_detector.cv2.circle")
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

    # @patch("pose.detector.cv2.putText")
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
        results = [{"bbox": [0, 0, 100, 100], "keypoints": None}]
        landmarks = self.detector.get_2d_landmarks(results)
        assert landmarks is None

    def test_select_target_prefers_largest_bbox(self):
        smaller_near_previous_target = {
            "bbox": [0.0, 0.0, 20.0, 20.0],
            "keypoints": [[5.0, 5.0], [10.0, 10.0]],
            "keypoint_scores": [1.0, 1.0],
        }
        larger_farther_target = {
            "bbox": [100.0, 100.0, 180.0, 190.0],
            "keypoints": [[120.0, 130.0], [150.0, 170.0]],
            "keypoint_scores": [1.0, 1.0],
        }
        self.detector._target_bbox_center = np.array((10.0, 10.0), dtype=np.float64)

        target = self.detector._select_target(
            [smaller_near_previous_target, larger_farther_target]
        )

        assert target is larger_farther_target

    def test_get_pose_method(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        detector_model = MagicMock(return_value=np.array([[10, 20, 200, 300]]))
        keypoints_3d = np.zeros((1, 17, 3), dtype=np.float64)
        keypoints_2d = np.zeros((1, 17, 2), dtype=np.float64)
        scores = np.ones((1, 17), dtype=np.float64)
        pose_model = MagicMock(
            return_value=(keypoints_3d, scores, None, keypoints_2d)
        )
        self.detector.model = MagicMock(
            det_model=detector_model,
            pose_model=pose_model,
        )

        result = self.detector.get_pose(img)

        detector_model.assert_called_once_with(img)
        pose_model.assert_called_once_with(img, bboxes=[[10.0, 20.0, 200.0, 300.0]])
        assert len(result) == 1
        assert len(result[0]["keypoints"]) == 17
        assert len(self.detector._last_wholebody_predictions) == 1

    def test_get_3d_landmarks_with_keypoints(self):
        results = [
            {
                "keypoints": [
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                ],
                "keypoint_scores": [1.0, 1.0],
            }
        ]
        landmarks = self.detector.get_3d_landmarks(results)

        assert isinstance(landmarks, dict)
        assert len(landmarks) == 2
        assert landmarks[COCOKeypoints.NOSE].shape == (3,)
        assert np.allclose(landmarks[COCOKeypoints.NOSE], [0.1, 0.2, 0.3])

    @patch("badminton_analysis.services.pose_detector.cv2.ellipse")
    @patch.object(PoseDetector, "_PoseDetector__add_text_with_pillow")
    def test_show_angle_arc(self, mock_add_text, mock_ellipse):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        point_a = (100, 200)
        point_b = (150, 150)
        point_c = (200, 200)
        angle = 90.0
        
        self.detector.show_angle_arc(img, point_a, point_b, point_c, angle)
        
        mock_ellipse.assert_called_once()
        mock_add_text.assert_called_once()

    @patch("badminton_analysis.services.pose_detector.Image.fromarray")
    @patch("badminton_analysis.services.pose_detector.cv2.cvtColor")
    @patch("badminton_analysis.services.pose_detector.np.copyto")
    def test_add_text_with_pillow(self, mock_copyto, mock_cvtcolor, mock_from_array):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        text = "Test°"
        position = (100, 100)
        
        mock_pil_image = MagicMock()
        mock_draw = MagicMock()
        mock_from_array.return_value = mock_pil_image
        
        with patch("badminton_analysis.services.pose_detector.ImageDraw.Draw", return_value=mock_draw):
            with patch("badminton_analysis.services.pose_detector.ImageFont.load_default") as mock_font:
                mock_font.return_value = "mock_font"
                
                # Access the private method for testing
                self.detector._PoseDetector__add_text_with_pillow(
                    img, text, position
                )
                
                mock_draw.text.assert_called_once_with(
                    position, text, font="mock_font", fill=(255, 255, 255)
                )
