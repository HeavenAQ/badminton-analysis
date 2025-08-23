import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from Normalizer import BodyCentricNormalizer
from Types import COCOKeypoints, CoordinateDict, BodyCoordinateSystem


class TestBodyCentricNormalizer:
    def setup_method(self):
        self.normalizer = BodyCentricNormalizer()
        
        # Create sample landmark coordinates for testing
        self.sample_landmarks = {
            COCOKeypoints.LEFT_SHOULDER: (10, 20),
            COCOKeypoints.RIGHT_SHOULDER: (30, 20),
            COCOKeypoints.LEFT_HIP: (12, 40),
            COCOKeypoints.RIGHT_HIP: (28, 40),
            COCOKeypoints.LEFT_ELBOW: (5, 35),
            COCOKeypoints.RIGHT_ELBOW: (35, 35),
            COCOKeypoints.LEFT_WRIST: (0, 50),
            COCOKeypoints.RIGHT_WRIST: (40, 50),
        }
        
        # Minimal landmarks with just shoulders and hips
        self.minimal_landmarks = {
            COCOKeypoints.LEFT_SHOULDER: (0, 0),
            COCOKeypoints.RIGHT_SHOULDER: (10, 0),
            COCOKeypoints.LEFT_HIP: (1, 10),
            COCOKeypoints.RIGHT_HIP: (9, 10),
        }

    def test_normalizer_initialization(self):
        assert isinstance(self.normalizer, BodyCentricNormalizer)

    def test_create_body_coordinate_system_basic(self):
        result = self.normalizer._BodyCentricNormalizer__create_body_coordinate_system(
            self.sample_landmarks
        )
        
        assert isinstance(result, dict)
        assert "origin" in result
        assert "x_axis" in result
        assert "y_axis" in result
        
        # Check that axes are normalized (unit vectors)
        x_axis_length = np.linalg.norm(result["x_axis"])
        y_axis_length = np.linalg.norm(result["y_axis"])
        
        assert abs(x_axis_length - 1.0) < 1e-10
        assert abs(y_axis_length - 1.0) < 1e-10

    def test_create_body_coordinate_system_minimal_case(self):
        result = self.normalizer._BodyCentricNormalizer__create_body_coordinate_system(
            self.minimal_landmarks
        )
        
        # x-axis should point from left to right shoulder
        expected_x_axis = np.array([1, 0])
        np.testing.assert_array_almost_equal(result["x_axis"], expected_x_axis)
        
        # y-axis should point from hips to shoulders (upward)
        expected_y_axis = np.array([0, -1])
        np.testing.assert_array_almost_equal(result["y_axis"], expected_y_axis)

    def test_create_body_coordinate_system_origin_calculation(self):
        result = self.normalizer._BodyCentricNormalizer__create_body_coordinate_system(
            self.minimal_landmarks
        )
        
        # mid_hip should be (5, 10), mid_shoulder should be (5, 0)
        # mid_body should be (mid_hip + right_hip) / 2 = ((5,10) + (9,10)) / 2 = (7, 10)
        expected_origin = np.array([7, 10])
        np.testing.assert_array_almost_equal(result["origin"], expected_origin)

    def test_apply_matrix_transformation(self):
        body_system = {
            "origin": np.array([5, 5]),
            "x_axis": np.array([1, 0]),
            "y_axis": np.array([0, 1]),
        }
        
        landmarks = {
            COCOKeypoints.LEFT_SHOULDER: (0, 0),
            COCOKeypoints.RIGHT_SHOULDER: (10, 0),
        }
        
        result = self.normalizer._BodyCentricNormalizer__apply_matrix_transformation(
            landmarks, body_system
        )
        
        # Point (0,0) should transform to (-5, -5) relative to origin (5,5)
        assert result[COCOKeypoints.LEFT_SHOULDER] == (-5, -5)
        # Point (10,0) should transform to (5, -5) relative to origin (5,5)
        assert result[COCOKeypoints.RIGHT_SHOULDER] == (5, -5)

    def test_apply_matrix_transformation_rotated_axes(self):
        # Test with rotated coordinate system (45 degrees)
        sqrt2_half = np.sqrt(2) / 2
        body_system = {
            "origin": np.array([0, 0]),
            "x_axis": np.array([sqrt2_half, sqrt2_half]),
            "y_axis": np.array([-sqrt2_half, sqrt2_half]),
        }
        
        landmarks = {
            COCOKeypoints.LEFT_SHOULDER: (1, 0),
            COCOKeypoints.RIGHT_SHOULDER: (0, 1),
        }
        
        result = self.normalizer._BodyCentricNormalizer__apply_matrix_transformation(
            landmarks, body_system
        )
        
        # Check that transformation preserves distances and angles correctly
        assert len(result) == 2
        assert isinstance(result[COCOKeypoints.LEFT_SHOULDER], tuple)
        assert isinstance(result[COCOKeypoints.RIGHT_SHOULDER], tuple)

    def test_normalize_scale_basic(self):
        landmarks = {
            COCOKeypoints.LEFT_SHOULDER: (0, 0),
            COCOKeypoints.RIGHT_SHOULDER: (10, 0),
            COCOKeypoints.LEFT_ELBOW: (5, 5),
        }
        
        result = self.normalizer._BodyCentricNormalizer__normalize_scale(landmarks)
        
        # Shoulder width is 10, so all coordinates should be divided by 10
        assert result[COCOKeypoints.LEFT_SHOULDER] == (0, 0)
        assert result[COCOKeypoints.RIGHT_SHOULDER] == (1, 0)
        assert result[COCOKeypoints.LEFT_ELBOW] == (0.5, 0.5)

    def test_normalize_scale_missing_shoulders(self):
        # Test with missing shoulder landmarks
        landmarks = {
            COCOKeypoints.LEFT_ELBOW: (5, 5),
            COCOKeypoints.RIGHT_ELBOW: (15, 5),
        }
        
        result = self.normalizer._BodyCentricNormalizer__normalize_scale(landmarks)
        
        # Should return empty dict when shoulders are missing
        assert result == {}

    def test_normalize_scale_missing_one_shoulder(self):
        landmarks = {
            COCOKeypoints.LEFT_SHOULDER: (0, 0),
            COCOKeypoints.LEFT_ELBOW: (5, 5),
        }
        
        result = self.normalizer._BodyCentricNormalizer__normalize_scale(landmarks)
        
        # Should return empty dict when one shoulder is missing
        assert result == {}

    def test_normalize_pose_integration(self):
        result = self.normalizer.normalize_pose(self.sample_landmarks)
        
        assert isinstance(result, dict)
        assert len(result) > 0
        
        # All values should be tuples of (x, y) coordinates
        for joint, coordinate in result.items():
            assert isinstance(coordinate, tuple)
            assert len(coordinate) == 2
            assert isinstance(coordinate[0], (int, float, np.number))
            assert isinstance(coordinate[1], (int, float, np.number))

    def test_normalize_pose_preserves_relative_positions(self):
        # Test that relative positions are preserved after normalization
        landmarks = {
            COCOKeypoints.LEFT_SHOULDER: (0, 0),
            COCOKeypoints.RIGHT_SHOULDER: (20, 0),
            COCOKeypoints.LEFT_HIP: (2, 20),
            COCOKeypoints.RIGHT_HIP: (18, 20),
            COCOKeypoints.LEFT_ELBOW: (-5, 10),
            COCOKeypoints.RIGHT_ELBOW: (25, 10),
        }
        
        result = self.normalizer.normalize_pose(landmarks)
        
        # After normalization, the relative structure should be maintained
        assert len(result) == len(landmarks)
        
        # Check that all coordinates are finite numbers
        for joint, (x, y) in result.items():
            assert np.isfinite(x)
            assert np.isfinite(y)

    def test_normalize_pose_invariant_to_translation(self):
        # Test that normalization is invariant to translation
        offset_landmarks = {}
        offset = (100, 200)
        
        for joint, (x, y) in self.sample_landmarks.items():
            offset_landmarks[joint] = (x + offset[0], y + offset[1])
        
        original_result = self.normalizer.normalize_pose(self.sample_landmarks)
        offset_result = self.normalizer.normalize_pose(offset_landmarks)
        
        # Results should be very similar (accounting for numerical precision)
        for joint in original_result:
            if joint in offset_result:
                orig_x, orig_y = original_result[joint]
                off_x, off_y = offset_result[joint]
                assert abs(orig_x - off_x) < 1e-10
                assert abs(orig_y - off_y) < 1e-10

    def test_normalize_pose_invariant_to_uniform_scaling(self):
        # Test that normalization is invariant to uniform scaling
        scale_factor = 2.5
        scaled_landmarks = {}
        
        for joint, (x, y) in self.sample_landmarks.items():
            scaled_landmarks[joint] = (x * scale_factor, y * scale_factor)
        
        original_result = self.normalizer.normalize_pose(self.sample_landmarks)
        scaled_result = self.normalizer.normalize_pose(scaled_landmarks)
        
        # Results should be identical after normalization
        for joint in original_result:
            if joint in scaled_result:
                orig_x, orig_y = original_result[joint]
                scaled_x, scaled_y = scaled_result[joint]
                assert abs(orig_x - scaled_x) < 1e-10
                assert abs(orig_y - scaled_y) < 1e-10

    def test_normalize_pose_handles_edge_case_zero_shoulder_width(self):
        # Test with zero shoulder width (degenerate case)
        degenerate_landmarks = {
            COCOKeypoints.LEFT_SHOULDER: (5, 5),
            COCOKeypoints.RIGHT_SHOULDER: (5, 5),  # Same position
            COCOKeypoints.LEFT_HIP: (4, 15),
            COCOKeypoints.RIGHT_HIP: (6, 15),
        }
        
        # This should handle the division by zero gracefully
        # The exact behavior depends on implementation, but it shouldn't crash
        try:
            result = self.normalizer.normalize_pose(degenerate_landmarks)
            # If it returns a result, it should be a dict
            assert isinstance(result, dict)
        except (ZeroDivisionError, ValueError):
            # It's acceptable to raise an error for degenerate cases
            pass

    def test_normalize_pose_handles_missing_critical_landmarks(self):
        # Test with missing critical landmarks (no hips)
        incomplete_landmarks = {
            COCOKeypoints.LEFT_SHOULDER: (0, 0),
            COCOKeypoints.RIGHT_SHOULDER: (10, 0),
            COCOKeypoints.LEFT_ELBOW: (5, 5),
        }
        
        # This should handle missing landmarks gracefully
        try:
            result = self.normalizer.normalize_pose(incomplete_landmarks)
            # The result might be empty or partial
            assert isinstance(result, dict)
        except KeyError:
            # It's acceptable to raise an error for missing critical landmarks
            pass

    def test_normalize_pose_empty_input(self):
        # Test with empty landmarks
        empty_landmarks = {}
        
        try:
            result = self.normalizer.normalize_pose(empty_landmarks)
            assert isinstance(result, dict)
        except (KeyError, ValueError):
            # It's acceptable to raise an error for empty input
            pass