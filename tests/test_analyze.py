import json
import numpy as np
from pathlib import Path

from tools.analyze import save_angles_to_json, pose_detector
from analysis import VideoAnalyzer
from core.types import COCOKeypoints


def _make_landmarks() -> dict:
    # Construct a minimal set of landmarks to compute the Right Elbow angle
    return {
        COCOKeypoints.RIGHT_SHOULDER: np.asarray([0.0, 0.0], dtype=float),
        COCOKeypoints.RIGHT_ELBOW: np.asarray([1.0, 0.0], dtype=float),
        COCOKeypoints.RIGHT_WRIST: np.asarray([1.0, 1.0], dtype=float),
    }


def test_save_angles_to_json(tmp_path: Path):
    analyzer = VideoAnalyzer()
    # Create five frames of identical normalized landmarks
    normalized_landmarks = [_make_landmarks() for _ in range(5)]

    output_dir = tmp_path
    video_name = "unit_test_video"

    # Key frames map to indices 0..4 below
    save_angles_to_json(
        normalized_landmarks,
        (0, 2, 4, 6, 8),
        output_dir=str(output_dir),
        video_name=video_name,
        analyzer=analyzer,
    )

    output_json = output_dir / "training_data.json"
    assert output_json.exists()
    data = json.loads(output_json.read_text())
    assert video_name in data
    # Should have 5 key frames saved
    assert len(data[video_name]) == 5
    # Each element is a dict of angles or None; ensure at least one dict present
    assert any(isinstance(entry, dict) for entry in data[video_name])
