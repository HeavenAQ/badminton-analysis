from typing import Tuple

from analysis import VideoAnalyzer
from core.types import GraderResult, Handedness, Skill, TrackingData
from graders.registry import GraderRegistry
from pose import PoseDetector


class PlayerGrader:
    """Grades a player's performance using extracted tracking data."""

    def __init__(self) -> None:
        pass

    def grade(
        self,
        skill: Skill,
        handedness: Handedness,
        tracking: TrackingData,
        pose_detector: PoseDetector | None = None,
    ) -> tuple[GraderResult, Tuple[int, int, int]]:
        hand_positions = tracking["hand_positions"]
        if len(hand_positions) <= 2:
            return {"total_grade": 0, "grading_details": []}, (0, 0, 0)

        start_index, peak_frame, end_index = VideoAnalyzer.find_acc_analysis_window(hand_positions)
        key_frames_indices = (start_index, (start_index + peak_frame) // 2, peak_frame, (peak_frame + end_index) // 2, end_index)

        normalized_landmarks = tracking["normalized_landmarks"]
        angle_lists = [
            VideoAnalyzer.compute_angles(i, normalized_landmarks, pose_detector)
            for i in key_frames_indices
        ]
        grader = GraderRegistry.get(skill, handedness)
        result = grader.grade(angle_lists)
        return result, (start_index, peak_frame, end_index)

