from badminton_analysis.analysis import VideoAnalyzer
from badminton_analysis.core.types import GradingOutcome, Handedness, Skill, TrackingData
from badminton_analysis.graders.registry import GraderRegistry
from badminton_analysis.pose import PoseDetector


class PlayerGrader:
    """Grades a player's performance using extracted tracking data."""

    def __init__(self) -> None:
        pass

    def grade(
        self,
        skill: Skill,
        handedness: Handedness,
        tracking: TrackingData,
    ) -> tuple[GradingOutcome, tuple[int, int, int]]:
        hand_positions = tracking["hand_positions"]
        elbow_positions = tracking["elbow_positions"]
        if len(hand_positions) <= 2:
            return {"total_grade": 0, "grading_details": []}, (0, 0, 0)

        start_index, peak_frame, end_index = VideoAnalyzer.find_analysis_window(
            skill=skill,
            hand_positions=hand_positions,
            elbow_positions=elbow_positions,
        )
        landmark_list = [
            tracking["original_landmarks"][i]
            for i in (
                start_index,
                (start_index + peak_frame) // 2,
                peak_frame,
                (peak_frame + end_index) // 2,
                end_index,
            )
        ]

        angle_list = list(map(VideoAnalyzer.compute_angles, landmark_list))
        grader = GraderRegistry.get(skill, handedness)
        result = grader.grade(angle_list, landmark_list)
        return result, (start_index, peak_frame, end_index)
