from badminton_analysis.services.body_normalizer import BodyCentricNormalizer
from badminton_analysis.services.video_analyzer import VideoAnalyzer
from badminton_analysis.models.types import GradingOutcome, Handedness, Skill, TrackingData
from badminton_analysis.services.graders.registry import GraderRegistry


class PlayerGrader:
    """Grades a player's performance using extracted tracking data."""

    def __init__(self) -> None:
        pass

    def grade(
        self,
        skill: Skill,
        handedness: Handedness,
        tracking: TrackingData,
        *,
        footwork_reference_path: str | None = None,
    ) -> tuple[GradingOutcome, tuple[int, int, int]]:
        if skill == Skill.FOOTWORK:
            landmark_list = tracking["original_landmarks"]
            if len(landmark_list) < 2:
                return {"total_grade": 0, "grading_details": []}, (0, 0, 0)
            grader = GraderRegistry.get(
                skill,
                handedness,
                reference_data_path=footwork_reference_path,
            )
            result = grader.grade([], landmark_list)
            return result, (0, 0, len(landmark_list) - 1)

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

        normalizer = BodyCentricNormalizer(handedness == Handedness.RIGHT)
        normalized_landmarks = [
            normalizer.normalize_pose(landmark)
            for landmark in landmark_list
        ]
        angle_list = list(map(VideoAnalyzer.compute_angles, normalized_landmarks))
        if handedness == Handedness.LEFT:
            angle_list = [VideoAnalyzer.mirror_angles(angles) for angles in angle_list]
        grader = GraderRegistry.get(skill, handedness)
        result = grader.grade(angle_list, landmark_list)
        return result, (start_index, peak_frame, end_index)
