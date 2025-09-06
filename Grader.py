from abc import ABC, abstractmethod

from Logger import Logger
from Types import GraderInput, GraderResult, Handedness


# Error Response
EMPTY_GRADER_RESULT: GraderResult = {
    "grading_details": [],
    "total_grade": 0,
}


class Grader(ABC):
    """
    Base class for all graders. Each grader should implement the `grade` method.
    """

    def __init__(self, handedness: Handedness):
        self.handedness = handedness
        self.logger = Logger(self.__class__.__name__)

    @abstractmethod
    def grade(self, grader_input: GraderInput) -> GraderResult:
        """
        Abstract method to grade the performance based on angles.

        Args:
            angles (list[dict[str, float]]): list of angles for the frames to be graded.

        Returns:
            float: Grading score.
        """
        pass
