from abc import ABC, abstractmethod

from core.logger import Logger
from core.types import GraderInput, GraderResult, Handedness


EMPTY_GRADER_RESULT: GraderResult = {
    "grading_details": [],
    "total_grade": 0,
}


class Grader(ABC):
    def __init__(self, handedness: Handedness):
        self.handedness = handedness
        self.logger = Logger(self.__class__.__name__)

    @abstractmethod
    def grade(self, grader_input: GraderInput) -> GraderResult:
        raise NotImplementedError

