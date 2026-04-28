from typing import Dict, Tuple, Type

from core.logger import Logger
from core.types import Handedness, Skill
from .base import Grader
from .footwork import BackCourtFootworkGrader
from .serve import ServeGrader


class GraderRegistry:
    _registry: Dict[Tuple[Skill, Handedness], Type[Grader]] = {}

    @classmethod
    def register(
        cls, skill: Skill, handedness: Handedness, grader_class: Type[Grader]
    ) -> None:
        logger = Logger("GraderRegistry")
        logger.info(f"Registering grader for skill: {skill}, handedness: {handedness}")
        cls._registry[(skill, handedness)] = grader_class

    @classmethod
    def get(cls, skill: Skill, handedness: Handedness) -> Grader:
        logger = Logger("GraderRegistry")
        logger.debug(f"Getting grader for skill: {skill}, handedness: {handedness}")
        grader_class = cls._registry.get((skill, handedness))
        if not grader_class:
            logger.error(
                f"No grader registered for skill={skill}, handedness={handedness}"
            )
            raise ValueError(
                f"No grader registered for skill={skill}, handedness={handedness}"
            )
        logger.info(f"Retrieved grader: {grader_class.__name__}")
        return grader_class(handedness)


GraderRegistry.register(Skill.SERVE, Handedness.LEFT, ServeGrader)
GraderRegistry.register(Skill.SERVE, Handedness.RIGHT, ServeGrader)
GraderRegistry.register(Skill.FOOTWORK, Handedness.LEFT, BackCourtFootworkGrader)
GraderRegistry.register(Skill.FOOTWORK, Handedness.RIGHT, BackCourtFootworkGrader)

