from typing import Dict, Tuple, Type
from FootworkGrader import BackCourtFootworkGrader
from Grader import Grader
from Logger import Logger
from ServeGrader import ServeGrader
from Types import (
    Handedness,
    Skill,
)


class GraderRegistry:
    _registry: Dict[Tuple[Skill, Handedness], Type[Grader]] = {}

    @classmethod
    def register(
        cls, skill: Skill, handedness: Handedness, grader_class: Type[Grader]
    ) -> None:
        """
        Register a grader class for a specific skill and handedness.

        Args:
            skill (str): Badminton skill (e.g., 'serve', 'clear', 'smash').
            handedness (str): Handedness (e.g., 'left', 'right').
            grader_class (type): The grader class to register.
        """
        logger = Logger("GraderRegistry")
        logger.info(f"Registering grader for skill: {skill}, handedness: {handedness}")
        cls._registry[(skill, handedness)] = grader_class

    @classmethod
    def get(cls, skill: Skill, handedness: Handedness) -> Grader:
        """
        Retrieve the grader class for the given skill and handedness.

        Args:
            skill (str): Badminton skill.
            handedness (str): Handedness.

        Returns:
            Grader: An instance of the appropriate grader.
        """
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
