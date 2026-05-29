from typing import Any

from badminton_analysis.core.logger import Logger
from badminton_analysis.models.types import Handedness, Skill
from badminton_analysis.services.graders.drive import (
    BackhandDriveGrader,
    ForehandDriveGrader,
)
from .base import Grader
from .serve import ServeGrader
from .smash import SmashGrader
from .clear import ClearGrader
from .netkill import ForehandNetKillGrader, BackhandNetKillGrader
from .footwork import BackCourtFootworkGrader
from .lift import LiftGrader


class GraderRegistry:
    _registry: dict[tuple[Skill, Handedness], type[Grader]] = {}

    @classmethod
    def register(
        cls, skill: Skill, handedness: Handedness, grader_class: type[Grader]
    ) -> None:
        logger = Logger("GraderRegistry")
        logger.info(f"Registering grader for skill: {skill}, handedness: {handedness}")
        cls._registry[(skill, handedness)] = grader_class

    @classmethod
    def get(
        cls,
        skill: Skill,
        handedness: Handedness,
        **grader_kwargs: Any,
    ) -> Grader:
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
        if grader_kwargs:
            return grader_class(handedness, **grader_kwargs)
        return grader_class(handedness)


GraderRegistry.register(Skill.SERVE, Handedness.LEFT, ServeGrader)
GraderRegistry.register(Skill.SERVE, Handedness.RIGHT, ServeGrader)
GraderRegistry.register(Skill.SMASH, Handedness.RIGHT, SmashGrader)
GraderRegistry.register(Skill.SMASH, Handedness.LEFT, SmashGrader)
GraderRegistry.register(Skill.LIFT, Handedness.RIGHT, LiftGrader)
GraderRegistry.register(Skill.LIFT, Handedness.LEFT, LiftGrader)
GraderRegistry.register(Skill.BACKHAND_DRIVE, Handedness.RIGHT, BackhandDriveGrader)
GraderRegistry.register(Skill.BACKHAND_DRIVE, Handedness.LEFT, BackhandDriveGrader)
GraderRegistry.register(Skill.FOREHAND_DRIVE, Handedness.RIGHT, ForehandDriveGrader)
GraderRegistry.register(Skill.FOREHAND_DRIVE, Handedness.LEFT, ForehandDriveGrader)
GraderRegistry.register(Skill.BACKHAND_NETKILL, Handedness.RIGHT, BackhandNetKillGrader)
GraderRegistry.register(Skill.BACKHAND_NETKILL, Handedness.LEFT, BackhandNetKillGrader)
GraderRegistry.register(Skill.FOREHAND_NETKILL, Handedness.RIGHT, ForehandNetKillGrader)
GraderRegistry.register(Skill.FOREHAND_NETKILL, Handedness.LEFT, ForehandNetKillGrader)
GraderRegistry.register(Skill.FOOTWORK, Handedness.LEFT, BackCourtFootworkGrader)
GraderRegistry.register(Skill.FOOTWORK, Handedness.RIGHT, BackCourtFootworkGrader)
GraderRegistry.register(Skill.CLEAR, Handedness.RIGHT, ClearGrader)
GraderRegistry.register(Skill.CLEAR, Handedness.LEFT, ClearGrader)
