import pytest
from Types import COCOKeypoints, Skill, Handedness


class TestCOCOKeypoints:
    def test_coco_keypoints_values(self):
        assert COCOKeypoints.NOSE == 0
        assert COCOKeypoints.LEFT_EYE == 1
        assert COCOKeypoints.RIGHT_EYE == 2
        assert COCOKeypoints.RIGHT_ANKLE == 16

    def test_coco_keypoints_total_count(self):
        assert len(COCOKeypoints) == 17


class TestSkill:
    def test_skill_values(self):
        assert Skill.SERVE == 0
        assert Skill.CLEAR == 1

    def test_skill_convert_to_enum(self):
        assert Skill.convert_to_enum("serve") == Skill.SERVE
        assert Skill.convert_to_enum("SERVE") == Skill.SERVE
        assert Skill.convert_to_enum("clear") == Skill.CLEAR

    def test_skill_str_representation(self):
        assert str(Skill.SERVE) == "serve"
        assert str(Skill.CLEAR) == "clear"

    def test_skill_convert_invalid(self):
        with pytest.raises(KeyError):
            Skill.convert_to_enum("invalid")


class TestHandedness:
    def test_handedness_values(self):
        assert Handedness.RIGHT == 0
        assert Handedness.LEFT == 1

    def test_handedness_convert_to_enum(self):
        assert Handedness.convert_to_enum("right") == Handedness.RIGHT
        assert Handedness.convert_to_enum("LEFT") == Handedness.LEFT

    def test_handedness_str_representation(self):
        assert str(Handedness.RIGHT) == "right"
        assert str(Handedness.LEFT) == "left"

    def test_handedness_convert_invalid(self):
        with pytest.raises(KeyError):
            Handedness.convert_to_enum("invalid")