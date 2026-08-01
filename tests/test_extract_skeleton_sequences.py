from pathlib import Path

import pytest

from badminton_analysis.models.types import Handedness
from scripts.extract_skeleton_sequences import (
    _resolve_handedness,
    _sequence_identity,
    build_parser,
)


def test_known_handedness_overrides_motion_estimate_and_filename_override() -> None:
    handedness, source = _resolve_handedness(
        Path("left-looking.mp4"),
        Handedness.LEFT,
        Handedness.RIGHT,
        {"left-looking.mp4": Handedness.LEFT},
    )

    assert handedness == Handedness.RIGHT
    assert source == "known_metadata"


def test_filename_override_precedes_motion_estimate() -> None:
    handedness, source = _resolve_handedness(
        Path("sample.mp4"),
        Handedness.RIGHT,
        None,
        {"sample.mp4": Handedness.LEFT},
    )

    assert handedness == Handedness.LEFT
    assert source == "metadata_override"


def test_nstc_sequence_prefix_prevents_left_right_collisions() -> None:
    left = _sequence_identity(Path("1.mp4"), "nstc_left_")
    right = _sequence_identity(Path("1.mp4"), "nstc_right_")

    assert left == ("nstc_left_1", "nstc_left_1.mp4")
    assert right == ("nstc_right_1", "nstc_right_1.mp4")
    assert left != right


def test_sequence_prefix_rejects_path_separators() -> None:
    with pytest.raises(ValueError, match="path separator"):
        _sequence_identity(Path("1.mp4"), "nstc/left/")


def test_cli_accepts_authoritative_handedness_and_id_prefix() -> None:
    args = build_parser().parse_args(
        [
            "--known-handedness",
            "left",
            "--id-prefix",
            "nstc_left_",
        ]
    )

    assert args.known_handedness == "left"
    assert args.id_prefix == "nstc_left_"
