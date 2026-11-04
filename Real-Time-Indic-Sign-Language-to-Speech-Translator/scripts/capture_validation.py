"""Validation helpers for developer-captured landmark sequences."""

from dataclasses import dataclass

import numpy as np

from scripts.sequence_contract import (
    HAND_DIM,
    POSE_DIM,
    PREPROCESSING_VERSION,
    SEQUENCE_SHAPE,
    SequenceContractError,
    validate_sequence,
)

MIN_HAND_FRAME_RATIO = 0.8
LEFT_HAND_START = POSE_DIM + 120
RIGHT_HAND_START = LEFT_HAND_START + HAND_DIM


@dataclass(frozen=True)
class CaptureQuality:
    frames: int
    hand_frames: int
    hand_frame_ratio: float


def validate_capture_sequence(
    sequence, preprocessing_version: str = PREPROCESSING_VERSION
) -> tuple[np.ndarray, CaptureQuality]:
    array = validate_sequence(sequence, preprocessing_version)
    left_hand = array[:, LEFT_HAND_START:RIGHT_HAND_START]
    right_hand = array[:, RIGHT_HAND_START:]
    hand_visible = np.any(left_hand != 0, axis=1) | np.any(right_hand != 0, axis=1)
    hand_frames = int(np.count_nonzero(hand_visible))
    ratio = hand_frames / SEQUENCE_SHAPE[0]
    quality = CaptureQuality(
        frames=SEQUENCE_SHAPE[0],
        hand_frames=hand_frames,
        hand_frame_ratio=ratio,
    )
    if ratio < MIN_HAND_FRAME_RATIO:
        raise SequenceContractError(
            "Hands must be detected in at least "
            f"{MIN_HAND_FRAME_RATIO:.0%} of frames; detected in "
            f"{hand_frames}/{SEQUENCE_SHAPE[0]}"
        )
    return array, quality
