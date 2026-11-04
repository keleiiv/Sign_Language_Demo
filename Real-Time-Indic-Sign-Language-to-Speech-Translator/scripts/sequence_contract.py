"""Versioned MediaPipe sequence contract used by the live 50-class model."""

from collections.abc import Iterable

import numpy as np

PREPROCESSING_VERSION = "isl-holistic-30x378-v1"
MAX_FRAMES = 30
POSE_DIM = 33 * 4
FACE_KEY_INDICES = (
    70, 63, 105, 66, 107, 55, 65, 52,
    300, 293, 334, 296, 336, 285, 295, 282,
    33, 133, 159, 145,
    362, 263, 386, 374,
    61, 37, 0, 267, 291, 321, 314, 17,
    78, 81, 13, 311, 308, 317, 14, 87,
)
FACE_DIM = len(FACE_KEY_INDICES) * 3
HAND_DIM = 21 * 3
FEATURE_DIM = POSE_DIM + FACE_DIM + HAND_DIM * 2
SEQUENCE_SHAPE = (MAX_FRAMES, FEATURE_DIM)


class SequenceContractError(ValueError):
    """Raised when a frame or sequence does not meet its declared contract."""


def _wrist_relative(landmarks) -> np.ndarray:
    coordinates = np.asarray(
        [[point.x, point.y, point.z] for point in landmarks], dtype=np.float32
    )
    if coordinates.shape != (21, 3):
        raise SequenceContractError(
            f"Hand landmarks must have shape (21, 3), got {coordinates.shape}"
        )
    coordinates -= coordinates[0]
    return coordinates.reshape(-1)


def extract_frame_features(result) -> np.ndarray:
    """Extract pose, selected nose-relative face, and wrist-relative hands."""
    if result.pose_landmarks:
        pose = np.asarray(
            [[point.x, point.y, point.z, point.visibility]
             for point in result.pose_landmarks],
            dtype=np.float32,
        )
        if pose.shape != (33, 4):
            raise SequenceContractError(
                f"Pose landmarks must have shape (33, 4), got {pose.shape}"
            )
        pose = pose.reshape(-1)
    else:
        pose = np.zeros(POSE_DIM, dtype=np.float32)

    if (result.face_landmarks
            and len(result.face_landmarks) > max(FACE_KEY_INDICES)):
        landmarks = result.face_landmarks
        nose = np.asarray(
            [landmarks[1].x, landmarks[1].y, landmarks[1].z],
            dtype=np.float32,
        )
        face_points = np.asarray(
            [[landmarks[index].x, landmarks[index].y, landmarks[index].z]
             for index in FACE_KEY_INDICES],
            dtype=np.float32,
        )
        face_points -= nose
        face = face_points.reshape(-1)
    else:
        face = np.zeros(FACE_DIM, dtype=np.float32)

    left_hand = (_wrist_relative(result.left_hand_landmarks)
                 if result.left_hand_landmarks
                 else np.zeros(HAND_DIM, dtype=np.float32))
    right_hand = (_wrist_relative(result.right_hand_landmarks)
                  if result.right_hand_landmarks
                  else np.zeros(HAND_DIM, dtype=np.float32))

    frame = np.concatenate((pose, face, left_hand, right_hand)).astype(
        np.float32, copy=False
    )
    if frame.shape != (FEATURE_DIM,) or not np.isfinite(frame).all():
        raise SequenceContractError(
            f"Extracted frame must be finite with shape ({FEATURE_DIM},)"
        )
    return frame


def validate_sequence(sequence, preprocessing_version: str) -> np.ndarray:
    """Validate and return a sequence without padding, truncating, or reshaping."""
    if preprocessing_version != PREPROCESSING_VERSION:
        raise SequenceContractError(
            f"Preprocessing version {preprocessing_version!r} is incompatible; "
            f"expected {PREPROCESSING_VERSION!r}"
        )
    array = np.asarray(sequence)
    if array.shape != SEQUENCE_SHAPE:
        raise SequenceContractError(
            f"Sequence must have shape {SEQUENCE_SHAPE}, got {array.shape}"
        )
    if (not np.issubdtype(array.dtype, np.number)
            or np.issubdtype(array.dtype, np.complexfloating)):
        raise SequenceContractError("Sequence values must be numeric")
    converted = np.asarray(array, dtype=np.float32)
    if not np.isfinite(converted).all():
        raise SequenceContractError("Sequence values must all be finite")
    return converted


def validate_label_id(label_id: int) -> int:
    if type(label_id) is not int or label_id < 0:
        raise ValueError("label_id must be a non-negative integer")
    return label_id


def build_sequence(frames: Iterable, preprocessing_version: str) -> np.ndarray:
    """Build a sequence only from exactly 30 already-extracted feature frames."""
    if preprocessing_version != PREPROCESSING_VERSION:
        raise SequenceContractError(
            f"Preprocessing version {preprocessing_version!r} is incompatible; "
            f"expected {PREPROCESSING_VERSION!r}"
        )
    array = np.asarray(list(frames))
    return validate_sequence(array, preprocessing_version)
