"""Isolated CPU learning engine for developer-approved vocabulary examples."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import uuid

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.sequence_contract import (
    PREPROCESSING_VERSION,
    SEQUENCE_SHAPE,
    SequenceContractError,
    validate_sequence,
)
from scripts.vocabulary_store import TrainingExample, VocabularyStore

CLASSIFIER_VERSION = 1
MIN_TRAIN_SESSIONS_PER_CLASS = 2
MIN_EVALUATION_SESSIONS_PER_CLASS = 4
VALIDATION_FRACTION = 0.25
TEMPORAL_BINS = 5
LEARNING_RATE = 0.01
EPOCHS = 400
L2_STRENGTH = 0.01
ABSTENTION_THRESHOLD = 0.55


class InsufficientDataError(ValueError):
    """Raised when independent capture sessions are insufficient."""


class ArtifactCompatibilityError(ValueError):
    """Raised when an artifact is incompatible or corrupted."""


@dataclass(frozen=True)
class SplitResult:
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    train_sessions: dict[str, tuple[str, ...]]
    validation_sessions: dict[str, tuple[str, ...]]
    evaluation_ready: bool
    insufficiencies: tuple[str, ...]


def temporal_features(sequence) -> np.ndarray:
    array = validate_sequence(sequence, PREPROCESSING_VERSION)
    bins = np.array_split(array, TEMPORAL_BINS, axis=0)
    means = [chunk.mean(axis=0) for chunk in bins]
    transitions = [means[index + 1] - means[index]
                   for index in range(len(means) - 1)]
    result = np.concatenate((*means, *transitions)).astype(np.float32)
    return result


def split_by_session(
    examples: list[TrainingExample],
    seed: int = 42,
    validation_fraction: float = VALIDATION_FRACTION,
) -> SplitResult:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if not examples:
        raise InsufficientDataError("No approved samples are available.")

    scopes = {example.scope for example in examples}
    versions = {example.preprocessing_version for example in examples}
    if len(scopes) != 1:
        raise ValueError("Training data must contain exactly one vocabulary scope.")
    if scopes - {"shared", "personal"}:
        raise ValueError("Training data contains an invalid vocabulary scope.")
    if versions != {PREPROCESSING_VERSION}:
        raise SequenceContractError("Training data contains incompatible versions.")

    labels: dict[int, str] = {}
    sessions_by_label: dict[int, set[str]] = defaultdict(set)
    indices_by_label_session: dict[tuple[int, str], list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        validate_sequence(example.sequence, example.preprocessing_version)
        existing_label = labels.setdefault(example.label_id, example.label)
        if existing_label != example.label:
            raise ValueError(f"Conflicting label name for ID {example.label_id}")
        sessions_by_label[example.label_id].add(example.session_id)
        indices_by_label_session[(example.label_id, example.session_id)].append(index)

    rng = np.random.default_rng(seed)
    train_indices = []
    validation_indices = []
    train_sessions: dict[str, tuple[str, ...]] = {}
    validation_sessions: dict[str, tuple[str, ...]] = {}
    insufficiencies = []

    for label_id in sorted(labels):
        label = labels[label_id]
        sessions = sorted(sessions_by_label[label_id])
        if len(sessions) < MIN_TRAIN_SESSIONS_PER_CLASS:
            insufficiencies.append(
                f"{label}: {len(sessions)} independent session(s); "
                f"{MIN_TRAIN_SESSIONS_PER_CLASS} required to train."
            )
            continue
        if len(sessions) < MIN_EVALUATION_SESSIONS_PER_CLASS:
            insufficiencies.append(
                f"{label}: {len(sessions)} independent session(s); "
                f"{MIN_EVALUATION_SESSIONS_PER_CLASS} required for a "
                "two-session-per-side validation evaluation."
            )
            train_for_class = tuple(sessions)
            validation_sessions[label] = ()
            val_sessions = ()
        else:
            order = rng.permutation(len(sessions))
            shuffled = [sessions[position] for position in order]
            validation_count = max(
                2, int(round(len(shuffled) * validation_fraction))
            )
            validation_count = min(validation_count, len(shuffled) - 2)
            val_sessions = tuple(sorted(shuffled[:validation_count]))
            train_for_class = tuple(sorted(shuffled[validation_count:]))
            validation_sessions[label] = val_sessions
        train_sessions[label] = train_for_class
        for session_id in train_for_class:
            train_indices.extend(indices_by_label_session[(label_id, session_id)])
        for session_id in val_sessions:
            validation_indices.extend(indices_by_label_session[(label_id, session_id)])

    return SplitResult(
        train_indices=tuple(sorted(train_indices)),
        validation_indices=tuple(sorted(validation_indices)),
        train_sessions=train_sessions,
        validation_sessions=validation_sessions,
        evaluation_ready=not insufficiencies,
        insufficiencies=tuple(insufficiencies),
    )


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(np.clip(shifted, -80, 80))
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def _fit_standardizer(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = features.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    return mean, scale


def _fit_softmax(
    features: np.ndarray,
    targets: np.ndarray,
    class_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.001, (features.shape[1], class_count)).astype(
        np.float32
    )
    bias = np.zeros(class_count, dtype=np.float32)
    first_w = np.zeros_like(weights)
    second_w = np.zeros_like(weights)
    first_b = np.zeros_like(bias)
    second_b = np.zeros_like(bias)
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    sample_count = len(features)
    one_hot = np.eye(class_count, dtype=np.float32)[targets]

    for step in range(1, EPOCHS + 1):
        probabilities = _softmax(features @ weights + bias)
        error = (probabilities - one_hot) / sample_count
        grad_w = features.T @ error + L2_STRENGTH * weights
        grad_b = error.sum(axis=0)
        first_w = beta1 * first_w + (1 - beta1) * grad_w
        second_w = beta2 * second_w + (1 - beta2) * (grad_w * grad_w)
        first_b = beta1 * first_b + (1 - beta1) * grad_b
        second_b = beta2 * second_b + (1 - beta2) * (grad_b * grad_b)
        corrected_first_w = first_w / (1 - beta1**step)
        corrected_second_w = second_w / (1 - beta2**step)
        corrected_first_b = first_b / (1 - beta1**step)
        corrected_second_b = second_b / (1 - beta2**step)
        weights -= LEARNING_RATE * corrected_first_w / (
            np.sqrt(corrected_second_w) + epsilon
        )
        bias -= LEARNING_RATE * corrected_first_b / (
            np.sqrt(corrected_second_b) + epsilon
        )
    return weights, bias


def _class_metrics(
    true_ids: np.ndarray,
    predicted_ids: np.ndarray,
    label_map: list[dict],
) -> dict:
    ids = [item["label_id"] for item in label_map]
    matrix = np.zeros((len(ids), len(ids)), dtype=np.int64)
    id_to_index = {label_id: index for index, label_id in enumerate(ids)}
    for truth, prediction in zip(true_ids, predicted_ids):
        matrix[id_to_index[int(truth)], id_to_index[int(prediction)]] += 1
    per_class = {}
    f1_scores = []
    for index, item in enumerate(label_map):
        true_positive = int(matrix[index, index])
        false_positive = int(matrix[:, index].sum() - true_positive)
        false_negative = int(matrix[index, :].sum() - true_positive)
        precision = (true_positive / (true_positive + false_positive)
                     if true_positive + false_positive else 0.0)
        recall = (true_positive / (true_positive + false_negative)
                  if true_positive + false_negative else 0.0)
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        per_class[item["label"]] = {
            "support": int(matrix[index, :].sum()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        f1_scores.append(f1)
    total = int(matrix.sum())
    accuracy = float(np.trace(matrix) / total) if total else None
    return {
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "label_order": [item["label"] for item in label_map],
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_scores)) if f1_scores else None,
        "support": total,
    }


def _artifact_root(store: VocabularyStore, scope: str) -> Path:
    if scope not in {"shared", "personal"}:
        raise ValueError("scope must be shared or personal")
    return store.root / scope / "models"


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_manifest(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactCompatibilityError(f"Cannot read artifact metadata: {error}") from error
    if (not isinstance(payload, dict)
            or payload.get("artifact_schema_version") != CLASSIFIER_VERSION
            or payload.get("preprocessing_version") != PREPROCESSING_VERSION
            or payload.get("sequence_shape") != list(SEQUENCE_SHAPE)):
        raise ArtifactCompatibilityError("Artifact metadata is incompatible.")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_fingerprint(examples: list[TrainingExample]) -> str:
    """Stable identifier for the exact scoped sequences and their labels."""
    digest = hashlib.sha256()
    for example in sorted(examples, key=lambda item: item.sample_id):
        record = {
            "sample_id": example.sample_id,
            "entry_id": example.entry_id,
            "label_id": example.label_id,
            "label": example.label,
            "scope": example.scope,
            "session_id": example.session_id,
            "preprocessing_version": example.preprocessing_version,
        }
        digest.update(json.dumps(record, sort_keys=True, separators=(",", ":")).encode())
        sequence = np.ascontiguousarray(example.sequence, dtype="<f4")
        digest.update(sequence.tobytes())
    return digest.hexdigest()


def train_and_evaluate(
    store: VocabularyStore,
    scope: str,
    seed: int = 42,
    unknown_sequences: list[tuple[str, np.ndarray]] | None = None,
    promote: bool = False,
) -> dict:
    examples = store.load_training_examples(scope)
    if not examples:
        raise InsufficientDataError(f"No saved {scope} examples are available.")
    split = split_by_session(examples, seed=seed)
    if not split.train_indices:
        details = "; ".join(split.insufficiencies)
        raise InsufficientDataError(
            f"Not enough independent sessions to train {scope} scope. {details}"
        )

    labels_by_id = {}
    for example in examples:
        previous = labels_by_id.setdefault(example.label_id, example.label)
        if previous != example.label:
            raise ValueError(f"Conflicting labels for label ID {example.label_id}")
    label_map = [
        {"label_id": label_id, "label": labels_by_id[label_id]}
        for label_id in sorted(labels_by_id)
    ]
    id_to_target = {item["label_id"]: index
                    for index, item in enumerate(label_map)}

    train_examples = [examples[index] for index in split.train_indices]
    train_x = np.stack([temporal_features(item.sequence) for item in train_examples])
    train_y = np.asarray(
        [id_to_target[item.label_id] for item in train_examples], dtype=np.int64
    )
    train_counts = Counter(int(value) for value in train_y)
    missing_train = [
        item["label"] for item in label_map
        if train_counts[id_to_target[item["label_id"]]] == 0
    ]
    if missing_train:
        raise InsufficientDataError(
            "No training samples after session split for: "
            + ", ".join(missing_train)
        )
    if len(label_map) < 2:
        raise InsufficientDataError("At least two vocabulary classes are required.")

    mean, scale = _fit_standardizer(train_x)
    normalized_train = (train_x - mean) / scale
    weights, bias = _fit_softmax(
        normalized_train, train_y, len(label_map), seed
    )

    report = {
        "scope": scope,
        "preprocessing_version": PREPROCESSING_VERSION,
        "classifier": "temporal-pooled-linear-softmax",
        "seed": seed,
        "split": {
            "method": "capture-session-grouped",
            "validation_fraction": VALIDATION_FRACTION,
            "train_sessions": split.train_sessions,
            "validation_sessions": split.validation_sessions,
        },
        "data": {
            "total_samples": len(examples),
            "fingerprint": dataset_fingerprint(examples),
            "train_samples": len(split.train_indices),
            "validation_samples": len(split.validation_indices),
            "class_session_counts": {
                label: len({
                    example.session_id for example in examples
                    if example.label == label
                })
                for label in labels_by_id.values()
            },
        },
        "evaluation_ready": split.evaluation_ready,
        "evaluation_reliability": (
            "session minimums met; validation remains experimental"
            if split.evaluation_ready else "insufficient independent sessions"
        ),
        "insufficiencies": list(split.insufficiencies),
        "validation": None,
        "unknown_evaluation": {
            "available": False,
            "reason": "No separate unknown/background evaluation set supplied.",
            "false_accept_rate": None,
            "abstention_rate": None,
        },
    }
    if split.validation_indices:
        validation_examples = [examples[index] for index in split.validation_indices]
        validation_x = np.stack(
            [temporal_features(item.sequence) for item in validation_examples]
        )
        validation_y = np.asarray(
            [item.label_id for item in validation_examples], dtype=np.int64
        )
        target_y = np.asarray(
            [id_to_target[int(value)] for value in validation_y], dtype=np.int64
        )
        probabilities = _softmax(((validation_x - mean) / scale) @ weights + bias)
        prediction_targets = np.argmax(probabilities, axis=1)
        prediction_ids = np.asarray(
            [label_map[index]["label_id"] for index in prediction_targets],
            dtype=np.int64,
        )
        report["validation"] = _class_metrics(
            validation_y, prediction_ids, label_map
        )
        report["validation"]["abstention_threshold"] = ABSTENTION_THRESHOLD
        abstained = probabilities.max(axis=1) < ABSTENTION_THRESHOLD
        report["validation"]["abstained_count"] = int(np.count_nonzero(abstained))
        report["validation"]["abstention_rate"] = (
            float(abstained.mean()) if len(abstained) else None
        )
    else:
        validation_y = np.empty(0, dtype=np.int64)
        probabilities = np.empty((0, len(label_map)), dtype=np.float32)

    if unknown_sequences:
        unknown_probabilities = []
        unknown_sessions = set()
        unknown_groups: dict[str, list[float]] = defaultdict(list)
        training_sessions = {
            example.session_id for example in examples
        }
        for version, sequence, session_id, category in unknown_sequences:
            if version != PREPROCESSING_VERSION:
                raise SequenceContractError(
                    f"Unknown evaluation version {version!r} is incompatible."
                )
            if not isinstance(session_id, str) or not session_id.strip():
                raise ValueError("Unknown samples require an independent session ID.")
            if session_id in training_sessions:
                raise ValueError(
                    "Unknown/background evaluation sessions must not overlap "
                    "with vocabulary training sessions."
                )
            if category not in {"unknown", "background"}:
                raise ValueError("Unknown sample class must be unknown or background.")
            features = temporal_features(sequence).reshape(1, -1)
            probability = _softmax(((features - mean) / scale) @ weights + bias)[0]
            unknown_probabilities.append(probability)
            unknown_sessions.add(session_id)
            unknown_groups[category].append(float(probability.max()))
        confidence = np.asarray(unknown_probabilities).max(axis=1)
        accepted = confidence >= ABSTENTION_THRESHOLD
        report["unknown_evaluation"] = {
            "available": True,
            "sample_count": int(len(confidence)),
            "independent_session_count": len(unknown_sessions),
            "false_accept_rate": float(accepted.mean()) if len(accepted) else None,
            "abstention_rate": float((~accepted).mean()) if len(accepted) else None,
            "threshold": ABSTENTION_THRESHOLD,
            "per_category": {
                category: {
                    "sample_count": len(scores),
                    "false_accept_rate": float(
                        (np.asarray(scores) >= ABSTENTION_THRESHOLD).mean()
                    ),
                }
                for category, scores in unknown_groups.items()
            },
            "evaluation_reliability": (
                "at least two independent unknown/background sessions"
                if len(unknown_sessions) >= 2
                else "insufficient independent unknown/background sessions"
            ),
            "threshold_note": (
                "Fixed conservative threshold, not calibrated; unknown set must "
                "be independent and labeled by the developer."
            ),
        }

    artifact_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    artifact_dir = _artifact_root(store, scope) / artifact_id
    artifact_dir.mkdir(parents=True, exist_ok=False)
    parameters_path = artifact_dir / "classifier.npz"
    metadata_path = artifact_dir / "metadata.json"
    label_map_path = artifact_dir / "label_map.json"
    preprocessing_path = artifact_dir / "preprocessing.json"
    training_metadata_path = artifact_dir / "training_metadata.json"
    evaluation_path = artifact_dir / "evaluation_report.json"
    temporary_parameters = artifact_dir / f".classifier.{uuid.uuid4().hex}.tmp"
    try:
        with temporary_parameters.open("wb") as handle:
            np.savez(
                handle,
                weights=weights.astype(np.float32),
                bias=bias.astype(np.float32),
                feature_mean=mean.astype(np.float32),
                feature_scale=scale.astype(np.float32),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_parameters, parameters_path)
        metadata = {
            "artifact_schema_version": CLASSIFIER_VERSION,
            "artifact_id": artifact_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "scope": scope,
            "preprocessing_version": PREPROCESSING_VERSION,
            "sequence_shape": list(SEQUENCE_SHAPE),
            "classifier": report["classifier"],
            "feature_definition": {
                "temporal_bins": TEMPORAL_BINS,
                "feature_dimension": int(weights.shape[0]),
                "description": (
                    "five ordered bin means followed by four adjacent-bin "
                    "mean differences"
                ),
            },
            "label_map": [{**item, "scope": scope} for item in label_map],
            "parameters_file": parameters_path.name,
            "parameters_sha256": _sha256(parameters_path),
            "training": {
                "seed": seed,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "l2_strength": L2_STRENGTH,
                "abstention_threshold": ABSTENTION_THRESHOLD,
                "split": report["split"],
                "data": report["data"],
            },
            "evaluation": report,
        }
        _atomic_json(label_map_path, {
            "schema_version": CLASSIFIER_VERSION,
            "scope": scope,
            "label_map": metadata["label_map"],
        })
        _atomic_json(preprocessing_path, {
            "schema_version": CLASSIFIER_VERSION,
            "preprocessing_version": PREPROCESSING_VERSION,
            "sequence_shape": list(SEQUENCE_SHAPE),
            "feature_definition": metadata["feature_definition"],
        })
        _atomic_json(training_metadata_path, {
            "schema_version": CLASSIFIER_VERSION,
            "scope": scope,
            "artifact_id": artifact_id,
            "training": metadata["training"],
        })
        _atomic_json(evaluation_path, {
            "schema_version": CLASSIFIER_VERSION,
            "scope": scope,
            "artifact_id": artifact_id,
            "evaluation": report,
        })
        _atomic_json(metadata_path, metadata)
        if promote:
            activate_artifact(store, scope, artifact_id)
    except Exception:
        shutil.rmtree(artifact_dir, ignore_errors=True)
        raise

    report["artifact"] = {
        "artifact_id": artifact_id,
        "directory": str(artifact_dir),
        "metadata": str(metadata_path),
        "parameters": str(parameters_path),
        "label_map": str(label_map_path),
        "preprocessing": str(preprocessing_path),
        "training_metadata": str(training_metadata_path),
        "evaluation_report": str(evaluation_path),
        "promoted": promote,
    }
    return report


def list_artifacts(store: VocabularyStore, scope: str) -> list[dict]:
    root = _artifact_root(store, scope)
    if not root.is_dir():
        return []
    artifacts = []
    for metadata_path in sorted(root.glob("*/metadata.json")):
        metadata = _load_manifest(metadata_path)
        artifacts.append({
            "artifact_id": metadata["artifact_id"],
            "created_at": metadata["created_at"],
            "scope": metadata["scope"],
            "path": str(metadata_path.parent),
        })
    return artifacts


def activate_artifact(store: VocabularyStore, scope: str, artifact_id: str) -> Path:
    if not re.fullmatch(r"\d{8}T\d{12}Z", artifact_id):
        raise ValueError("Invalid artifact ID.")
    artifact_dir = _artifact_root(store, scope) / artifact_id
    metadata_path = artifact_dir / "metadata.json"
    metadata = _load_manifest(metadata_path)
    if metadata.get("scope") != scope:
        raise ArtifactCompatibilityError("Artifact belongs to a different scope.")
    parameters_path = artifact_dir / metadata["parameters_file"]
    if _sha256(parameters_path) != metadata.get("parameters_sha256"):
        raise ArtifactCompatibilityError("Artifact parameter checksum mismatch.")
    pointer = _artifact_root(store, scope) / "active.json"
    _atomic_json(
        pointer,
        {
            "artifact_schema_version": CLASSIFIER_VERSION,
            "scope": scope,
            "artifact_id": artifact_id,
        },
    )
    return artifact_dir


class ExperimentalClassifier:
    def __init__(self, scope: str, metadata: dict, parameters: dict):
        self.scope = scope
        self.metadata = metadata
        self.label_map = metadata["label_map"]
        self.weights = parameters["weights"]
        self.bias = parameters["bias"]
        self.feature_mean = parameters["feature_mean"]
        self.feature_scale = parameters["feature_scale"]

    def predict(self, sequence) -> dict:
        features = temporal_features(sequence).reshape(1, -1)
        normalized = (features - self.feature_mean) / self.feature_scale
        probabilities = _softmax(normalized @ self.weights + self.bias)[0]
        best_index = int(np.argmax(probabilities))
        confidence = float(probabilities[best_index])
        abstain = confidence < float(
            self.metadata["training"]["abstention_threshold"]
        )
        return {
            "label": None if abstain else self.label_map[best_index]["label"],
            "label_id": (
                None if abstain else self.label_map[best_index]["label_id"]
            ),
            "confidence": confidence,
            "abstained": abstain,
            "scope": self.scope,
            "preprocessing_version": PREPROCESSING_VERSION,
            "artifact_id": self.metadata["artifact_id"],
        }


def load_artifact(
    store: VocabularyStore,
    scope: str,
    artifact_id: str | None = None,
) -> ExperimentalClassifier:
    root = _artifact_root(store, scope)
    if artifact_id is None:
        pointer_path = root / "active.json"
        try:
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ArtifactCompatibilityError(
                f"No active artifact is configured for {scope} scope."
            ) from error
        if pointer.get("scope") != scope:
            raise ArtifactCompatibilityError("Active artifact scope mismatch.")
        artifact_id = pointer.get("artifact_id")
    if not isinstance(artifact_id, str) or not re.fullmatch(
        r"\d{8}T\d{12}Z", artifact_id
    ):
        raise ArtifactCompatibilityError("Invalid artifact ID.")
    artifact_dir = root / artifact_id
    metadata = _load_manifest(artifact_dir / "metadata.json")
    if metadata.get("scope") != scope:
        raise ArtifactCompatibilityError("Artifact belongs to a different scope.")
    parameters_path = artifact_dir / metadata["parameters_file"]
    if _sha256(parameters_path) != metadata.get("parameters_sha256"):
        raise ArtifactCompatibilityError("Artifact parameter checksum mismatch.")
    try:
        with np.load(parameters_path, allow_pickle=False) as archive:
            parameters = {name: archive[name] for name in archive.files}
    except (OSError, ValueError) as error:
        raise ArtifactCompatibilityError(
            f"Cannot load model parameters: {error}"
        ) from error
    expected_dimension = metadata["feature_definition"]["feature_dimension"]
    class_count = len(metadata["label_map"])
    if (parameters["weights"].shape != (expected_dimension, class_count)
            or parameters["bias"].shape != (class_count,)
            or parameters["feature_mean"].shape != (expected_dimension,)
            or parameters["feature_scale"].shape != (expected_dimension,)
            or not all(np.isfinite(value).all() for value in parameters.values())):
        raise ArtifactCompatibilityError("Model parameter shapes or values are invalid.")
    if any(item.get("scope") != scope for item in metadata["label_map"]):
        raise ArtifactCompatibilityError("Artifact label map scope mismatch.")
    return ExperimentalClassifier(scope, metadata, parameters)


def load_unknown_sequences(
    path: Path,
) -> list[tuple[str, np.ndarray, str, str]]:
    """Load a JSON manifest of independently collected unknown/background arrays."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read unknown-set manifest: {error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise ValueError("Unknown-set manifest must contain a samples list.")
    sequences = []
    for index, sample in enumerate(payload["samples"]):
        if not isinstance(sample, dict):
            raise ValueError(f"Unknown sample {index} must be an object.")
        version = sample.get("preprocessing_version")
        if version != PREPROCESSING_VERSION:
            raise SequenceContractError(
                f"Unknown sample {index} has incompatible preprocessing version."
            )
        file_path = (path.parent / sample.get("file", "")).resolve()
        if path.parent.resolve() not in file_path.parents:
            raise ValueError("Unknown sample path escapes the manifest directory.")
        sequence = np.load(file_path, allow_pickle=False)
        session_id = sample.get("session_id")
        category = sample.get("class")
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError(f"Unknown sample {index} requires a session_id.")
        if category not in {"unknown", "background"}:
            raise ValueError(
                f"Unknown sample {index} class must be unknown or background."
            )
        sequences.append(
            (version, validate_sequence(sequence, version), session_id, category)
        )
    return sequences


def _store_from_args(args) -> VocabularyStore:
    root = Path(args.store).resolve()
    return VocabularyStore(root)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train/evaluate an isolated expandable-vocabulary classifier."
    )
    parser.add_argument(
        "--store",
        default=str(Path(__file__).resolve().parents[1] / "data" / "vocabulary_drafts"),
        help="VocabularyStore root (shared and personal scopes remain isolated).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train and evaluate by capture session.")
    train.add_argument("--scope", choices=("shared", "personal"), required=True)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--unknown-manifest", type=Path)
    train.add_argument("--promote", action="store_true",
                       help="Set this scope's active artifact pointer after saving.")

    evaluate = subparsers.add_parser("evaluate", help="Report saved validation metrics.")
    evaluate.add_argument("--scope", choices=("shared", "personal"), required=True)
    evaluate.add_argument("--artifact-id")

    infer = subparsers.add_parser("infer", help="Predict one saved sequence.")
    infer.add_argument("--scope", choices=("shared", "personal"), required=True)
    infer.add_argument("--sequence", type=Path, required=True)
    infer.add_argument("--preprocessing-version", default=PREPROCESSING_VERSION)
    infer.add_argument("--artifact-id")

    activate = subparsers.add_parser("activate", help="Select an older experimental artifact.")
    activate.add_argument("--scope", choices=("shared", "personal"), required=True)
    activate.add_argument("--artifact-id", required=True)

    listing = subparsers.add_parser("list", help="List immutable scoped artifacts.")
    listing.add_argument("--scope", choices=("shared", "personal"), required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    store = _store_from_args(args)
    try:
        if args.command == "train":
            unknown = (
                load_unknown_sequences(args.unknown_manifest)
                if args.unknown_manifest else None
            )
            report = train_and_evaluate(
                store,
                args.scope,
                seed=args.seed,
                unknown_sequences=unknown,
                promote=args.promote,
            )
            print(json.dumps(report, indent=2, ensure_ascii=False))
            if not report["evaluation_ready"]:
                return 2
        elif args.command == "evaluate":
            if args.artifact_id:
                artifact = load_artifact(store, args.scope, args.artifact_id)
                print(json.dumps(artifact.metadata["evaluation"], indent=2))
            else:
                models_root = _artifact_root(store, args.scope)
                artifacts = list_artifacts(store, args.scope)
                if not artifacts:
                    raise ArtifactCompatibilityError("No artifacts exist for this scope.")
                latest = artifacts[-1]
                metadata = _load_manifest(
                    Path(latest["path"]) / "metadata.json"
                )
                print(json.dumps(metadata["evaluation"], indent=2))
        elif args.command == "infer":
            classifier = load_artifact(store, args.scope, args.artifact_id)
            sequence = np.load(args.sequence, allow_pickle=False)
            validate_sequence(sequence, args.preprocessing_version)
            print(json.dumps(classifier.predict(sequence), indent=2))
        elif args.command == "activate":
            path = activate_artifact(store, args.scope, args.artifact_id)
            print(f"Active {args.scope} artifact: {path}")
        elif args.command == "list":
            print(json.dumps(list_artifacts(store, args.scope), indent=2))
        return 0
    except (InsufficientDataError, ArtifactCompatibilityError,
            SequenceContractError, ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
