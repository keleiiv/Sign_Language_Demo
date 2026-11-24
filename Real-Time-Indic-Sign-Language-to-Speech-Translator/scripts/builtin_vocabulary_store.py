"""Isolated 30x378 capture storage for immutable built-in vocabulary labels.

This store intentionally does not create VocabularyEntry records and never writes
to the legacy ``data/extracted_features`` tree.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import uuid

import numpy as np

from scripts.sequence_contract import (
    PREPROCESSING_VERSION,
    SEQUENCE_SHAPE,
    validate_sequence,
)

SCHEMA_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_timestamp(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("created_at must be a timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("created_at must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError("created_at must include a timezone")
    return value


@dataclass(frozen=True)
class BuiltinCaptureSession:
    session_id: str
    label: str
    created_at: str
    preprocessing_version: str = PREPROCESSING_VERSION
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        if not isinstance(self.session_id, str) or not self.session_id.isalnum():
            raise ValueError("session_id must be a safe identifier")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be non-empty")
        _validate_timestamp(self.created_at)
        if self.preprocessing_version != PREPROCESSING_VERSION:
            raise ValueError("Unsupported built-in capture preprocessing version")
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError("Unsupported built-in capture schema version")


@dataclass(frozen=True)
class BuiltinSample:
    sample_id: str
    session_id: str
    label: str
    file_name: str
    sha256: str
    created_at: str
    shape: tuple[int, int] = SEQUENCE_SHAPE
    preprocessing_version: str = PREPROCESSING_VERSION
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        for field_name, value in (("sample_id", self.sample_id), ("session_id", self.session_id)):
            if not isinstance(value, str) or not value.isalnum():
                raise ValueError(f"{field_name} must be a safe identifier")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be non-empty")
        if self.file_name != f"{self.sample_id}.npy":
            raise ValueError("file_name must match sample_id")
        if not isinstance(self.sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("sha256 must be a SHA-256 digest")
        _validate_timestamp(self.created_at)
        if tuple(self.shape) != SEQUENCE_SHAPE:
            raise ValueError(f"Sample shape must be {SEQUENCE_SHAPE}")
        object.__setattr__(self, "shape", SEQUENCE_SHAPE)
        if self.preprocessing_version != PREPROCESSING_VERSION:
            raise ValueError("Unsupported sample preprocessing version")
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError("Unsupported built-in sample schema version")


def _load_records(path: Path, record_type):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if (not isinstance(payload, dict)
            or type(payload.get("schema_version")) is not int
            or payload["schema_version"] != SCHEMA_VERSION
            or not isinstance(payload.get("records"), list)):
        raise ValueError(f"Invalid built-in dataset manifest: {path}")
    return [record_type(**record) for record in payload["records"]]


def _write_records(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(
                {"schema_version": SCHEMA_VERSION,
                 "records": [asdict(record) for record in records]},
                handle,
                ensure_ascii=False,
                indent=2,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class BuiltinVocabularyStore:
    """Append-only sequence storage keyed by canonical built-in label."""

    def __init__(self, root: Path | str, labels_provider):
        self.root = Path(root)
        self.labels_provider = labels_provider
        self._lock = threading.RLock()

    @property
    def sessions_path(self) -> Path:
        return self.root / "shared" / "sessions.json"

    @property
    def samples_path(self) -> Path:
        return self.root / "shared" / "samples.json"

    @property
    def sample_dir(self) -> Path:
        return self.root / "shared" / "samples"

    def canonical_label(self, requested: str) -> str:
        if not isinstance(requested, str) or not requested.strip():
            raise ValueError("Select a built-in vocabulary label.")
        labels = self.labels_provider()
        matches = [label for label in labels
                   if isinstance(label, str)
                   and label.casefold() == requested.strip().casefold()]
        if not matches:
            raise ValueError("Label is not in the built-in vocabulary.")
        if len(matches) != 1:
            raise ValueError("Built-in vocabulary contains ambiguous case-insensitive labels.")
        return matches[0]

    def list_sessions(self) -> list[BuiltinCaptureSession]:
        sessions = _load_records(self.sessions_path, BuiltinCaptureSession)
        for session in sessions:
            if self.canonical_label(session.label) != session.label:
                raise ValueError("Built-in session label is not canonical.")
        return sessions

    def list_samples(self, label: str | None = None) -> list[BuiltinSample]:
        samples = _load_records(self.samples_path, BuiltinSample)
        sessions = {session.session_id: session for session in self.list_sessions()}
        for sample in samples:
            canonical = self.canonical_label(sample.label)
            session = sessions.get(sample.session_id)
            if canonical != sample.label or session is None or session.label != sample.label:
                raise ValueError("Built-in sample references an invalid label or session.")
        if label is None:
            return samples
        canonical = self.canonical_label(label)
        return [sample for sample in samples if sample.label == canonical]

    def create_session(self, label: str) -> BuiltinCaptureSession:
        canonical = self.canonical_label(label)
        with self._lock:
            sessions = self.list_sessions()
            session = BuiltinCaptureSession(
                session_id=uuid.uuid4().hex,
                label=canonical,
                created_at=_now(),
            )
            _write_records(self.sessions_path, sessions + [session])
            return session

    def save_sample(
        self,
        session_id: str,
        sequence,
        preprocessing_version: str = PREPROCESSING_VERSION,
    ) -> BuiltinSample:
        array = validate_sequence(sequence, preprocessing_version)
        with self._lock:
            sessions = self.list_sessions()
            session = next((item for item in sessions
                            if item.session_id == session_id), None)
            if session is None:
                raise ValueError("Built-in capture session does not exist.")
            samples = self.list_samples()
            sample_id = uuid.uuid4().hex
            sample_path = self.sample_dir / f"{sample_id}.npy"
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            created = False
            try:
                with sample_path.open("xb") as handle:
                    created = True
                    np.save(handle, array, allow_pickle=False)
                    handle.flush()
                    os.fsync(handle.fileno())
                record = BuiltinSample(
                    sample_id=sample_id,
                    session_id=session_id,
                    label=session.label,
                    file_name=sample_path.name,
                    sha256=hashlib.sha256(sample_path.read_bytes()).hexdigest(),
                    created_at=_now(),
                )
                _write_records(self.samples_path, samples + [record])
                return record
            except Exception:
                if created:
                    sample_path.unlink(missing_ok=True)
                raise

    def load_sample(self, sample: BuiltinSample) -> np.ndarray:
        if sample not in self.list_samples():
            raise ValueError("Sample is not registered in the built-in dataset.")
        path = self.sample_dir / sample.file_name
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != sample.sha256:
            raise ValueError(f"Sample integrity check failed: {sample.sample_id}")
        with path.open("rb") as handle:
            return validate_sequence(
                np.load(handle, allow_pickle=False), sample.preprocessing_version
            )

    def delete_label_samples(self, label: str) -> dict:
        """Delete only one built-in label's samples and sessions, with rollback."""
        canonical = self.canonical_label(label)
        with self._lock:
            sessions = self.list_sessions()
            samples = self.list_samples()
            target_sessions = [item for item in sessions if item.label == canonical]
            target_session_ids = {item.session_id for item in target_sessions}
            target_samples = [
                item for item in samples
                if item.label == canonical or item.session_id in target_session_ids
            ]

            originals: dict[str, bytes] = {}
            for sample in target_samples:
                path = self.sample_dir / sample.file_name
                if not path.is_file():
                    raise ValueError(f"Sample file is missing: {sample.sample_id}")
                payload = path.read_bytes()
                if hashlib.sha256(payload).hexdigest() != sample.sha256:
                    raise ValueError(f"Sample integrity check failed: {sample.sample_id}")
                originals[sample.sample_id] = payload

            moved: list[tuple[Path, Path]] = []
            try:
                for sample in target_samples:
                    source = self.sample_dir / sample.file_name
                    tombstone = source.with_name(
                        f".{sample.sample_id}.{uuid.uuid4().hex}.delete"
                    )
                    os.replace(source, tombstone)
                    moved.append((source, tombstone))

                target_ids = {sample.sample_id for sample in target_samples}
                _write_records(
                    self.samples_path,
                    [sample for sample in samples if sample.sample_id not in target_ids],
                )
                _write_records(
                    self.sessions_path,
                    [session for session in sessions if session.label != canonical],
                )
                for _, tombstone in moved:
                    tombstone.unlink()
            except Exception as error:
                rollback_errors = []
                for source, tombstone in reversed(moved):
                    try:
                        if tombstone.exists():
                            if source.exists():
                                raise OSError(f"Refusing to overwrite existing sample: {source.name}")
                            os.replace(tombstone, source)
                        elif not source.exists():
                            sample_id = source.stem
                            recovery = source.with_name(f".{sample_id}.{uuid.uuid4().hex}.restore")
                            try:
                                with recovery.open("xb") as handle:
                                    handle.write(originals[sample_id])
                                    handle.flush()
                                    os.fsync(handle.fileno())
                                os.replace(recovery, source)
                            finally:
                                recovery.unlink(missing_ok=True)
                    except OSError as rollback_error:
                        rollback_errors.append(str(rollback_error))
                try:
                    _write_records(self.samples_path, samples)
                    _write_records(self.sessions_path, sessions)
                except Exception as rollback_error:
                    rollback_errors.append(str(rollback_error))
                if rollback_errors:
                    raise OSError(
                        "Built-in sample deletion failed and rollback needs attention: "
                        + "; ".join(rollback_errors)
                    ) from error
                raise OSError(
                    "Built-in sample deletion failed; the selected label's data was restored."
                ) from error

            return {
                "label": canonical,
                "deleted_samples": len(target_samples),
                "deleted_sessions": len(target_sessions),
            }

    def label_summary(self) -> list[dict]:
        with self._lock:
            sessions = self.list_sessions()
            samples = self.list_samples()
            result = []
            for label in self.labels_provider():
                label_samples = [sample for sample in samples if sample.label == label]
                sample_session_ids = {sample.session_id for sample in label_samples}
                result.append({
                    "label": label,
                    "sample_count": len(label_samples),
                    "session_count": len(sample_session_ids),
                    "capture_session_count": sum(
                        session.label == label for session in sessions
                    ),
                })
            return result
