"""Versioned, scope-isolated persistence for expandable-vocabulary data."""

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
    validate_label_id,
    validate_sequence,
)

SCHEMA_VERSION = 1
SCOPES = frozenset({"shared", "personal"})
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _validate_id(value: str, field: str) -> str:
    if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value):
        raise ValueError(f"{field} must be a safe non-empty identifier")
    return value


def validate_entry_id(value: str) -> str:
    """Validate the public entry identifier received by API routes."""
    return _validate_id(value, "entry_id")


def _validate_scope(scope: str) -> str:
    if not isinstance(scope, str) or scope not in SCOPES:
        raise ValueError(f"scope must be one of {sorted(SCOPES)}")
    return scope


def _validate_timestamp(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("created_at must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("created_at must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError("created_at must include a timezone")
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class VocabularyEntry:
    entry_id: str
    label_id: int
    label: str
    scope: str
    preprocessing_version: str = PREPROCESSING_VERSION
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        _validate_id(self.entry_id, "entry_id")
        validate_label_id(self.label_id)
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be a non-empty string")
        _validate_scope(self.scope)
        if self.preprocessing_version != PREPROCESSING_VERSION:
            raise ValueError("Unsupported vocabulary preprocessing version")
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError("Unsupported vocabulary schema version")


@dataclass(frozen=True)
class CaptureSession:
    session_id: str
    entry_id: str
    scope: str
    created_at: str
    preprocessing_version: str = PREPROCESSING_VERSION
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def create(cls, entry: VocabularyEntry) -> "CaptureSession":
        return cls(
            session_id=uuid.uuid4().hex,
            entry_id=entry.entry_id,
            scope=entry.scope,
            created_at=_now(),
            preprocessing_version=entry.preprocessing_version,
        )

    def __post_init__(self):
        _validate_id(self.session_id, "session_id")
        _validate_id(self.entry_id, "entry_id")
        _validate_scope(self.scope)
        _validate_timestamp(self.created_at)
        if self.preprocessing_version != PREPROCESSING_VERSION:
            raise ValueError("Unsupported capture-session preprocessing version")
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError("Unsupported capture-session schema version")


@dataclass(frozen=True)
class Sample:
    sample_id: str
    session_id: str
    entry_id: str
    label_id: int
    scope: str
    file_name: str
    sha256: str
    created_at: str
    shape: tuple[int, int] = SEQUENCE_SHAPE
    preprocessing_version: str = PREPROCESSING_VERSION
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        _validate_id(self.sample_id, "sample_id")
        _validate_id(self.session_id, "session_id")
        _validate_id(self.entry_id, "entry_id")
        validate_label_id(self.label_id)
        _validate_scope(self.scope)
        if not isinstance(self.file_name, str):
            raise ValueError("file_name must be a string")
        _validate_id(self.file_name.removesuffix(".npy"), "file_name")
        if self.file_name != f"{self.sample_id}.npy":
            raise ValueError("file_name must be the sample ID with a .npy suffix")
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("sha256 must be a lowercase SHA-256 digest")
        _validate_timestamp(self.created_at)
        if tuple(self.shape) != SEQUENCE_SHAPE:
            raise ValueError(f"Sample shape must be {SEQUENCE_SHAPE}")
        object.__setattr__(self, "shape", SEQUENCE_SHAPE)
        if self.preprocessing_version != PREPROCESSING_VERSION:
            raise ValueError("Unsupported sample preprocessing version")
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError("Unsupported sample schema version")


@dataclass(frozen=True)
class TrainingExample:
    sequence: np.ndarray
    label_id: int
    label: str
    scope: str
    entry_id: str
    session_id: str
    sample_id: str
    preprocessing_version: str


def _load_records(path: Path, record_type):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if (not isinstance(payload, dict)
            or type(payload.get("schema_version")) is not int
            or payload.get("schema_version") != SCHEMA_VERSION):
        raise ValueError(f"Unsupported or invalid manifest: {path}")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"Invalid records list in manifest: {path}")
    return [record_type(**record) for record in records]


def _write_records(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
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
        if temporary.exists():
            temporary.unlink()


class VocabularyStore:
    """Store shared and personal records in distinct manifests and directories."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self._lock = threading.RLock()

    def _scope_root(self, scope: str) -> Path:
        return self.root / _validate_scope(scope)

    def _manifest(self, scope: str, kind: str) -> Path:
        return self._scope_root(scope) / f"{kind}.json"

    def list_entries(self, scope: str) -> list[VocabularyEntry]:
        return _load_records(self._manifest(scope, "vocabulary"), VocabularyEntry)

    def add_entry(self, entry: VocabularyEntry) -> None:
        with self._lock:
            entries = self.list_entries(entry.scope)
            if any(old.entry_id == entry.entry_id for old in entries):
                raise ValueError(f"Vocabulary entry already exists: {entry.entry_id}")
            if any(old.label_id == entry.label_id for old in entries):
                raise ValueError(f"Label ID already exists in {entry.scope} scope")
            _write_records(
                self._manifest(entry.scope, "vocabulary"), entries + [entry]
            )

    def create_session(self, session: CaptureSession) -> None:
        with self._lock:
            entries = self.list_entries(session.scope)
            if not any(entry.entry_id == session.entry_id for entry in entries):
                raise ValueError("Capture session must reference an entry in its scope")
            sessions = _load_records(
                self._manifest(session.scope, "sessions"), CaptureSession
            )
            if any(old.session_id == session.session_id for old in sessions):
                raise ValueError(f"Capture session already exists: {session.session_id}")
            _write_records(
                self._manifest(session.scope, "sessions"), sessions + [session]
            )

    def list_sessions(self, scope: str) -> list[CaptureSession]:
        return _load_records(self._manifest(scope, "sessions"), CaptureSession)

    def save_sample(
        self,
        session_id: str,
        entry_id: str,
        sequence,
        preprocessing_version: str,
        scope: str,
    ) -> Sample:
        with self._lock:
            return self._save_sample(
                session_id,
                entry_id,
                sequence,
                preprocessing_version,
                scope,
            )

    def _save_sample(
        self,
        session_id: str,
        entry_id: str,
        sequence,
        preprocessing_version: str,
        scope: str,
    ) -> Sample:
        scope_root = self._scope_root(scope)
        sessions = _load_records(
            self._manifest(scope, "sessions"), CaptureSession
        )
        session = next((item for item in sessions
                        if item.session_id == session_id), None)
        if session is None:
            raise ValueError("Capture session does not exist in the requested scope")
        if session.entry_id != entry_id:
            raise ValueError("Sample vocabulary entry does not match its session")
        entries = self.list_entries(scope)
        entry = next((item for item in entries if item.entry_id == entry_id), None)
        if entry is None:
            raise ValueError("Vocabulary entry does not exist in the requested scope")
        if session.preprocessing_version != preprocessing_version:
            raise ValueError("Sample preprocessing version does not match its session")
        array = validate_sequence(sequence, preprocessing_version)

        manifest_path = self._manifest(scope, "samples")
        samples = _load_records(manifest_path, Sample)
        sample_id = uuid.uuid4().hex
        sample_path = scope_root / "samples" / f"{sample_id}.npy"
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = sample_path.with_name(f".{sample_id}.tmp")
        try:
            with temporary.open("xb") as handle:
                np.save(handle, array, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, sample_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        committed = False
        try:
            sample = Sample(
                sample_id=sample_id,
                session_id=session_id,
                entry_id=entry_id,
                label_id=entry.label_id,
                scope=scope,
                file_name=sample_path.name,
                sha256=hashlib.sha256(sample_path.read_bytes()).hexdigest(),
                created_at=_now(),
            )
            _write_records(manifest_path, samples + [sample])
            committed = True
        finally:
            if not committed:
                sample_path.unlink(missing_ok=True)
        return sample

    def list_samples(self, scope: str) -> list[Sample]:
        return _load_records(self._manifest(scope, "samples"), Sample)

    def delete_samples(
        self, entry_id: str, sample_ids: list[str], scope: str
    ) -> list[Sample]:
        with self._lock:
            _validate_scope(scope)
            _validate_id(entry_id, "entry_id")
            if not isinstance(sample_ids, list) or not sample_ids:
                raise ValueError("At least one sample ID is required")
            for sample_id in sample_ids:
                _validate_id(sample_id, "sample_id")
            if len(set(sample_ids)) != len(sample_ids):
                raise ValueError("Sample IDs must be unique")
            requested_ids = set(sample_ids)

            if not any(
                entry.entry_id == entry_id for entry in self.list_entries(scope)
            ):
                raise ValueError("Vocabulary entry does not exist in the requested scope")

            manifest_path = self._manifest(scope, "samples")
            samples = _load_records(manifest_path, Sample)
            targets = [
                sample for sample in samples
                if sample.sample_id in requested_ids
            ]
            if (len(targets) != len(sample_ids)
                    or any(sample.entry_id != entry_id for sample in targets)):
                raise ValueError(
                    "Every sample ID must belong to the requested vocabulary entry and scope"
                )

            sample_dir = self._scope_root(scope) / "samples"
            staged = []
            try:
                for sample in targets:
                    source = sample_dir / sample.file_name
                    if not source.is_file():
                        raise ValueError(f"Sample file is missing: {sample.sample_id}")
                    tombstone = source.with_name(
                        f".{sample.file_name}.{uuid.uuid4().hex}.delete"
                    )
                    os.replace(source, tombstone)
                    staged.append((source, tombstone))
                _write_records(
                    manifest_path,
                    [sample for sample in samples
                     if sample.sample_id not in requested_ids],
                )
            except Exception:
                for source, tombstone in reversed(staged):
                    if tombstone.exists():
                        os.replace(tombstone, source)
                raise

            for _, tombstone in staged:
                try:
                    tombstone.unlink()
                except OSError as error:
                    raise OSError(
                        "Sample records were removed, but a sample file could not be fully deleted."
                    ) from error
            return targets

    def delete_entry(self, entry_id: str, scope: str) -> dict[str, int | str]:
        """Delete one vocabulary entry and only its scoped sessions and samples."""
        with self._lock:
            _validate_scope(scope)
            _validate_id(entry_id, "entry_id")

            vocabulary_path = self._manifest(scope, "vocabulary")
            sessions_path = self._manifest(scope, "sessions")
            samples_path = self._manifest(scope, "samples")
            entries = _load_records(vocabulary_path, VocabularyEntry)
            sessions = _load_records(sessions_path, CaptureSession)
            samples = _load_records(samples_path, Sample)
            entry = next(
                (item for item in entries if item.entry_id == entry_id), None
            )
            if entry is None:
                raise ValueError("Vocabulary entry does not exist in the requested scope")

            target_sessions = [
                item for item in sessions if item.entry_id == entry_id
            ]
            target_session_ids = {item.session_id for item in target_sessions}
            target_samples = [
                item for item in samples
                if item.entry_id == entry_id or item.session_id in target_session_ids
            ]
            if any(
                item.entry_id != entry_id or item.scope != scope
                or item.session_id not in target_session_ids
                for item in target_samples
            ):
                raise ValueError(
                    "Draft references are inconsistent; no vocabulary data was deleted"
                )

            sample_dir = self._scope_root(scope) / "samples"
            staged = []
            for sample in target_samples:
                source = sample_dir / sample.file_name
                if not source.is_file():
                    raise ValueError(f"Sample file is missing: {sample.sample_id}")

            try:
                for sample in target_samples:
                    source = sample_dir / sample.file_name
                    tombstone = source.with_name(
                        f".{sample.file_name}.{uuid.uuid4().hex}.delete"
                    )
                    os.replace(source, tombstone)
                    staged.append((source, tombstone))

                _write_records(
                    samples_path,
                    [item for item in samples if item not in target_samples],
                )
                _write_records(
                    sessions_path,
                    [item for item in sessions if item.entry_id != entry_id],
                )
                _write_records(
                    vocabulary_path,
                    [item for item in entries if item.entry_id != entry_id],
                )
            except Exception:
                _write_records(samples_path, samples)
                _write_records(sessions_path, sessions)
                _write_records(vocabulary_path, entries)
                for source, tombstone in reversed(staged):
                    if tombstone.exists():
                        os.replace(tombstone, source)
                raise

            for _, tombstone in staged:
                try:
                    tombstone.unlink()
                except OSError as error:
                    raise OSError(
                        "Vocabulary records were removed, but a sample file could not be fully deleted."
                    ) from error

            return {
                "entry_id": entry.entry_id,
                "label": entry.label,
                "deleted_sessions": len(target_sessions),
                "deleted_samples": len(target_samples),
            }

    def load_training_examples(self, scope: str) -> list[TrainingExample]:
        _validate_scope(scope)
        with self._lock:
            entries = {entry.entry_id: entry for entry in self.list_entries(scope)}
            sessions = {
                session.session_id: session
                for session in self.list_sessions(scope)
            }
            examples = []
            for sample in self.list_samples(scope):
                if sample.scope != scope:
                    raise ValueError(
                        f"Sample {sample.sample_id} is registered in the wrong scope"
                    )
                entry = entries.get(sample.entry_id)
                session = sessions.get(sample.session_id)
                if entry is None or session is None:
                    raise ValueError(
                        f"Sample {sample.sample_id} references a missing entry or session"
                    )
                if (entry.label_id != sample.label_id
                        or session.entry_id != sample.entry_id
                        or session.scope != scope
                        or session.preprocessing_version != sample.preprocessing_version):
                    raise ValueError(
                        f"Sample {sample.sample_id} metadata is inconsistent"
                    )
                examples.append(
                    TrainingExample(
                        sequence=self.load_sample(sample, scope),
                        label_id=entry.label_id,
                        label=entry.label,
                        scope=scope,
                        entry_id=entry.entry_id,
                        session_id=session.session_id,
                        sample_id=sample.sample_id,
                        preprocessing_version=sample.preprocessing_version,
                    )
                )
            return examples

    def count_samples(self, scope: str, entry_id: str) -> int:
        _validate_id(entry_id, "entry_id")
        return sum(sample.entry_id == entry_id for sample in self.list_samples(scope))

    def load_sample(self, sample: Sample, scope: str) -> np.ndarray:
        _validate_scope(scope)
        if sample.scope != scope:
            raise ValueError("Sample cannot be loaded from a different scope")
        if sample not in self.list_samples(scope):
            raise ValueError(f"Sample is not registered in {scope} scope")
        path = self._scope_root(scope) / "samples" / sample.file_name
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != sample.sha256:
            raise ValueError(f"Sample checksum mismatch: {sample.sample_id}")
        sequence = np.load(path, allow_pickle=False)
        return validate_sequence(sequence, sample.preprocessing_version)
