"""Tests for isolated capture storage of existing built-in vocabulary labels."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.builtin_vocabulary_store import BuiltinVocabularyStore
from scripts.sequence_contract import PREPROCESSING_VERSION, SEQUENCE_SHAPE


def valid_sequence():
    sequence = np.zeros(SEQUENCE_SHAPE, dtype=np.float32)
    sequence[:, 252:315] = 0.25
    sequence[:, 315:378] = 0.5
    return sequence


class BuiltinVocabularyStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.store = BuiltinVocabularyStore(
            Path(self.temp_dir.name) / "builtin-store",
            labels_provider=lambda: ["Doctor", "Teacher"],
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_doctor_samples_append_and_keep_session_and_contract_metadata(self):
        first_session = self.store.create_session("doctor")
        first = self.store.save_sample(first_session.session_id, valid_sequence())
        first_path = self.store.sample_dir / first.file_name
        original_bytes = first_path.read_bytes()
        second = self.store.save_sample(first_session.session_id, valid_sequence())

        samples = self.store.list_samples("Doctor")
        summary = {item["label"]: item for item in self.store.label_summary()}
        self.assertEqual(first.label, "Doctor")
        self.assertEqual(second.label, "Doctor")
        self.assertEqual(first.sample_id != second.sample_id, True)
        self.assertEqual(hashlib.sha256(first_path.read_bytes()).digest(),
                         hashlib.sha256(original_bytes).digest())
        self.assertEqual(len(samples), 2)
        self.assertEqual(summary["Doctor"]["sample_count"], 2)
        self.assertEqual(summary["Doctor"]["session_count"], 1)
        self.assertEqual(self.store.load_sample(first).shape, (30, 378))
        self.assertEqual(first.preprocessing_version, PREPROCESSING_VERSION)
        self.assertFalse((self.store.root / "Doctor").exists())
        reopened = BuiltinVocabularyStore(
            self.store.root, labels_provider=lambda: ["Doctor", "Teacher"]
        )
        self.assertEqual(len(reopened.list_samples("doctor")), 2)
        self.assertEqual(reopened.load_sample(second).shape, (30, 378))

    def test_labels_without_samples_are_reported_without_becoming_samples(self):
        summary = {item["label"]: item for item in self.store.label_summary()}
        self.assertEqual(summary["Doctor"]["sample_count"], 0)
        self.assertEqual(summary["Teacher"]["sample_count"], 0)
        self.assertEqual(self.store.list_samples(), [])

    def test_unknown_label_wrong_shape_and_missing_session_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "not in the built-in"):
            self.store.create_session("new label")
        session = self.store.create_session("Doctor")
        with self.assertRaises(ValueError):
            self.store.save_sample(session.session_id, np.zeros((29, 378)))
        with self.assertRaisesRegex(ValueError, "session does not exist"):
            self.store.save_sample("missing-session", valid_sequence())

    def test_four_session_eligibility_uses_independent_sessions_with_samples(self):
        for expected in range(1, 5):
            session = self.store.create_session("Doctor")
            self.store.save_sample(session.session_id, valid_sequence())
            doctor = next(item for item in self.store.label_summary()
                          if item["label"] == "Doctor")
            self.assertEqual(doctor["session_count"], expected)
            self.assertEqual(doctor["sample_count"], expected)
            self.assertEqual(doctor["session_count"] >= 4, expected == 4)

    def test_delete_removes_only_target_label_and_related_sessions(self):
        doctor_session = self.store.create_session("Doctor")
        doctor_sample = self.store.save_sample(doctor_session.session_id, valid_sequence())
        teacher_session = self.store.create_session("Teacher")
        teacher_sample = self.store.save_sample(teacher_session.session_id, valid_sequence())

        result = self.store.delete_label_samples("doctor")
        self.assertEqual(result["deleted_samples"], 1)
        self.assertEqual(result["deleted_sessions"], 1)
        self.assertFalse((self.store.sample_dir / doctor_sample.file_name).exists())
        self.assertEqual(self.store.load_sample(teacher_sample).shape, (30, 378))
        counts = {item["label"]: item for item in self.store.label_summary()}
        self.assertEqual(counts["Doctor"]["sample_count"], 0)
        self.assertEqual(counts["Doctor"]["capture_session_count"], 0)
        self.assertEqual(counts["Teacher"]["sample_count"], 1)
        self.assertEqual(counts["Teacher"]["session_count"], 1)

    def test_delete_failure_rolls_back_files_and_manifests(self):
        session = self.store.create_session("Doctor")
        samples = [
            self.store.save_sample(session.session_id, valid_sequence()),
            self.store.save_sample(session.session_id, valid_sequence()),
        ]
        original_bytes = {
            sample.sample_id: (self.store.sample_dir / sample.file_name).read_bytes()
            for sample in samples
        }
        original_unlink = Path.unlink
        tombstones_removed = 0

        def fail_tombstone_unlink(path, *args, **kwargs):
            nonlocal tombstones_removed
            if path.name.endswith(".delete"):
                tombstones_removed += 1
                if tombstones_removed == 2:
                    raise OSError("simulated deletion failure after one removal")
            return original_unlink(path, *args, **kwargs)

        with patch.object(Path, "unlink", fail_tombstone_unlink):
            with self.assertRaisesRegex(OSError, "data was restored"):
                self.store.delete_label_samples("Doctor")

        for sample in samples:
            self.assertEqual(
                (self.store.sample_dir / sample.file_name).read_bytes(),
                original_bytes[sample.sample_id],
            )
        self.assertEqual(len(self.store.list_samples("Doctor")), 2)
        self.assertEqual(len(self.store.list_sessions()), 1)
        self.assertFalse(any(path.name.endswith(".delete")
                             for path in self.store.sample_dir.iterdir()))


class BuiltinVocabularyCaptureApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.root = Path(self.temp_dir.name)
        self.features = self.root / "features"
        (self.features / "Doctor").mkdir(parents=True)
        (self.features / "Teacher").mkdir()
        self.store = BuiltinVocabularyStore(
            self.root / "builtin-store",
            labels_provider=lambda: sorted(path.name for path in self.features.iterdir()
                                           if path.is_dir()),
        )
        self.draft_store = None
        import app
        from scripts.vocabulary_store import VocabularyStore

        self.app_module = app
        self.draft_store = VocabularyStore(self.root / "draft-store")
        self.patches = [
            patch.object(app, "FEATURES_DIR", self.features),
            patch.object(app, "BUILTIN_VOCABULARY_STORE", self.store),
            patch.object(app, "VOCABULARY_STORE", self.draft_store),
        ]
        for item in self.patches:
            item.start()
        self.client = app.app.test_client()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp_dir.cleanup()

    def test_doctor_capture_api_saves_separately_without_creating_draft(self):
        headers = {"Host": "localhost"}
        initial = self.client.get(
            "/developer/api/builtin-training-data", headers=headers
        )
        self.assertEqual(initial.status_code, 200, initial.get_json())
        label = next(item for item in initial.get_json()["labels"]
                     if item["label"] == "Doctor")
        self.assertEqual(label["sample_count"], 0)

        session_response = self.client.post(
            "/developer/api/builtin-training-data/sessions",
            json={"label": "doctor"}, headers=headers,
        )
        self.assertEqual(session_response.status_code, 201, session_response.get_json())
        session = session_response.get_json()
        self.assertEqual(session["label"], "Doctor")

        sample_response = self.client.post(
            f"/developer/api/builtin-training-data/sessions/{session['session_id']}/samples",
            json={
                "sequence": valid_sequence().tolist(),
                "preprocessing_version": PREPROCESSING_VERSION,
            },
            headers=headers,
        )
        self.assertEqual(sample_response.status_code, 201, sample_response.get_json())
        self.assertEqual(sample_response.get_json()["shape"], [30, 378])
        self.assertEqual(sample_response.get_json()["sample_count"], 1)

        refreshed = self.client.get(
            "/developer/api/builtin-training-data", headers=headers
        ).get_json()
        doctor = next(item for item in refreshed["labels"]
                      if item["label"] == "Doctor")
        self.assertEqual(doctor["sample_count"], 1)
        self.assertEqual(doctor["session_count"], 1)
        self.assertEqual(self.draft_store.list_entries("shared"), [])
        self.assertEqual(list((self.features / "Doctor").iterdir()), [])
        self.assertEqual(refreshed["training_integrated"], True)

    def test_delete_api_requires_confirmation_and_isolates_label(self):
        headers = {"Host": "localhost"}
        stored = {}
        for label in ("Doctor", "Teacher"):
            session = self.store.create_session(label)
            stored[label] = session
            self.store.save_sample(session.session_id, valid_sequence())

        missing_confirmation = self.client.delete(
            "/developer/api/builtin-training-data",
            json={"label": "Doctor"}, headers=headers,
        )
        self.assertEqual(missing_confirmation.status_code, 400)
        self.assertEqual(len(self.store.list_samples("Doctor")), 1)

        deleted = self.client.delete(
            "/developer/api/builtin-training-data",
            json={"label": "doctor", "confirmed": True}, headers=headers,
        )
        self.assertEqual(deleted.status_code, 200, deleted.get_json())
        self.assertEqual(deleted.get_json()["deleted_samples"], 1)
        self.assertEqual(deleted.get_json()["deleted_sessions"], 1)
        summary = {item["label"]: item for item in self.store.label_summary()}
        self.assertEqual(summary["Doctor"]["sample_count"], 0)
        self.assertEqual(summary["Teacher"]["sample_count"], 1)
        self.assertEqual(self.draft_store.list_entries("shared"), [])
        self.assertTrue((self.features / "Doctor").is_dir())


if __name__ == "__main__":
    unittest.main()
