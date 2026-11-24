import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.builtin_vocabulary_store import BuiltinVocabularyStore
from scripts.expandable_classifier import (
    ArtifactCompatibilityError,
    build_integrated_training_data,
    dataset_fingerprint,
    load_artifact,
    train_and_evaluate,
)
from scripts.sequence_contract import PREPROCESSING_VERSION, SEQUENCE_SHAPE
from scripts.vocabulary_store import CaptureSession, VocabularyEntry, VocabularyStore


def sequence(value):
    result = np.zeros(SEQUENCE_SHAPE, dtype=np.float32)
    result[:, 0:63] = value
    result[:, 252:315] = value / 2
    return result


def add_builtin_sessions(store, label, count=4, samples_per_session=1):
    for session_number in range(count):
        session = store.create_session(label)
        for sample_number in range(samples_per_session):
            store.save_sample(
                session.session_id,
                sequence(0.15 + 0.01 * session_number + 0.001 * sample_number),
                PREPROCESSING_VERSION,
            )


def add_draft_sessions(store, label, label_id, count=4):
    entry = VocabularyEntry(label.casefold(), label_id, label, "shared")
    store.add_entry(entry)
    for session_number in range(count):
        session = CaptureSession.create(entry)
        store.create_session(session)
        store.save_sample(
            session.session_id,
            entry.entry_id,
            sequence(0.75 + 0.01 * session_number),
            PREPROCESSING_VERSION,
            "shared",
        )


class BuiltinClassifierIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.drafts = VocabularyStore(self.root / "drafts")
        self.builtins = BuiltinVocabularyStore(
            self.root / "builtins", labels_provider=lambda: ["Doctor", "Hello"]
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_combines_sources_once_with_stable_canonical_ids(self):
        add_builtin_sessions(self.builtins, "Doctor")
        add_builtin_sessions(self.builtins, "Hello")
        add_draft_sessions(self.drafts, "friend", 2)
        add_draft_sessions(self.drafts, "hello", 3)

        dataset = build_integrated_training_data(
            self.drafts, self.builtins, "shared", ["Doctor", "Hello"]
        )
        names = [item["label"].casefold() for item in dataset["label_map"]]
        ids = [item["label_id"] for item in dataset["label_map"]]
        self.assertEqual(names.count("hello"), 1)
        self.assertEqual(len(ids), len(set(ids)))
        hello = next(item for item in dataset["label_map"] if item["label"] == "Hello")
        self.assertEqual(hello["label_id"], 1)
        self.assertEqual(hello["source"], "built-in+draft")
        self.assertEqual(dataset["source_sample_counts"], {"builtin": 8, "draft": 8})
        self.assertEqual(set(dataset["source_fingerprints"]), {"builtin", "draft"})
        self.assertEqual(len(dataset_fingerprint(dataset["examples"])), 64)

    def test_insufficient_builtin_classes_are_excluded_and_reported(self):
        add_builtin_sessions(self.builtins, "Doctor", count=3)
        add_draft_sessions(self.drafts, "friend", 2)
        add_draft_sessions(self.drafts, "student", 3)
        dataset = build_integrated_training_data(
            self.drafts, self.builtins, "shared", ["Doctor", "Hello"]
        )
        self.assertNotIn("Doctor", [item["label"] for item in dataset["label_map"]])
        doctor = next(item for item in dataset["excluded_labels"] if item["label"] == "Doctor")
        self.assertEqual(doctor["session_count"], 3)
        self.assertEqual(doctor["minimum_sessions"], 4)

    def test_duplicate_canonical_draft_labels_are_rejected(self):
        add_draft_sessions(self.drafts, "friend", 2)
        duplicate = VocabularyEntry("friend-duplicate", 3, " Friend ", "shared")
        self.drafts.add_entry(duplicate)
        session = CaptureSession.create(duplicate)
        self.drafts.create_session(session)
        self.drafts.save_sample(
            session.session_id,
            duplicate.entry_id,
            sequence(0.6),
            PREPROCESSING_VERSION,
            "shared",
        )
        with self.assertRaisesRegex(ValueError, "duplicate canonical label"):
            build_integrated_training_data(
                self.drafts, self.builtins, "shared", ["Doctor", "Hello"]
            )

    def test_integrated_fingerprint_tracks_new_builtin_samples(self):
        add_builtin_sessions(self.builtins, "Doctor")
        add_draft_sessions(self.drafts, "friend", 2)
        before = build_integrated_training_data(
            self.drafts, self.builtins, "shared", ["Doctor", "Hello"]
        )
        session = next(item for item in self.builtins.list_sessions() if item.label == "Doctor")
        self.builtins.save_sample(session.session_id, sequence(0.33), PREPROCESSING_VERSION)
        after = build_integrated_training_data(
            self.drafts, self.builtins, "shared", ["Doctor", "Hello"]
        )
        self.assertNotEqual(
            dataset_fingerprint(before["examples"]),
            dataset_fingerprint(after["examples"]),
        )

    def test_candidate_contains_mapping_and_is_not_activated(self):
        add_builtin_sessions(self.builtins, "Doctor")
        add_draft_sessions(self.drafts, "friend", 2)
        dataset = build_integrated_training_data(
            self.drafts, self.builtins, "shared", ["Doctor", "Hello"]
        )
        report = train_and_evaluate(
            self.drafts,
            "shared",
            examples=dataset["examples"],
            label_map=dataset["label_map"],
            dataset_details=dataset,
        )
        artifact = Path(report["artifact"]["directory"])
        metadata = json.loads((artifact / "metadata.json").read_text(encoding="utf-8"))
        label_manifest = json.loads((artifact / "label_map.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["training"]["data"]["dataset_mode"], "built-in-and-draft")
        self.assertEqual(
            set(metadata["training"]["data"]["source_fingerprints"]),
            {"builtin", "draft"},
        )
        self.assertEqual(label_manifest["label_map"], metadata["label_map"])
        self.assertEqual(
            label_manifest["fingerprint"],
            metadata["training"]["data"]["label_map_fingerprint"],
        )
        self.assertIn("Doctor", [item["label"] for item in metadata["label_map"]])
        self.assertFalse((self.root / "drafts" / "shared" / "models" / "active.json").exists())
        loaded = load_artifact(self.drafts, "shared", metadata["artifact_id"])
        self.assertEqual(loaded.label_map, metadata["label_map"])

        label_manifest["label_map"][0]["label_id"] += 100
        (artifact / "label_map.json").write_text(json.dumps(label_manifest), encoding="utf-8")
        with self.assertRaises(ArtifactCompatibilityError):
            load_artifact(self.drafts, "shared", metadata["artifact_id"])

    def test_train_api_creates_combined_candidate_and_explicit_activation(self):
        import app

        add_builtin_sessions(self.builtins, "Doctor")
        add_draft_sessions(self.drafts, "friend", 2)
        with (
            patch.object(app, "VOCABULARY_STORE", self.drafts),
            patch.object(app, "BUILTIN_VOCABULARY_STORE", self.builtins),
            patch.object(app, "_base_label_names", return_value=["Doctor", "Hello"]),
        ):
            client = app.app.test_client()
            response = client.post(
                "/api/experimental/train",
                json={"scope": "shared", "target_label": "Doctor"},
                headers={"Host": "localhost"},
            )
            self.assertEqual(response.status_code, 201, response.get_json())
            result = response.get_json()
            candidate = result["candidate"]
            self.assertFalse(result["activated"])
            self.assertIn("Doctor", [item["label"] for item in candidate["label_map"]])
            self.assertEqual(candidate["data"]["source_sample_counts"], {"builtin": 4, "draft": 4})
            self.assertFalse((self.drafts.root / "shared" / "models" / "active.json").exists())

            activation = client.post(
                f"/api/experimental/activate/{candidate['artifact_id']}",
                json={"scope": "shared"},
                headers={"Host": "localhost"},
            )
            self.assertEqual(activation.status_code, 200, activation.get_json())
            self.assertTrue((self.drafts.root / "shared" / "models" / "active.json").is_file())
            status = app._experimental_status("shared", include_candidates=False)
            self.assertTrue(status["active_model"]["usable"], status["active_model"])
            supported = {item["label"] for item in status["active_model"]["supports"]}
            self.assertEqual(supported, {"Doctor", "friend"})

            doctor_session = next(
                item for item in self.builtins.list_sessions()
                if item.label == "Doctor"
            )
            self.builtins.save_sample(
                doctor_session.session_id, sequence(0.4), PREPROCESSING_VERSION
            )
            stale = client.post(
                f"/api/experimental/activate/{candidate['artifact_id']}",
                json={"scope": "shared"},
                headers={"Host": "localhost"},
            )
            self.assertEqual(stale.status_code, 409)

    def test_train_api_enforces_four_session_target_and_duplicate_job_guard(self):
        import app

        add_builtin_sessions(self.builtins, "Doctor", count=3)
        with (
            patch.object(app, "VOCABULARY_STORE", self.drafts),
            patch.object(app, "BUILTIN_VOCABULARY_STORE", self.builtins),
            patch.object(app, "_base_label_names", return_value=["Doctor", "Hello"]),
        ):
            client = app.app.test_client()
            response = client.post(
                "/api/experimental/train",
                json={"scope": "shared", "target_label": "Doctor"},
                headers={"Host": "localhost"},
            )
            self.assertEqual(response.status_code, 422)
            self.assertEqual(response.get_json()["session_count"], 3)

            add_builtin_sessions(self.builtins, "Doctor", count=1)
            add_draft_sessions(self.drafts, "friend", 2)
            dataset = build_integrated_training_data(
                self.drafts, self.builtins, "shared", ["Doctor", "Hello"]
            )
            job_key = ("shared", dataset_fingerprint(dataset["examples"]))
            with app._experimental_training_lock:
                app._experimental_training_jobs.add(job_key)
            try:
                duplicate = client.post(
                    "/api/experimental/train",
                    json={"scope": "shared", "target_label": "Doctor"},
                    headers={"Host": "localhost"},
                )
            finally:
                with app._experimental_training_lock:
                    app._experimental_training_jobs.discard(job_key)
            self.assertEqual(duplicate.status_code, 409)


if __name__ == "__main__":
    unittest.main()
