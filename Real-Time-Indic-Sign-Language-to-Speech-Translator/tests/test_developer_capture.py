import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from scripts.capture_validation import validate_capture_sequence
from scripts.expandable_classifier import dataset_fingerprint
from scripts.sequence_contract import (
    PREPROCESSING_VERSION,
    SEQUENCE_SHAPE,
    SequenceContractError,
)
from scripts.vocabulary_store import VocabularyStore


def valid_capture():
    sequence = np.zeros(SEQUENCE_SHAPE, dtype=np.float32)
    sequence[:, 252:315] = 0.25
    sequence[:, 315:378] = 0.5
    return sequence


class CaptureValidationTests(unittest.TestCase):
    def test_complete_hand_sequence_passes_quality_gate(self):
        sequence, quality = validate_capture_sequence(valid_capture())
        self.assertEqual(sequence.shape, SEQUENCE_SHAPE)
        self.assertEqual(quality.frames, 30)
        self.assertEqual(quality.hand_frames, 30)
        self.assertEqual(quality.hand_frame_ratio, 1.0)

    def test_missing_or_partial_sequence_is_rejected(self):
        with self.assertRaises(SequenceContractError):
            validate_capture_sequence(np.zeros((29, 378), dtype=np.float32))

        sequence = valid_capture()
        sequence[:7, 252:] = 0
        with self.assertRaisesRegex(SequenceContractError, "at least 80%"):
            validate_capture_sequence(sequence)

    def test_incompatible_preprocessing_version_is_rejected(self):
        with self.assertRaises(SequenceContractError):
            validate_capture_sequence(valid_capture(), "legacy-1692")


class DeveloperDraftEndpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.store = VocabularyStore(Path(self.temporary_directory.name))
        import app

        self.app_module = app
        self.store_patch = patch.object(app, "VOCABULARY_STORE", self.store)
        self.store_patch.start()
        with app._lock:
            app._state["camera_active"] = False
            app._developer_capture_lease_until = 0.0
        self.client = app.app.test_client()

    def tearDown(self):
        self.store_patch.stop()
        self.temporary_directory.cleanup()

    def create_draft(self, label, scope):
        response = self.client.post(
            "/developer/api/drafts",
            json={"label": label, "scope": scope},
            headers={"Host": "localhost"},
        )
        self.assertIn(response.status_code, {200, 201}, response.get_json())
        return response.get_json()

    def test_missing_legacy_keras_file_keeps_browser_only_startup_available(self):
        original_model = self.app_module._model
        original_labels = self.app_module._labels
        try:
            with (
                patch.object(
                    self.app_module,
                    "KERAS_MODEL",
                    str(Path(self.temporary_directory.name) / "missing.keras"),
                ),
                patch.object(
                    self.app_module,
                    "FEATURES_DIR",
                    Path(self.temporary_directory.name) / "missing-features",
                ),
            ):
                self.app_module._load_resources()
                self.assertIsNone(self.app_module._model)
                self.assertEqual(self.app_module._labels, [])
        finally:
            self.app_module._model = original_model
            self.app_module._labels = original_labels

    def save_example(self, draft):
        response = self.client.post(
            f"/developer/api/drafts/{draft['draft']['entry_id']}/samples",
            json={
                "scope": draft["draft"]["scope"],
                "session_id": draft["session_id"],
                "preprocessing_version": PREPROCESSING_VERSION,
                "sequence": valid_capture().tolist(),
            },
            headers={"Host": "localhost"},
        )
        self.assertEqual(response.status_code, 201, response.get_json())
        return response.get_json()["sample_id"]

    def start_session(self, draft, scope=None):
        return self.client.post(
            f"/developer/api/drafts/{draft['draft']['entry_id']}/sessions",
            json={"scope": scope or draft["draft"]["scope"]},
            headers={"Host": "localhost"},
        )

    def test_new_session_is_unique_and_new_samples_keep_old_session_unchanged(self):
        draft = self.create_draft("SessionWord", "shared")
        entry_id = draft["draft"]["entry_id"]
        first_session = draft["session_id"]
        original_sample = self.save_example(draft)
        original_record = next(
            sample for sample in self.store.list_samples("shared")
            if sample.sample_id == original_sample
        )

        started = self.start_session(draft)
        self.assertEqual(started.status_code, 201, started.get_json())
        second_session = started.get_json()["session_id"]
        self.assertNotEqual(first_session, second_session)
        self.assertEqual(
            [item.session_id for item in self.store.list_sessions("shared")],
            [first_session, second_session],
        )
        new_sample = self.client.post(
            f"/developer/api/drafts/{entry_id}/samples",
            json={
                "scope": "shared",
                "session_id": second_session,
                "preprocessing_version": PREPROCESSING_VERSION,
                "sequence": valid_capture().tolist(),
            },
            headers={"Host": "localhost"},
        )
        self.assertEqual(new_sample.status_code, 201, new_sample.get_json())
        samples = {item.sample_id: item for item in self.store.list_samples("shared")}
        self.assertEqual(samples[original_sample].session_id, first_session)
        self.assertEqual(samples[original_sample], original_record)
        self.assertEqual(samples[new_sample.get_json()["sample_id"]].session_id, second_session)
        self.assertEqual(self.store.count_samples("shared", entry_id), 2)

    def test_resume_does_not_create_or_switch_session(self):
        draft = self.create_draft("ResumeSession", "personal")
        first_session = draft["session_id"]
        second = self.start_session(draft)
        self.assertEqual(second.status_code, 201, second.get_json())
        second_session = second.get_json()["session_id"]
        session_ids_before = [item.session_id for item in self.store.list_sessions("personal")]

        resumed = self.client.post(
            "/developer/api/drafts",
            json={"label": "ResumeSession", "scope": "personal"},
            headers={"Host": "localhost"},
        )
        self.assertEqual(resumed.status_code, 200, resumed.get_json())
        self.assertEqual(resumed.get_json()["session_id"], second_session)
        self.assertNotEqual(resumed.get_json()["session_id"], first_session)
        self.assertEqual(
            [item.session_id for item in self.store.list_sessions("personal")],
            session_ids_before,
        )

    def test_shared_and_personal_sessions_are_isolated(self):
        shared = self.create_draft("SameWord", "shared")
        personal = self.create_draft("SameWord", "personal")
        shared_sessions_before = [
            item.session_id for item in self.store.list_sessions("shared")
        ]
        personal_sessions_before = [
            item.session_id for item in self.store.list_sessions("personal")
        ]

        started = self.start_session(shared)
        self.assertEqual(started.status_code, 201, started.get_json())
        self.assertEqual(
            [item.session_id for item in self.store.list_sessions("shared")],
            shared_sessions_before + [started.get_json()["session_id"]],
        )
        self.assertEqual(
            [item.session_id for item in self.store.list_sessions("personal")],
            personal_sessions_before,
        )
        self.assertEqual(shared["draft"]["entry_id"], personal["draft"]["entry_id"])

    def test_invalid_session_requests_leave_manifests_and_samples_unchanged(self):
        draft = self.create_draft("SafeSession", "shared")
        sample_id = self.save_example(draft)
        sessions_before = self.store.list_sessions("shared")
        samples_before = self.store.list_samples("shared")
        invalid_id = self.client.post(
            "/developer/api/drafts/not%20safe/sessions",
            json={"scope": "shared"}, headers={"Host": "localhost"},
        )
        invalid_scope = self.start_session(draft, "personal")
        missing_scope = self.client.post(
            f"/developer/api/drafts/{draft['draft']['entry_id']}/sessions",
            json={"scope": "other"}, headers={"Host": "localhost"},
        )
        self.assertEqual(invalid_id.status_code, 422)
        self.assertEqual(invalid_scope.status_code, 404)
        self.assertEqual(missing_scope.status_code, 400)
        self.assertEqual(self.store.list_sessions("shared"), sessions_before)
        self.assertEqual(self.store.list_samples("shared"), samples_before)
        self.assertEqual(self.store.count_samples("shared", draft["draft"]["entry_id"]), 1)
        self.assertEqual(samples_before[0].sample_id, sample_id)

    def test_classifier_readiness_tracks_additional_session(self):
        draft = self.create_draft("ReadyWord", "shared")
        entry_id = draft["draft"]["entry_id"]
        before = self.client.get(
            "/developer/api/drafts?scope=shared", headers={"Host": "localhost"}
        ).get_json()["drafts"][0]
        self.assertEqual(before["session_count"], 0)
        self.assertFalse(before["training_ready"])
        self.assertFalse(before["evaluation_ready"])

        second_session = self.start_session(draft)
        self.assertEqual(second_session.status_code, 201)
        new_sample = self.client.post(
            f"/developer/api/drafts/{entry_id}/samples",
            json={
                "scope": "shared",
                "session_id": second_session.get_json()["session_id"],
                "preprocessing_version": PREPROCESSING_VERSION,
                "sequence": valid_capture().tolist(),
            },
            headers={"Host": "localhost"},
        )
        self.assertEqual(new_sample.status_code, 201, new_sample.get_json())
        after = next(
            item for item in self.client.get(
                "/developer/api/drafts?scope=shared", headers={"Host": "localhost"}
            ).get_json()["drafts"] if item["entry_id"] == entry_id
        )
        self.assertEqual(after["session_count"], 1)
        self.assertEqual(after["capture_session_count"], 2)
        self.assertFalse(after["training_ready"])

        second = self.start_session(draft)
        self.assertEqual(second.status_code, 201)
        self.client.post(
            f"/developer/api/drafts/{entry_id}/samples",
            json={
                "scope": "shared",
                "session_id": second.get_json()["session_id"],
                "preprocessing_version": PREPROCESSING_VERSION,
                "sequence": valid_capture().tolist(),
            },
            headers={"Host": "localhost"},
        )
        after = next(
            item for item in self.client.get(
                "/developer/api/drafts?scope=shared", headers={"Host": "localhost"}
            ).get_json()["drafts"] if item["entry_id"] == entry_id
        )
        self.assertEqual(after["session_count"], 2)
        self.assertTrue(after["training_ready"])
        self.assertFalse(after["evaluation_ready"])

    def test_public_vocabulary_refresh_exposes_backend_words_and_session_counts(self):
        draft = self.create_draft("WebsiteWord", "shared")
        self.save_example(draft)
        first = self.client.get(
            "/api/vocabulary?scope=shared", headers={"Host": "localhost"}
        )
        self.assertEqual(first.status_code, 200, first.get_json())
        word = next(
            item for item in first.get_json()["words"]
            if item["entry_id"] == draft["draft"]["entry_id"]
        )
        self.assertEqual(word["label_id"], draft["draft"]["label_id"])
        self.assertEqual(word["sample_count"], 1)
        self.assertEqual(word["session_count"], 1)
        self.assertFalse(word["recognizable_by_active_model"])
        self.assertFalse((self.store.root / "shared" / "models").exists())

        started = self.start_session(draft)
        self.assertEqual(started.status_code, 201)
        saved_again = self.client.post(
            f"/developer/api/drafts/{draft['draft']['entry_id']}/samples",
            json={
                "scope": "shared",
                "session_id": started.get_json()["session_id"],
                "preprocessing_version": PREPROCESSING_VERSION,
                "sequence": valid_capture().tolist(),
            },
            headers={"Host": "localhost"},
        )
        self.assertEqual(saved_again.status_code, 201, saved_again.get_json())
        refreshed = self.client.get(
            "/api/vocabulary?scope=shared", headers={"Host": "localhost"}
        ).get_json()
        word_after = next(
            item for item in refreshed["words"]
            if item["entry_id"] == draft["draft"]["entry_id"]
        )
        self.assertEqual(word_after["session_count"], 2)
        self.assertEqual(word_after["capture_session_count"], 2)
        self.assertEqual(word_after["sample_count"], 2)

    def test_public_vocabulary_reports_recognizers_and_changes_storage_version(self):
        response = self.client.get(
            "/api/vocabulary?scope=shared", headers={"Host": "localhost"}
        )
        self.assertEqual(response.status_code, 200, response.get_json())
        initial = response.get_json()
        self.assertEqual(
            initial["recognizers"]["browser_knn"]["vocabulary"],
            ["Hello", "Thank you", "Sorry", "Yes", "No"],
        )
        self.assertTrue(initial["recognizers"]["browser_knn"]["fixed_vocabulary"])
        self.assertFalse(initial["recognizers"]["experimental"]["available"])
        self.assertEqual(
            initial["recognizers"]["experimental"]["supported_label_count"], 0
        )
        self.assertFalse(
            initial["recognizers"]["experimental"]["full_vocabulary_supported"]
        )

        draft = self.create_draft("DynamicWord", "shared")
        after_entry = self.client.get(
            "/api/vocabulary?scope=shared", headers={"Host": "localhost"}
        ).get_json()
        self.assertNotEqual(
            initial["storage_version"], after_entry["storage_version"]
        )
        self.save_example(draft)
        after_sample = self.client.get(
            "/api/vocabulary?scope=shared", headers={"Host": "localhost"}
        ).get_json()
        self.assertNotEqual(
            after_entry["storage_version"], after_sample["storage_version"]
        )
        self.assertIn(
            "DynamicWord", [item["label"] for item in after_sample["words"]]
        )

    def test_public_vocabulary_refresh_tracks_active_model_changes(self):
        inactive = {
            "scope": "shared",
            "preprocessing_version": PREPROCESSING_VERSION,
            "vocabulary_version": "vocab-v1",
            "active_model": {
                "usable": False, "artifact_id": None, "supports": [],
            },
            "candidates": [],
            "candidate_list_error": None,
        }
        active = {
            **inactive,
            "active_model": {
                "usable": True,
                "artifact_id": "20261103T120000000000Z",
                "supports": [{
                    "entry_id": "hello-entry",
                    "label_id": 1,
                    "label": "Hello",
                }],
            },
        }
        with (
            patch.object(
                self.app_module, "_experimental_status",
                side_effect=[inactive, active],
            ),
            patch.object(self.app_module, "_draft_payload", return_value=[]),
        ):
            before = self.client.get(
                "/api/vocabulary?scope=shared", headers={"Host": "localhost"}
            ).get_json()
            after = self.client.get(
                "/api/vocabulary?scope=shared", headers={"Host": "localhost"}
            ).get_json()
        self.assertFalse(before["recognizers"]["experimental"]["available"])
        self.assertTrue(after["recognizers"]["experimental"]["available"])
        self.assertEqual(
            after["recognizers"]["experimental"]["active_artifact_id"],
            "20261103T120000000000Z",
        )
        self.assertEqual(
            after["recognizers"]["experimental"]["vocabulary"], ["Hello"]
        )

    def test_prediction_returns_consistent_model_and_vocabulary_versions(self):
        draft = self.create_draft("ModelWord", "personal")
        entry = draft["draft"]
        artifact_id = "20261025T010203000000Z"
        label_map = [{
            "label_id": entry["label_id"],
            "label": entry["label"],
            "scope": "personal",
        }]

        def fake_predict(sequence):
            if np.asarray(sequence).shape != (30, 378):
                raise AssertionError("The saved-sequence contract was not preserved")
            return {
                "label": entry["label"],
                "label_id": entry["label_id"],
                "confidence": 0.78,
                "abstained": False,
                "scope": "personal",
                "preprocessing_version": PREPROCESSING_VERSION,
                "artifact_id": artifact_id,
            }

        fake_model = SimpleNamespace(
            metadata={
                "artifact_id": artifact_id,
                "scope": "personal",
                "preprocessing_version": PREPROCESSING_VERSION,
                "label_map": label_map,
                "training": {"data": {"fingerprint": dataset_fingerprint([])}},
                "evaluation": {"evaluation_ready": True, "validation": {"accuracy": 1.0}},
            },
            label_map=label_map,
            predict=fake_predict,
        )

        quality = SimpleNamespace(frames=30, hand_frames=30)
        with patch.object(self.app_module, "load_artifact", return_value=fake_model), \
                patch.object(
                    self.app_module,
                    "_sequence_from_frame_data_urls",
                    return_value=(valid_capture(), quality),
                ):
            response = self.client.post(
                "/api/experimental/predict",
                json={
                    "scope": "personal",
                    "artifact_id": artifact_id,
                    "frames": ["unused"] * 30,
                },
                headers={"Host": "localhost"},
            )
            vocabulary = self.client.get(
                "/api/vocabulary?scope=personal", headers={"Host": "localhost"}
            ).get_json()
        self.assertEqual(response.status_code, 200, response.get_json())
        prediction = response.get_json()
        self.assertEqual(prediction["model_version"], artifact_id)
        self.assertEqual(prediction["vocabulary_version"], vocabulary["vocabulary_version"])
        self.assertEqual(prediction["current_vocabulary_version"], vocabulary["vocabulary_version"])
        self.assertEqual(
            prediction["current_storage_version"], vocabulary["storage_version"]
        )
        self.assertEqual(prediction["entry_id"], entry["entry_id"])
        self.assertTrue(prediction["test_only"])

    def test_classifier_label_and_label_id_must_match_exact_map(self):
        first = self.create_draft("FirstLabel", "personal")
        second = self.create_draft("SecondLabel", "personal")
        entries = self.store.list_entries("personal")
        expected = self.app_module._expected_label_map(entries, "personal")
        self.assertTrue(
            self.app_module._model_label_map_matches(expected, entries, "personal")
        )
        self.assertFalse(
            self.app_module._model_label_map_matches(expected[:1], entries, "personal")
        )
        self.assertFalse(
            self.app_module._model_label_map_matches(list(reversed(expected)), entries, "personal")
        )
        wrong_name = [dict(item) for item in expected]
        wrong_name[0]["label"] = "MisindexedLabel"
        self.assertFalse(
            self.app_module._model_label_map_matches(wrong_name, entries, "personal")
        )
        self.assertNotEqual(first["draft"]["label_id"], second["draft"]["label_id"])

    def test_candidate_prediction_rejects_label_index_label_disagreement(self):
        draft = self.create_draft("MappedWord", "personal")
        self.save_example(draft)
        next_session = self.start_session(draft)
        second = {"draft": draft["draft"], "session_id": next_session.get_json()["session_id"]}
        self.save_example(second)
        entry = draft["draft"]
        artifact_id = "20261102T010203000002Z"
        label_map = [{
            "label_id": entry["label_id"],
            "label": entry["label"],
            "scope": "personal",
        }]
        fake_model = SimpleNamespace(
            metadata={
                "artifact_id": artifact_id,
                "scope": "personal",
                "training": {
                    "data": {
                        "fingerprint": dataset_fingerprint(
                            self.store.load_training_examples("personal")
                        )
                    }
                },
                "evaluation": {"evaluation_ready": False, "validation": None},
            },
            label_map=label_map,
            predict=lambda sequence: {
                "label": "WrongLabel",
                "label_id": entry["label_id"],
                "confidence": 0.9,
                "abstained": False,
                "scope": "personal",
                "preprocessing_version": PREPROCESSING_VERSION,
                "artifact_id": artifact_id,
            },
        )
        quality = SimpleNamespace(frames=30, hand_frames=30)
        holistic_asset = Path(self.temporary_directory.name) / "holistic.task"
        holistic_asset.touch()
        with patch.object(self.app_module, "load_artifact", return_value=fake_model), \
                patch.object(self.app_module, "HOLISTIC_MODEL", str(holistic_asset)), \
                patch.object(
                    self.app_module,
                    "_sequence_from_frame_data_urls",
                    return_value=(valid_capture(), quality),
                ):
            response = self.client.post(
                "/api/experimental/predict",
                json={
                    "scope": "personal",
                    "artifact_id": artifact_id,
                    "frames": ["unused"] * 30,
                },
                headers={"Host": "localhost"},
            )
        self.assertEqual(response.status_code, 502, response.get_json())
        self.assertIn("label index", response.get_json()["error"])

    def test_live_prediction_rejects_an_unusable_active_model(self):
        draft = self.create_draft("StaleActive", "personal")
        entry = draft["draft"]
        fake_model = SimpleNamespace(
            metadata={"artifact_id": "20261102T010203000003Z"},
            label_map=[{
                "label_id": entry["label_id"],
                "label": entry["label"],
                "scope": "personal",
            }],
        )
        holistic_asset = Path(self.temporary_directory.name) / "holistic.task"
        holistic_asset.touch()
        with patch.object(self.app_module, "load_artifact", return_value=fake_model), \
                patch.object(self.app_module, "HOLISTIC_MODEL", str(holistic_asset)), \
                patch.object(
                    self.app_module,
                    "_experimental_status",
                    return_value={"active_model": {"usable": False}},
                ):
            response = self.client.post(
                "/api/experimental/predict",
                json={"scope": "personal", "frames": ["unused"] * 30},
                headers={"Host": "localhost"},
            )
        self.assertEqual(response.status_code, 409, response.get_json())
        self.assertIn("not usable", response.get_json()["error"])

    def test_ai_sentence_endpoint_rejects_a_non_active_candidate(self):
        response = self.client.post(
            "/api/ai-sentence",
            json={
                "recognizer": "experimental",
                "scope": "personal",
                "artifact_id": "20261102T010203000004Z",
                "signs": [{"label": "Food", "confidence": 0.7}],
            },
            headers={"Host": "localhost"},
        )
        self.assertEqual(response.status_code, 409, response.get_json())
        self.assertIn("active model", response.get_json()["error"])

    def test_unvalidated_candidate_can_be_tested_without_activation(self):
        draft = self.create_draft("CandidateWord", "personal")
        self.save_example(draft)
        next_session = self.start_session(draft)
        self.assertEqual(next_session.status_code, 201)
        second_draft = {
            "draft": draft["draft"],
            "session_id": next_session.get_json()["session_id"],
        }
        self.save_example(second_draft)

        entry = draft["draft"]
        artifact_id = "20261102T010203000000Z"
        label_map = [{
            "label_id": entry["label_id"],
            "label": entry["label"],
            "scope": "personal",
        }]

        def fake_predict(sequence):
            self.assertEqual(np.asarray(sequence).shape, (30, 378))
            return {
                "label": entry["label"],
                "label_id": entry["label_id"],
                "confidence": 0.72,
                "abstained": False,
                "scope": "personal",
                "preprocessing_version": PREPROCESSING_VERSION,
                "artifact_id": artifact_id,
            }

        fake_model = SimpleNamespace(
            metadata={
                "artifact_id": artifact_id,
                "scope": "personal",
                "training": {
                    "data": {
                        "fingerprint": dataset_fingerprint(
                            self.store.load_training_examples("personal")
                        )
                    }
                },
                "evaluation": {
                    "evaluation_ready": False,
                    "validation": None,
                },
            },
            label_map=label_map,
            predict=fake_predict,
        )
        quality = SimpleNamespace(frames=30, hand_frames=30)
        holistic_asset = Path(self.temporary_directory.name) / "holistic.task"
        holistic_asset.touch()
        with patch.object(self.app_module, "load_artifact", return_value=fake_model), \
                patch.object(self.app_module, "HOLISTIC_MODEL", str(holistic_asset)), \
                patch.object(
                    self.app_module,
                    "_sequence_from_frame_data_urls",
                    return_value=(valid_capture(), quality),
                ):
            response = self.client.post(
                "/api/experimental/predict",
                json={
                    "scope": "personal",
                    "artifact_id": artifact_id,
                    "frames": ["unused"] * 30,
                },
                headers={"Host": "localhost"},
            )

        self.assertEqual(response.status_code, 200, response.get_json())
        prediction = response.get_json()
        self.assertEqual(prediction["artifact_id"], artifact_id)
        self.assertEqual(prediction["label"], entry["label"])
        self.assertEqual(prediction["confidence"], 0.72)
        self.assertFalse(prediction["abstained"])
        self.assertFalse(prediction["evaluation_ready"])
        self.assertIn("unvalidated", prediction["evaluation_status"])
        self.assertTrue(prediction["test_only"])
        self.assertEqual(prediction["sequence_quality"]["preprocessing_version"], PREPROCESSING_VERSION)
        self.assertFalse((self.store.root / "personal" / "models" / "active.json").exists())

    def test_candidate_testing_rejects_stale_training_data(self):
        draft = self.create_draft("StaleCandidate", "personal")
        self.save_example(draft)
        entry = draft["draft"]
        artifact_id = "20261102T010203000001Z"
        fake_model = SimpleNamespace(
            metadata={
                "artifact_id": artifact_id,
                "scope": "personal",
                "training": {"data": {"fingerprint": "stale-fingerprint"}},
                "evaluation": {"evaluation_ready": False, "validation": None},
            },
            label_map=[{
                "label_id": entry["label_id"],
                "label": entry["label"],
                "scope": "personal",
            }],
        )
        holistic_asset = Path(self.temporary_directory.name) / "holistic.task"
        holistic_asset.touch()
        with patch.object(self.app_module, "load_artifact", return_value=fake_model), \
                patch.object(self.app_module, "HOLISTIC_MODEL", str(holistic_asset)):
            response = self.client.post(
                "/api/experimental/predict",
                json={
                    "scope": "personal",
                    "artifact_id": artifact_id,
                    "frames": ["unused"] * 30,
                },
                headers={"Host": "localhost"},
            )
        self.assertEqual(response.status_code, 409)
        self.assertIn("changed", response.get_json()["error"])

    def test_explicit_training_blocks_unready_words_without_model_writes(self):
        self.create_draft("WordOne", "shared")
        self.create_draft("WordTwo", "shared")
        response = self.client.post(
            "/api/experimental/train",
            json={"scope": "shared"}, headers={"Host": "localhost"},
        )
        self.assertEqual(response.status_code, 422, response.get_json())
        self.assertIn("unready_words", response.get_json())
        self.assertFalse((self.store.root / "shared" / "models").exists())

    def test_explicit_training_saves_two_session_candidate_without_validation(self):
        drafts = [
            self.create_draft("WordOne", "shared"),
            self.create_draft("WordTwo", "shared"),
        ]
        for draft in drafts:
            self.save_example(draft)
            session = self.start_session(draft)
            self.assertEqual(session.status_code, 201)
            self.save_example({
                "draft": draft["draft"],
                "session_id": session.get_json()["session_id"],
            })

        trained = self.client.post(
            "/api/experimental/train",
            json={"scope": "shared"},
            headers={"Host": "localhost"},
        )
        self.assertEqual(trained.status_code, 201, trained.get_json())
        candidate = trained.get_json()["candidate"]
        self.assertFalse(candidate["evaluation_ready"])
        self.assertIsNone(candidate["validation"])
        self.assertEqual(candidate["data"]["total_samples"], 4)
        self.assertFalse(trained.get_json()["activated"])
        self.assertFalse((self.store.root / "shared" / "models" / "active.json").exists())

        activation = self.client.post(
            f"/api/experimental/activate/{candidate['artifact_id']}",
            json={"scope": "shared"},
            headers={"Host": "localhost"},
        )
        self.assertEqual(activation.status_code, 409)

    def test_model_unavailable_and_invalid_activation_fail_safely(self):
        self.create_draft("SafeWord", "personal")
        status = self.client.get(
            "/api/vocabulary?scope=personal", headers={"Host": "localhost"}
        )
        self.assertEqual(status.status_code, 200)
        self.assertFalse(status.get_json()["active_model"]["available"])
        rejected = self.client.post(
            "/api/experimental/activate/invalid-id",
            json={"scope": "personal"}, headers={"Host": "localhost"},
        )
        unavailable = self.client.post(
            "/api/experimental/predict",
            json={"scope": "personal", "frames": ["invalid"] * 30},
            headers={"Host": "localhost"},
        )
        self.assertEqual(rejected.status_code, 422)
        self.assertEqual(unavailable.status_code, 503)
        self.assertFalse((self.store.root / "personal" / "models").exists())

    def test_saving_and_reopening_draft_preserves_scope_and_count(self):
        response = self.client.post(
            "/developer/api/drafts",
            json={"label": "Fish", "scope": "shared"},
            headers={"Host": "localhost"},
        )
        self.assertEqual(response.status_code, 201, response.get_json())
        created = response.get_json()
        entry_id = created["draft"]["entry_id"]
        base_labels = [
            path for path in (self.app_module.ROOT / "data" / "extracted_features").iterdir()
            if path.is_dir()
        ]
        self.assertGreaterEqual(created["draft"]["label_id"], len(base_labels))

        saved = self.client.post(
            f"/developer/api/drafts/{entry_id}/samples",
            json={
                "scope": "shared",
                "session_id": created["session_id"],
                "preprocessing_version": PREPROCESSING_VERSION,
                "sequence": valid_capture().tolist(),
            },
            headers={"Host": "localhost"},
        )
        self.assertEqual(saved.status_code, 201, saved.get_json())
        self.assertEqual(saved.get_json()["sample_count"], 1)

        reopened_store = VocabularyStore(Path(self.temporary_directory.name))
        self.assertEqual(reopened_store.count_samples("shared", entry_id), 1)
        self.assertEqual(reopened_store.count_samples("personal", entry_id), 0)
        drafts = self.client.get(
            "/developer/api/drafts?scope=shared",
            headers={"Host": "localhost"},
        ).get_json()["drafts"]
        self.assertEqual(drafts[0]["sample_count"], 1)
        self.assertEqual(drafts[0]["session_id"], created["session_id"])

    def test_create_resume_uses_flask_post_route_not_frontend_server_origin(self):
        post_rules = [
            rule for rule in self.app_module.app.url_map.iter_rules()
            if rule.rule == "/developer/api/drafts" and "POST" in rule.methods
        ]
        self.assertEqual(len(post_rules), 1)

        developer_script = (
            Path(__file__).parent.parent / "Frontend" / "developer.js"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'const API = `http://${API_HOST}:5000`;',
            developer_script,
        )
        self.assertIn(
            'new Set(["localhost", "127.0.0.1", "[::1]"])',
            developer_script,
        )
        self.assertIn(
            '? window.location.hostname\n    : "localhost";',
            developer_script,
        )
        self.assertIn(
            'api("/developer/api/drafts", {\n        method: "POST",',
            developer_script,
        )
        self.assertNotIn("const API = window.location.origin;", developer_script)

        created = self.client.post(
            "/developer/api/drafts",
            json={"label": "Route check", "scope": "personal"},
            headers={"Host": "localhost"},
        )
        self.assertEqual(created.status_code, 201, created.get_json())
        resumed = self.client.post(
            "/developer/api/drafts",
            json={"label": "Route check", "scope": "personal"},
            headers={"Host": "localhost"},
        )
        self.assertEqual(resumed.status_code, 200, resumed.get_json())

    def test_shared_label_cannot_duplicate_builtin_label(self):
        response = self.client.post(
            "/developer/api/drafts",
            json={"label": "Hello", "scope": "shared"},
            headers={"Host": "localhost"},
        )
        self.assertEqual(response.status_code, 409)

    def test_resuming_entry_keeps_scope_and_label_id(self):
        shared_response = self.client.post(
            "/developer/api/drafts",
            json={"label": "Fish", "scope": "shared"},
            headers={"Host": "localhost"},
        )
        personal_response = self.client.post(
            "/developer/api/drafts",
            json={"label": "Fish", "scope": "personal"},
            headers={"Host": "localhost"},
        )
        self.assertEqual(shared_response.status_code, 201)
        self.assertEqual(personal_response.status_code, 201)
        shared = shared_response.get_json()
        personal = personal_response.get_json()
        self.assertEqual(shared["draft"]["entry_id"], personal["draft"]["entry_id"])
        self.assertEqual(shared["draft"]["scope"], "shared")
        self.assertEqual(personal["draft"]["scope"], "personal")
        self.assertGreater(shared["draft"]["label_id"], personal["draft"]["label_id"])

        resumed = self.client.post(
            "/developer/api/drafts",
            json={"label": "Fish", "scope": "shared"},
            headers={"Host": "localhost"},
        )
        self.assertEqual(resumed.status_code, 200)
        reopened = resumed.get_json()
        self.assertEqual(reopened["draft"]["entry_id"], shared["draft"]["entry_id"])
        self.assertEqual(reopened["draft"]["label_id"], shared["draft"]["label_id"])

    def test_invalid_sequence_and_cross_scope_write_are_rejected(self):
        created_response = self.client.post(
            "/developer/api/drafts",
            json={"label": "Sorry", "scope": "shared"},
            headers={"Host": "localhost"},
        )
        created = created_response.get_json()
        entry_id = created["draft"]["entry_id"]

        invalid = self.client.post(
            f"/developer/api/drafts/{entry_id}/samples",
            json={
                "scope": "shared",
                "session_id": created["session_id"],
                "preprocessing_version": PREPROCESSING_VERSION,
                "sequence": np.zeros((29, 378), dtype=np.float32).tolist(),
            },
            headers={"Host": "localhost"},
        )
        self.assertEqual(invalid.status_code, 422)
        self.assertEqual(self.store.count_samples("shared", entry_id), 0)

        cross_scope = self.client.post(
            f"/developer/api/drafts/{entry_id}/samples",
            json={
                "scope": "personal",
                "session_id": created["session_id"],
                "preprocessing_version": PREPROCESSING_VERSION,
                "sequence": valid_capture().tolist(),
            },
            headers={"Host": "localhost"},
        )
        self.assertEqual(cross_scope.status_code, 422)
        self.assertEqual(self.store.count_samples("personal", entry_id), 0)

    def test_selected_sample_deletion_preserves_draft_and_unselected_examples(self):
        created_response = self.client.post(
            "/developer/api/drafts",
            json={"label": "Okay", "scope": "shared"},
            headers={"Host": "localhost"},
        )
        created = created_response.get_json()
        entry_id = created["draft"]["entry_id"]
        saved_ids = []
        for _ in range(3):
            response = self.client.post(
                f"/developer/api/drafts/{entry_id}/samples",
                json={
                    "scope": "shared",
                    "session_id": created["session_id"],
                    "preprocessing_version": PREPROCESSING_VERSION,
                    "sequence": valid_capture().tolist(),
                },
                headers={"Host": "localhost"},
            )
            self.assertEqual(response.status_code, 201, response.get_json())
            saved_ids.append(response.get_json()["sample_id"])

        deleted = self.client.delete(
            f"/developer/api/drafts/{entry_id}/samples",
            json={"scope": "shared", "sample_ids": saved_ids[:2]},
            headers={"Host": "localhost"},
        )
        self.assertEqual(deleted.status_code, 200, deleted.get_json())
        self.assertEqual(deleted.get_json()["deleted_count"], 2)
        self.assertEqual(deleted.get_json()["sample_count"], 1)
        self.assertEqual(self.store.count_samples("shared", entry_id), 1)
        self.assertEqual(
            [sample.sample_id for sample in self.store.list_samples("shared")],
            [saved_ids[2]],
        )
        self.assertEqual(len(self.store.list_entries("shared")), 1)
        self.assertEqual(len(self.store.list_sessions("shared")), 1)

    def test_sample_deletion_validates_scope_ids_and_local_client(self):
        created_response = self.client.post(
            "/developer/api/drafts",
            json={"label": "Okay", "scope": "shared"},
            headers={"Host": "localhost"},
        )
        created = created_response.get_json()
        entry_id = created["draft"]["entry_id"]
        saved = self.client.post(
            f"/developer/api/drafts/{entry_id}/samples",
            json={
                "scope": "shared",
                "session_id": created["session_id"],
                "preprocessing_version": PREPROCESSING_VERSION,
                "sequence": valid_capture().tolist(),
            },
            headers={"Host": "localhost"},
        ).get_json()
        sample_id = saved["sample_id"]

        invalid_scope = self.client.delete(
            f"/developer/api/drafts/{entry_id}/samples",
            json={"scope": "personal", "sample_ids": [sample_id]},
            headers={"Host": "localhost"},
        )
        missing_ids = self.client.delete(
            f"/developer/api/drafts/{entry_id}/samples",
            json={"scope": "shared", "sample_ids": []},
            headers={"Host": "localhost"},
        )
        invalid_id = self.client.delete(
            f"/developer/api/drafts/{entry_id}/samples",
            json={"scope": "shared", "sample_ids": ["../other"]},
            headers={"Host": "localhost"},
        )
        remote = self.client.delete(
            f"/developer/api/drafts/{entry_id}/samples",
            json={"scope": "shared", "sample_ids": [sample_id]},
            headers={"Host": "localhost"},
            environ_base={"REMOTE_ADDR": "192.0.2.10"},
        )
        self.assertEqual(invalid_scope.status_code, 404)
        self.assertEqual(missing_ids.status_code, 400)
        self.assertEqual(invalid_id.status_code, 422)
        self.assertEqual(remote.status_code, 403)
        self.assertEqual(self.store.count_samples("shared", entry_id), 1)

    def test_delete_word_removes_only_selected_scope_entry_data(self):
        shared_delete = self.create_draft("TemporaryDelete", "shared")
        shared_keep = self.create_draft("KeepWord", "shared")
        personal_same_label = self.create_draft("TemporaryDelete", "personal")
        deleted_sample_ids = [
            self.save_example(shared_delete), self.save_example(shared_delete)
        ]
        shared_keep_sample = self.save_example(shared_keep)
        personal_sample = self.save_example(personal_same_label)
        entry_id = shared_delete["draft"]["entry_id"]
        response = self.client.delete(
            f"/developer/api/drafts/{entry_id}",
            json={"scope": "shared", "confirmed": True},
            headers={"Host": "localhost"},
        )

        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual(response.get_json()["deleted"]["deleted_samples"], 2)
        self.assertEqual(response.get_json()["deleted"]["deleted_sessions"], 1)
        self.assertEqual(
            [entry.label for entry in self.store.list_entries("shared")],
            ["KeepWord"],
        )
        self.assertEqual(
            [entry.label for entry in self.store.list_entries("personal")],
            ["TemporaryDelete"],
        )
        self.assertEqual(
            [session.entry_id for session in self.store.list_sessions("shared")],
            [shared_keep["draft"]["entry_id"]],
        )
        self.assertEqual(
            [sample.sample_id for sample in self.store.list_samples("shared")],
            [shared_keep_sample],
        )
        self.assertEqual(
            [sample.sample_id for sample in self.store.list_samples("personal")],
            [personal_sample],
        )
        for sample_id in deleted_sample_ids:
            self.assertFalse(
                (self.store.root / "shared" / "samples" / f"{sample_id}.npy").exists()
            )

    def test_unconfirmed_word_deletion_leaves_all_records_and_files_unchanged(self):
        draft = self.create_draft("CancelDelete", "personal")
        sample_id = self.save_example(draft)
        entry_id = draft["draft"]["entry_id"]
        response = self.client.delete(
            f"/developer/api/drafts/{entry_id}",
            json={"scope": "personal", "confirmed": False},
            headers={"Host": "localhost"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(len(self.store.list_entries("personal")), 1)
        self.assertEqual(len(self.store.list_sessions("personal")), 1)
        self.assertEqual(self.store.count_samples("personal", entry_id), 1)
        self.assertTrue(
            (self.store.root / "personal" / "samples" / f"{sample_id}.npy").is_file()
        )

    def test_word_deletion_validates_entry_id_and_scope(self):
        invalid_id = self.client.delete(
            "/developer/api/drafts/not%20safe",
            json={"scope": "shared", "confirmed": True},
            headers={"Host": "localhost"},
        )
        invalid_scope = self.client.delete(
            "/developer/api/drafts/valid-id",
            json={"scope": "other", "confirmed": True},
            headers={"Host": "localhost"},
        )
        self.assertEqual(invalid_id.status_code, 422)
        self.assertEqual(invalid_scope.status_code, 400)
        self.assertEqual(self.store.list_entries("shared"), [])

    def test_draft_lists_model_artifacts_that_reference_the_word(self):
        draft = self.create_draft("ModelReference", "shared")
        entry = draft["draft"]
        label_map = (
            self.store.root / "shared" / "models" / "artifact-1" / "label_map.json"
        )
        label_map.parent.mkdir(parents=True)
        label_map.write_text(
            '{"label_map": [{"label_id": '
            f'{entry["label_id"]}, "label": "{entry["label"]}"' +
            ', "scope": "shared"}]}',
            encoding="utf-8",
        )

        response = self.client.get(
            "/developer/api/drafts?scope=shared",
            headers={"Host": "localhost"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json()["drafts"][0]["model_references"],
            [{"artifact_id": "artifact-1", "unreadable": False}],
        )

    def test_remote_client_is_denied(self):
        response = self.client.get(
            "/developer/api/drafts?scope=shared",
            headers={"Host": "localhost"},
            environ_base={"REMOTE_ADDR": "192.0.2.10"},
        )
        self.assertEqual(response.status_code, 403)

    def test_cross_origin_developer_request_is_denied(self):
        response = self.client.get(
            "/developer/api/drafts?scope=shared",
            headers={"Host": "localhost", "Origin": "https://attacker.example"},
        )
        self.assertEqual(response.status_code, 403)

    def test_capture_endpoint_validates_and_extracts_complete_sequence(self):
        image = np.full((32, 32, 3), 127, dtype=np.uint8)
        encoded_ok, encoded_image = cv2.imencode(".jpg", image)
        self.assertTrue(encoded_ok)
        import base64

        data_url = "data:image/jpeg;base64," + base64.b64encode(
            encoded_image.tobytes()
        ).decode("ascii")
        landmarks = [
            SimpleNamespace(x=index / 21, y=index / 21, z=0.1)
            for index in range(21)
        ]
        detection = SimpleNamespace(
            pose_landmarks=[],
            face_landmarks=[],
            left_hand_landmarks=landmarks,
            right_hand_landmarks=[],
        )
        detector = SimpleNamespace(detect=lambda image: detection, close=lambda: None)
        with self.app_module._lock:
            self.app_module._developer_capture_lease_until = (
                time.monotonic() + 60
            )
        with patch.object(self.app_module, "_build_holistic", return_value=detector):
            response = self.client.post(
                "/developer/api/capture/process",
                json={"frames": [data_url] * 30},
                headers={"Host": "localhost"},
            )
        self.assertEqual(response.status_code, 200, response.get_json())
        result = response.get_json()
        self.assertEqual(len(result["sequence"]), 30)
        self.assertTrue(all(len(frame) == 378 for frame in result["sequence"]))
        self.assertEqual(
            result["quality"]["preprocessing_version"], PREPROCESSING_VERSION
        )

    def test_camera_lease_prevents_regular_camera_start(self):
        with self.app_module._lock:
            self.app_module._developer_capture_lease_until = (
                time.monotonic() + 60
            )
        response = self.client.post(
            "/control",
            json={"action": "start_camera"},
            headers={"Host": "localhost"},
        )
        self.assertEqual(response.status_code, 409)
        self.assertFalse(self.app_module._state["camera_active"])

    def test_developer_capture_reports_missing_holistic_asset(self):
        with patch.object(
            self.app_module,
            "HOLISTIC_MODEL",
            str(Path(self.temporary_directory.name) / "missing.task"),
        ):
            response = self.client.post(
                "/developer/api/capture/process",
                json={"frames": []},
                headers={"Host": "localhost"},
            )
        self.assertEqual(response.status_code, 503)
        self.assertIn("MediaPipe Holistic task file", response.get_json()["error"])

    def test_browser_site_remains_available_without_legacy_model_files(self):
        with (
            patch.object(self.app_module, "_model", None),
            patch.object(
                self.app_module,
                "HOLISTIC_MODEL",
                str(Path(self.temporary_directory.name) / "missing.task"),
            ),
        ):
            page = self.client.get("/")
            self.assertEqual(page.status_code, 200)
            self.assertIn(b"knn-recognizer.bundle.js", page.data)
            page.close()

            response = self.client.post(
                "/control",
                json={"action": "start_camera"},
                headers={"Host": "localhost"},
            )
        self.assertEqual(response.status_code, 503)
        self.assertIn("model files", response.get_json()["error"])
        self.assertFalse(self.app_module._state["camera_active"])


if __name__ == "__main__":
    unittest.main()
