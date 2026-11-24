"""Tests for Sentence Demo Mode configuration, validation, and API endpoints."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from app import app, _state, _lock
from scripts.sentence_demo_manager import SentenceDemoManager, normalize_label
from scripts.sequence_contract import PREPROCESSING_VERSION
from scripts.vocabulary_store import CaptureSession, VocabularyEntry, VocabularyStore


class SentenceDemoManagerTestCase(unittest.TestCase):
    """Unit tests for SentenceDemoManager configuration handling and validation."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_path = self.temp_dir / "test_mappings.json"
        self.mock_supported_labels = ["sorry", "yes", "no", "please", "help", "water"]
        self.manager = SentenceDemoManager(
            config_path=self.config_path,
            supported_labels_provider=lambda: self.mock_supported_labels,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_auto_create_default_config_if_missing(self):
        """If config file is missing, default config is created and loaded."""
        self.assertTrue(self.config_path.is_file())
        mappings = self.manager.load_mappings()
        self.assertIn("help", mappings)
        self.assertEqual(mappings["help"], "Could you please help me?")

    def test_lookup_case_insensitive_and_trimmed(self):
        """Lookup matches case-insensitively and ignores leading/trailing spaces."""
        res = self.manager.lookup_sentence("  HELP  ")
        self.assertIsNotNone(res)
        self.assertEqual(res["matched_sign"], "help")
        self.assertEqual(res["sentence"], "Could you please help me?")

        res2 = self.manager.lookup_sentence("Water")
        self.assertIsNotNone(res2)
        self.assertEqual(res2["sentence"], "May I please have a glass of water?")

    def test_lookup_unmapped_sign_returns_none(self):
        """Signs without a configured mapping return None without invent or fallback."""
        res = self.manager.lookup_sentence("unknown_sign")
        self.assertIsNone(res)

    def test_save_and_reload_mappings(self):
        """Adding and removing mappings without code changes persists to disk."""
        new_mappings = {
            "help": "I need assistance right away.",
            "water": "Could I have some cold water?",
            "custom_sign": "This is a custom demonstration phrase.",
        }
        ok, err = self.manager.save_mappings(new_mappings)
        self.assertTrue(ok)
        self.assertIsNone(err)

        # Create new manager pointing to same file
        reloaded_manager = SentenceDemoManager(
            config_path=self.config_path,
            supported_labels_provider=lambda: self.mock_supported_labels,
        )
        loaded = reloaded_manager.load_mappings(force=True)
        self.assertEqual(len(loaded), 3)
        self.assertEqual(loaded["help"], "I need assistance right away.")
        self.assertEqual(loaded["custom_sign"], "This is a custom demonstration phrase.")

    def test_save_mapping_merges_without_changing_original_mappings(self):
        original_path = Path(__file__).resolve().parents[1] / "data" / "sentence_demo_mappings.json"
        original = json.loads(original_path.read_text(encoding="utf-8"))["mappings"]
        self.assertTrue(self.manager.save_mappings(original)[0])
        ok, error = self.manager.save_mapping("good morning", "Good morning! Have a great day.")
        self.assertTrue(ok, error)
        loaded = self.manager.load_mappings(force=True)
        for label, sentence in original.items():
            self.assertEqual(loaded[label], sentence)
        self.assertEqual(loaded["good morning"], "Good morning! Have a great day.")

    def test_save_mapping_rejects_empty_sentence_and_invalid_label(self):
        self.assertFalse(self.manager.save_mapping("valid label", "   ")[0])
        self.assertFalse(self.manager.save_mapping("bad/label", "A sentence.")[0])

    def test_validation_identifies_unsupported_labels(self):
        """Validation report accurately flags labels not in the active model vocabulary."""
        custom_mappings = {
            "help": "Help requested.",       # supported
            "water": "Water requested.",     # supported
            "rocket": "Fly to space.",       # unsupported by mock_supported_labels
        }
        self.manager.save_mappings(custom_mappings)
        report = self.manager.validate_current_configuration()

        self.assertEqual(report["total_mappings"], 3)
        self.assertEqual(report["supported_count"], 2)
        self.assertEqual(report["unsupported_count"], 1)

        entries_by_sign = {e["sign"]: e for e in report["entries"]}
        self.assertTrue(entries_by_sign["help"]["supported"])
        self.assertFalse(entries_by_sign["rocket"]["supported"])
        self.assertIn("not in the active model vocabulary", entries_by_sign["rocket"]["reason"])

    def test_malformed_json_handled_gracefully(self):
        """Corrupted JSON file does not crash the application and reports the error."""
        with open(self.config_path, "w", encoding="utf-8") as f:
            f.write("{ invalid json content :::")

        mgr = SentenceDemoManager(config_path=self.config_path)
        # Does not raise exception
        mappings = mgr.load_mappings(force=True)
        self.assertIsNotNone(mgr._last_load_error)
        self.assertIn("Malformed JSON", mgr._last_load_error)

    def test_empty_keys_and_sentences_filtered(self):
        """Empty or whitespace-only keys and sentences are safely filtered."""
        invalid_payload = {
            "mappings": {
                " ": "Sentence with empty key",
                "valid_key": "   ",
                "good_sign": "Valid sentence here.",
            }
        }
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(invalid_payload, f)

        mgr = SentenceDemoManager(config_path=self.config_path)
        mappings = mgr.load_mappings(force=True)
        self.assertIn("good_sign", mappings)
        self.assertNotIn(" ", mappings)
        self.assertNotIn("valid_key", mappings)

    def test_array_format_supported(self):
        """Array of {sign, sentence} objects is also parsed correctly."""
        array_payload = {
            "mappings": [
                {"sign": "help", "sentence": "Help me please."},
                {"sign": "yes", "sentence": "Yes indeed."},
            ]
        }
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(array_payload, f)

        mgr = SentenceDemoManager(config_path=self.config_path)
        mappings = mgr.load_mappings(force=True)
        self.assertEqual(len(mappings), 2)
        self.assertEqual(mappings["help"], "Help me please.")
        self.assertEqual(mappings["yes"], "Yes indeed.")


class SentenceDemoApiTestCase(unittest.TestCase):
    """Integration tests for Flask API endpoints."""

    def setUp(self):
        self.client = app.test_client()
        from app import SENTENCE_DEMO_MANAGER
        self.mgr = SENTENCE_DEMO_MANAGER
        self.orig_content = None
        if self.mgr.config_path.is_file():
            with open(self.mgr.config_path, "r", encoding="utf-8") as f:
                self.orig_content = f.read()

    def tearDown(self):
        if self.orig_content is not None and self.mgr.config_path.is_file():
            with open(self.mgr.config_path, "w", encoding="utf-8") as f:
                f.write(self.orig_content)
            self.mgr.load_mappings(force=True)

    def test_get_config_endpoint(self):
        """GET /api/sentence-demo/config returns 200 with mappings and validation."""
        res = self.client.get("/api/sentence-demo/config")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("status"), "ok")
        self.assertIn("mappings", data)
        self.assertIn("validation", data)
        self.assertIsInstance(data["validation"].get("entries"), list)

    def test_post_config_valid_and_invalid(self):
        """POST /api/sentence-demo/config saves valid mappings or returns 400 on error."""
        # Invalid body
        bad_res = self.client.post("/api/sentence-demo/config", json="not a dict")
        self.assertEqual(bad_res.status_code, 400)

        # Empty mappings
        empty_res = self.client.post("/api/sentence-demo/config", json={"mappings": {}})
        self.assertEqual(empty_res.status_code, 400)

        # Valid update
        good_payload = {
            "mappings": {
                "help": "Could you please help me?",
                "water": "May I please have a glass of water?",
                "sorry": "I apologize for the misunderstanding.",
            }
        }
        good_res = self.client.post("/api/sentence-demo/config", json=good_payload)
        self.assertEqual(good_res.status_code, 200)
        saved_data = good_res.get_json()
        self.assertEqual(saved_data.get("status"), "saved")
        self.assertIn("help", saved_data.get("mappings", {}))

    def test_trigger_endpoint_confidence_threshold(self):
        """POST /api/sentence-demo/trigger respects confidence threshold and mappings."""
        # Low confidence rejected
        low_res = self.client.post(
            "/api/sentence-demo/trigger",
            json={"label": "help", "confidence": 0.30, "min_confidence": 0.55, "speak": False},
        )
        self.assertEqual(low_res.status_code, 200)
        self.assertFalse(low_res.get_json().get("triggered"))

        # Unknown label rejected
        unmapped_res = self.client.post(
            "/api/sentence-demo/trigger",
            json={"label": "unmapped_xyz", "confidence": 0.95, "speak": False},
        )
        self.assertEqual(unmapped_res.status_code, 200)
        self.assertFalse(unmapped_res.get_json().get("triggered"))
        self.assertIn("No sentence mapping configured", unmapped_res.get_json().get("reason", ""))

        # Valid mapped trigger
        valid_res = self.client.post(
            "/api/sentence-demo/trigger",
            json={"label": "help", "confidence": 0.88, "speak": False},
        )
        self.assertEqual(valid_res.status_code, 200)
        data = valid_res.get_json()
        self.assertTrue(data.get("triggered"))
        self.assertEqual(data.get("sentence"), "Could you please help me?")

    def test_control_mode_toggle_and_clear(self):
        """POST /control enables sentence demo mode and resets state."""
        # Enable demo mode
        toggle_res = self.client.post(
            "/control",
            json={"action": "set_sentence_demo_mode", "enabled": True},
        )
        self.assertEqual(toggle_res.status_code, 200)
        status_res = self.client.get("/status")
        self.assertTrue(status_res.get_json().get("sentence_demo_mode"))

        # Clear demo state
        clear_res = self.client.post(
            "/control",
            json={"action": "clear_sentence_demo"},
        )
        self.assertEqual(clear_res.status_code, 200)
        status_after = self.client.get("/status").get_json()
        self.assertEqual(status_after.get("sentence_demo_sentence"), "")
        self.assertEqual(status_after.get("sentence_demo_trigger"), "")

        # Disable demo mode
        self.client.post(
            "/control",
            json={"action": "set_sentence_demo_mode", "enabled": False},
        )
        status_final = self.client.get("/status").get_json()
        self.assertFalse(status_final.get("sentence_demo_mode"))


class DeveloperSentenceMappingApiTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = VocabularyStore(Path(self.temp_dir.name) / "vocabulary")
        self.manager = SentenceDemoManager(Path(self.temp_dir.name) / "mappings.json")
        self.entry = VocabularyEntry(entry_id="morning-sign", label_id=1, label="good morning", scope="shared")
        self.store.add_entry(self.entry)
        self.client = app.test_client()
        self.store_patch = patch("app.VOCABULARY_STORE", self.store)
        self.manager_patch = patch("app.SENTENCE_DEMO_MANAGER", self.manager)
        self.status_patch = patch("app._experimental_status", side_effect=self.fake_status)
        self.store_patch.start()
        self.manager_patch.start()
        self.status_patch.start()

    def tearDown(self):
        self.status_patch.stop()
        self.manager_patch.stop()
        self.store_patch.stop()
        self.temp_dir.cleanup()

    def fake_status(self, scope, include_candidates=True):
        supported = getattr(self, "active", False) and scope == self.entry.scope
        supports = [{"entry_id": self.entry.entry_id, "label": self.entry.label}] if supported else []
        return {"active_model": {"usable": bool(supports), "supports": supports}}

    def put_mapping(self, sentence):
        return self.client.put(
            f"/developer/api/drafts/{self.entry.entry_id}/sentence-mapping",
            json={"scope": "shared", "sentence": sentence},
            headers={"Host": "localhost"},
        )

    def test_active_label_mapping_is_saved_and_available(self):
        self.active = True
        response = self.put_mapping("Good morning!")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["state"], "available")

    def test_unactivated_label_mapping_is_pending_and_not_recognizable(self):
        self.active = False
        session = CaptureSession.create(self.entry)
        self.store.create_session(session)
        self.store.save_sample(
            session.session_id,
            self.entry.entry_id,
            np.zeros((30, 378), dtype=np.float32),
            PREPROCESSING_VERSION,
            "shared",
        )
        response = self.put_mapping("Good morning!")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["state"], "pending_activation")
        self.assertFalse(response.get_json()["recognizable_by_active_model"])
        vocabulary = self.client.get("/api/vocabulary?scope=shared&include_candidates=false").get_json()
        self.assertFalse(vocabulary["words"][0]["recognizable_by_active_model"])

    def test_mapping_can_be_edited_without_recording_again(self):
        self.assertEqual(self.put_mapping("First sentence.").status_code, 200)
        sample_count = self.store.count_samples("shared", self.entry.entry_id)
        response = self.put_mapping("Edited sentence.")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.manager.lookup_sentence(self.entry.label)["sentence"], "Edited sentence.")
        self.assertEqual(self.store.count_samples("shared", self.entry.entry_id), sample_count)

    def test_mapping_becomes_available_after_activation_and_is_loaded_by_translator_config(self):
        self.active = False
        self.assertEqual(self.put_mapping("Good morning!").get_json()["state"], "pending_activation")
        self.active = True
        status = self.client.get(
            f"/developer/api/drafts/{self.entry.entry_id}/sentence-mapping?scope=shared",
            headers={"Host": "localhost"},
        ).get_json()
        self.assertEqual(status["state"], "available")
        config = self.client.get("/api/sentence-demo/config").get_json()
        self.assertEqual(config["mappings"]["good morning"], "Good morning!")

    def test_empty_or_oversized_sentences_are_rejected(self):
        self.assertEqual(self.put_mapping("   ").status_code, 400)
        self.assertEqual(self.put_mapping("x" * 1001).status_code, 400)

    def test_builtin_label_mapping_saves_without_creating_vocabulary_entry(self):
        with patch("app._base_label_names", return_value=["Doctor"]):
            labels = self.client.get(
                "/developer/api/builtin-sentence-mapping",
                headers={"Host": "localhost"},
            )
            self.assertEqual(labels.status_code, 200)
            self.assertEqual(labels.get_json()["labels"], ["Doctor"])

            saved = self.client.put(
                "/developer/api/builtin-sentence-mapping",
                json={"label": "doctor", "sentence": "I need a doctor."},
                headers={"Host": "localhost"},
            )
            self.assertEqual(saved.status_code, 200)
            self.assertEqual(saved.get_json()["mapping"]["sentence"], "I need a doctor.")
            self.assertEqual(
                self.manager.lookup_sentence("doctor")["sentence"], "I need a doctor."
            )
            config = self.client.get("/api/sentence-demo/config").get_json()
            self.assertEqual(
                next(value for key, value in config["mappings"].items()
                     if key.casefold() == "doctor"),
                "I need a doctor.",
            )
            self.assertEqual(len(self.store.list_entries("shared")), 1)

            duplicate_draft = self.client.post(
                "/developer/api/drafts",
                json={"label": "Doctor", "scope": "shared"},
                headers={"Host": "localhost"},
            )
            self.assertEqual(duplicate_draft.status_code, 409)
            self.assertEqual(len(self.store.list_entries("shared")), 1)


if __name__ == "__main__":
    unittest.main()
