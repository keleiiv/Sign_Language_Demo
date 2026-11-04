import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.expandable_classifier import (
    ABSTENTION_THRESHOLD,
    ArtifactCompatibilityError,
    InsufficientDataError,
    activate_artifact,
    dataset_fingerprint,
    load_artifact,
    split_by_session,
    temporal_features,
    train_and_evaluate,
)
from scripts.sequence_contract import (
    PREPROCESSING_VERSION,
    SEQUENCE_SHAPE,
    SequenceContractError,
)
from scripts.vocabulary_store import (
    CaptureSession,
    VocabularyEntry,
    VocabularyStore,
)


def make_sequence(value):
    sequence = np.zeros(SEQUENCE_SHAPE, dtype=np.float32)
    sequence[:, 252:315] = value
    sequence[15:, 315:378] = value / 2
    return sequence


def populate_scope(store, scope, label_offset=0, sessions_per_class=4):
    entries = [
        VocabularyEntry("hello", label_offset, "Hello", scope),
        VocabularyEntry("fish", label_offset + 1, "Fish", scope),
    ]
    for entry in entries:
        store.add_entry(entry)
        for session_index in range(sessions_per_class):
            session = CaptureSession.create(entry)
            store.create_session(session)
            value = 0.2 if entry.label == "Hello" else 0.8
            sequence = make_sequence(value + session_index * 0.001)
            store.save_sample(
                session.session_id,
                entry.entry_id,
                sequence,
                PREPROCESSING_VERSION,
                scope,
            )


class TemporalClassifierTests(unittest.TestCase):
    def test_dataset_fingerprint_is_order_independent_and_tracks_sequence_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            store = VocabularyStore(directory)
            populate_scope(store, "shared", sessions_per_class=2)
            examples = store.load_training_examples("shared")
            baseline = dataset_fingerprint(examples)
            self.assertEqual(dataset_fingerprint(list(reversed(examples))), baseline)
            examples[0].sequence[0, 0] = 0.125
            self.assertNotEqual(dataset_fingerprint(examples), baseline)

    def test_temporal_features_encode_ordered_bins_and_differences(self):
        sequence = make_sequence(0.4)
        features = temporal_features(sequence)
        self.assertEqual(features.shape, (9 * 378,))
        self.assertTrue(np.isfinite(features).all())
        with self.assertRaises(SequenceContractError):
            temporal_features(np.zeros((29, 378), dtype=np.float32))

    def test_session_split_keeps_sessions_disjoint_and_requires_four_for_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            store = VocabularyStore(directory)
            populate_scope(store, "shared", sessions_per_class=4)
            examples = store.load_training_examples("shared")
            split = split_by_session(examples, seed=10)
            self.assertTrue(split.evaluation_ready)
            for label in ("Hello", "Fish"):
                train = set(split.train_sessions[label])
                validation = set(split.validation_sessions[label])
                self.assertEqual(len(train), 2)
                self.assertEqual(len(validation), 2)
                self.assertFalse(train & validation)

    def test_training_saves_loadable_versioned_artifact_and_rollback_pointer(self):
        with tempfile.TemporaryDirectory() as directory:
            store = VocabularyStore(directory)
            populate_scope(store, "shared")
            first_report = train_and_evaluate(store, "shared", seed=123, promote=True)
            first_id = first_report["artifact"]["artifact_id"]
            self.assertTrue(first_report["evaluation_ready"])
            self.assertEqual(
                len(first_report["data"]["fingerprint"]), 64
            )
            self.assertIsNotNone(first_report["validation"]["macro_f1"])
            self.assertIn("Hello", first_report["validation"]["per_class"])
            self.assertIn("Fish", first_report["validation"]["per_class"])
            for key in (
                "metadata", "parameters", "label_map", "preprocessing",
                "training_metadata", "evaluation_report",
            ):
                self.assertTrue(Path(first_report["artifact"][key]).is_file())

            classifier = load_artifact(store, "shared")
            prediction = classifier.predict(make_sequence(0.2))
            self.assertIn(prediction["label"], {"Hello", "Fish", None})
            self.assertGreaterEqual(prediction["confidence"], 0.0)
            self.assertLessEqual(prediction["confidence"], 1.0)
            self.assertEqual(prediction["scope"], "shared")

            second_report = train_and_evaluate(store, "shared", seed=456)
            second_id = second_report["artifact"]["artifact_id"]
            self.assertNotEqual(first_id, second_id)
            activate_artifact(store, "shared", first_id)
            self.assertEqual(load_artifact(store, "shared").metadata["artifact_id"], first_id)
            self.assertEqual(len(list((Path(directory) / "shared" / "models").glob("*/metadata.json"))), 2)

    def test_personal_and_shared_training_data_are_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            store = VocabularyStore(directory)
            populate_scope(store, "shared", label_offset=0)
            populate_scope(store, "personal", label_offset=0)
            shared = store.load_training_examples("shared")
            personal = store.load_training_examples("personal")
            self.assertTrue(all(item.scope == "shared" for item in shared))
            self.assertTrue(all(item.scope == "personal" for item in personal))
            with self.assertRaises(ValueError):
                split_by_session(shared + personal)
            report = train_and_evaluate(store, "personal", seed=1, promote=True)
            with self.assertRaises(ArtifactCompatibilityError):
                load_artifact(store, "shared", report["artifact"]["artifact_id"])

    def test_unknown_background_metrics_require_separate_sessions(self):
        with tempfile.TemporaryDirectory() as directory:
            store = VocabularyStore(directory)
            populate_scope(store, "shared")
            unknown = [
                (
                    PREPROCESSING_VERSION,
                    make_sequence(0.5),
                    "unknown-session-1",
                    "unknown",
                ),
                (
                    PREPROCESSING_VERSION,
                    make_sequence(0.55),
                    "background-session-2",
                    "background",
                ),
            ]
            report = train_and_evaluate(
                store, "shared", seed=17, unknown_sequences=unknown
            )
            unknown_report = report["unknown_evaluation"]
            self.assertTrue(unknown_report["available"])
            self.assertEqual(unknown_report["independent_session_count"], 2)
            self.assertIn("false_accept_rate", unknown_report)
            self.assertIn("unknown", unknown_report["per_category"])
            self.assertIn("background", unknown_report["per_category"])

            overlapping = [
                (
                    PREPROCESSING_VERSION,
                    make_sequence(0.5),
                    report["split"]["train_sessions"]["Hello"][0],
                    "unknown",
                )
            ]
            with self.assertRaisesRegex(ValueError, "must not overlap"):
                train_and_evaluate(
                    store, "shared", seed=17, unknown_sequences=overlapping
                )

    def test_two_sessions_can_train_but_are_not_reported_as_evaluation_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            store = VocabularyStore(directory)
            populate_scope(store, "shared", sessions_per_class=2)
            report = train_and_evaluate(store, "shared")
            self.assertFalse(report["evaluation_ready"])
            self.assertIsNone(report["validation"])
            self.assertTrue(report["insufficiencies"])
            self.assertFalse(report["unknown_evaluation"]["available"])

    def test_one_session_per_class_blocks_training(self):
        with tempfile.TemporaryDirectory() as directory:
            store = VocabularyStore(directory)
            populate_scope(store, "shared", sessions_per_class=1)
            with self.assertRaises(InsufficientDataError):
                train_and_evaluate(store, "shared")

    def test_abstention_threshold_is_not_presented_as_unknown_validation(self):
        self.assertEqual(ABSTENTION_THRESHOLD, 0.55)


if __name__ == "__main__":
    unittest.main()
