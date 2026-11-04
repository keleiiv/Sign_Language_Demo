import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.sequence_contract import (
    FACE_DIM,
    FEATURE_DIM,
    MAX_FRAMES,
    PREPROCESSING_VERSION,
    POSE_DIM,
    SEQUENCE_SHAPE,
    HAND_DIM,
    SequenceContractError,
    build_sequence,
    extract_frame_features,
    validate_label_id,
    validate_sequence,
)
from scripts.vocabulary_store import (
    CaptureSession,
    VocabularyEntry,
    VocabularyStore,
)


def _landmarks(count, visibility=False):
    return [
        SimpleNamespace(
            x=float(index),
            y=float(index + 1),
            z=float(index + 2),
            **({"visibility": float(index + 3)} if visibility else {}),
        )
        for index in range(count)
    ]


class SequenceContractTests(unittest.TestCase):
    def test_extraction_preserves_live_layout_and_float32(self):
        result = SimpleNamespace(
            pose_landmarks=_landmarks(33, visibility=True),
            face_landmarks=_landmarks(478),
            left_hand_landmarks=_landmarks(21),
            right_hand_landmarks=[],
        )

        frame = extract_frame_features(result)

        self.assertEqual(frame.shape, (FEATURE_DIM,))
        self.assertEqual(frame.dtype, np.float32)
        self.assertEqual(FEATURE_DIM, 378)
        self.assertEqual(FACE_DIM, 120)
        np.testing.assert_array_equal(frame[:4], [0, 1, 2, 3])
        self.assertEqual(frame[POSE_DIM], 69)
        self.assertEqual(frame[-HAND_DIM:].shape[0], HAND_DIM)
        np.testing.assert_array_equal(frame[-HAND_DIM:], np.zeros(HAND_DIM))

    def test_missing_landmarks_produce_zero_filled_frame(self):
        result = SimpleNamespace(
            pose_landmarks=[],
            face_landmarks=[],
            left_hand_landmarks=[],
            right_hand_landmarks=[],
        )
        np.testing.assert_array_equal(
            extract_frame_features(result), np.zeros(FEATURE_DIM, dtype=np.float32)
        )

    def test_sequence_shape_version_and_finite_values_are_enforced(self):
        valid = np.zeros(SEQUENCE_SHAPE, dtype=np.float32)
        self.assertEqual(
            validate_sequence(valid, PREPROCESSING_VERSION).shape, SEQUENCE_SHAPE
        )
        self.assertEqual(
            build_sequence([row for row in valid], PREPROCESSING_VERSION).shape,
            (MAX_FRAMES, FEATURE_DIM),
        )
        with self.assertRaises(SequenceContractError):
            validate_sequence(valid[:-1], PREPROCESSING_VERSION)
        with self.assertRaises(SequenceContractError):
            validate_sequence(valid, "legacy-1692")
        invalid = valid.copy()
        invalid[0, 0] = np.nan
        with self.assertRaises(SequenceContractError):
            validate_sequence(invalid, PREPROCESSING_VERSION)

    def test_label_ids_must_be_non_negative_integers(self):
        self.assertEqual(validate_label_id(0), 0)
        for invalid in (-1, True, 1.0, "1"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_label_id(invalid)


class VocabularyStoreTests(unittest.TestCase):
    def test_records_and_sample_persist_with_scope_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = VocabularyStore(root)
            shared = VocabularyEntry("hello", 0, "Hello", "shared")
            personal = VocabularyEntry("my-sign", 0, "My sign", "personal")
            store.add_entry(shared)
            store.add_entry(personal)
            shared_session = CaptureSession.create(shared)
            personal_session = CaptureSession.create(personal)
            store.create_session(shared_session)
            store.create_session(personal_session)

            sequence = np.zeros(SEQUENCE_SHAPE, dtype=np.float32)
            sample = store.save_sample(
                shared_session.session_id,
                shared.entry_id,
                sequence,
                PREPROCESSING_VERSION,
                "shared",
            )
            store.save_sample(
                personal_session.session_id,
                personal.entry_id,
                sequence + 1,
                PREPROCESSING_VERSION,
                "personal",
            )

            reopened = VocabularyStore(root)
            self.assertEqual(reopened.list_entries("shared"), [shared])
            self.assertEqual(reopened.list_entries("personal"), [personal])
            shared_samples = reopened.list_samples("shared")
            personal_samples = reopened.list_samples("personal")
            self.assertEqual(len(shared_samples), 1)
            self.assertEqual(len(personal_samples), 1)
            self.assertEqual(shared_samples[0].scope, "shared")
            self.assertEqual(personal_samples[0].scope, "personal")
            np.testing.assert_array_equal(
                reopened.load_sample(shared_samples[0], "shared"), sequence
            )
            with self.assertRaises(ValueError):
                reopened.load_sample(shared_samples[0], "personal")
            self.assertNotEqual(
                sample.sample_id, personal_samples[0].sample_id
            )

    def test_entries_reject_duplicate_ids_and_label_ids_per_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            store = VocabularyStore(directory)
            store.add_entry(VocabularyEntry("first", 0, "First", "shared"))
            with self.assertRaises(ValueError):
                store.add_entry(VocabularyEntry("second", 0, "Second", "shared"))
            with self.assertRaises(ValueError):
                store.add_entry(VocabularyEntry("first", 1, "Other", "shared"))
            store.add_entry(VocabularyEntry("first", 0, "Personal", "personal"))

    def test_store_rejects_wrong_scope_entry_and_preprocessing(self):
        with tempfile.TemporaryDirectory() as directory:
            store = VocabularyStore(directory)
            entry = VocabularyEntry("hello", 0, "Hello", "shared")
            store.add_entry(entry)
            session = CaptureSession.create(entry)
            store.create_session(session)
            with self.assertRaises(ValueError):
                store.save_sample(
                    session.session_id,
                    entry.entry_id,
                    np.zeros(SEQUENCE_SHAPE, dtype=np.float32),
                    "legacy-1692",
                    "shared",
                )
            with self.assertRaises(ValueError):
                store.save_sample(
                    session.session_id,
                    entry.entry_id,
                    np.zeros(SEQUENCE_SHAPE, dtype=np.float32),
                    PREPROCESSING_VERSION,
                    "personal",
                )

    def test_deleting_selected_samples_preserves_other_records_and_sessions(self):
        with tempfile.TemporaryDirectory() as directory:
            store = VocabularyStore(directory)
            okay = VocabularyEntry("okay", 0, "Okay", "shared")
            other = VocabularyEntry("other", 1, "Other", "shared")
            personal = VocabularyEntry("okay", 0, "Okay", "personal")
            for entry in (okay, other, personal):
                store.add_entry(entry)
            sessions = {
                (entry.scope, entry.entry_id): CaptureSession.create(entry)
                for entry in (okay, other, personal)
            }
            for session in sessions.values():
                store.create_session(session)

            sequence = np.zeros(SEQUENCE_SHAPE, dtype=np.float32)
            target_samples = [
                store.save_sample(
                    sessions[("shared", "okay")].session_id,
                    "okay",
                    sequence + value,
                    PREPROCESSING_VERSION,
                    "shared",
                )
                for value in range(3)
            ]
            other_sample = store.save_sample(
                sessions[("shared", "other")].session_id,
                "other",
                sequence + 10,
                PREPROCESSING_VERSION,
                "shared",
            )
            personal_sample = store.save_sample(
                sessions[("personal", "okay")].session_id,
                "okay",
                sequence + 20,
                PREPROCESSING_VERSION,
                "personal",
            )
            removed = store.delete_samples(
                "okay",
                [target_samples[0].sample_id, target_samples[2].sample_id],
                "shared",
            )

            self.assertEqual(
                {sample.sample_id for sample in removed},
                {target_samples[0].sample_id, target_samples[2].sample_id},
            )
            reopened = VocabularyStore(directory)
            self.assertEqual(reopened.list_entries("shared"), [okay, other])
            self.assertEqual(reopened.list_entries("personal"), [personal])
            self.assertEqual(
                {session.session_id for session in reopened.list_sessions("shared")},
                {sessions[("shared", "okay")].session_id,
                 sessions[("shared", "other")].session_id},
            )
            remaining_shared = reopened.list_samples("shared")
            self.assertEqual(
                {sample.sample_id for sample in remaining_shared},
                {target_samples[1].sample_id, other_sample.sample_id},
            )
            self.assertEqual(
                [sample.sample_id for sample in reopened.list_samples("personal")],
                [personal_sample.sample_id],
            )
            self.assertEqual(
                [session.session_id for session in reopened.list_sessions("personal")],
                [sessions[("personal", "okay")].session_id],
            )
            np.testing.assert_array_equal(
                reopened.load_sample(target_samples[1], "shared"), sequence + 1
            )
            np.testing.assert_array_equal(
                reopened.load_sample(personal_sample, "personal"), sequence + 20
            )
            self.assertFalse(
                (Path(directory) / "shared" / "samples"
                 / target_samples[0].file_name).exists()
            )
            self.assertEqual(
                reopened.count_samples("shared", "okay"), 1
            )
            self.assertEqual(
                reopened.count_samples("shared", "other"), 1
            )
            self.assertEqual(
                reopened.count_samples("personal", "okay"), 1
            )

    def test_deleting_samples_rejects_invalid_or_cross_entry_ids_without_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            store = VocabularyStore(directory)
            entry = VocabularyEntry("okay", 0, "Okay", "shared")
            other = VocabularyEntry("other", 1, "Other", "shared")
            store.add_entry(entry)
            store.add_entry(other)
            session = CaptureSession.create(entry)
            other_session = CaptureSession.create(other)
            store.create_session(session)
            store.create_session(other_session)
            sample = store.save_sample(
                session.session_id, "okay",
                np.zeros(SEQUENCE_SHAPE, dtype=np.float32),
                PREPROCESSING_VERSION, "shared",
            )
            other_sample = store.save_sample(
                other_session.session_id, "other",
                np.ones(SEQUENCE_SHAPE, dtype=np.float32),
                PREPROCESSING_VERSION, "shared",
            )

            for ids in ([], [sample.sample_id, sample.sample_id],
                        [sample.sample_id, other_sample.sample_id], ["../outside"]):
                with self.subTest(ids=ids), self.assertRaises(ValueError):
                    store.delete_samples("okay", ids, "shared")
            self.assertEqual(store.count_samples("shared", "okay"), 1)
            self.assertEqual(store.count_samples("shared", "other"), 1)

    def test_deleting_vocabulary_entry_removes_only_its_scoped_data(self):
        with tempfile.TemporaryDirectory() as directory:
            store = VocabularyStore(directory)
            target = VocabularyEntry("target", 10, "Target", "shared")
            other = VocabularyEntry("other", 11, "Other", "shared")
            personal = VocabularyEntry("target", 10, "Target", "personal")
            entries = (target, other, personal)
            sessions = {}
            sample_ids = {}
            for entry in entries:
                store.add_entry(entry)
                session = CaptureSession.create(entry)
                store.create_session(session)
                sessions[(entry.scope, entry.entry_id)] = session
                sample = store.save_sample(
                    session.session_id,
                    entry.entry_id,
                    np.zeros(SEQUENCE_SHAPE, dtype=np.float32),
                    PREPROCESSING_VERSION,
                    entry.scope,
                )
                sample_ids[(entry.scope, entry.entry_id)] = sample.sample_id

            result = store.delete_entry("target", "shared")

            self.assertEqual(result["deleted_samples"], 1)
            self.assertEqual(result["deleted_sessions"], 1)
            self.assertEqual(store.list_entries("shared"), [other])
            self.assertEqual(store.list_sessions("shared"), [sessions[("shared", "other")]])
            self.assertEqual(
                [sample.sample_id for sample in store.list_samples("shared")],
                [sample_ids[("shared", "other")]],
            )
            shared_other_sample_id = sample_ids[("shared", "other")]
            shared_target_sample_id = sample_ids[("shared", "target")]
            self.assertTrue(
                (Path(directory) / "shared" / "samples" /
                 f"{shared_other_sample_id}.npy").is_file()
            )
            self.assertFalse(
                (Path(directory) / "shared" / "samples" /
                 f"{shared_target_sample_id}.npy").exists()
            )
            self.assertEqual(store.list_entries("personal"), [personal])
            self.assertEqual(store.list_sessions("personal"), [sessions[("personal", "target")]])
            self.assertEqual(store.count_samples("personal", "target"), 1)

    def test_deleting_vocabulary_entry_rejects_invalid_scope_or_id_without_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            store = VocabularyStore(directory)
            entry = VocabularyEntry("target", 10, "Target", "shared")
            store.add_entry(entry)
            session = CaptureSession.create(entry)
            store.create_session(session)

            for entry_id, scope in (("not safe", "shared"), ("target", "unknown")):
                with self.subTest(entry_id=entry_id, scope=scope):
                    with self.assertRaises(ValueError):
                        store.delete_entry(entry_id, scope)
            self.assertEqual(store.list_entries("shared"), [entry])
            self.assertEqual(store.list_sessions("shared"), [session])


if __name__ == "__main__":
    unittest.main()
