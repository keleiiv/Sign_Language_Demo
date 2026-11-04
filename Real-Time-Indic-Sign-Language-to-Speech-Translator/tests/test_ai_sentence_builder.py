import json
import unittest
from unittest.mock import patch

from scripts.ai_sentence_builder import (
    OllamaSentenceBuilder,
    OllamaUnavailableError,
    SentenceBuilderError,
    RESPONSE_SCHEMA,
    validate_request,
    validate_response,
)


def valid_model_result(**updates):
    result = {
        "sentence": "I want some food.",
        "corrections": [
            {
                "index": 2,
                "original_label": "where",
                "suggested_label": "food",
                "reason": "The surrounding sequence makes a food request more plausible.",
            }
        ],
        "grammar_changes": ["Added natural English grammar."],
        "ambiguous": False,
        "uncertainty_note": "",
    }
    result.update(updates)
    return result


class FakeResponse:
    status = 200

    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _limit):
        return json.dumps(self.value).encode("utf-8")


class SentenceBuilderValidationTests(unittest.TestCase):
    def test_empty_invalid_or_oversized_sequences_are_rejected(self):
        with self.assertRaisesRegex(SentenceBuilderError, "at least one"):
            validate_request([], ["food"])
        with self.assertRaisesRegex(SentenceBuilderError, "at most"):
            validate_request([{"label": "food"}] * 33, ["food"])
        with self.assertRaisesRegex(SentenceBuilderError, "confidence"):
            validate_request(
                [{"label": "food", "confidence": 1.5}], ["food"]
            )
        with self.assertRaisesRegex(SentenceBuilderError, "uncertainty"):
            validate_request([{"label": None}], ["food"])
        with self.assertRaisesRegex(SentenceBuilderError, "not in"):
            validate_request(
                [{"label": "food", "user_corrected_label": "invented"}],
                ["food"],
            )

    def test_valid_request_marks_confidence_as_uncalibrated(self):
        signs, vocabulary = validate_request(
            [{"label": "food", "confidence": 0.72, "uncertain": False}],
            ["food", "where"],
        )
        self.assertEqual(vocabulary, ["food", "where"])
        self.assertEqual(signs[0]["confidence_kind"], "uncalibrated_raw_score")
        self.assertEqual(signs[0]["confidence"], 0.72)

    def test_response_preserves_original_label_and_separates_grammar(self):
        signs, vocabulary = validate_request(
            [
                {"label": "I", "confidence": 0.8},
                {"label": "want", "confidence": 0.7},
                {"label": "where", "confidence": 0.6},
            ],
            ["I", "want", "food", "where"],
        )
        result = validate_response(
            valid_model_result(), signs, vocabulary
        )
        self.assertEqual(result["original_sequence"][2]["label"], "where")
        self.assertEqual(result["corrections"][0]["original_label"], "where")
        self.assertEqual(result["corrections"][0]["suggested_label"], "food")
        self.assertEqual(result["grammar_changes"], ["Added natural English grammar."])

    def test_unverified_or_mismatched_corrections_are_rejected(self):
        signs, vocabulary = validate_request(
            [{"label": "where"}], ["where", "food"]
        )
        with self.assertRaisesRegex(SentenceBuilderError, "preserve"):
            validate_response(
                valid_model_result(corrections=[{
                    "index": 0,
                    "original_label": "food",
                    "suggested_label": "where",
                    "reason": "wrong original",
                }]),
                signs,
                vocabulary,
            )
        with self.assertRaisesRegex(SentenceBuilderError, "supported"):
            validate_response(
                valid_model_result(corrections=[{
                    "index": 0,
                    "original_label": "where",
                    "suggested_label": "invented",
                    "reason": "unsupported",
                }]),
                signs,
                vocabulary,
            )

    def test_noop_model_corrections_are_not_presented_as_proposals(self):
        signs, vocabulary = validate_request(
            [{"label": "where"}], ["where", "food"]
        )
        result = validate_response(
            valid_model_result(corrections=[{
                "index": 0,
                "original_label": "where",
                "suggested_label": "where",
                "reason": "No correction needed.",
            }]),
            signs,
            vocabulary,
        )
        self.assertEqual(result["corrections"], [])

    def test_user_correction_cannot_be_overwritten_by_ai(self):
        signs, vocabulary = validate_request(
            [{
                "label": "where",
                "user_corrected_label": "food",
            }],
            ["where", "food"],
        )
        with self.assertRaisesRegex(SentenceBuilderError, "preserve"):
            validate_response(
                valid_model_result(corrections=[{
                    "index": 0,
                    "original_label": "where",
                    "suggested_label": "where",
                    "reason": "revert",
                }]),
                signs,
                vocabulary,
            )

    def test_ambiguous_interpretation_requires_an_uncertainty_note(self):
        signs, vocabulary = validate_request(
            [{"label": "food"}], ["food"]
        )
        with self.assertRaisesRegex(SentenceBuilderError, "uncertainty"):
            validate_response(
                valid_model_result(
                    corrections=[],
                    ambiguous=True,
                    uncertainty_note="",
                ),
                signs,
                vocabulary,
            )
        result = validate_response(
            valid_model_result(
                corrections=[],
                ambiguous=True,
                uncertainty_note="The intended request is unclear.",
            ),
            signs,
            vocabulary,
        )
        self.assertTrue(result["ambiguous"])

    def test_uncertainty_note_sets_ambiguous_even_if_model_flag_is_false(self):
        signs, vocabulary = validate_request(
            [{"label": "food"}], ["food"]
        )
        result = validate_response(
            valid_model_result(
                corrections=[],
                ambiguous=False,
                uncertainty_note="The intended meaning is unclear.",
            ),
            signs,
            vocabulary,
        )
        self.assertTrue(result["ambiguous"])

    def test_all_abstentions_return_uncertain_result_without_model_call(self):
        provider = OllamaSentenceBuilder("http://127.0.0.1:11434", "test-model")
        with patch.object(provider, "_request_json") as request:
            result = provider.build_sentence(
                [{"label": None, "confidence": 0.2, "uncertain": True}],
                ["food"],
            )
        request.assert_not_called()
        self.assertTrue(result["ambiguous"])
        self.assertIsNone(result["original_sequence"][0]["label"])


class OllamaProviderTests(unittest.TestCase):
    def setUp(self):
        self.provider = OllamaSentenceBuilder(
            "http://127.0.0.1:11434", "qwen2.5:3b"
        )

    def test_builds_structured_payload_with_only_sign_sequence(self):
        response_body = {
            "message": {"content": json.dumps(valid_model_result())}
        }
        signs = [
            {"label": "I", "confidence": 0.8, "uncertain": False},
            {"label": "want", "confidence": 0.7, "uncertain": False},
            {"label": "where", "confidence": 0.6, "uncertain": True},
        ]
        with patch(
            "scripts.ai_sentence_builder.urllib.request.urlopen",
            return_value=FakeResponse(response_body),
        ) as urlopen:
            result = self.provider.build_sentence(
                signs, ["I", "want", "food", "where"]
            )
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        user_content = json.loads(payload["messages"][-1]["content"])
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/chat")
        self.assertEqual(payload["format"], RESPONSE_SCHEMA)
        self.assertEqual(payload["model"], "qwen2.5:3b")
        self.assertIn("ISL has its own grammar", payload["messages"][0]["content"])
        self.assertIn("do not propose corrections", payload["messages"][1]["content"])
        self.assertEqual([item["label"] for item in user_content["signs"]],
                         ["I", "want", "where"])
        self.assertNotIn("frames", user_content)
        self.assertEqual(result["sentence"], "I want some food.")

    def test_timeout_and_unavailable_service_are_explicit(self):
        with patch(
            "scripts.ai_sentence_builder.urllib.request.urlopen",
            side_effect=TimeoutError("timed out"),
        ):
            with self.assertRaises(OllamaUnavailableError):
                self.provider.build_sentence(
                    [{"label": "food"}], ["food"]
                )
            status = self.provider.status()
        self.assertFalse(status["available"])
        self.assertIn("Install and start Ollama", status["message"])

    def test_model_status_reports_missing_model(self):
        with patch.object(
            self.provider,
            "_request_json",
            return_value={"models": [{"name": "other:latest"}]},
        ):
            status = self.provider.status()
        self.assertTrue(status["reachable"])
        self.assertFalse(status["available"])
        self.assertIn("ollama pull qwen2.5:3b", status["message"])

    def test_invalid_provider_configuration_is_reported_without_import_failure(self):
        provider = OllamaSentenceBuilder("not-a-url", "")
        status = provider.status()
        self.assertFalse(status["available"])
        self.assertIn("HTTP(S) URL", status["message"])

    def test_malformed_model_output_is_rejected(self):
        with patch(
            "scripts.ai_sentence_builder.urllib.request.urlopen",
            return_value=FakeResponse({"message": {"content": "{not json"}}),
        ):
            with self.assertRaisesRegex(SentenceBuilderError, "malformed"):
                self.provider.build_sentence(
                    [{"label": "food"}], ["food"]
                )

    def test_short_sequence_corrections_are_suppressed_as_ambiguous(self):
        response_body = {
            "message": {"content": json.dumps(valid_model_result(
                corrections=[{
                    "index": 0,
                    "original_label": "where",
                    "suggested_label": "what",
                    "reason": "A context-based suggestion.",
                }]
            ))}
        }
        with patch(
            "scripts.ai_sentence_builder.urllib.request.urlopen",
            return_value=FakeResponse(response_body),
        ):
            result = self.provider.build_sentence(
                [{"label": "where"}, {"label": "food"}],
                ["where", "food", "what", "water"],
            )
        self.assertEqual(result["corrections"], [])
        self.assertTrue(result["ambiguous"])
        self.assertIn("too little context", result["uncertainty_note"])

    def test_two_sign_sequence_is_marked_ambiguous_without_corrections(self):
        response_body = {
            "message": {"content": json.dumps(valid_model_result(
                sentence="Where is the food?",
                corrections=[],
                grammar_changes=["Added English grammar."],
                ambiguous=False,
                uncertainty_note="",
            ))}
        }
        with patch(
            "scripts.ai_sentence_builder.urllib.request.urlopen",
            return_value=FakeResponse(response_body),
        ):
            result = self.provider.build_sentence(
                [{"label": "where"}, {"label": "food"}],
                ["where", "food"],
            )
        self.assertTrue(result["ambiguous"])
        self.assertIn("two-sign", result["uncertainty_note"])
        self.assertEqual(result["sentence"], "where food")

    def test_unresolved_multi_sign_gloss_echo_is_marked_ambiguous(self):
        response_body = {
            "message": {"content": json.dumps(valid_model_result(
                sentence="where food please",
                corrections=[],
                grammar_changes=[],
                ambiguous=False,
                uncertainty_note="",
            ))}
        }
        with patch(
            "scripts.ai_sentence_builder.urllib.request.urlopen",
            return_value=FakeResponse(response_body),
        ):
            result = self.provider.build_sentence(
                [{"label": label} for label in ("where", "food", "please")],
                ["where", "food", "please"],
            )
        self.assertTrue(result["ambiguous"])
        self.assertIn("echoed the gloss sequence", result["uncertainty_note"])

    def test_where_food_please_does_not_trigger_unsupported_correction(self):
        response_body = {
            "message": {"content": json.dumps(valid_model_result(
                sentence="Where food please?",
                corrections=[{
                    "index": 1,
                    "original_label": "food",
                    "suggested_label": "where",
                    "reason": "The sequence sounds like a question about a location.",
                }],
                grammar_changes=[],
                ambiguous=False,
                uncertainty_note="",
            ))}
        }
        signs = [{"label": label, "uncertain": False} for label in (
            "where", "food", "please"
        )]
        with patch(
            "scripts.ai_sentence_builder.urllib.request.urlopen",
            return_value=FakeResponse(response_body),
        ):
            result = self.provider.build_sentence(
                signs, ["where", "food", "please"]
            )
        self.assertEqual(result["sentence"], "where food please")
        self.assertEqual(result["corrections"], [])
        self.assertEqual(
            [sign["label"] for sign in result["original_sequence"]],
            ["where", "food", "please"],
        )
        self.assertTrue(result["ambiguous"])
        self.assertIn("gloss", result["uncertainty_note"])

    def test_ambiguous_model_output_never_exposes_its_corrections(self):
        response_body = {
            "message": {"content": json.dumps(valid_model_result(
                sentence="The food is somewhere.",
                corrections=[{
                    "index": 1,
                    "original_label": "food",
                    "suggested_label": "where",
                    "reason": "This may be a location question.",
                }],
                grammar_changes=["Added English grammar."],
                ambiguous=True,
                uncertainty_note="The intended meaning is unclear.",
            ))}
        }
        signs = [{"label": label, "uncertain": True} for label in (
            "where", "food", "please"
        )]
        with patch(
            "scripts.ai_sentence_builder.urllib.request.urlopen",
            return_value=FakeResponse(response_body),
        ):
            result = self.provider.build_sentence(
                signs, ["where", "food", "please"]
            )
        self.assertEqual(result["corrections"], [])
        self.assertEqual(result["sentence"], "where food please")
        self.assertEqual(
            [sign["label"] for sign in result["original_sequence"]],
            ["where", "food", "please"],
        )
        self.assertTrue(result["ambiguous"])

    def test_no_correction_is_added_when_model_does_not_justify_one(self):
        response_body = {
            "message": {"content": json.dumps(valid_model_result(
                sentence="I want some food.",
                corrections=[],
                grammar_changes=["Added English function words."],
                ambiguous=False,
                uncertainty_note="",
            ))}
        }
        signs = [{"label": label, "uncertain": False} for label in (
            "I", "want", "food"
        )]
        with patch(
            "scripts.ai_sentence_builder.urllib.request.urlopen",
            return_value=FakeResponse(response_body),
        ):
            result = self.provider.build_sentence(
                signs, ["I", "want", "food"]
            )
        self.assertEqual(result["corrections"], [])
        self.assertEqual(result["sentence"], "I want some food.")
        self.assertFalse(result["ambiguous"])
        self.assertEqual(
            [sign["label"] for sign in result["original_sequence"]],
            ["I", "want", "food"],
        )

    def test_context_alone_cannot_justify_a_sign_correction(self):
        response_body = {
            "message": {"content": json.dumps(valid_model_result(
                sentence="I want some food.",
                corrections=[{
                    "index": 2,
                    "original_label": "where",
                    "suggested_label": "food",
                    "reason": "Food makes this sentence more natural.",
                }],
                grammar_changes=["Added English function words."],
                ambiguous=False,
                uncertainty_note="",
            ))}
        }
        signs = [{"label": label, "uncertain": False} for label in (
            "I", "want", "where"
        )]
        with patch(
            "scripts.ai_sentence_builder.urllib.request.urlopen",
            return_value=FakeResponse(response_body),
        ):
            result = self.provider.build_sentence(
                signs, ["I", "want", "food", "where"]
            )
        self.assertEqual(result["corrections"], [])
        self.assertEqual(result["sentence"], "I want where")
        self.assertTrue(result["ambiguous"])
        self.assertIn("explicitly marked uncertain", result["uncertainty_note"])
        self.assertEqual(result["original_sequence"][2]["label"], "where")


class SentenceBuilderApiTests(unittest.TestCase):
    def setUp(self):
        import app

        self.app_module = app
        self.client = app.app.test_client()

    def test_api_returns_valid_structured_sentence(self):
        expected = {
            "sentence": "I want some food.",
            "original_sequence": [{"label": "food"}],
            "corrections": [],
            "grammar_changes": [],
            "ambiguous": False,
            "uncertainty_note": "",
        }
        with patch.object(
            self.app_module.AI_SENTENCE_BUILDER,
            "build_sentence",
            return_value=expected,
        ) as build:
            response = self.client.post(
                "/api/ai-sentence",
                json={
                    "recognizer": "browser_knn",
                    "signs": [{"label": "food", "confidence": 0.7}],
                },
                headers={"Host": "localhost"},
            )
        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual(response.get_json(), expected)
        self.assertEqual(build.call_args.args[1], ["Hello", "No", "Sorry", "Thank you", "Yes"])

    def test_api_rejects_empty_or_invalid_requests(self):
        empty = self.client.post(
            "/api/ai-sentence",
            json={"recognizer": "browser_knn", "signs": []},
            headers={"Host": "localhost"},
        )
        invalid = self.client.post(
            "/api/ai-sentence",
            json={"recognizer": "other", "signs": [{"label": "food"}]},
            headers={"Host": "localhost"},
        )
        self.assertEqual(empty.status_code, 422)
        self.assertEqual(invalid.status_code, 400)

    def test_api_surfaces_unavailable_provider(self):
        with patch.object(
            self.app_module.AI_SENTENCE_BUILDER,
            "build_sentence",
            side_effect=OllamaUnavailableError("Ollama timed out."),
        ):
            response = self.client.post(
                "/api/ai-sentence",
                json={
                    "recognizer": "browser_knn",
                    "signs": [{"label": "Hello"}],
                },
                headers={"Host": "localhost"},
            )
        self.assertEqual(response.status_code, 503)
        self.assertIn("timed out", response.get_json()["error"])

    def test_api_status_reports_missing_local_model(self):
        with patch.object(
            self.app_module.AI_SENTENCE_BUILDER,
            "status",
            return_value={
                "available": False,
                "reachable": False,
                "model": "qwen2.5:3b",
                "message": "Install and start Ollama.",
            },
        ):
            response = self.client.get(
                "/api/ai-sentence/status",
                headers={"Host": "localhost"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.get_json()["available"])

    def test_ai_endpoint_is_local_only(self):
        response = self.client.post(
            "/api/ai-sentence",
            json={"recognizer": "browser_knn", "signs": [{"label": "Hello"}]},
            headers={"Host": "example.com"},
        )
        self.assertEqual(response.status_code, 403)


if __name__ == "__main__":
    unittest.main()
