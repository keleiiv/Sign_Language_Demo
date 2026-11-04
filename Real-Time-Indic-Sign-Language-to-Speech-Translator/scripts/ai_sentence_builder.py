"""Local Ollama-backed sentence formation for recognized ISL gloss sequences."""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"
MAX_SIGNS = 32
MAX_LABEL_LENGTH = 80
MAX_VOCABULARY_SIZE = 200
MAX_REQUEST_BYTES = 16 * 1024
REQUEST_TIMEOUT_SECONDS = 45

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "sentence": {"type": "string"},
        "corrections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "original_label": {"type": "string"},
                    "suggested_label": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": [
                    "index", "original_label", "suggested_label", "reason",
                ],
                "additionalProperties": False,
            },
        },
        "grammar_changes": {"type": "array", "items": {"type": "string"}},
        "ambiguous": {"type": "boolean"},
        "uncertainty_note": {"type": "string"},
    },
    "required": [
        "sentence", "corrections", "grammar_changes", "ambiguous",
        "uncertainty_note",
    ],
    "additionalProperties": False,
}


class SentenceBuilderError(ValueError):
    """Raised when input or generated structured output is invalid."""


class OllamaUnavailableError(RuntimeError):
    """Raised when the configured local Ollama service/model cannot be used."""


def _configured_timeout() -> float:
    value = os.environ.get(
        "ISL_OLLAMA_TIMEOUT_SECONDS", str(REQUEST_TIMEOUT_SECONDS)
    )
    try:
        timeout = float(value)
    except ValueError as error:
        raise ValueError("ISL_OLLAMA_TIMEOUT_SECONDS must be numeric.") from error
    if not 1 <= timeout <= 120:
        raise ValueError("ISL_OLLAMA_TIMEOUT_SECONDS must be between 1 and 120.")
    return timeout


def validate_request(signs, vocabulary) -> tuple[list[dict], list[str]]:
    if not isinstance(signs, list) or not signs:
        raise SentenceBuilderError("Add at least one recognized sign first.")
    if len(signs) > MAX_SIGNS:
        raise SentenceBuilderError(f"A sentence can contain at most {MAX_SIGNS} signs.")
    if not isinstance(vocabulary, list) or len(vocabulary) > MAX_VOCABULARY_SIZE:
        raise SentenceBuilderError("The supported vocabulary list is invalid.")

    clean_vocabulary = []
    seen = set()
    for label in vocabulary:
        if (not isinstance(label, str) or not label.strip()
                or len(label.strip()) > MAX_LABEL_LENGTH):
            raise SentenceBuilderError("Supported vocabulary contains an invalid label.")
        clean_label = label.strip()
        if clean_label.casefold() not in seen:
            clean_vocabulary.append(clean_label)
            seen.add(clean_label.casefold())

    clean_signs = []
    for index, sign in enumerate(signs):
        if not isinstance(sign, dict):
            raise SentenceBuilderError(f"Sign {index + 1} must be an object.")
        label = sign.get("label")
        if label is not None and (
            not isinstance(label, str) or not label.strip()
            or len(label.strip()) > MAX_LABEL_LENGTH
        ):
            raise SentenceBuilderError(f"Sign {index + 1} has an invalid label.")
        if (isinstance(label, str) and clean_vocabulary
                and label.strip().casefold() not in seen):
            raise SentenceBuilderError(
                f"Sign {index + 1} is not in the selected recognizer vocabulary."
            )
        confidence = sign.get("confidence")
        if confidence is not None and (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            raise SentenceBuilderError(
                f"Sign {index + 1} has an invalid raw confidence score."
            )
        uncertain = sign.get("uncertain", False)
        if type(uncertain) is not bool:
            raise SentenceBuilderError(f"Sign {index + 1} uncertainty must be boolean.")
        if label is None and not uncertain:
            raise SentenceBuilderError(
                f"Sign {index + 1} needs a label or an uncertainty flag."
            )
        corrected = sign.get("user_corrected_label")
        if corrected is not None:
            if (not isinstance(corrected, str) or not corrected.strip()
                    or len(corrected.strip()) > MAX_LABEL_LENGTH):
                raise SentenceBuilderError(
                    f"Sign {index + 1} has an invalid user correction."
                )
            corrected = corrected.strip()
            if (clean_vocabulary and
                    corrected.casefold() not in seen):
                raise SentenceBuilderError(
                    f"User correction for sign {index + 1} is not in the supported vocabulary."
                )
        clean_signs.append({
            "label": label.strip() if isinstance(label, str) else None,
            "confidence": float(confidence) if confidence is not None else None,
            "confidence_kind": "uncalibrated_raw_score",
            "uncertain": uncertain,
            "user_corrected_label": corrected,
        })

    payload_size = len(json.dumps(
        {"signs": clean_signs, "supported_vocabulary": clean_vocabulary},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8"))
    if payload_size > MAX_REQUEST_BYTES:
        raise SentenceBuilderError("Sentence request exceeds the size limit.")
    return clean_signs, clean_vocabulary


def validate_response(result, signs, vocabulary) -> dict:
    if not isinstance(result, dict) or set(result) != {
        "sentence", "corrections", "grammar_changes", "ambiguous",
        "uncertainty_note",
    }:
        raise SentenceBuilderError("The language model returned an invalid response shape.")
    sentence = result["sentence"]
    if not isinstance(sentence, str) or not sentence.strip() or len(sentence) > 500:
        raise SentenceBuilderError("The language model returned an invalid sentence.")
    corrections = result["corrections"]
    if not isinstance(corrections, list) or len(corrections) > len(signs):
        raise SentenceBuilderError("The language model returned invalid corrections.")
    vocabulary_by_case = {label.casefold(): label for label in vocabulary}
    validated_corrections = []
    used_indices = set()
    for correction in corrections:
        if not isinstance(correction, dict) or set(correction) != {
            "index", "original_label", "suggested_label", "reason",
        }:
            raise SentenceBuilderError("The language model returned an invalid correction.")
        index = correction["index"]
        if type(index) is not int or not 0 <= index < len(signs):
            raise SentenceBuilderError("A proposed correction has an invalid sign index.")
        original = signs[index]["label"]
        if (index in used_indices or original is None
                or signs[index]["user_corrected_label"] is not None
                or correction["original_label"] != original):
            raise SentenceBuilderError(
                "A proposed correction does not preserve the original recognized label."
            )
        suggested = correction["suggested_label"]
        canonical_suggestion = vocabulary_by_case.get(
            suggested.casefold() if isinstance(suggested, str) else ""
        )
        reason = correction["reason"]
        if canonical_suggestion is None:
            raise SentenceBuilderError(
                "A proposed correction is not supported or lacks a brief reason."
            )
        if canonical_suggestion == original:
            continue
        if (not isinstance(reason, str) or not reason.strip() or len(reason) > 240):
            raise SentenceBuilderError(
                "A proposed correction is not supported or lacks a brief reason."
            )
        used_indices.add(index)
        validated_corrections.append({
            "index": index,
            "original_label": original,
            "suggested_label": canonical_suggestion,
            "reason": reason.strip(),
        })

    grammar_changes = result["grammar_changes"]
    if (not isinstance(grammar_changes, list) or len(grammar_changes) > 8
            or any(not isinstance(item, str) or not item.strip() or len(item) > 160
                   for item in grammar_changes)):
        raise SentenceBuilderError("The language model returned invalid grammar notes.")
    ambiguous = result["ambiguous"]
    uncertainty_note = result["uncertainty_note"]
    if (type(ambiguous) is not bool or not isinstance(uncertainty_note, str)
            or len(uncertainty_note) > 240):
        raise SentenceBuilderError("The language model returned invalid uncertainty information.")
    if uncertainty_note.strip():
        ambiguous = True
    if ambiguous and not uncertainty_note.strip():
        raise SentenceBuilderError("The language model returned invalid uncertainty information.")

    return {
        "sentence": sentence.strip(),
        "original_sequence": signs,
        "corrections": validated_corrections,
        "grammar_changes": [item.strip() for item in grammar_changes],
        "ambiguous": ambiguous,
        "uncertainty_note": uncertainty_note.strip(),
        "confidence_notice": (
            "Classifier confidence values are uncalibrated raw scores, not probabilities."
        ),
    }


def _normalized_words(text: str) -> list[str]:
    return "".join(
        character.casefold() if character.isalnum() else " "
        for character in text
    ).split()


class OllamaSentenceBuilder:
    """Small provider boundary; the rest of the app does not depend on Ollama."""

    def __init__(self, base_url=None, model=None):
        self.base_url = (
            base_url or os.environ.get("ISL_OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL)
        ).rstrip("/")
        self.model = model or os.environ.get(
            "ISL_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL
        )

    def _validate_configuration(self) -> None:
        if not isinstance(self.base_url, str) or not self.base_url.startswith(
            ("http://", "https://")
        ):
            raise OllamaUnavailableError(
                "ISL_OLLAMA_BASE_URL must be an HTTP(S) URL."
            )
        if (not isinstance(self.model, str) or not self.model.strip()
                or len(self.model) > 128):
            raise OllamaUnavailableError(
                "ISL_OLLAMA_MODEL must be a non-empty model name."
            )

    def _request_json(self, path, body=None, timeout=3):
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST" if body is not None else "GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if response.status != 200:
                    raise OllamaUnavailableError(
                        f"Ollama returned HTTP {response.status}."
                    )
                return json.loads(response.read(2 * 1024 * 1024))
        except urllib.error.HTTPError as error:
            detail = error.read(2048).decode("utf-8", errors="replace")
            raise OllamaUnavailableError(
                f"Ollama returned HTTP {error.code}: {detail or error.reason}"
            ) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise OllamaUnavailableError(
                f"Cannot reach local Ollama at {self.base_url}: {error}"
            ) from error
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise OllamaUnavailableError("Ollama returned malformed JSON.") from error

    def status(self) -> dict:
        try:
            self._validate_configuration()
        except OllamaUnavailableError as error:
            return {
                "available": False,
                "reachable": False,
                "model": self.model,
                "message": str(error),
            }
        try:
            result = self._request_json("/api/tags", timeout=3)
        except OllamaUnavailableError as error:
            return {
                "available": False,
                "reachable": False,
                "model": self.model,
                "message": (
                    f"Ollama is unavailable at {self.base_url}. Install and start "
                    "Ollama, then pull the configured model."
                ),
                "detail": str(error),
            }
        models = result.get("models") if isinstance(result, dict) else None
        if not isinstance(models, list):
            return {
                "available": False,
                "reachable": True,
                "model": self.model,
                "message": "Ollama is reachable but returned an invalid model list.",
            }
        names = {
            item.get("name") for item in models
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        }
        model_present = self.model in names or f"{self.model}:latest" in names
        return {
            "available": model_present,
            "reachable": True,
            "model": self.model,
            "message": (
                "Local sentence model is ready."
                if model_present
                else f"Ollama is running, but model {self.model!r} is not installed. "
                     f"Run `ollama pull {self.model}`."
            ),
        }

    def build_sentence(self, signs, vocabulary) -> dict:
        self._validate_configuration()
        clean_signs, clean_vocabulary = validate_request(signs, vocabulary)
        if not any(
            sign["user_corrected_label"] or sign["label"]
            for sign in clean_signs
        ):
            return {
                "sentence": "I could not infer a sentence from the uncertain signs.",
                "original_sequence": clean_signs,
                "corrections": [],
                "grammar_changes": [],
                "ambiguous": True,
                "uncertainty_note": (
                    "No sign label was available; add or correct a label before interpreting."
                ),
                "confidence_notice": (
                    "Classifier confidence values are uncalibrated raw scores, not probabilities."
                ),
            }

        try:
            timeout = _configured_timeout()
        except ValueError as error:
            raise OllamaUnavailableError(str(error)) from error
        if len(clean_vocabulary) == 0:
            raise SentenceBuilderError("No supported vocabulary is available.")
        user_payload = {
            "signs": clean_signs,
            "supported_vocabulary": clean_vocabulary,
            "rules": [
                "Treat labels as isolated experimental glosses, not English words in English word order or a complete translation.",
                "ISL grammar and ordering differ from English; the gloss sequence alone may not establish the speaker's intended meaning.",
                "Do not infer a different sign merely because it makes a more fluent or familiar English sentence.",
                "A recognition correction is optional and may be proposed only for an input sign explicitly marked uncertain=true, and only when the other supplied signs provide strong, specific evidence for one supported alternative.",
                "Low or high raw confidence is not proof that a sign is wrong; confidence is uncalibrated.",
                "A correction must use an existing sign index and a supported vocabulary label.",
                "Never claim an abstained sign was recognized or invent additional recognized signs.",
                "Do not change a user-corrected label.",
                "Separate recognition corrections from grammar changes.",
                "If more than one interpretation is plausible, or the labels do not establish a clear meaning, keep the recognized sequence unchanged, return no corrections, set ambiguous=true, and explain briefly.",
                "When meaning is unclear, do not force a natural-sounding sentence; use a concise gloss rendering rather than presenting a guess as the intended meaning.",
                "Confidence values are uncalibrated raw classifier scores, not probabilities.",
                "Do not invent facts about ISL signs or make claims not supported by this payload.",
                "With fewer than three labeled signs, do not make a context-based recognition correction; mark interpretation uncertainty instead.",
            ],
        }
        try:
            response = self._request_json(
                "/api/chat",
                {
                    "model": self.model,
                    "stream": False,
                    "format": RESPONSE_SCHEMA,
                    "options": {"temperature": 0.1, "num_predict": 192},
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Interpret only the supplied experimental ISL glosses; they "
                                "are not an English sentence and their order must not be "
                                "assumed to follow English grammar. ISL has its own grammar, "
                                "and this limited vocabulary may not reveal the intended "
                                "meaning. Return the requested JSON only. Preserve all "
                                "original labels and never invent or silently replace a sign. "
                                "A correction is optional: propose one only if that exact "
                                "input sign is marked uncertain=true and the remaining "
                                "sequence gives strong, specific evidence for a single "
                                "supported alternative. Fluency, plausibility, and "
                                "uncalibrated confidence alone are not evidence. Otherwise "
                                "return no corrections. If meaning is ambiguous, keep the "
                                "sequence unchanged, provide a concise gloss rendering, set "
                                "ambiguous=true, and explain the uncertainty. Grammar words "
                                "may be added only to a clearly supported English rendering "
                                "and must be listed separately as grammar changes. Never "
                                "claim complete ISL translation or describe added grammar "
                                "words as recognized signs."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                "Example: glosses where, food, please, with no sign marked "
                                "uncertain. Their sequence alone does not establish whether "
                                "this is a question, a request, or something else. Preserve "
                                "the labels, do not propose corrections, and mark the "
                                "interpretation ambiguous instead of guessing."
                            ),
                        },
                        {
                            "role": "assistant",
                            "content": json.dumps({
                                "sentence": "where food please",
                                "corrections": [],
                                "grammar_changes": [],
                                "ambiguous": True,
                                "uncertainty_note": (
                                    "The glosses do not establish a unique intended meaning."
                                ),
                            }),
                        },
                        {
                            "role": "user",
                            "content": json.dumps(
                                user_payload, ensure_ascii=False, separators=(",", ":")
                            ),
                        },
                    ],
                },
                timeout=timeout,
            )
        except OllamaUnavailableError:
            raise
        except (TimeoutError, OSError) as error:
            raise OllamaUnavailableError(
                f"Local sentence generation failed: {error}"
            ) from error
        message = response.get("message") if isinstance(response, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise SentenceBuilderError("Ollama response did not contain structured content.")
        try:
            generated = json.loads(content)
        except json.JSONDecodeError as error:
            raise SentenceBuilderError(
                "Ollama returned malformed sentence JSON."
            ) from error
        result = validate_response(generated, clean_signs, clean_vocabulary)
        labeled_count = sum(
            bool(sign["user_corrected_label"] or sign["label"])
            for sign in clean_signs
        )
        if labeled_count < 3 and result["corrections"]:
            result["corrections"] = []
            result["ambiguous"] = True
            result["uncertainty_note"] = (
                "Fewer than three labeled signs provide too little context to support "
                "an AI recognition correction."
            )
        elif labeled_count == 2 and not result["ambiguous"]:
            result["ambiguous"] = True
            result["uncertainty_note"] = (
                "A two-sign sequence may have more than one interpretation; no "
                "recognition correction was verified."
            )
        effective_labels = [
            sign["user_corrected_label"] or sign["label"]
            for sign in clean_signs
            if sign["user_corrected_label"] or sign["label"]
        ]
        echoed_glosses = (
            labeled_count >= 3
            and not result["grammar_changes"]
            and _normalized_words(result["sentence"])
            == _normalized_words(" ".join(effective_labels))
        )
        correction_evidence_missing = any(
            not clean_signs[correction["index"]]["uncertain"]
            for correction in result["corrections"]
        )
        if result["ambiguous"] or echoed_glosses or correction_evidence_missing:
            result["corrections"] = []
        if correction_evidence_missing and not result["ambiguous"]:
            result["ambiguous"] = True
            result["uncertainty_note"] = (
                "No corrected sign was explicitly marked uncertain; label-only context "
                "is insufficient evidence to propose a recognition correction."
            )
        if echoed_glosses and not result["ambiguous"]:
            result["ambiguous"] = True
            result["uncertainty_note"] = (
                "The model echoed the gloss sequence without resolving it into "
                "English sentence form; review or edit the suggestion."
            )
        elif echoed_glosses:
            result["uncertainty_note"] = (
                "The model echoed the gloss sequence; it does not establish a unique "
                "English meaning. Review or edit the suggestion."
            )
        if result["ambiguous"]:
            result["sentence"] = " ".join(effective_labels)
            result["corrections"] = []
            result["grammar_changes"] = []
        return result
