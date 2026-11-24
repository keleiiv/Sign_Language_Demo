"""Sentence Demo Mode Configuration & Mapping Manager.

Provides robust, configurable mapping between recognized ISL signs and predefined
demonstration sentences for college evaluation.

Design principles:
- Configuration stored in a separate JSON file with no hardcoded limit on mapping count.
- Thread-safe loading, validating, and saving.
- Strict validation: checks for missing files, malformed JSON, duplicate entries,
  and empty sentences. A saved vocabulary label may be pending model activation.
- Never silently substitutes an unsupported label or invents predictions.
- Supports both dictionary {"sign": "sentence"} and list [{"sign": "...", "sentence": "..."}] formats.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Default path relative to project root
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "data" / "sentence_demo_mappings.json"

# Default demonstration mappings matching the active validated 13-sign model
DEFAULT_DEMO_MAPPINGS = {
    "help": "Could you please help me?",
    "water": "May I please have a glass of water?",
    "bathroom": "Excuse me, where is the restroom?",
    "food": "I am hungry and need something to eat.",
    "thank you very much": "Thank you so much for your assistance.",
    "sorry": "I apologize for the misunderstanding.",
    "please": "Please give me a moment.",
    "goodbye": "Goodbye, have a wonderful day!",
    "yes": "Yes, that is correct.",
    "no": "No, thank you.",
    "what": "Could you explain what is happening?",
    "where": "Could you tell me where to go?",
    "okay": "Everything is okay, thank you.",
    "hello": "Hello! Welcome to our sign language demonstration."
}


def normalize_label(label: str) -> str:
    """Normalize sign label for robust case-insensitive comparison."""
    if not isinstance(label, str):
        return ""
    # Strip whitespace and collapse multiple spaces, lowercase
    cleaned = re.sub(r"\s+", " ", label.strip())
    return cleaned.lower()


class SentenceDemoManager:
    """Manages sentence mappings configuration, validation, and sentence resolution."""

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        supported_labels_provider: Optional[callable] = None
    ) -> None:
        self.config_path = (
            Path(config_path)
            if config_path
            else Path(os.environ.get("ISL_SENTENCE_MAPPINGS_PATH", DEFAULT_CONFIG_PATH))
        )
        self.supported_labels_provider = supported_labels_provider
        self._mappings: Dict[str, str] = {}
        self._raw_metadata: Dict[str, Any] = {}
        self._last_load_error: Optional[str] = None
        self._last_loaded_mtime: float = 0.0

        # Load or create default
        self.load_mappings()

    def get_supported_labels(self) -> List[str]:
        """Fetch list of verified labels currently supported by the active model."""
        if callable(self.supported_labels_provider):
            try:
                labels = self.supported_labels_provider()
                if isinstance(labels, (list, tuple, set)):
                    return sorted(str(lbl).strip() for lbl in labels if str(lbl).strip())
            except Exception as error:
                logger.warning("Could not fetch supported labels from provider: %s", error)
        return []

    def load_mappings(self, force: bool = False) -> Dict[str, str]:
        """Load mappings from JSON file with validation and graceful recovery."""
        if not self.config_path.is_file():
            # Create default configuration file if missing
            logger.info("Sentence demo config missing at %s; creating default.", self.config_path)
            self._create_default_file()

        try:
            mtime = self.config_path.stat().st_mtime
            if not force and self._mappings and mtime == self._last_loaded_mtime and self._last_load_error is None:
                return dict(self._mappings)

            with open(self.config_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    raise ValueError("Sentence mapping file is empty.")
                data = json.loads(content)

            validated_mappings, metadata, errors, warnings = self._parse_and_validate_payload(data)
            if errors:
                self._last_load_error = "; ".join(errors)
                # Keep previously loaded mappings if available, or validated subset
                if not self._mappings and validated_mappings:
                    self._mappings = validated_mappings
                logger.warning("Errors validating sentence demo mappings: %s", self._last_load_error)
            else:
                self._mappings = validated_mappings
                self._raw_metadata = metadata
                self._last_load_error = None
                self._last_loaded_mtime = mtime

        except json.JSONDecodeError as err:
            self._last_load_error = f"Malformed JSON in mapping file ({self.config_path.name}: line {err.lineno}, col {err.colno}): {err.msg}"
            logger.error(self._last_load_error)
        except Exception as err:
            self._last_load_error = f"Failed to load sentence mappings: {err}"
            logger.error(self._last_load_error)

        return dict(self._mappings)

    def _create_default_file(self) -> None:
        """Create a documented default configuration file."""
        payload = {
            "$schema_version": 1,
            "description": (
                "Sentence Demo Mode mappings for Indian Sign Language Translator. "
                "Maps verified model sign labels to predefined demonstration sentences. "
                "This mode demonstrates intent-based phrase mapping and is not general-purpose "
                "continuous sign-to-sentence translation."
            ),
            "instructions": (
                "Add, edit, or remove mappings in the 'mappings' object below. "
                "Keys must match labels in the saved vocabulary. Pending labels are not "
                "recognizable until a compatible model is trained, evaluated, and activated. "
                "Sentences must be non-empty strings. Any number of mappings is supported."
            ),
            "mappings": DEFAULT_DEMO_MAPPINGS
        }
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")

    def _parse_and_validate_payload(
        self,
        data: Any
    ) -> Tuple[Dict[str, str], Dict[str, Any], List[str], List[str]]:
        """Parse dictionary or list format and validate entries."""
        errors: List[str] = []
        warnings: List[str] = []
        mappings: Dict[str, str] = {}
        metadata: Dict[str, Any] = {}

        raw_dict: Dict[Any, Any] = {}
        if isinstance(data, dict):
            metadata = {k: v for k, v in data.items() if k != "mappings"}
            if "mappings" in data:
                raw_mappings = data["mappings"]
                if isinstance(raw_mappings, dict):
                    raw_dict = raw_mappings
                elif isinstance(raw_mappings, list):
                    for idx, item in enumerate(raw_mappings):
                        if isinstance(item, dict) and "sign" in item and "sentence" in item:
                            raw_dict[item["sign"]] = item["sentence"]
                        else:
                            errors.append(f"Item {idx + 1} in mappings array must be an object with 'sign' and 'sentence'.")
                else:
                    errors.append("'mappings' key must be a JSON object or array of {sign, sentence} objects.")
            else:
                # Top-level dict treated as mapping
                raw_dict = data
        elif isinstance(data, list):
            for idx, item in enumerate(data):
                if isinstance(item, dict) and "sign" in item and "sentence" in item:
                    raw_dict[item["sign"]] = item["sentence"]
                else:
                    errors.append(f"Item {idx + 1} must be an object with 'sign' and 'sentence'.")
        else:
            errors.append("Configuration root must be a JSON object or array.")
            return mappings, metadata, errors, warnings

        seen_keys: set[str] = set()
        for raw_k, raw_v in raw_dict.items():
            if not isinstance(raw_k, str) or not raw_k.strip():
                warnings.append(f"Ignored non-string or empty sign key: {repr(raw_k)}")
                continue

            clean_k = re.sub(r"\s+", " ", raw_k.strip())
            norm_k = clean_k.lower()

            if norm_k in seen_keys:
                warnings.append(f"Duplicate trigger sign '{clean_k}' detected; later definition takes precedence.")
            seen_keys.add(norm_k)

            if not isinstance(raw_v, str):
                warnings.append(f"Ignored non-string sentence for sign '{clean_k}': {repr(raw_v)}")
                continue

            clean_sentence = raw_v.strip()
            if not clean_sentence:
                warnings.append(f"Ignored empty sentence for sign '{clean_k}'.")
                continue

            mappings[clean_k] = clean_sentence

        return mappings, metadata, errors, warnings

    def validate_current_configuration(self) -> Dict[str, Any]:
        """Generate a complete diagnostic validation report against active model vocabulary."""
        supported_labels = self.get_supported_labels()
        norm_supported = {normalize_label(lbl): lbl for lbl in supported_labels}

        report_entries = []
        supported_count = 0
        unsupported_count = 0

        for sign, sentence in self._mappings.items():
            norm_sign = normalize_label(sign)
            is_supported = (not norm_supported) or (norm_sign in norm_supported)

            if is_supported:
                supported_count += 1
                report_entries.append({
                    "sign": sign,
                    "normalized_sign": norm_sign,
                    "sentence": sentence,
                    "supported": True,
                    "active_label": norm_supported.get(norm_sign, sign),
                    "reason": None
                })
            else:
                unsupported_count += 1
                report_entries.append({
                    "sign": sign,
                    "normalized_sign": norm_sign,
                    "sentence": sentence,
                    "supported": False,
                    "active_label": None,
                    "reason": (
                        f"Sign '{sign}' is not in the active model vocabulary. "
                        f"Active model supports: {', '.join(supported_labels[:8])}"
                        + (f" and {len(supported_labels) - 8} more" if len(supported_labels) > 8 else "")
                    )
                })

        return {
            "config_path": str(self.config_path),
            "file_exists": self.config_path.is_file(),
            "load_error": self._last_load_error,
            "total_mappings": len(self._mappings),
            "supported_count": supported_count,
            "unsupported_count": unsupported_count,
            "active_model_labels_count": len(supported_labels),
            "active_model_labels": supported_labels,
            "entries": report_entries
        }

    def save_mappings(self, new_mappings: Dict[str, str]) -> Tuple[bool, Optional[str]]:
        """Safely validate and save new mappings to the JSON configuration file."""
        if not isinstance(new_mappings, dict):
            return False, "Mappings payload must be a JSON object."

        validated_mappings, _, errors, warnings = self._parse_and_validate_payload(new_mappings)
        if errors:
            return False, "; ".join(errors)

        if not validated_mappings:
            return False, "Cannot save an empty mappings configuration. Provide at least one valid mapping."

        payload = {
            "$schema_version": 1,
            "description": self._raw_metadata.get(
                "description",
                "Sentence Demo Mode mappings for Indian Sign Language Translator."
            ),
            "instructions": (
                "Add, edit, or remove mappings in the 'mappings' object below. "
                "Keys must match labels in the saved vocabulary. Pending labels are not "
                "recognizable until a compatible model is trained, evaluated, and activated. "
                "Sentences must be non-empty strings. Any number of mappings is supported."
            ),
            "mappings": validated_mappings
        }

        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            # Atomic write via temporary file
            with tempfile.NamedTemporaryFile(
                "w",
                dir=str(self.config_path.parent),
                delete=False,
                encoding="utf-8"
            ) as tmp:
                json.dump(payload, tmp, indent=2, ensure_ascii=False)
                tmp.write("\n")
                tmp_path = Path(tmp.name)

            tmp_path.replace(self.config_path)
            self._mappings = validated_mappings
            self._last_load_error = None
            self._last_loaded_mtime = self.config_path.stat().st_mtime
            logger.info("Saved %d sentence demo mappings to %s", len(self._mappings), self.config_path)
            return True, None
        except Exception as error:
            logger.exception("Failed to write sentence mappings to %s", self.config_path)
            return False, f"Could not write mappings file: {error}"

    def save_mapping(self, sign_label: str, sentence: str) -> Tuple[bool, Optional[str]]:
        """Insert or edit one label mapping without replacing other mappings."""
        if not isinstance(sign_label, str):
            return False, "Sign label must be text."
        clean_label = re.sub(r"\s+", " ", sign_label.strip())
        if not clean_label or len(clean_label) > 80 or not re.fullmatch(
            r"[\w][\w '\-]{0,79}", clean_label, flags=re.UNICODE
        ):
            return False, "Sign label must contain 1 to 80 supported characters."
        if not isinstance(sentence, str):
            return False, "Predefined sentence must be text."
        clean_sentence = sentence.strip()
        if not clean_sentence:
            return False, "Predefined sentence cannot be empty."
        if len(clean_sentence) > 1000:
            return False, "Predefined sentence must be 1000 characters or fewer."

        mappings = self.load_mappings(force=True)
        if self._last_load_error:
            return False, f"Existing mappings could not be loaded: {self._last_load_error}"
        matching_key = next(
            (key for key in mappings if normalize_label(key) == normalize_label(clean_label)),
            clean_label,
        )
        mappings[matching_key] = clean_sentence
        return self.save_mappings(mappings)

    def lookup_sentence(self, sign_label: str) -> Optional[Dict[str, str]]:
        """Look up mapped sentence for a recognized sign (case-insensitive & trimmed)."""
        if not isinstance(sign_label, str) or not sign_label.strip():
            return None

        norm_query = normalize_label(sign_label)
        for sign, sentence in self._mappings.items():
            if normalize_label(sign) == norm_query:
                return {
                    "matched_sign": sign,
                    "query_label": sign_label,
                    "sentence": sentence
                }
        return None
