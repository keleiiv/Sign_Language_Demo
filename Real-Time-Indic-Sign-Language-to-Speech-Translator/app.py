"""
app.py — Flask backend for ISL Translator.

Runs the camera server-side (same pipeline as sentence_builder.py),
streams annotated video as MJPEG to the browser, and exposes
/status + /control endpoints for the frontend UI.

Run:
    venv312\\Scripts\\python.exe app.py
Then open: http://localhost:5000
"""

import os
import base64
import binascii
import hashlib
import ipaddress
import json
import queue
import re
# Load .env if present
try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass
import tempfile
import threading
import time
from collections import Counter, deque
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mv
from scripts.sequence_contract import (
    FEATURE_DIM as KEYPOINT_DIM,
    MAX_FRAMES,
    PREPROCESSING_VERSION,
    SEQUENCE_SHAPE,
    extract_frame_features,
)
from scripts.capture_validation import validate_capture_sequence
from scripts.ai_sentence_builder import (
    OllamaSentenceBuilder,
    OllamaUnavailableError,
    SentenceBuilderError,
)
from scripts.expandable_classifier import (
    ArtifactCompatibilityError,
    InsufficientDataError,
    MIN_EVALUATION_SESSIONS_PER_CLASS,
    MIN_TRAIN_SESSIONS_PER_CLASS,
    activate_artifact,
    dataset_fingerprint,
    list_artifacts,
    load_artifact,
    split_by_session,
    train_and_evaluate,
)
from scripts.vocabulary_store import (
    CaptureSession,
    VocabularyEntry,
    VocabularyStore,
    validate_entry_id,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).parent
HOLISTIC_MODEL = os.environ.get("ISL_HOLISTIC_MODEL_PATH") or str(
    ROOT / "mediapipe_models" / "holistic_landmarker.task"
)
KERAS_MODEL    = os.environ.get("ISL_KERAS_MODEL_PATH") or str(
    ROOT / "models" / "isl_model_solo.keras"
)
FEATURES_DIR   = ROOT / "data" / "extracted_features"
FRONTEND_DIR   = ROOT / "Frontend"
VOCABULARY_DIR = ROOT / "data" / "vocabulary_drafts"
VOCABULARY_STORE = VocabularyStore(VOCABULARY_DIR)
AI_SENTENCE_BUILDER = OllamaSentenceBuilder()

# ── Sentence builder params ───────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.60
VOTE_WINDOW          = 6
VOTE_THRESHOLD       = 4
LOW_CONF_TOLERANCE   = 2
PREDICT_EVERY        = 2
SENTENCE_TIMEOUT     = 2.5
WORD_COOLDOWN        = 1.0

# ── Languages ─────────────────────────────────────────────────────────────────
LANGUAGES = {
    "en": "English", "hi": "Hindi",  "bn": "Bengali",
    "ta": "Tamil",   "te": "Telugu", "mr": "Marathi", "gu": "Gujarati",
}

# ── Overlay colours (BGR) ─────────────────────────────────────────────────────
C_GREEN  = (80, 220, 120)
C_ORANGE = (60, 160, 255)
C_CYAN   = (220, 200, 80)
C_YELLOW = (60, 220, 240)
C_GRAY   = (160, 160, 160)
C_WHITE  = (240, 240, 240)


def _dark_bar(frame, y1, y2, alpha=0.65):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y1), (w, y2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


# ── Grammar helpers ───────────────────────────────────────────────────────────
# Multi-word ISL signs that must never be grammatically modified
FIXED_PHRASES = {"Good Morning", "How are you", "Thank you"}
KNN_CLASS_LABELS = ("Hello", "Thank you", "Sorry", "Yes", "No")
KNN_GLOSS_WORDS = set(KNN_CLASS_LABELS)
KNN_GLOSS_WORDS_CASEFOLD = {word.casefold() for word in KNN_GLOSS_WORDS}
GRAMMAR_LINKING_WORDS = {"am", "is", "are", "a", "an", "in", "the"}

SUBJECTS_AM  = {"I"}
SUBJECTS_IS  = {"he", "she", "it", "Mother", "Father", "Brother", "Sister",
                "Friend", "Man", "Woman", "Teacher", "Doctor", "Police",
                "Student", "Patient"}
SUBJECTS_ARE = {"you", "we", "they", "Family"}
PURE_PRONOUNS  = {"I", "he", "she", "they", "we", "you", "it"}
ARTICLE_NOUNS  = {"Teacher", "Doctor", "Police", "Student", "Patient",
                  "Man", "Woman", "Friend", "Dream", "Sign", "Sport"}
PLACE_NOUNS    = {"Hospital", "School", "House"}


def inject_grammar(words):
    if not words:
        return []
    result = []
    for i, word in enumerate(words):
        result.append(word)
        if i < len(words) - 1:
            nxt = words[i + 1]
            if nxt in PURE_PRONOUNS:
                continue
            if word in SUBJECTS_AM:
                result.append("am")
                if nxt in ARTICLE_NOUNS:  result.append("a")
                elif nxt in PLACE_NOUNS:  result.extend(["in", "the"])
            elif word in SUBJECTS_IS:
                result.append("is")
                if nxt in ARTICLE_NOUNS:  result.append("a")
                elif nxt in PLACE_NOUNS:  result.extend(["in", "the"])
            elif word in SUBJECTS_ARE:
                result.append("are")
                if nxt in ARTICLE_NOUNS:  result.append("a")
                elif nxt in PLACE_NOUNS:  result.extend(["in", "the"])
    return result


def enhance_grammar(words):
    if not words:
        return "."
    # Single fixed phrase — return exactly as signed, no LLM needed
    # Check both single-item and joined form, case-insensitive
    joined = " ".join(words)
    print(f"  [Grammar] words={words!r}  joined={joined!r}")
    for phrase in FIXED_PHRASES:
        if joined.lower() == phrase.lower() or (len(words) == 1 and words[0].lower() == phrase.lower()):
            return phrase + "."
    # Multi-word sentence containing a fixed phrase — protect it from rewriting
    # by passing it through but adding a strong hint to the LLM
    fallback = " ".join(inject_grammar(words)) + "."
    try:
        from groq import Groq
        client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": (
                    "You convert Indian Sign Language (ISL) gloss word sequences into natural English sentences. "
                    "STRICT RULES — follow exactly:\n"
                    "(1) Keep the original word order. Never reorder words.\n"
                    "(2) Only add linking words (am/is/are/a/an/in/the) if WITHOUT them the sentence makes no grammatical sense. "
                    "If the words already form a meaningful phrase, output them as-is.\n"
                    "(3) NEVER add linking words inside fixed greetings or multi-word phrases like "
                    "'Good Morning', 'How are you', 'Thank you', 'Good Night'.\n"
                    "(4) Do NOT add any new nouns, verbs, or content words not present in the input.\n"
                    "(5) Reply with the final sentence only — no explanation, no quotes, no commentary."
                )},
                {"role": "user",      "content": "Gloss words: I Teacher"},
                {"role": "assistant", "content": "I am a Teacher."},
                {"role": "user",      "content": "Gloss words: he Doctor"},
                {"role": "assistant", "content": "He is a Doctor."},
                {"role": "user",      "content": "Gloss words: Mother Hospital"},
                {"role": "assistant", "content": "Mother is in the Hospital."},
                {"role": "user",      "content": "Gloss words: Good Morning"},
                {"role": "assistant", "content": "Good Morning."},
                {"role": "user",      "content": "Gloss words: How are you"},
                {"role": "assistant", "content": "How are you."},
                {"role": "user",      "content": "Gloss words: Thank you"},
                {"role": "assistant", "content": "Thank you."},
                {"role": "user",      "content": f"Gloss words: {' '.join(words)}"},
            ],
            temperature=0.1, max_tokens=60,
        )
        result = response.choices[0].message.content.strip()
        if not result.endswith("."): result += "."
        print(f"  [Groq]  →  {result}")
        return result
    except Exception as e:
        print(f"  [Groq] {e} — using rule-based fallback")
        return fallback


def _punctuate_sentence(text):
    text = text.strip().rstrip(".!?")
    if not text:
        return "."
    return text[0].upper() + text[1:] + "."


def _grammar_result_preserves_gloss(result, words):
    source_tokens = [
        token.casefold()
        for word in words
        for token in re.findall(r"[A-Za-z]+(?:['’-][A-Za-z]+)*", word)
    ]
    result_tokens = [
        token.casefold()
        for token in re.findall(r"[A-Za-z]+(?:['’-][A-Za-z]+)*", result)
    ]
    source_index = 0
    inserted_linking_words = 0
    for token in result_tokens:
        if source_index < len(source_tokens) and token == source_tokens[source_index]:
            source_index += 1
        elif token not in GRAMMAR_LINKING_WORDS:
            return False
        else:
            inserted_linking_words += 1
    return source_index == len(source_tokens) and inserted_linking_words <= 4


def process_sentence(words):
    """Format glosses conservatively without treating them as full ISL translation."""
    unique_words = []
    for word in words:
        if not isinstance(word, str):
            continue
        word = word.strip()
        if word and (not unique_words or
                     word.casefold() != unique_words[-1].casefold()):
            unique_words.append(word)

    if not unique_words:
        return "."

    if all(word.casefold() in KNN_GLOSS_WORDS_CASEFOLD for word in unique_words):
        return _punctuate_sentence(" ".join(unique_words))

    corrected = enhance_grammar(unique_words)
    if _grammar_result_preserves_gloss(corrected, unique_words):
        return _punctuate_sentence(corrected)

    fallback = " ".join(inject_grammar(unique_words))
    return _punctuate_sentence(fallback)


# ── TTS ───────────────────────────────────────────────────────────────────────
_tts_queue: queue.Queue = queue.Queue()


def _tts_worker():
    import pygame
    from deep_translator import GoogleTranslator
    from gtts import gTTS
    pygame.mixer.init()
    while True:
        item = _tts_queue.get()
        if item is None:
            break
        english_text, lang_code = item
        try:
            translated = english_text if lang_code == "en" else \
                GoogleTranslator(source="en", target=lang_code).translate(english_text)
            tts = gTTS(text=translated, lang=lang_code, slow=False)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp_path = f.name
            tts.save(tmp_path)
            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)
        except Exception as e:
            print(f"  [TTS] {e}")
        finally:
            _tts_queue.task_done()


def speak(text, lang_code):
    _tts_queue.put((text, lang_code))


def extract_keypoints(result):
    return extract_frame_features(result)


def _build_holistic():
    options = mv.HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HOLISTIC_MODEL),
        running_mode=mv.RunningMode.IMAGE,
        min_face_detection_confidence=0.5,
        min_face_landmarks_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_pose_landmarks_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
    )
    return mv.HolisticLandmarker.create_from_options(options)


# ── Global state ──────────────────────────────────────────────────────────────
_state = {
    "sequence":           deque(maxlen=MAX_FRAMES),
    "sentence_words":     [],
    "completed_gloss_words": [],
    "vote_window":        deque(maxlen=VOTE_WINDOW),
    "cooldown_until":     0.0,
    "last_hand_time":     0.0,
    "completed_sentence": "",
    "frame_count":        0,
    "current_word":       "",
    "current_conf":       0.0,
    "top_vote_word":      "",
    "top_vote_count":     0,
    "low_conf_streak":    0,
    "lang_code":          "en",
    "camera_active":      False,
    "hands_present":      False,
}

_labels          = []
_model           = None
_holistic        = None   # loaded once at startup, reused by camera thread
_current_frame   = None   # latest annotated JPEG bytes for MJPEG stream
_lock            = threading.Lock()
_camera_thread   = None
_developer_capture_lease_until = 0.0

# Dark placeholder frame sent while camera is not yet started
def _make_placeholder():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (18, 26, 42)   # dark navy
    cv2.putText(img, "Camera not started", (160, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (74, 120, 155), 2)
    cv2.putText(img, "Click 'Start Camera'", (170, 270),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 90, 120), 1)
    _, jpg = cv2.imencode(".jpg", img)
    return jpg.tobytes()

_BLACK_FRAME = _make_placeholder()


def _load_resources():
    global _labels, _model
    print("Loading labels ...")
    _labels = (
        sorted(p.name for p in FEATURES_DIR.iterdir() if p.is_dir())
        if FEATURES_DIR.is_dir() else []
    )
    print(f"  {len(_labels)} classes.")
    if not Path(KERAS_MODEL).is_file():
        _model = None
        print(
            "  Keras model not found; Flask server-side recognition is disabled. "
            "The browser KNN recognizer remains available."
        )
        return
    print("Loading Keras model ...")
    try:
        from tensorflow.keras.models import load_model
    except ImportError as error:
        _model = None
        print(
            "  TensorFlow is unavailable; Flask server-side recognition is "
            f"disabled ({error}). The browser KNN recognizer remains available."
        )
        return
    _model = load_model(KERAS_MODEL)
    print("  Done.\n")


def _do_finalize(s, now):
    s["completed_gloss_words"] = s["sentence_words"][:]
    s["completed_sentence"] = process_sentence(s["completed_gloss_words"])
    lang_code = s["lang_code"]
    print(f"\nSentence: {s['completed_sentence']}  [{LANGUAGES.get(lang_code)}]")
    speak(s["completed_sentence"], lang_code)
    s["sentence_words"] = []
    s["sequence"].clear()
    s["vote_window"].clear()


# ── Camera loop (runs in background thread) ───────────────────────────────────
def _camera_loop(camera_index=0):
    global _current_frame

    print("  Opening camera ...")
    # Try DirectShow first (most reliable on Windows), fall back to default
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("  DirectShow failed, trying default backend ...")
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("  [ERROR] Cannot open camera at index", camera_index)
        with _lock:
            _state["camera_active"] = False
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Read one test frame to confirm camera works
    ok, test = cap.read()
    if not ok or test is None:
        print("  [ERROR] Camera opened but cannot read frames.")
        cap.release()
        with _lock:
            _state["camera_active"] = False
        return

    print("  Camera opened OK — frame size:", test.shape)

    # Build holistic fresh inside this thread — MediaPipe is not cross-thread safe
    print("  Loading MediaPipe in camera thread ...")
    holistic = _build_holistic()
    print("  Camera running.\n")

    sequence      = deque(maxlen=MAX_FRAMES)
    vote_window   = deque(maxlen=VOTE_WINDOW)
    last_hand_t   = 0.0
    cooldown_t    = 0.0
    frame_count   = 0
    cur_word      = ""
    cur_conf      = 0.0
    top_vote_word = ""
    top_vote_cnt  = 0
    low_conf_s    = 0

    while True:
        with _lock:
            if not _state["camera_active"]:
                break
            # pull mutable state we need
            sentence_words     = _state["sentence_words"]
            completed_sentence = _state["completed_sentence"]
            lang_code          = _state["lang_code"]

        ok, raw = cap.read()
        if not ok:
            continue

        frame = cv2.flip(raw, 1)
        h, w  = frame.shape[:2]
        now   = time.time()

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = holistic.detect(mp_img)

        hands_present = bool(result.left_hand_landmarks or result.right_hand_landmarks)

        # ── Draw landmarks overlay ────────────────────────────────────
        DRAW_LANDMARKS = True   # set False to hide overlay
        if DRAW_LANDMARKS:
            def _draw_connections(frm, landmarks, connections, color, radius=2):
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
                for a, b in connections:
                    cv2.line(frm, pts[a], pts[b], color, 1)
                for pt in pts:
                    cv2.circle(frm, pt, radius, color, -1)
            HAND_CONNECTIONS = [
                (0,1),(1,2),(2,3),(3,4),
                (0,5),(5,6),(6,7),(7,8),
                (0,9),(9,10),(10,11),(11,12),
                (0,13),(13,14),(14,15),(15,16),
                (0,17),(17,18),(18,19),(19,20),
                (5,9),(9,13),(13,17),
            ]
            POSE_CONNECTIONS = [
                (11,12),(11,13),(13,15),(12,14),(14,16),
                (11,23),(12,24),(23,24),(23,25),(24,26),
            ]
            if result.left_hand_landmarks:
                _draw_connections(frame, result.left_hand_landmarks, HAND_CONNECTIONS, (0,255,0))
            if result.right_hand_landmarks:
                _draw_connections(frame, result.right_hand_landmarks, HAND_CONNECTIONS, (0,255,0))
            if result.pose_landmarks:
                _draw_connections(frame, result.pose_landmarks, POSE_CONNECTIONS, (255,100,0))

        if hands_present:
            last_hand_t = now
            sequence.append(extract_keypoints(result))
        else:
            if now - last_hand_t > 0.5:
                sequence.clear()
                vote_window.clear()

        # Auto-finalize
        if sentence_words and now - last_hand_t > SENTENCE_TIMEOUT and now > cooldown_t:
            with _lock:
                _do_finalize(_state, now)
                sentence_words = _state["sentence_words"]
                completed_sentence = _state["completed_sentence"]
            sequence.clear()
            vote_window.clear()

        # Predict
        frame_count += 1
        if len(sequence) == MAX_FRAMES and now > cooldown_t and frame_count % PREDICT_EVERY == 0:
            X     = np.expand_dims(np.array(sequence, dtype=np.float32), 0)
            probs = _model.predict(X, verbose=0)[0]
            top_i = int(np.argmax(probs))
            cur_conf = float(probs[top_i])
            cur_word = _labels[top_i]

            if cur_conf >= CONFIDENCE_THRESHOLD:
                vote_window.append(cur_word)
                low_conf_s = 0
            else:
                low_conf_s += 1
                if low_conf_s >= LOW_CONF_TOLERANCE:
                    vote_window.clear()
                    low_conf_s = 0

            if vote_window:
                top_vote_word, top_vote_cnt = Counter(vote_window).most_common(1)[0]
            else:
                top_vote_word, top_vote_cnt = "", 0

            if top_vote_cnt >= VOTE_THRESHOLD:
                with _lock:
                    if not _state["sentence_words"] or _state["sentence_words"][-1] != top_vote_word:
                        _state["sentence_words"].append(top_vote_word)
                        sentence_words = _state["sentence_words"]
                        print(f"  + {top_vote_word}")
                cooldown_t = now + WORD_COOLDOWN
                vote_window.clear()
                low_conf_s = 0

        # ── Update shared state ───────────────────────────────────────────────
        buf_pct  = len(sequence) / MAX_FRAMES
        vote_pct = top_vote_cnt / VOTE_WINDOW if top_vote_cnt > 0 else 0
        with _lock:
            _state["current_word"]    = cur_word
            _state["current_conf"]    = cur_conf
            _state["top_vote_word"]   = top_vote_word
            _state["top_vote_count"]  = top_vote_cnt
            _state["hands_present"]   = hands_present
            _state["buf_pct"]         = round(buf_pct, 2)
            _state["vote_pct"]        = round(vote_pct, 2)
            _state["in_cooldown"]     = now < cooldown_t

        # ── Draw overlays (same as sentence_builder.py) ───────────────────────
        bar_w = w - 20

        # top bar
        _dark_bar(frame, 0, 48)
        buf_ready = len(sequence) == MAX_FRAMES
        if not hands_present:
            cv2.putText(frame, "Waiting ...", (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, C_GRAY, 2)
        elif not buf_ready:
            if cur_word:
                cv2.putText(frame, cur_word, (12, 34),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (70, 70, 70), 2)
            cv2.putText(frame, f"{len(sequence)}/{MAX_FRAMES}",
                        (w - 90, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (70, 70, 70), 1)
        else:
            colour = C_GREEN if cur_conf >= CONFIDENCE_THRESHOLD else C_GRAY
            cv2.putText(frame, cur_word, (12, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2)
            cv2.putText(frame, f"{cur_conf*100:.0f}%",
                        (w - 80, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)

        # progress bar
        in_cooldown = now < cooldown_t
        if top_vote_cnt > 0 and not in_cooldown:
            bar_col = C_ORANGE
            bar_pct = top_vote_cnt / VOTE_WINDOW
        else:
            bar_col = C_CYAN
            bar_pct = len(sequence) / MAX_FRAMES
        cv2.rectangle(frame, (10, 50), (10 + bar_w, 54), C_GRAY, 1)
        cv2.rectangle(frame, (10, 50),
                      (10 + int(bar_w * min(bar_pct, 1.0)), 54), bar_col, -1)

        # bottom strip
        _dark_bar(frame, h - 72, h, 0.75)
        with _lock:
            sw = _state["sentence_words"][:]
            cs = _state["completed_sentence"]
        lang_label = f"Lang: {LANGUAGES.get(lang_code, 'English')}  [1=En 2=Hi 3=Bn 4=Ta 5=Te 6=Mr 7=Gu]"
        building = " ".join(sw)

        if cs and not sw:
            cv2.putText(frame, cs, (12, h - 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_GREEN, 2)
        elif building:
            cv2.putText(frame, building, (12, h - 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_YELLOW, 2)
        else:
            cv2.putText(frame, "Start signing ...", (12, h - 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_GRAY, 1)

        cv2.putText(frame, lang_label, (12, h - 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_CYAN, 1)
        cv2.putText(frame, "Use website controls to Confirm / Speak / Undo / Clear",
                    (12, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_GRAY, 1)

        # Encode to JPEG and store
        _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        _current_frame = jpg.tobytes()

    cap.release()
    # Do NOT close holistic — it is the global singleton, reused on next start
    _current_frame = None
    print("  Camera stopped.")


# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app)


@app.after_request
def log_developer_api_response(response):
    if request.path.startswith("/developer/api/"):
        print(
            f"[Developer API] {request.method} {request.base_url} "
            f"-> {response.status_code}",
            flush=True,
        )
    return response


@app.route("/")
def index():
    return send_from_directory(str(FRONTEND_DIR), "index.html")


def _developer_request_is_local() -> bool:
    try:
        remote = ipaddress.ip_address(request.remote_addr or "")
        host = urlsplit(f"//{request.host}").hostname
    except ValueError:
        return False
    if not remote.is_loopback or host not in {"localhost", "127.0.0.1", "::1"}:
        return False
    origin = request.headers.get("Origin")
    if origin:
        try:
            origin_host = urlsplit(origin).hostname
        except ValueError:
            return False
        if origin_host != host:
            return False
    return True


def _developer_forbidden_response():
    if not _developer_request_is_local():
        return jsonify({"error": "Developer tools are available only on localhost."}), 403
    return None


def _canonical_label_id(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", label.casefold()).strip("-")[:100]


def _base_label_names() -> list[str]:
    if not FEATURES_DIR.is_dir():
        return []
    return sorted(path.name for path in FEATURES_DIR.iterdir() if path.is_dir())


def _model_artifact_references(scope: str, entry: VocabularyEntry) -> list[dict]:
    models_root = VOCABULARY_STORE.root / scope / "models"
    references = []
    if not models_root.is_dir():
        return references
    for label_map_path in sorted(models_root.glob("*/label_map.json")):
        try:
            with label_map_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            references.append({
                "artifact_id": label_map_path.parent.name,
                "unreadable": True,
            })
            continue
        labels = payload.get("label_map") if isinstance(payload, dict) else None
        if not isinstance(labels, list):
            references.append({
                "artifact_id": label_map_path.parent.name,
                "unreadable": True,
            })
            continue
        if any(
            isinstance(item, dict)
            and item.get("label_id") == entry.label_id
            and isinstance(item.get("label"), str)
            and item["label"].casefold() == entry.label.casefold()
            for item in labels
        ):
            references.append({
                "artifact_id": label_map_path.parent.name,
                "unreadable": False,
            })
    return references


def _draft_payload(scope: str) -> list[dict]:
    entries = VOCABULARY_STORE.list_entries(scope)
    sessions = VOCABULARY_STORE.list_sessions(scope)
    samples = VOCABULARY_STORE.list_samples(scope)
    latest_session = {}
    sessions_by_entry = Counter(session.entry_id for session in sessions)
    samples_by_session = Counter(sample.session_id for sample in samples)
    recorded_sessions_by_entry = {}
    for sample in samples:
        recorded_sessions_by_entry.setdefault(sample.entry_id, set()).add(sample.session_id)
    recorded_session_counts = {
        entry_id: len(session_ids)
        for entry_id, session_ids in recorded_sessions_by_entry.items()
    }
    for session in sessions:
        previous = latest_session.get(session.entry_id)
        if previous is None or session.created_at > previous.created_at:
            latest_session[session.entry_id] = session
    return [
        {
            "entry_id": entry.entry_id,
            "label_id": entry.label_id,
            "label": entry.label,
            "scope": entry.scope,
            "sample_count": VOCABULARY_STORE.count_samples(scope, entry.entry_id),
            "session_count": recorded_session_counts.get(entry.entry_id, 0),
            "capture_session_count": sessions_by_entry[entry.entry_id],
            "session_id": (latest_session[entry.entry_id].session_id
                           if entry.entry_id in latest_session else None),
            "session_sample_count": (
                samples_by_session[latest_session[entry.entry_id].session_id]
                if entry.entry_id in latest_session else 0
            ),
            "minimum_training_sessions": MIN_TRAIN_SESSIONS_PER_CLASS,
            "minimum_evaluation_sessions": MIN_EVALUATION_SESSIONS_PER_CLASS,
            "training_ready": (
                recorded_session_counts.get(entry.entry_id, 0)
                >= MIN_TRAIN_SESSIONS_PER_CLASS
            ),
            "evaluation_ready": (
                recorded_session_counts.get(entry.entry_id, 0)
                >= MIN_EVALUATION_SESSIONS_PER_CLASS
            ),
            "preprocessing_version": entry.preprocessing_version,
            "model_references": _model_artifact_references(scope, entry),
        }
        for entry in entries
    ]


def _vocabulary_version(scope: str) -> str:
    labels = [
        {"entry_id": entry.entry_id, "label_id": entry.label_id, "label": entry.label}
        for entry in sorted(
            VOCABULARY_STORE.list_entries(scope),
            key=lambda item: item.label_id,
        )
    ]
    payload = json.dumps(labels, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _storage_version(scope: str) -> str:
    digest = hashlib.sha256()
    scope_root = VOCABULARY_STORE.root / scope
    for name in ("vocabulary.json", "sessions.json", "samples.json"):
        path = scope_root / name
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes() if path.is_file() else b"<missing>")
    return digest.hexdigest()


def _expected_label_map(entries, scope: str) -> list[dict]:
    return [
        {"label_id": entry.label_id, "label": entry.label, "scope": scope}
        for entry in sorted(entries, key=lambda item: item.label_id)
    ]


def _model_label_map_matches(label_map, entries, scope: str) -> bool:
    return (
        isinstance(label_map, list)
        and bool(label_map)
        and label_map == _expected_label_map(entries, scope)
    )


def _evaluation_matches_label_map(evaluation: dict, label_map: list[dict]) -> bool:
    if not isinstance(evaluation, dict):
        return False
    validation = evaluation.get("validation")
    if (
        not isinstance(label_map, list)
        or not label_map
        or any(not isinstance(item, dict) for item in label_map)
        or any(
            not isinstance(item.get("label"), str)
            or type(item.get("label_id")) is not int
            for item in label_map
        )
        or not evaluation.get("evaluation_ready")
        or not isinstance(validation, dict)
    ):
        return False
    labels = [item["label"] for item in label_map]
    matrix = validation.get("confusion_matrix")
    support = validation.get("support")
    per_class = validation.get("per_class")
    if not isinstance(per_class, dict):
        return False
    return (
        validation.get("label_order") == labels
        and isinstance(matrix, list)
        and len(matrix) == len(labels)
        and all(
            isinstance(row, list)
            and len(row) == len(labels)
            and all(type(value) is int and value >= 0 for value in row)
            for row in matrix
        )
        and type(support) is int
        and support > 0
        and sum(sum(row) for row in matrix) == support
        and set(per_class) == set(labels)
        and all(
            isinstance(per_class[label], dict)
            and type(per_class[label].get("support")) is int
            and per_class[label]["support"] > 0
            for label in labels
        )
    )


def _experimental_status(scope: str) -> dict:
    live_entries = VOCABULARY_STORE.list_entries(scope)
    live_map = {
        (entry.label_id, entry.label): entry.entry_id
        for entry in live_entries
    }
    active = {
        "available": False,
        "artifact_id": None,
        "preprocessing_version": None,
        "trained_data_version": None,
        "vocabulary_version": None,
        "compatible": False,
        "usable": False,
        "data_current": False,
        "current_data_version": None,
        "supports": [],
        "coverage_complete": False,
        "evaluation_ready": False,
        "validation": None,
        "unusable_reason": "No active model is configured for this scope.",
        "error": None,
    }
    current_data_version = None
    try:
        classifier = load_artifact(VOCABULARY_STORE, scope)
        metadata = classifier.metadata
        model_map = metadata.get("label_map", [])
        compatible = _model_label_map_matches(model_map, live_entries, scope)
        supports = [
            {
                "entry_id": live_map[(item["label_id"], item["label"])],
                "label_id": item["label_id"],
                "label": item["label"],
            }
            for item in model_map
        ] if compatible else []
        evaluation = metadata.get("evaluation", {})
        evaluation_valid = _evaluation_matches_label_map(evaluation, model_map)
        current_data_version = dataset_fingerprint(
            VOCABULARY_STORE.load_training_examples(scope)
        )
        trained_data_version = metadata.get("training", {}).get(
            "data", {}
        ).get("fingerprint")
        active.update({
            "available": True,
            "artifact_id": metadata.get("artifact_id"),
            "preprocessing_version": metadata.get("preprocessing_version"),
            "trained_data_version": trained_data_version,
            "current_data_version": current_data_version,
            "data_current": bool(trained_data_version)
                and trained_data_version == current_data_version,
            "vocabulary_version": hashlib.sha256(json.dumps(
                model_map, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")).hexdigest(),
            "compatible": compatible,
            "usable": (
                compatible
                and trained_data_version == current_data_version
                and evaluation_valid
            ),
            "supports": supports,
            "coverage_complete": compatible,
            "evaluation_ready": bool(evaluation.get("evaluation_ready")),
            "validation": evaluation.get("validation"),
            "unusable_reason": (
                None if compatible and trained_data_version == current_data_version
                and evaluation_valid
                else (
                    "Model labels do not exactly match the current saved vocabulary."
                    if not compatible else
                    "Saved recordings changed after this model was trained."
                    if trained_data_version != current_data_version else
                    "A complete session-separated validation report is required."
                )
            ),
        })
    except (ArtifactCompatibilityError, OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        active["error"] = str(error)

    candidates = []
    try:
        artifacts = list_artifacts(VOCABULARY_STORE, scope)
    except (ArtifactCompatibilityError, OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        artifacts = []
        candidate_list_error = str(error)
    else:
        candidate_list_error = None
    for artifact in artifacts:
        try:
            candidate = load_artifact(
                VOCABULARY_STORE, scope, artifact["artifact_id"]
            )
            evaluation = candidate.metadata.get("evaluation", {})
            if current_data_version is None:
                current_data_version = dataset_fingerprint(
                    VOCABULARY_STORE.load_training_examples(scope)
                )
            candidate_fingerprint = candidate.metadata.get("training", {}) \
                .get("data", {}).get("fingerprint")
            vocabulary_matches = _model_label_map_matches(
                candidate.label_map, live_entries, scope
            )
            candidates.append({
                "artifact_id": artifact["artifact_id"],
                "created_at": artifact["created_at"],
                "evaluation_ready": bool(evaluation.get("evaluation_ready")),
                "validation": evaluation.get("validation"),
                "active": artifact["artifact_id"] == active["artifact_id"],
                "data_current": bool(candidate_fingerprint)
                    and candidate_fingerprint == current_data_version,
                "vocabulary_matches": vocabulary_matches,
                "coverage_complete": vocabulary_matches,
                "label_map": candidate.label_map,
                "supported_label_count": len(candidate.label_map),
                "preprocessing_compatible": (
                    candidate.metadata.get("preprocessing_version")
                    == PREPROCESSING_VERSION
                    and candidate.metadata.get("sequence_shape")
                    == list(SEQUENCE_SHAPE)
                ),
                "evaluation_valid": _evaluation_matches_label_map(
                    evaluation, candidate.label_map
                ),
            })
        except (ArtifactCompatibilityError, OSError, ValueError, KeyError, TypeError, AttributeError):
            candidates.append({
                "artifact_id": artifact.get("artifact_id"),
                "created_at": artifact.get("created_at"),
                "evaluation_ready": False,
                "validation": None,
                "active": False,
                "data_current": False,
                "vocabulary_matches": False,
                "invalid": True,
            })
    return {
        "scope": scope,
        "preprocessing_version": PREPROCESSING_VERSION,
        "vocabulary_version": _vocabulary_version(scope),
        "active_model": active,
        "candidates": candidates,
        "candidate_list_error": candidate_list_error,
    }


def _sequence_from_frame_data_urls(frames):
    decoded_frames = []
    for index, frame_data in enumerate(frames):
        if not isinstance(frame_data, str) or not frame_data.startswith(
            "data:image/jpeg;base64,"
        ):
            raise ValueError(f"Frame {index + 1} is not a JPEG data URL.")
        try:
            image_bytes = base64.b64decode(frame_data.partition(",")[2], validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValueError(f"Frame {index + 1} has invalid base64 data.") from error
        if len(image_bytes) > 300_000:
            raise ValueError(f"Frame {index + 1} is too large.")
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Frame {index + 1} is not a decodable image.")
        decoded_frames.append(image)

    holistic = _build_holistic()
    try:
        features = []
        for image in decoded_frames:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            features.append(extract_frame_features(holistic.detect(mp_image)))
    finally:
        holistic.close()
    sequence = np.asarray(features, dtype=np.float32)
    return validate_capture_sequence(sequence, PREPROCESSING_VERSION)


@app.route("/developer")
@app.route("/developer.html")
def developer_page():
    forbidden = _developer_forbidden_response()
    if forbidden:
        return forbidden
    return send_from_directory(str(FRONTEND_DIR), "developer.html")


@app.route("/developer/api/drafts", methods=["GET"])
def list_developer_drafts():
    forbidden = _developer_forbidden_response()
    if forbidden:
        return forbidden
    scope = request.args.get("scope", "")
    if scope not in {"shared", "personal"}:
        return jsonify({"error": "scope must be shared or personal"}), 400
    return jsonify({"drafts": _draft_payload(scope)})


@app.route("/api/vocabulary", methods=["GET"])
def list_public_vocabulary():
    forbidden = _developer_forbidden_response()
    if forbidden:
        return forbidden
    scope = request.args.get("scope", "shared")
    if scope not in {"shared", "personal"}:
        return jsonify({"error": "scope must be shared or personal"}), 400
    drafts = _draft_payload(scope)
    supported = set()
    status = _experimental_status(scope)
    if status["active_model"]["usable"]:
        supported = {item["entry_id"] for item in status["active_model"]["supports"]}
    words = [
        {
            **draft,
            "recognizable_by_active_model": draft["entry_id"] in supported,
        }
        for draft in drafts
    ]
    knn_labels = list(KNN_CLASS_LABELS)
    return jsonify({
        "scope": scope,
        "words": words,
        "storage_version": _storage_version(scope),
        "recognizers": {
            "browser_knn": {
                "id": "browser_knn",
                "label": "Browser KNN",
                "available": True,
                "fixed_vocabulary": True,
                "vocabulary": knn_labels,
            },
            "experimental": {
                "id": "experimental",
                "label": "Active sequence classifier",
                "available": bool(status["active_model"]["usable"]),
                "active_artifact_id": status["active_model"]["artifact_id"],
                "active_model": status["active_model"],
                "vocabulary": [
                    item["label"] for item in status["active_model"]["supports"]
                ],
                "supported_label_count": len(status["active_model"]["supports"]),
                "saved_vocabulary_count": len(words),
                "full_vocabulary_supported": (
                    bool(status["active_model"]["usable"])
                    and len(status["active_model"]["supports"]) == len(words)
                ),
                "unavailable_reason": status["active_model"].get("unusable_reason"),
            },
        },
        **status,
    })


@app.route("/api/experimental/train", methods=["POST"])
def train_experimental_candidate():
    forbidden = _developer_forbidden_response()
    if forbidden:
        return forbidden
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or data.get("scope") not in {"shared", "personal"}:
        return jsonify({"error": "scope must be shared or personal"}), 400
    scope = data["scope"]
    drafts = _draft_payload(scope)
    if len(drafts) < 2:
        return jsonify({"error": "At least two vocabulary words are required."}), 422
    unready = [
        {"label": item["label"], "recorded_sessions": item["session_count"]}
        for item in drafts if not item["training_ready"]
    ]
    if unready:
        return jsonify({
            "error": "Every word needs at least "
                     f"{MIN_TRAIN_SESSIONS_PER_CLASS} recorded sessions before training.",
            "unready_words": unready,
        }), 422
    try:
        examples = VOCABULARY_STORE.load_training_examples(scope)
        split = split_by_session(examples)
        if not split.train_indices:
            raise InsufficientDataError("No eligible training examples are available.")
        report = train_and_evaluate(VOCABULARY_STORE, scope, promote=False)
    except (InsufficientDataError, ArtifactCompatibilityError, ValueError, OSError) as error:
        return jsonify({"error": str(error)}), 422
    return jsonify({
        "candidate": {
            "artifact_id": report["artifact"]["artifact_id"],
            "scope": report["scope"],
            "evaluation_ready": report["evaluation_ready"],
            "validation": report["validation"],
            "insufficiencies": report["insufficiencies"],
            "data": report["data"],
        },
        "activated": False,
        "message": (
            "Candidate saved without changing the active model. "
            "Session-separated evaluation is required before activation."
        ),
    }), 201


@app.route("/api/experimental/activate/<artifact_id>", methods=["POST"])
def activate_experimental_candidate(artifact_id):
    forbidden = _developer_forbidden_response()
    if forbidden:
        return forbidden
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or data.get("scope") not in {"shared", "personal"}:
        return jsonify({"error": "scope must be shared or personal"}), 400
    scope = data["scope"]
    try:
        candidate = load_artifact(VOCABULARY_STORE, scope, artifact_id)
        evaluation = candidate.metadata.get("evaluation", {})
        if not _evaluation_matches_label_map(evaluation, candidate.label_map):
            return jsonify({
                "error": "This candidate has no session-separated validation report and cannot be activated."
            }), 409
        with VOCABULARY_STORE._lock:
            current_examples = VOCABULARY_STORE.load_training_examples(scope)
            current_fingerprint = dataset_fingerprint(current_examples)
            candidate_fingerprint = candidate.metadata.get("training", {}).get(
                "data", {}
            ).get("fingerprint")
            if not candidate_fingerprint or candidate_fingerprint != current_fingerprint:
                return jsonify({
                    "error": "The saved examples changed after this candidate was trained. Train a fresh candidate before activation."
                }), 409
            current_entries = VOCABULARY_STORE.list_entries(scope)
            if not _model_label_map_matches(
                candidate.label_map, current_entries, scope
            ):
                return jsonify({
                    "error": "The candidate label map no longer matches the current vocabulary. Train a fresh candidate."
                }), 409
            activate_artifact(VOCABULARY_STORE, scope, artifact_id)
    except (ArtifactCompatibilityError, OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        return jsonify({"error": str(error)}), 422
    return jsonify({
        "activated": True,
        "scope": scope,
        "artifact_id": artifact_id,
        "vocabulary_version": _vocabulary_version(scope),
    })


@app.route("/developer/api/drafts", methods=["POST"])
def create_developer_draft():
    forbidden = _developer_forbidden_response()
    if forbidden:
        return forbidden
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "A JSON object is required."}), 400
    label = data.get("label")
    scope = data.get("scope")
    if (not isinstance(label, str) or not label.strip()
            or len(label.strip()) > 80):
        return jsonify({"error": "Enter a canonical label of 1 to 80 characters."}), 400
    label = label.strip()
    if not re.fullmatch(r"[\w][\w '\-]{0,79}", label, flags=re.UNICODE):
        return jsonify({"error": "The label contains unsupported characters."}), 400
    if scope not in {"shared", "personal"}:
        return jsonify({"error": "scope must be shared or personal"}), 400

    entry_id = _canonical_label_id(label)
    if not entry_id:
        return jsonify({"error": "The label must contain letters or numbers."}), 400
    entries = VOCABULARY_STORE.list_entries(scope)
    base_labels = _base_label_names() if scope == "shared" else []
    if any(label.casefold() == item.casefold() for item in base_labels):
        return jsonify({"error": "This label is already in the built-in vocabulary."}), 409
    existing = next(
        (entry for entry in entries if entry.label.casefold() == label.casefold()),
        None,
    )
    if existing:
        sessions = [
            session for session in VOCABULARY_STORE.list_sessions(scope)
            if session.entry_id == existing.entry_id
        ]
        if sessions:
            session = max(sessions, key=lambda item: item.created_at)
        else:
            session = CaptureSession.create(existing)
            VOCABULARY_STORE.create_session(session)
        return jsonify({
            "draft": _draft_payload(scope)[
                next(i for i, item in enumerate(entries)
                     if item.entry_id == existing.entry_id)
            ],
            "session_id": session.session_id,
        })
    if any(entry.entry_id == entry_id for entry in entries):
        return jsonify({"error": "A conflicting canonical entry ID already exists."}), 409

    entry = VocabularyEntry(
        entry_id=entry_id,
        label_id=max(
            len(base_labels) - 1,
            max((item.label_id for item in entries), default=-1),
        ) + 1,
        label=label,
        scope=scope,
    )
    VOCABULARY_STORE.add_entry(entry)
    session = CaptureSession.create(entry)
    VOCABULARY_STORE.create_session(session)
    draft = next(
        item for item in _draft_payload(scope)
        if item["entry_id"] == entry.entry_id
    )
    return jsonify({"draft": draft, "session_id": session.session_id}), 201


@app.route("/developer/api/drafts/<entry_id>/sessions", methods=["POST"])
def start_developer_capture_session(entry_id):
    forbidden = _developer_forbidden_response()
    if forbidden:
        return forbidden
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "A JSON object is required."}), 400
    scope = data.get("scope")
    if not isinstance(scope, str) or scope not in {"shared", "personal"}:
        return jsonify({"error": "scope must be shared or personal"}), 400
    try:
        validate_entry_id(entry_id)
    except ValueError as error:
        return jsonify({"error": str(error)}), 422

    entry = next(
        (item for item in VOCABULARY_STORE.list_entries(scope)
         if item.entry_id == entry_id),
        None,
    )
    if entry is None:
        return jsonify({"error": "Draft does not exist in the requested scope."}), 404
    session = CaptureSession.create(entry)
    try:
        VOCABULARY_STORE.create_session(session)
    except ValueError as error:
        return jsonify({"error": str(error)}), 409

    draft = next(
        item for item in _draft_payload(scope)
        if item["entry_id"] == entry_id
    )
    return jsonify({"draft": draft, "session_id": session.session_id}), 201


@app.route("/developer/api/drafts/<entry_id>", methods=["DELETE"])
def delete_developer_draft(entry_id):
    forbidden = _developer_forbidden_response()
    if forbidden:
        return forbidden
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "A JSON object is required."}), 400
    scope = data.get("scope")
    if not isinstance(scope, str) or scope not in {"shared", "personal"}:
        return jsonify({"error": "scope must be shared or personal"}), 400
    if data.get("confirmed") is not True:
        return jsonify({"error": "Explicit deletion confirmation is required."}), 400
    try:
        validate_entry_id(entry_id)
    except ValueError as error:
        return jsonify({"error": str(error)}), 422

    entry = next(
        (item for item in VOCABULARY_STORE.list_entries(scope)
         if item.entry_id == entry_id),
        None,
    )
    if entry is None:
        return jsonify({"error": "Draft does not exist in the requested scope."}), 404
    if scope == "shared" and any(
        entry.label.casefold() == label.casefold()
        for label in _base_label_names()
    ):
        return jsonify({"error": "Built-in vocabulary labels cannot be deleted here."}), 409

    model_references = _model_artifact_references(scope, entry)
    try:
        deleted = VOCABULARY_STORE.delete_entry(entry_id, scope)
    except ValueError as error:
        return jsonify({"error": str(error)}), 422
    except OSError as error:
        return jsonify({"error": str(error)}), 500
    return jsonify({
        "deleted": deleted,
        "model_references": model_references,
    })


@app.route("/developer/api/capture/process", methods=["POST"])
def process_developer_capture():
    forbidden = _developer_forbidden_response()
    if forbidden:
        return forbidden
    if not Path(HOLISTIC_MODEL).is_file():
        return jsonify({
            "error": (
                "Developer capture needs the configured MediaPipe Holistic task "
                "file. Set ISL_HOLISTIC_MODEL_PATH to an authorized local copy."
            )
        }), 503
    with _lock:
        if _state["camera_active"]:
            return jsonify({
                "error": "Stop the regular Flask camera before developer capture."
            }), 409
        if _developer_capture_lease_until < time.monotonic():
            return jsonify({"error": "Acquire the local camera lease before capture."}), 409
    if request.content_length and request.content_length > 12 * 1024 * 1024:
        return jsonify({"error": "Capture payload is too large."}), 413
    data = request.get_json(silent=True)
    frames = data.get("frames") if isinstance(data, dict) else None
    if not isinstance(frames, list) or len(frames) != MAX_FRAMES:
        return jsonify({"error": f"Exactly {MAX_FRAMES} webcam frames are required."}), 400
    try:
        sequence, quality = _sequence_from_frame_data_urls(frames)
    except ValueError as error:
        return jsonify({"error": str(error)}), 422
    return jsonify({
        "sequence": sequence.tolist(),
        "quality": {
            "frames": quality.frames,
            "hand_frames": quality.hand_frames,
            "hand_frame_ratio": quality.hand_frame_ratio,
            "preprocessing_version": PREPROCESSING_VERSION,
        },
    })


@app.route("/api/experimental/predict", methods=["POST"])
def predict_experimental_sign():
    forbidden = _developer_forbidden_response()
    if forbidden:
        return forbidden
    if request.content_length and request.content_length > 12 * 1024 * 1024:
        return jsonify({"error": "Capture payload is too large."}), 413
    data = request.get_json(silent=True)
    if (not isinstance(data, dict)
            or not isinstance(data.get("scope"), str)
            or data.get("scope") not in {"shared", "personal"}):
        return jsonify({"error": "scope must be shared or personal"}), 400
    if not Path(HOLISTIC_MODEL).is_file():
        return jsonify({
            "error": (
                "Experimental live capture needs the configured MediaPipe "
                "Holistic task file. Set ISL_HOLISTIC_MODEL_PATH to an "
                "authorized local copy."
            )
        }), 503
    scope = data["scope"]
    artifact_id = data.get("artifact_id")
    if artifact_id is not None and (
        not isinstance(artifact_id, str)
        or not re.fullmatch(r"\d{8}T\d{12}Z", artifact_id)
    ):
        return jsonify({"error": "artifact_id is invalid."}), 400
    frames = data.get("frames")
    if not isinstance(frames, list) or len(frames) != MAX_FRAMES:
        return jsonify({"error": f"Exactly {MAX_FRAMES} webcam frames are required."}), 400
    try:
        classifier = load_artifact(VOCABULARY_STORE, scope, artifact_id)
    except (ArtifactCompatibilityError, OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        return jsonify({"error": f"No usable active experimental model: {error}"}), 503
    evaluation = classifier.metadata.get("evaluation", {})
    test_only = artifact_id is not None
    if artifact_id is None:
        status = _experimental_status(scope)["active_model"]
        if not status["usable"]:
            return jsonify({
                "error": "The active model is not usable for this vocabulary until its label map and session-separated evaluation are validated."
            }), 409
        if status["artifact_id"] != classifier.metadata.get("artifact_id"):
            return jsonify({"error": "The active model changed while loading. Refresh and retry."}), 409
        supports = status["supports"]
        vocabulary_version = status["vocabulary_version"]
    else:
        if classifier.metadata.get("scope") != scope:
            return jsonify({
                "error": "Candidate scope does not match the requested vocabulary."
            }), 409
        try:
            with VOCABULARY_STORE._lock:
                current_examples = VOCABULARY_STORE.load_training_examples(scope)
                current_fingerprint = dataset_fingerprint(current_examples)
                current_entries = sorted(
                    VOCABULARY_STORE.list_entries(scope),
                    key=lambda item: item.label_id,
                )
                current_map = [
                    {"label_id": entry.label_id, "label": entry.label, "scope": scope}
                    for entry in current_entries
                ]
        except (OSError, ValueError, KeyError, TypeError) as error:
            return jsonify({"error": f"Could not verify candidate data: {error}"}), 422
        candidate_fingerprint = classifier.metadata.get("training", {}).get(
            "data", {}
        ).get("fingerprint")
        if candidate_fingerprint != current_fingerprint:
            return jsonify({
                "error": "Saved examples changed after this candidate was trained. Train a fresh candidate before testing."
            }), 409
        if not _model_label_map_matches(classifier.label_map, current_entries, scope):
            return jsonify({
                "error": "Candidate labels no longer match the current vocabulary. Train a fresh candidate before testing."
            }), 409
        supports = [
            {"label_id": entry.label_id, "entry_id": entry.entry_id}
            for entry in current_entries
        ]
        vocabulary_version = _vocabulary_version(scope)
    request_vocabulary_version = _vocabulary_version(scope)
    request_storage_version = _storage_version(scope)
    vocabulary_version = request_vocabulary_version
    if not _model_label_map_matches(
        classifier.label_map,
        VOCABULARY_STORE.list_entries(scope),
        scope,
    ):
        return jsonify({"error": "Model labels do not match the current vocabulary."}), 409
    try:
        sequence, quality = _sequence_from_frame_data_urls(frames)
        prediction = classifier.predict(sequence)
    except (TypeError, ValueError, OSError) as error:
        return jsonify({"error": str(error)}), 422
    if not isinstance(prediction, dict):
        return jsonify({"error": "The classifier returned an invalid prediction."}), 502
    label_id = prediction.get("label_id")
    confidence = prediction.get("confidence")
    if (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or not np.isfinite(confidence)
        or not 0 <= confidence <= 1
        or not isinstance(prediction.get("abstained"), bool)
        or prediction.get("scope") != scope
        or prediction.get("artifact_id") != classifier.metadata.get("artifact_id")
        or prediction.get("preprocessing_version") != PREPROCESSING_VERSION
    ):
        return jsonify({"error": "The classifier returned an invalid prediction contract."}), 502
    if prediction["abstained"]:
        if prediction.get("label") is not None or label_id is not None:
            return jsonify({"error": "The classifier returned a label for an abstained prediction."}), 502
        entry_id = None
    else:
        mapped = next(
            (item for item in classifier.label_map if item["label_id"] == label_id),
            None,
        )
        if (
            type(label_id) is not int
            or mapped is None
            or prediction.get("label") != mapped["label"]
        ):
            return jsonify({"error": "The classifier label index does not match the current label map."}), 502
        supported = {item["label_id"]: item["entry_id"] for item in supports}
        entry_id = supported.get(label_id)
        if entry_id is None:
            return jsonify({"error": "The classifier predicted a label outside the supported vocabulary."}), 502
    if (
        request_vocabulary_version != _vocabulary_version(scope)
        or request_storage_version != _storage_version(scope)
    ):
        return jsonify({"error": "Saved vocabulary or examples changed during prediction. Refresh and retry."}), 409
    prediction.update({
        "entry_id": entry_id,
        "model_version": prediction["artifact_id"],
        "vocabulary_version": vocabulary_version,
        "current_vocabulary_version": request_vocabulary_version,
        "current_storage_version": request_storage_version,
        "evaluation_ready": bool(evaluation.get("evaluation_ready")),
        "evaluation_status": (
            "session-separated evaluation available; experimental"
            if _evaluation_matches_label_map(evaluation, classifier.label_map)
            else "unvalidated: four independent sessions per word are required"
        ),
        "test_only": test_only,
        "sequence_quality": {
            "frames": quality.frames,
            "hand_frames": quality.hand_frames,
            "preprocessing_version": PREPROCESSING_VERSION,
        },
    })
    return jsonify(prediction)


@app.route("/api/ai-sentence/status", methods=["GET"])
def ai_sentence_status():
    forbidden = _developer_forbidden_response()
    if forbidden:
        return forbidden
    return jsonify(AI_SENTENCE_BUILDER.status())


@app.route("/api/ai-sentence", methods=["POST"])
def build_ai_sentence():
    forbidden = _developer_forbidden_response()
    if forbidden:
        return forbidden
    if request.content_length and request.content_length > 20 * 1024:
        return jsonify({"error": "Sentence request exceeds the 20 KB limit."}), 413
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "A JSON object is required."}), 400
    recognizer = data.get("recognizer")
    if not isinstance(recognizer, str) or recognizer not in {
        "browser_knn", "experimental",
    }:
        return jsonify({"error": "Select a supported recognizer."}), 400

    if recognizer == "browser_knn":
        vocabulary = sorted(KNN_GLOSS_WORDS, key=str.casefold)
    else:
        scope = data.get("scope")
        artifact_id = data.get("artifact_id")
        if not isinstance(scope, str) or scope not in {"shared", "personal"}:
            return jsonify({"error": "scope must be shared or personal."}), 400
        if (not isinstance(artifact_id, str)
                or not re.fullmatch(r"\d{8}T\d{12}Z", artifact_id)):
            return jsonify({"error": "A current active sequence model is required."}), 400
        try:
            active = _experimental_status(scope)["active_model"]
            if not active["usable"] or active["artifact_id"] != artifact_id:
                return jsonify({
                    "error": "The active model is unavailable or changed. Refresh and collect the sequence again."
                }), 409
            classifier = load_artifact(VOCABULARY_STORE, scope, artifact_id)
            current_examples = VOCABULARY_STORE.load_training_examples(scope)
            current_fingerprint = dataset_fingerprint(current_examples)
            entries = VOCABULARY_STORE.list_entries(scope)
        except (ArtifactCompatibilityError, OSError, ValueError, KeyError, TypeError) as error:
            return jsonify({"error": f"Could not verify the active model: {error}"}), 409
        if classifier.metadata.get("scope") != scope:
            return jsonify({"error": "Active model scope does not match the request."}), 409
        if (classifier.metadata.get("training", {}).get("data", {}).get("fingerprint")
                != current_fingerprint
                or not _model_label_map_matches(classifier.label_map, entries, scope)
                or not _evaluation_matches_label_map(
                    classifier.metadata.get("evaluation", {}), classifier.label_map
                )):
            return jsonify({
                "error": "The active model is stale or no longer validated. Refresh and activate a current validated model before sentence formation."
            }), 409
        vocabulary = [item["label"] for item in classifier.label_map]

    signs = data.get("signs")
    try:
        result = AI_SENTENCE_BUILDER.build_sentence(signs, vocabulary)
    except SentenceBuilderError as error:
        return jsonify({"error": str(error)}), 422
    except OllamaUnavailableError as error:
        return jsonify({"error": str(error)}), 503
    return jsonify(result)


@app.route("/developer/api/capture/lease", methods=["POST"])
def developer_capture_lease():
    global _developer_capture_lease_until
    forbidden = _developer_forbidden_response()
    if forbidden:
        return forbidden
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or type(data.get("active")) is not bool:
        return jsonify({"error": "active must be a boolean"}), 400
    with _lock:
        if data["active"] and _state["camera_active"]:
            return jsonify({
                "error": "Stop the regular Flask camera before developer capture."
            }), 409
        _developer_capture_lease_until = (
            time.monotonic() + 60.0 if data["active"] else 0.0
        )
    return jsonify({"active": data["active"]})


@app.route("/developer/api/drafts/<entry_id>/samples", methods=["POST"])
def save_developer_sample(entry_id):
    forbidden = _developer_forbidden_response()
    if forbidden:
        return forbidden
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "A JSON object is required."}), 400
    scope = data.get("scope")
    session_id = data.get("session_id")
    version = data.get("preprocessing_version")
    if scope not in {"shared", "personal"}:
        return jsonify({"error": "scope must be shared or personal"}), 400
    try:
        sequence, quality = validate_capture_sequence(
            data.get("sequence"), version
        )
        sample = VOCABULARY_STORE.save_sample(
            session_id=session_id,
            entry_id=entry_id,
            sequence=sequence,
            preprocessing_version=version,
            scope=scope,
        )
    except (TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 422
    return jsonify({
        "sample_id": sample.sample_id,
        "sample_count": VOCABULARY_STORE.count_samples(scope, entry_id),
        "quality": {
            "frames": quality.frames,
            "hand_frames": quality.hand_frames,
            "hand_frame_ratio": quality.hand_frame_ratio,
        },
    }), 201


@app.route("/developer/api/drafts/<entry_id>/samples", methods=["GET"])
def list_developer_samples(entry_id):
    forbidden = _developer_forbidden_response()
    if forbidden:
        return forbidden
    scope = request.args.get("scope", "")
    if scope not in {"shared", "personal"}:
        return jsonify({"error": "scope must be shared or personal"}), 400
    if not any(
        entry.entry_id == entry_id
        for entry in VOCABULARY_STORE.list_entries(scope)
    ):
        return jsonify({"error": "Draft does not exist in the requested scope."}), 404
    summaries = []
    for sample in VOCABULARY_STORE.list_samples(scope):
        if sample.entry_id != entry_id:
            continue
        sequence = VOCABULARY_STORE.load_sample(sample, scope)
        _, quality = validate_capture_sequence(
            sequence, sample.preprocessing_version
        )
        summaries.append({
            "sample_id": sample.sample_id,
            "session_id": sample.session_id,
            "created_at": sample.created_at,
            "sha256": sample.sha256,
            "frames": quality.frames,
            "hand_frames": quality.hand_frames,
            "preprocessing_version": sample.preprocessing_version,
        })
    return jsonify({"samples": summaries})


@app.route("/developer/api/drafts/<entry_id>/samples", methods=["DELETE"])
def delete_developer_samples(entry_id):
    forbidden = _developer_forbidden_response()
    if forbidden:
        return forbidden
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "A JSON object is required."}), 400
    scope = data.get("scope")
    sample_ids = data.get("sample_ids")
    if not isinstance(scope, str) or scope not in {"shared", "personal"}:
        return jsonify({"error": "scope must be shared or personal"}), 400
    if not isinstance(sample_ids, list) or not sample_ids:
        return jsonify({"error": "Select at least one saved example."}), 400
    if not any(
        entry.entry_id == entry_id
        for entry in VOCABULARY_STORE.list_entries(scope)
    ):
        return jsonify({"error": "Draft does not exist in the requested scope."}), 404
    try:
        deleted = VOCABULARY_STORE.delete_samples(entry_id, sample_ids, scope)
    except (TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 422
    except OSError as error:
        return jsonify({"error": str(error)}), 500
    return jsonify({
        "deleted_count": len(deleted),
        "sample_count": VOCABULARY_STORE.count_samples(scope, entry_id),
    })


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(str(FRONTEND_DIR), filename)


@app.route("/video_feed")
def video_feed():
    """MJPEG stream — point an <img> tag at this endpoint."""
    def generate():
        while True:
            frame = _current_frame if _current_frame is not None else _BLACK_FRAME
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.033)   # ~30 fps
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    """Current recognition state — polled by the frontend."""
    with _lock:
        s = _state
        return jsonify({
            "camera_active":      s["camera_active"],
            "hands":              s.get("hands_present", False),
            "current_word":       s["current_word"],
            "current_conf":       round(s["current_conf"] * 100),
            "top_vote_word":      s["top_vote_word"],
            "top_vote_count":     s["top_vote_count"],
            "sentence_words":     s["sentence_words"][:],
            "completed_gloss_words": s["completed_gloss_words"][:],
            "completed_sentence": s["completed_sentence"],
            "buf_pct":            s.get("buf_pct", 0),
            "vote_pct":           s.get("vote_pct", 0),
            "in_cooldown":        s.get("in_cooldown", False),
            "lang_code":          s["lang_code"],
            "lang_name":          LANGUAGES.get(s["lang_code"], "English"),
        })


@app.route("/control", methods=["POST"])
def control():
    global _camera_thread
    data   = request.get_json(silent=True) or {}
    action = data.get("action", "")

    with _lock:
        s   = _state
        now = time.time()

        if action == "start_camera":
            if _developer_capture_lease_until >= time.monotonic():
                return jsonify({
                    "error": "Stop developer capture before starting the Flask camera."
                }), 409
            if _model is None or not Path(HOLISTIC_MODEL).is_file():
                return jsonify({
                    "error": (
                        "Flask camera recognition needs the configured Keras and "
                        "MediaPipe model files. The browser KNN camera is separate."
                    )
                }), 503
            if not s["camera_active"]:
                s["camera_active"] = True
                t = threading.Thread(target=_camera_loop, daemon=True)
                t.start()
                _camera_thread = t

        elif action == "stop_camera":
            s["camera_active"] = False
            s["sequence"].clear()
            s["vote_window"].clear()
            s["current_word"]  = ""
            s["current_conf"]  = 0.0


        elif action == "add_word":
            word = data.get("word", "")

            if isinstance(word, str):
                word = word.strip()

                if word and len(word) <= 80:
                    s["sentence_words"].append(word)
                    s["completed_sentence"] = ""
                    s["completed_gloss_words"] = []
                    s["vote_window"].clear()
                    print(f"  + {word} (browser KNN)")



        elif action == "finalize":
            if s["sentence_words"]:
                _do_finalize(s, now)
            elif s["completed_sentence"]:
                speak(s["completed_sentence"], s["lang_code"])

        elif action == "clear":
            s["sentence_words"]     = []
            s["completed_gloss_words"] = []
            s["vote_window"].clear()
            s["completed_sentence"] = ""
            s["sequence"].clear()
            s["current_word"]       = ""
            s["current_conf"]       = 0.0

        elif action == "undo":
            if s["sentence_words"]:
                removed = s["sentence_words"].pop()
                print(f"  - {removed} (undone)")
                s["vote_window"].clear()

        elif action == "language":
            lang = data.get("lang", "en")
            if lang in LANGUAGES:
                s["lang_code"] = lang
                print(f"  Language: {LANGUAGES[lang]}")

        elif action == "speak":
            if s["completed_sentence"]:
                speak(s["completed_sentence"], s["lang_code"])

        return jsonify({
            "camera_active":      s["camera_active"],
            "sentence_words":     s["sentence_words"][:],
            "completed_gloss_words": s["completed_gloss_words"][:],
            "completed_sentence": s["completed_sentence"],
            "lang_code":          s["lang_code"],
            "lang_name":          LANGUAGES.get(s["lang_code"], "English"),
        })


if __name__ == "__main__":
    _load_resources()

    tts_thread = threading.Thread(target=_tts_worker, daemon=True)
    tts_thread.start()

    print("Starting Flask server at http://localhost:5000")
    print("Open http://localhost:5000 in your browser.\n")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
