// ═══════ ISL Translator — Starry Night Frontend ═══════

const API = window.location.origin;
let cameraActive = false;
let pollTimer    = null;
let cameraTransition = false;
let mainCameraStream = null;
let cameraRecognizerSource = null;
let experimentalStream = null;
let experimentalBusy = false;
let experimentalStatus = null;
let lastExperimentalPrediction = null;
let aiSignSequence = [];
let aiSentenceBusy = false;
let sequenceCollecting = false;
let sequenceStale = false;
let sequenceRunId = 0;
let sequenceContext = null;
let backendRecognizerInfo = null;
let vocabularyRefreshId = 0;
let vocabularyRefreshPromise = null;
let vocabularyRefreshPromiseScope = null;
let vocabularyRefreshPromiseIncludesCandidates = false;
let recognizerSelectionMade = false;
const RECOGNIZER_STATUS_TIMEOUT_MS = 20000;
let experimentalFramePumpActive = false;
let experimentalFrameBuffer = [];
let experimentalFrameTimes = [];
let experimentalFrameSerial = 0;
let cameraRunId = 0;
let lastSequenceWindowSerial = null;
const sequenceAccumulator = new window.ISLSequenceAccumulator.SequenceAccumulator();
const QUICK_PHRASE_CATEGORIES = [
  { name: 'Everyday', phrases: ['Hello!', 'Good morning!', 'Good afternoon!', 'Thank you!', "You're welcome.", 'Goodbye!'] },
  { name: 'Basic Needs', phrases: ['I need some water.', 'I am hungry.', 'I need to use the restroom.', 'I need a break.', 'I am feeling tired.'] },
  { name: 'Help and Assistance', phrases: ['Could you please help me?', 'I need some assistance.', "I don't feel well.", 'Please call a doctor.', 'This is an emergency.'] },
  { name: 'Communication', phrases: ["I don't understand.", 'Please speak slowly.', 'Could you repeat that?', 'Please give me a moment.', 'Can you show me?'] },
  { name: 'Questions', phrases: ['Where is the nearest restroom?', 'What time is it?', 'Could you show me the way?', 'Can I ask you something?'] }
];
let activeQuickPhraseCategory = QUICK_PHRASE_CATEGORIES[0].name;
let selectedQuickPhrase = '';
let quickPhraseSpeechRunId = 0;
let quickPhraseSpeaking = false;

document.addEventListener('DOMContentLoaded', () => {
  initNavbar();
  initSmoothScroll();
  initAiSentenceBuilder();
  initExperimentalStudio();
  initSentenceDemo();
  initQuickPhrases();
  document.querySelectorAll('details.advanced-details').forEach((section) => {
    section.open = false;
  });
  window.addEventListener('focus', refreshVocabularyOnReturn);
  document.addEventListener('visibilitychange', refreshVocabularyOnReturn);
});

function aiSentenceMessage(id, message, kind = '') {
  const element = document.getElementById(id);
  if (!element) return;
  element.textContent = message;
  element.dataset.kind = kind;
}

function setLiveStatus(id, message, state) {
  const element = document.getElementById(id);
  if (!element) return;
  element.textContent = message;
  if (state) element.dataset.state = state;
}

function updateFrameProgress() {
  let count = Math.min(experimentalFrameBuffer.length, 30);
  let total = 30;
  let hintText = cameraActive ? 'Camera frames buffered' : 'Sign to begin';
  if (sequenceCollecting && experimentalBusy) {
    count = 30;
    hintText = 'Model analyzing this 30-frame window';
  } else if (sequenceCollecting && lastSequenceWindowSerial !== null) {
    count = Math.min(10, Math.max(0, experimentalFrameSerial - lastSequenceWindowSerial));
    total = 10;
    hintText = 'Next window uses 20 recent frames';
  } else if (sequenceCollecting) {
    hintText = 'Collecting first 30-frame window';
  } else if (cameraActive) {
    hintText = 'Recognition is not active';
  }
  const progress = document.getElementById('bufferBar');
  const text = document.getElementById('frameProgressText');
  const hint = document.getElementById('frameProgressHint');
  if (progress) progress.style.width = `${(count / total) * 100}%`;
  if (text) text.textContent = `${count} of ${total} ${total === 30 ? 'frames' : 'new frames'}`;
  if (hint) hint.textContent = hintText;
}

function showPredictionFailure(error) {
  const message = String(error?.message || '');
  const handsMatch = message.match(/detected in (\d+)\/30/i);
  if (handsMatch) {
    const detected = Number(handsMatch[1]);
    setLiveStatus(
      'handStatus',
      detected === 0 ? 'Hands not detected' : 'Hands not visible clearly enough',
      'warning'
    );
    setLiveStatus(
      'recognitionStatus',
      detected === 0 ? 'Hands not detected' : 'Hands not visible clearly enough',
      'warning'
    );
    const explanation = detected === 0
      ? 'No hands were found in this frame window. Check the camera view and keep your hands in frame.'
      : 'Hands were visible in too few frames. Keep your hands clearly in view throughout the sign.';
    const meta = document.getElementById('demoSignMeta');
    if (meta) meta.textContent = explanation;
    const trigger = document.getElementById('demoTriggerInfo');
    if (trigger) trigger.textContent = 'No new sentence was produced. Continue when your hands are visible.';
    return;
  }

  setLiveStatus('recognitionStatus', 'Request failed · retrying', 'error');
  const meta = document.getElementById('demoSignMeta');
  if (meta) meta.textContent = 'Could not check this sign. The camera will try again with a fresh frame window.';
  const trigger = document.getElementById('demoTriggerInfo');
  if (trigger) trigger.textContent = 'No new sentence was produced because this frame window could not be checked.';
}

function recordRecognitionTiming({
  windowIndex,
  frameCollectionMs,
  clientRequestMs,
  windowToResultMs,
  result,
  error,
  windowsSinceAccepted,
  qualityRejections,
  requestFailures
}) {
  const server = result?.timings || error?.data?.timings || {};
  const metrics = {
    window: windowIndex,
    frame_collection_ms: frameCollectionMs,
    estimated_capture_fps: frameCollectionMs > 0
      ? (29 * 1000) / frameCollectionMs
      : null,
    client_request_ms: clientRequestMs,
    window_to_result_ms: windowToResultMs,
    windows_since_last_accepted: windowsSinceAccepted,
    hand_quality_rejection_rate_pct: windowIndex
      ? (qualityRejections / windowIndex) * 100
      : 0,
    request_failure_rate_pct: windowIndex
      ? (requestFailures / windowIndex) * 100
      : 0,
    sentence_release_windows_seen: sentenceDemoReleaseCount,
    sentence_release_windows_required: SENTENCE_DEMO_RELEASE_WINDOWS,
    sentence_cooldown_remaining_ms: Math.max(
      0,
      SENTENCE_DEMO_COOLDOWN_MS - (Date.now() - sentenceDemoLastTriggerTime)
    ),
    hand_frames: result?.sequence_quality?.hand_frames ?? server.hand_frames ?? null,
    server_frame_decode_ms: server.frame_decode_ms ?? null,
    server_classifier_load_ms: server.classifier_load_ms ?? null,
    server_model_validation_ms: server.model_validation_ms ?? null,
    server_holistic_initialization_ms: server.holistic_initialization_ms ?? null,
    server_landmark_extraction_ms: server.landmark_extraction_ms ?? null,
    server_quality_validation_ms: server.quality_validation_ms ?? null,
    server_model_inference_ms: server.model_inference_ms ?? null,
    server_request_total_ms: server.request_total_ms ?? null,
    outcome: error
      ? (server.hand_quality_rejected ? 'hand_quality_rejected' : 'request_failed')
      : (result.abstained ? 'abstained' : 'predicted')
  };
  console.info('[ISL recognition timing]', metrics);
}

function initAiSentenceBuilder() {
  const recognizer = document.getElementById('sentenceRecognizer');
  if (!recognizer) return;
  recognizer.addEventListener('change', () => {
    if (cameraActive && cameraRecognizerSource !== recognizer.value) {
      recognizer.value = cameraRecognizerSource;
      aiSentenceMessage('aiSentenceStatus', 'Stop the camera before switching recognizers.', 'warning');
      return;
    }
    if (sequenceCollecting && sequenceContext) {
      recognizer.value = sequenceContext.source;
      aiSentenceMessage('aiSentenceStatus', 'Pause sequence capture before switching recognizers.', 'warning');
      return;
    }
    if (aiSignSequence.length && aiSignSequence.some((sign) => sign.source !== recognizer.value)) {
      recognizer.value = aiSignSequence[0].source;
      aiSentenceMessage('aiSentenceStatus', 'Clear the sequence before switching recognizers.', 'warning');
      return;
    }
    recognizerSelectionMade = true;
    updateSequenceControls();
    syncCameraControls();
  });
  document.getElementById('startAiSequence').addEventListener('click', startAiSequence);
  document.getElementById('pauseAiSequence').addEventListener('click', pauseAiSequence);
  document.getElementById('undoAiSign').addEventListener('click', undoAiSign);
  document.getElementById('clearAiSigns').addEventListener('click', clearAiSignSequence);
  document.getElementById('finishAiSentence').addEventListener('click', finishAiSentence);
  document.getElementById('acceptAiSentence').addEventListener('click', acceptAiSentence);
  document.getElementById('speakAiSentence').addEventListener('click', speakAiSentence);
  updateSequenceControls();
}

async function refreshAiProviderStatus() {
  try {
    const status = await experimentalApi('/api/ai-sentence/status');
    aiSentenceMessage(
      'aiProviderStatus',
      status.message || (status.available ? 'Local sentence model is ready.' : 'Local sentence model is unavailable.'),
      status.available ? 'good' : 'warning'
    );
  } catch (error) {
    aiSentenceMessage('aiProviderStatus', `Local AI status unavailable: ${error.message}`, 'error');
  }
}

function appendAiSign(label, confidence, uncertain = false, expectedSource) {
  const source = document.getElementById('sentenceRecognizer')?.value;
  if (!source || source !== expectedSource) {
    aiSentenceMessage(
      'aiSentenceStatus',
      `Select the ${expectedSource === 'browser_knn' ? 'Browser KNN' : 'experimental'} recognizer before adding its prediction.`,
      'warning'
    );
    return false;
  }
  if (aiSignSequence.length >= 32) {
    aiSentenceMessage('aiSentenceStatus', 'Sentence sequence limit reached (32 signs). Finish or clear the sequence before continuing.', 'warning');
    return false;
  }
  aiSignSequence.push({
    label,
    confidence: Number.isFinite(confidence) ? confidence : null,
    uncertain: Boolean(uncertain),
    user_corrected_label: null,
    source,
    scope: source === 'experimental'
      ? document.getElementById('experimentalScope').value
      : null,
    artifact_id: source === 'experimental'
      ? sequenceContext?.artifactId || null
      : null
  });
  document.getElementById('aiSentenceResult').hidden = true;
  aiSentenceMessage('aiSentenceStatus', 'Review the collected signs, then finish the sentence when ready.');
  renderAiSignSequence();
  return true;
}

function renderAiSignSequence() {
  const list = document.getElementById('aiSignSequence');
  const empty = document.getElementById('aiSequenceEmpty');
  list.replaceChildren();
  empty.hidden = aiSignSequence.length > 0;
  aiSignSequence.forEach((sign, index) => {
    const row = document.createElement('li');
    row.className = 'ai-sign-item';
    const number = document.createElement('span');
    number.className = 'ai-sign-index';
    number.textContent = String(index + 1);
    const label = document.createElement('input');
    label.className = `ai-sign-label${sign.user_corrected_label ? ' edited' : ''}`;
    label.type = 'text';
    label.maxLength = 80;
    label.value = sign.user_corrected_label ?? sign.label ?? '';
    label.placeholder = sign.uncertain ? 'Abstained / uncertain — enter a correction' : 'Sign label';
    label.setAttribute('aria-label', `Sign ${index + 1} label`);
    label.addEventListener('input', () => {
      const current = label.value.trim();
      sign.user_corrected_label = current &&
        current.toLocaleLowerCase() !== (sign.label || '').toLocaleLowerCase()
        ? current
        : null;
      label.classList.toggle('edited', Boolean(sign.user_corrected_label));
      document.getElementById('aiSentenceResult').hidden = true;
    });
    const confidence = document.createElement('span');
    confidence.className = 'ai-sign-confidence';
    confidence.textContent = sign.confidence === null
      ? 'No confidence score'
      : `Raw score ${(sign.confidence * 100).toFixed(1)}% · uncalibrated`;
    const remove = document.createElement('button');
    remove.className = 'ai-remove-sign';
    remove.type = 'button';
    remove.textContent = 'Remove';
    remove.setAttribute('aria-label', `Remove sign ${index + 1}`);
    remove.addEventListener('click', () => {
      aiSignSequence.splice(index, 1);
      document.getElementById('aiSentenceResult').hidden = true;
      renderAiSignSequence();
    });
    row.append(number, label, confidence, remove);
    list.append(row);
  });
  document.getElementById('finishAiSentence').disabled =
    aiSignSequence.length === 0 || aiSentenceBusy || sequenceStale;
  document.getElementById('undoAiSign').disabled = aiSignSequence.length === 0;
  updateSequenceControls();
}

function clearAiSignSequence() {
  pauseAiSequence(false);
  aiSignSequence = [];
  sequenceStale = false;
  sequenceContext = null;
  sequenceAccumulator.reset();
  document.getElementById('aiSentenceResult').hidden = true;
  aiSentenceMessage('aiSentenceStatus', 'Sequence cleared. Start capture when you are ready.');
  renderAiSignSequence();
}

function undoAiSign() {
  if (!aiSignSequence.length) return;
  aiSignSequence.pop();
  document.getElementById('aiSentenceResult').hidden = true;
  renderAiSignSequence();
  aiSentenceMessage('aiSentenceStatus', 'Removed the most recent sign. Review the sequence and continue when ready.');
}

function updateSequenceControls() {
  const start = document.getElementById('startAiSequence');
  const pause = document.getElementById('pauseAiSequence');
  if (!start || !pause) return;
  start.disabled = sequenceCollecting || sequenceStale || experimentalBusy;
  pause.disabled = !sequenceCollecting;
  start.textContent = sequenceCollecting ? 'Capturing sequence…' : 'Start sequence';
  document.getElementById('finishAiSentence').disabled =
    aiSignSequence.length === 0 || aiSentenceBusy || sequenceStale;
  document.getElementById('undoAiSign').disabled = aiSignSequence.length === 0;
}

function sequenceSignature(data, source, scope, artifactId) {
  if (source === 'browser_knn') {
    return JSON.stringify({
      source,
      labels: data.recognizers?.browser_knn?.vocabulary || []
    });
  }
  return JSON.stringify({
    source,
    scope,
    storage_version: data.storage_version,
    vocabulary_version: data.vocabulary_version,
    artifact_id: artifactId,
    active_artifact_id: data.active_model?.artifact_id || null,
    active_usable: Boolean(data.active_model?.usable)
  });
}

function markSequenceStale(message) {
  if (!sequenceContext || sequenceStale) return;
  sequenceStale = true;
  pauseAiSequence(false);
  sequenceContext = null;
  sequenceAccumulator.reset();
  aiSentenceMessage(
    'aiSentenceStatus',
    `${message} The sequence is preserved; clear it before starting with the updated model.`,
    'warning'
  );
  setLiveStatus('recognitionStatus', 'Model changed · refresh required', 'warning');
  updateSequenceControls();
  renderAiSignSequence();
}

async function startAiSequence() {
  if (sequenceCollecting || sequenceStale) return;
  const requestedScope = document.getElementById('experimentalScope').value;
  if (!experimentalStatus || experimentalStatus.scope !== requestedScope) {
    aiSentenceMessage('aiSentenceStatus', 'Checking the active sign model…');
    await loadExperimentalVocabulary({ quiet: true });
  }
  if (!experimentalStatus) {
    aiSentenceMessage('aiSentenceStatus', 'The active recognizer could not be verified. Check Flask and refresh the recognizer status before signing.', 'error');
    setLiveStatus('recognitionStatus', 'Unable to recognize', 'error');
    return;
  }
  if (experimentalStatus.scope !== requestedScope) {
    aiSentenceMessage('aiSentenceStatus', 'The selected vocabulary scope changed while refreshing. Try again.', 'warning');
    return;
  }
  const source = document.getElementById('sentenceRecognizer').value;

  if (source === 'browser_knn') {
    const labels = backendRecognizerInfo?.browser_knn?.vocabulary || [];
    const localLabels = window.knnRecognizer?.words?.slice(0, -1) || [];
    if (!window.knnRecognizer || JSON.stringify(labels) !== JSON.stringify(localLabels)) {
      aiSentenceMessage('aiSentenceStatus', 'The browser KNN labels do not match the backend. Refresh the page before collecting a sequence.', 'error');
      setLiveStatus('recognitionStatus', 'Fallback unavailable', 'error');
      const meta = document.getElementById('demoSignMeta');
      if (meta) meta.textContent = 'The Hello fallback is unavailable. Refresh the page or use a compatible active sequence model.';
      return;
    }
    if (!cameraActive) {
      aiSentenceMessage('aiSentenceStatus', 'Start the Browser KNN camera first, then start sequence capture.', 'warning');
      return;
    }
    if (!window.knnRecognizer.isRecognizing()) {
      window.knnRecognizer.startPrediction();
      if (!window.knnRecognizer.isRecognizing()) {
        aiSentenceMessage('aiSentenceStatus', 'The legacy Hello fallback could not start. Its browser-local saved examples may be missing or incompatible; use an active validated sequence model or prepare one in Developer Studio.', 'error');
        setLiveStatus('recognitionStatus', 'Fallback unavailable', 'error');
        const meta = document.getElementById('demoSignMeta');
        if (meta) meta.textContent = 'The Hello fallback could not start. Use a compatible active sequence model or check the saved browser examples.';
        return;
      }
    }
    sequenceContext = {
      source,
      signature: sequenceSignature(experimentalStatus, source)
    };
    sequenceCollecting = true;
    aiSentenceMessage('aiSentenceStatus', 'Listening for stable Browser KNN predictions. Pause between repeated signs so the recognizer can release its previous sign.');
    updateSequenceControls();
    return;
  }

  const scope = document.getElementById('experimentalScope').value;
  const activeModel = experimentalStatus.active_model;
  if (experimentalBusy) {
    aiSentenceMessage('aiSentenceStatus', 'Wait for the current experimental capture or prediction to finish before starting sequence capture.', 'warning');
    return;
  }
  if (!activeModel?.usable) {
    aiSentenceMessage('aiSentenceStatus', `No compatible active sequence model is available. ${activeModel?.unusable_reason || 'Prepare, evaluate, and activate a compatible model in Developer Studio, or choose the legacy Hello fallback if its browser-local examples are available.'}`, 'error');
    setLiveStatus('recognitionStatus', 'Model unavailable', 'error');
    const meta = document.getElementById('demoSignMeta');
    if (meta) meta.textContent = 'No compatible active model is available. Prepare and validate one in Developer Studio.';
    return;
  }
  if (!cameraActive || !experimentalStream || !experimentalFramePumpActive) {
    aiSentenceMessage('aiSentenceStatus', 'Start the shared camera first, then start sequence capture.', 'warning');
    return;
  }

  sequenceContext = {
    source,
    scope,
    artifactId: activeModel.artifact_id,
    signature: sequenceSignature(experimentalStatus, source, scope, activeModel.artifact_id)
  };
  sequenceCollecting = true;
  sequenceAccumulator.reset();
  updateFrameProgress();
  aiSentenceMessage('aiSentenceStatus', `Capturing consecutive 30-frame windows with active model ${activeModel.artifact_id}. Confidence scores are not accuracy estimates.`);
  updateSequenceControls();
  const runId = ++sequenceRunId;
  runExperimentalSequence(runId).catch((error) => {
    if (runId === sequenceRunId && sequenceCollecting) {
      if (error.status === 409) {
        markSequenceStale(`The backend rejected a stale candidate: ${error.message}`);
        loadExperimentalVocabulary({ quiet: true });
      } else {
        pauseAiSequence(false);
        aiSentenceMessage('aiSentenceStatus', `Continuous capture paused: ${error.message}`, 'error');
        setLiveStatus('recognitionStatus', 'Unable to recognize', 'error');
      }
    }
  });
}

function pauseAiSequence(showMessage = true) {
  sequenceCollecting = false;
  sequenceRunId += 1;
  sequenceAccumulator.reset();
  if (showMessage) {
    aiSentenceMessage('aiSentenceStatus', 'Sequence capture paused. Your reviewed signs are preserved.');
  }
  updateSequenceControls();
}

function collectStablePrediction(prediction, confirmations, source) {
  if (!sequenceCollecting || sequenceStale ||
      sequenceContext?.source !== source) return false;
  const accepted = sequenceAccumulator.observe(prediction, confirmations);
  if (!accepted) return false;
  const added = appendAiSign(
    accepted.label,
    accepted.confidence,
    Boolean(accepted.uncertain),
    source
  );
  if (added) addWordToSentence(accepted.label);
  return added;
}

function addWordToSentence(label) {
  experimentalApi('/control', {
    method: 'POST',
    body: JSON.stringify({ action: 'add_word', word: label })
  }).catch((error) => {
    experimentalMessage('experimentalPrediction', `The sign is in the reviewed AI sequence, but could not be copied to the existing sentence builder: ${error.message}`, 'warning');
  });
}

async function finishAiSentence() {
  if (!aiSignSequence.length || aiSentenceBusy || sequenceStale) return;
  pauseAiSequence(false);
  aiSentenceBusy = true;
  renderAiSignSequence();
  aiSentenceMessage('aiSentenceStatus', 'Forming a sentence with the local model…');
  try {
    const recognizer = document.getElementById('sentenceRecognizer').value;
    const scope = document.getElementById('experimentalScope').value;
    const artifactId = experimentalStatus?.active_model?.artifact_id;
    if (recognizer === 'experimental' && aiSignSequence.some(
      (sign) => sign.scope !== scope || sign.artifact_id !== artifactId
    )) {
      throw new Error('The vocabulary scope or active model changed. Clear the sequence and collect signs again.');
    }
    const requestBody = {
      recognizer,
      signs: aiSignSequence.map(({ label, confidence, uncertain, user_corrected_label }) => ({
        label,
        confidence,
        uncertain,
        user_corrected_label
      }))
    };
    if (recognizer === 'experimental') {
      requestBody.scope = scope;
      requestBody.artifact_id = artifactId;
    }
    const result = await experimentalApi('/api/ai-sentence', {
      method: 'POST',
      body: JSON.stringify(requestBody)
    });
    showAiSentenceResult(result);
    aiSentenceMessage('aiSentenceStatus', 'Sentence suggestion ready. Review all proposed corrections before accepting.', 'good');
  } catch (error) {
    aiSentenceMessage('aiSentenceStatus', `Could not form a sentence: ${error.message}`, 'error');
    if (error.message.includes('Ollama is unavailable') ||
        error.message.includes('not installed') ||
        error.message.includes('Cannot reach local Ollama')) {
      aiSentenceMessage('aiProviderStatus', error.message, 'warning');
    }
  } finally {
    aiSentenceBusy = false;
    renderAiSignSequence();
  }
}

function showAiSentenceResult(result) {
  const container = document.getElementById('aiSentenceResult');
  document.getElementById('aiOriginalSequence').textContent =
    result.original_sequence.map((sign, index) => {
      const raw = sign.label || 'abstained / unknown';
      return sign.user_corrected_label
        ? `${index + 1}. ${raw} (your correction: ${sign.user_corrected_label})`
        : `${index + 1}. ${raw}`;
    }).join(' · ');
  const corrections = document.getElementById('aiCorrections');
  corrections.replaceChildren();
  (result.corrections || []).forEach((correction) => {
    const item = document.createElement('li');
    item.textContent = `Sign ${correction.index + 1}: ${correction.original_label} → ${correction.suggested_label} (AI suggestion, not verified): ${correction.reason}`;
    corrections.append(item);
  });
  if (!corrections.children.length) {
    const item = document.createElement('li');
    item.textContent = 'No recognition corrections proposed.';
    corrections.append(item);
  }
  const uncertainty = document.getElementById('aiUncertainty');
  uncertainty.hidden = !result.ambiguous;
  uncertainty.textContent = result.ambiguous
    ? `Uncertain interpretation: ${result.uncertainty_note}`
    : '';
  document.getElementById('aiSentenceText').value = result.sentence;
  document.getElementById('aiGrammarNotes').textContent =
    (result.grammar_changes || []).length
      ? `Grammar changes (separate from sign corrections): ${result.grammar_changes.join('; ')}`
      : 'No separate grammar changes reported.';
  container.hidden = false;
}

function acceptAiSentence() {
  const sentence = document.getElementById('aiSentenceText').value.trim();
  if (!sentence) {
    aiSentenceMessage('aiAcceptedStatus', 'Enter a sentence before accepting it.', 'error');
    return;
  }
  showFinal(sentence);
  setOriginalGloss(aiSignSequence
    .map((sign) => sign.user_corrected_label || sign.label)
    .filter(Boolean));
  aiSentenceMessage('aiAcceptedStatus', 'Sentence copied to the existing output area. The recognized sequence was not changed.', 'good');
}

function speakAiSentence() {
  const sentence = document.getElementById('aiSentenceText').value.trim();
  if (!sentence) return;
  if (!('speechSynthesis' in window)) {
    aiSentenceMessage('aiAcceptedStatus', 'Browser speech output is unavailable on this device.', 'error');
    return;
  }
  const utterance = new SpeechSynthesisUtterance(sentence);
  const languages = {
    en: 'en-IN', hi: 'hi-IN', bn: 'bn-IN', ta: 'ta-IN',
    te: 'te-IN', mr: 'mr-IN', gu: 'gu-IN'
  };
  utterance.lang = languages[document.getElementById('langSelect').value] || 'en-IN';
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utterance);
}

function initExperimentalStudio() {
  const scope = document.getElementById('experimentalScope');
  if (!scope) return;
  const advancedTools = document.getElementById('advancedRecognizerTools');
  scope.addEventListener('change', () => loadExperimentalVocabulary({
    includeCandidates: false
  }));
  if (advancedTools) {
    advancedTools.addEventListener('toggle', () => {
      if (advancedTools.open) {
        loadExperimentalVocabulary({ includeCandidates: false });
        refreshAiProviderStatus();
      }
    });
  }
  loadExperimentalVocabulary({ includeCandidates: false });
}

function experimentalMessage(id, message, kind = '') {
  const element = document.getElementById(id);
  element.textContent = message;
  element.dataset.kind = kind;
}

async function experimentalApi(path, options = {}) {
  const response = await fetch(`${API}${path}`, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) }
  });
  const result = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(result.error || `Request failed (HTTP ${response.status})`);
    error.status = response.status;
    error.data = result;
    throw error;
  }
  return result;
}

async function loadExperimentalVocabulary({
  quiet = false,
  includeCandidates = false
} = {}) {
  const scope = document.getElementById('experimentalScope').value;
  if (vocabularyRefreshPromise) {
    const pending = vocabularyRefreshPromise;
    if (vocabularyRefreshPromiseScope === scope &&
        (vocabularyRefreshPromiseIncludesCandidates || !includeCandidates)) {
      return pending;
    }
    await pending;
    if (document.getElementById('experimentalScope').value !== scope) return;
    return loadExperimentalVocabulary({ quiet, includeCandidates });
  }
  const refreshId = ++vocabularyRefreshId;
  const previousStatus = experimentalStatus;
  if (!quiet) experimentalMessage('recognizerStateStatus', 'Refreshing vocabulary and model state from Flask…');
  const requestStartedAt = performance.now();
  const controller = new AbortController();
  const timeoutId = setTimeout(
    () => controller.abort(),
    RECOGNIZER_STATUS_TIMEOUT_MS
  );
  const request = (async () => {
    try {
      const data = await experimentalApi(
        `/api/vocabulary?scope=${encodeURIComponent(scope)}&include_candidates=${includeCandidates}`,
        { signal: controller.signal }
      );
      if (refreshId !== vocabularyRefreshId) return;
      console.info('[ISL recognizer status timing]', {
        client_request_ms: performance.now() - requestStartedAt,
        server_status_build_ms: data.timings?.status_build_ms ?? null
      });
      experimentalStatus = data;
      backendRecognizerInfo = data.recognizers || null;
      updatePublicModelStatus(data);
      renderRecognizerOptions(data);
      const active = data.active_model;
      experimentalMessage(
        'experimentalPrediction',
        active?.usable
          ? `Active validated model ${active.artifact_id} · ${active.supports.length} supported sign(s). Saved recordings and vocabulary match this model.`
          : `No usable active sequence model for ${scope}. ${active?.unusable_reason || 'Train and validate a candidate in Developer Studio before activation.'}`,
        active?.usable ? 'good' : 'warning'
      );
      if (sequenceCollecting && sequenceContext) {
        if (sequenceContext.source === 'experimental' &&
            data.scope !== sequenceContext.scope) {
          markSequenceStale('The selected vocabulary scope changed on the backend.');
        } else {
          const currentSignature = sequenceContext.source === 'browser_knn'
            ? sequenceSignature(data, 'browser_knn')
            : sequenceSignature(
              data,
              'experimental',
              sequenceContext.scope,
              sequenceContext.artifactId
            );
          if (currentSignature !== sequenceContext.signature) {
            markSequenceStale('Saved vocabulary or recognizer/model state changed on the backend.');
          }
        }
      } else if (sequenceContext?.source === 'experimental' &&
          sequenceContext.scope === scope) {
        const currentSignature = sequenceSignature(
          data,
          'experimental',
          sequenceContext.scope,
          sequenceContext.artifactId
        );
        if (currentSignature !== sequenceContext.signature) {
          markSequenceStale('Saved vocabulary or recognizer/model state changed on the backend.');
        }
      }
    } catch (error) {
      if (refreshId !== vocabularyRefreshId) return;
      experimentalStatus = null;
      backendRecognizerInfo = null;
      experimentalMessage('recognizerStateStatus', 'Live model status is unavailable. Check that the local service is running, then refresh.', 'error');
      const publicStatus = document.getElementById('projectModelStatus');
      if (publicStatus) {
        publicStatus.textContent = error.name === 'AbortError'
          ? 'Model status timed out. Check the local service and refresh.'
          : 'Model status could not be loaded. Check the local service and refresh.';
        publicStatus.dataset.kind = 'error';
      }
      const publicVocabulary = document.getElementById('projectVocabularyStatus');
      if (publicVocabulary) {
        publicVocabulary.textContent = 'Saved label and readiness counts are unavailable until the local service responds.';
        publicVocabulary.dataset.kind = 'error';
      }
      document.getElementById('sentenceRecognizer').replaceChildren(
        new Option('Recognizer status unavailable', 'browser_knn')
      );
      if (sequenceCollecting || sequenceContext || previousStatus) {
        markSequenceStale('The backend state could not be verified.');
      }
    }
  })();
  vocabularyRefreshPromise = request;
  vocabularyRefreshPromiseScope = scope;
  vocabularyRefreshPromiseIncludesCandidates = includeCandidates;
  try {
    await request;
  } finally {
    clearTimeout(timeoutId);
    if (vocabularyRefreshPromise === request) {
      vocabularyRefreshPromise = null;
      vocabularyRefreshPromiseScope = null;
      vocabularyRefreshPromiseIncludesCandidates = false;
    }
  }
}

function updatePublicModelStatus(data) {
  const modelStatus = document.getElementById('projectModelStatus');
  const vocabularyStatus = document.getElementById('projectVocabularyStatus');
  if (!modelStatus || !vocabularyStatus) return;

  const words = data.words || [];
  const active = data.active_model;
  modelStatus.textContent = active?.usable
    ? `Validated model active · ${active.supports.length} of ${words.length} saved labels supported`
    : `No usable active sequence model · ${active?.unusable_reason || 'The Browser KNN fallback remains available for Hello.'}`;
  modelStatus.dataset.kind = active?.usable ? 'good' : 'warning';

  const trainingReady = words.filter((word) => word.training_ready).length;
  const evaluationReady = words.filter((word) => word.evaluation_ready).length;
  const evaluationStatus = active?.evaluation_ready
    ? 'active model has session-separated evaluation'
    : 'active model has no session-separated evaluation';
  vocabularyStatus.textContent = `${words.length} saved labels · ${trainingReady} training ready · ${evaluationReady} evaluation ready · ${evaluationStatus} · ${data.scope} scope`;
  vocabularyStatus.dataset.kind = '';
}

function renderRecognizerOptions(data) {
  const select = document.getElementById('sentenceRecognizer');
  const previous = select.value;
  const knn = data.recognizers?.browser_knn;
  const experimental = data.recognizers?.experimental;
  select.replaceChildren();
  const knnLabels = knn?.vocabulary || [];
  const localLabels = window.knnRecognizer?.words?.slice(0, -1) || [];
  const knnMatches = JSON.stringify(knnLabels) === JSON.stringify(localLabels);
  const knnOption = new Option(
    knnMatches
      ? 'Hello fallback · Browser KNN'
      : 'Browser KNN · label mismatch',
    'browser_knn'
  );
  knnOption.disabled = !knnMatches;
  select.append(knnOption);

  const experimentalOption = new Option(
    experimental?.available
      ? `Active sequence model · ${experimental.supported_label_count} signs`
      : 'Active sequence model · unavailable',
    'experimental'
  );
  experimentalOption.disabled = !experimental?.available;
  select.append(experimentalOption);
  if (recognizerSelectionMade &&
      [...select.options].some((option) => option.value === previous && !option.disabled)) {
    select.value = previous;
  } else {
    select.value = experimental?.available ? 'experimental' : 'browser_knn';
  }

  const active = experimental?.active_model;
  const modelText = active?.usable
    ? `Validated model ready · ${active.supports.length} of ${data.words.length} saved labels supported.`
    : `No validated active model. ${active?.unusable_reason || 'Candidate testing and activation are available in Developer Studio.'}`;
  aiSentenceMessage(
    'recognizerStateStatus',
    `Browser KNN recognizes Hello only. ${modelText} New examples require explicit training, evaluation, and activation.`
  );
  syncCameraControls();
}

function refreshVocabularyOnReturn() {
  if (document.visibilityState === 'hidden') return;
  loadExperimentalVocabulary({ quiet: true, includeCandidates: false });
  loadSentenceDemoConfig();
}

function renderExperimentalVocabulary(data) {
  const container = document.getElementById('dynamicVocabulary');
  container.replaceChildren();
  const words = data.words || [];
  if (!words.length) {
    const empty = document.createElement('p');
    empty.className = 'experimental-status';
    empty.textContent = 'No vocabulary words are saved in this scope yet. Create one in Developer Studio.';
    container.append(empty);
  }
  for (const word of words) {
    const card = document.createElement('article');
    card.className = 'dynamic-word';
    const name = document.createElement('strong');
    name.textContent = `${word.label} · ID ${word.label_id}`;
    const samples = document.createElement('span');
    samples.textContent = `${word.sample_count} saved example(s) · ${word.session_count} independent recorded session(s)`;
    const training = document.createElement('span');
    training.textContent = word.training_ready
      ? 'Training data ready'
      : `Needs ${Math.max(0, word.minimum_training_sessions - word.session_count)} more session(s) with saved examples for training`;
    const evaluation = document.createElement('span');
    evaluation.textContent = word.evaluation_ready
      ? 'Session-separated evaluation data ready'
      : `Needs ${Math.max(0, word.minimum_evaluation_sessions - word.session_count)} more session(s) with saved examples for evaluation`;
    const recognition = document.createElement('span');
    recognition.textContent = word.recognizable_by_active_model
      ? 'Included in the active experimental model'
      : 'Not included in the active experimental model';
    card.append(name, samples, training, evaluation, recognition);
    container.append(card);
  }
  const allTrainReady = words.length >= 2 && words.every((word) => word.training_ready);
  document.getElementById('trainCandidate').disabled = !allTrainReady;
  experimentalMessage(
    'experimentalStatus',
    `${words.length} word(s) in ${data.scope} scope · vocabulary ${data.vocabulary_version.slice(0, 12)} · saved-data ${String(data.storage_version || '').slice(0, 12)} · preprocessing ${data.preprocessing_version}`
  );
}

function renderExperimentalModels(data) {
  const active = data.active_model;
  if (active && active.available) {
    const qualityNote = active.compatible
      ? (active.evaluation_ready ? 'Session-separated evaluation exists; results remain experimental.' : 'No session-separated evaluation is available yet.')
      : 'Vocabulary labels changed since this model was created; live testing is blocked.';
    const dataNote = active.data_current
      ? 'Model data matches current saved sequences.'
      : 'Saved sequences changed since training; this model is an older snapshot.';
    experimentalMessage(
      'experimentalModelStatus',
      `Active experimental model ${active.artifact_id} · ${active.supports.length} mapped word(s) · model vocabulary ${String(active.vocabulary_version || '').slice(0, 12)} · ${dataNote} ${qualityNote}`,
      active.compatible ? '' : 'error'
    );
  } else {
    experimentalMessage(
      'experimentalModelStatus',
      `No active experimental model for ${data.scope}. The browser KNN fallback is unchanged.`
    );
  }
  const select = document.getElementById('candidateSelect');
  const previous = select.value;
  select.replaceChildren();
  const candidates = data.candidates || [];
  if (!candidates.length) select.append(new Option('No candidates yet', ''));
  for (const candidate of candidates) {
    const score = candidate.validation && Number.isFinite(candidate.validation.accuracy)
      ? `${(candidate.validation.accuracy * 100).toFixed(1)}% validation`
      : 'UNVALIDATED · no validation score';
    const marks = [candidate.active ? 'active' : '', candidate.evaluation_ready ? 'evaluation available' : 'UNVALIDATED', score, candidate.data_current ? 'data current' : 'data changed', candidate.vocabulary_matches ? 'labels current' : 'labels changed']
      .filter(Boolean).join(', ');
    select.append(new Option(`${candidate.artifact_id} (${marks})`, candidate.artifact_id));
  }
  select.disabled = candidates.length === 0;
  if (candidates.some((candidate) => candidate.artifact_id === previous)) select.value = previous;
  select.dataset.candidates = JSON.stringify(candidates);
  updateCandidateControls();
}

function updateCandidateControls() {
  const select = document.getElementById('candidateSelect');
  const candidates = JSON.parse(select.dataset.candidates || '[]');
  const candidate = candidates.find((item) => item.artifact_id === select.value);
  const words = experimentalStatus?.words || [];
  document.getElementById('trainCandidate').disabled =
    words.length < 2 || !words.every((word) => word.training_ready) ||
    sequenceCollecting || experimentalBusy;
  document.getElementById('activateCandidate').disabled =
    !candidate || candidate.active || !candidate.evaluation_ready ||
    !candidate.data_current || !candidate.vocabulary_matches || Boolean(candidate.invalid);
  document.getElementById('testExperimentalSign').disabled =
    sequenceCollecting || !experimentalStream || !candidate || !candidate.data_current ||
    !candidate.vocabulary_matches || Boolean(candidate.invalid);
  document.getElementById('addExperimentalPrediction').disabled =
    sequenceCollecting || !lastExperimentalPrediction || lastExperimentalPrediction.abstained;
  document.getElementById('addExperimentalUncertain').disabled =
    sequenceCollecting || !lastExperimentalPrediction || !lastExperimentalPrediction.abstained;
}

async function trainExperimentalCandidate() {
  const button = document.getElementById('trainCandidate');
  button.disabled = true;
  experimentalMessage('experimentalModelStatus', 'Training a separate candidate from saved samples. The active model is unchanged…');
  try {
    const result = await experimentalApi('/api/experimental/train', {
      method: 'POST',
      body: JSON.stringify({ scope: document.getElementById('experimentalScope').value })
    });
    const candidate = result.candidate;
    const validation = candidate.validation;
    const metric = validation
      ? `Session-separated validation: ${(validation.accuracy * 100).toFixed(1)}% accuracy over ${validation.support} sample(s).`
      : 'No session-separated validation; this candidate cannot be activated yet.';
    experimentalMessage('experimentalModelStatus', `Saved candidate ${candidate.artifact_id}. ${metric} This is experimental and not a guarantee of accuracy.`);
    await loadExperimentalVocabulary();
  } catch (error) {
    experimentalMessage('experimentalModelStatus', `Candidate training did not complete: ${error.message}`, 'error');
    await loadExperimentalVocabulary();
  }
}

async function activateExperimentalCandidate() {
  const artifactId = document.getElementById('candidateSelect').value;
  if (!artifactId) return;
  const candidates = JSON.parse(document.getElementById('candidateSelect').dataset.candidates || '[]');
  const candidate = candidates.find((item) => item.artifact_id === artifactId);
  const validation = candidate?.validation;
  const score = validation && Number.isFinite(validation.accuracy)
    ? `${(validation.accuracy * 100).toFixed(1)}% accuracy and ${(validation.macro_f1 * 100).toFixed(1)}% macro F1`
    : 'no available validation scores';
  if (!window.confirm(`Activate candidate ${artifactId} with session-separated validation showing ${score}? Results remain experimental. The browser KNN stays unchanged, and older experimental artifacts remain available.`)) return;
  try {
    const result = await experimentalApi(`/api/experimental/activate/${encodeURIComponent(artifactId)}`, {
      method: 'POST',
      body: JSON.stringify({ scope: document.getElementById('experimentalScope').value })
    });
    experimentalMessage('experimentalModelStatus', `Activated ${result.artifact_id}. The KNN fallback remains unchanged.` , 'good');
    lastExperimentalPrediction = null;
    document.getElementById('addExperimentalPrediction').disabled = true;
    await loadExperimentalVocabulary();
  } catch (error) {
    experimentalMessage('experimentalModelStatus', `Candidate was not activated: ${error.message}`, 'error');
  }
}

async function startExperimentalCamera() {
  if (cameraActive || cameraTransition) {
    experimentalMessage('experimentalPrediction', 'Stop the browser KNN camera before enabling the separate experimental camera.', 'error');
    return;
  }
  cameraTransition = true;
  setLiveStatus('cameraStatus', 'Starting camera', 'working');
  try {
    const status = await experimentalApi('/status');
    if (status.camera_active) throw new Error('Stop the Flask camera before enabling experimental capture.');
    experimentalStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } }
    });
    const stream = experimentalStream;
    stream.getTracks().forEach((track) => {
      track.addEventListener('ended', () => {
        if (experimentalStream !== stream) return;
        stopExperimentalCamera();
        experimentalMessage('experimentalPrediction', 'The camera stream ended. Enable the camera again to continue.', 'warning');
      }, { once: true });
    });
    const video = document.getElementById('experimentalVideo');
    video.srcObject = experimentalStream;
    await video.play();
    startExperimentalFramePump();
    setLiveStatus('cameraStatus', 'Camera ready', 'ready');
    setLiveStatus('handStatus', 'Waiting for hands', 'idle');
    setLiveStatus('recognitionStatus', 'Collecting frames', 'working');
    document.getElementById('startExperimentalCamera').disabled = true;
    document.getElementById('stopExperimentalCamera').disabled = false;
    updateCandidateControls();
    experimentalMessage('experimentalPrediction', 'Camera ready. Select a current candidate, hold one sign clearly, then capture a 30-frame test. Candidates without four sessions per word are unvalidated.');
  } catch (error) {
    stopExperimentalCamera();
    setLiveStatus('cameraStatus', 'Camera error', 'error');
    setLiveStatus('recognitionStatus', 'Unable to start', 'error');
    experimentalMessage('experimentalPrediction', `Could not start experimental camera: ${error.message}`, 'error');
  } finally {
    cameraTransition = false;
  }
}

function stopExperimentalCamera() {
  cameraRunId += 1;
  if (sequenceContext?.source === 'experimental' && sequenceCollecting) {
    pauseAiSequence(false);
    aiSentenceMessage('aiSentenceStatus', 'Camera stopped; sequence capture paused. Reviewed signs are preserved.');
  }
  experimentalStream = null;
  experimentalFramePumpActive = false;
  experimentalFrameBuffer = [];
  experimentalFrameTimes = [];
  experimentalFrameSerial = 0;
  lastSequenceWindowSerial = null;
  lastExperimentalPrediction = null;
  updateFrameProgress();
}

function waitExperimentalFrame(video) {
  return new Promise((resolve) => {
    if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(() => resolve());
    else requestAnimationFrame(() => resolve());
  });
}

function startExperimentalFramePump() {
  if (experimentalFramePumpActive) return;
  experimentalFramePumpActive = true;
  experimentalFrameBuffer = [];
  experimentalFrameTimes = [];
  experimentalFrameSerial = 0;
  lastSequenceWindowSerial = null;
  const video = document.getElementById('knnVideo');
  const canvas = document.createElement('canvas');
  canvas.width = 320;
  canvas.height = 240;
  const context = canvas.getContext('2d');
  const pump = async () => {
    try {
      while (experimentalFramePumpActive && experimentalStream) {
        await waitExperimentalFrame(video);
        if (!experimentalFramePumpActive || !experimentalStream) break;
        context.save();
        context.translate(canvas.width, 0);
        context.scale(-1, 1);
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        context.restore();
        experimentalFrameBuffer.push(canvas.toDataURL('image/jpeg', 0.72));
        experimentalFrameTimes.push(performance.now());
        if (experimentalFrameBuffer.length > 30) {
          experimentalFrameBuffer.shift();
          experimentalFrameTimes.shift();
        }
        experimentalFrameSerial += 1;
        updateFrameProgress();
      }
    } catch (error) {
      if (experimentalFramePumpActive) {
        experimentalMessage('experimentalPrediction', `Camera frame capture stopped: ${error.message}`, 'error');
        if (sequenceCollecting && sequenceContext?.source === 'experimental') {
          pauseAiSequence(false);
          aiSentenceMessage('aiSentenceStatus', `Sequence capture paused because camera frames stopped: ${error.message}`, 'error');
        }
      }
    } finally {
      experimentalFramePumpActive = false;
    }
  };
  pump();
}

async function captureExperimentalWindow(onProgress) {
  const video = document.getElementById('knnVideo');
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');
  const frames = [];
  canvas.width = 320;
  canvas.height = 240;
  for (let index = 0; index < 30; index += 1) {
    if (!experimentalStream) throw new Error('Camera stopped during capture.');
    await waitExperimentalFrame(video);
    context.save();
    context.translate(canvas.width, 0);
    context.scale(-1, 1);
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    context.restore();
    frames.push(canvas.toDataURL('image/jpeg', 0.72));
    if (onProgress) onProgress(index + 1);
  }
  return frames;
}

function renderExperimentalPrediction(result) {
  if (!Number.isFinite(result.confidence) ||
      result.confidence < 0 || result.confidence > 1 ||
      typeof result.abstained !== 'boolean') {
    throw new Error('The experimental model returned an invalid confidence or abstention result.');
  }
  const percent = (result.confidence * 100).toFixed(1);
  const validationStatus = result.evaluation_ready
    ? 'Session-separated evaluation available (still experimental)'
    : 'UNVALIDATED: fewer than four sessions per word';
  const confidentPrediction = !result.abstained &&
    typeof result.label === 'string' &&
    result.confidence >= SENTENCE_DEMO_MIN_CONFIDENCE;
  const handFrames = result.sequence_quality?.hand_frames;
  const frameCount = result.sequence_quality?.frames;
  if (Number.isInteger(handFrames) && Number.isInteger(frameCount)) {
    setLiveStatus('handStatus', `Hands detected · ${handFrames}/${frameCount} frames`, 'ready');
  } else {
    setLiveStatus('handStatus', 'Hand detected', 'ready');
  }
  if (!confidentPrediction) {
    lastExperimentalPrediction = result;
    document.getElementById('detectedWord').textContent = 'No confident prediction';
    document.getElementById('signMeta').textContent = `Raw model score ${percent}%`;
    setLiveStatus('recognitionStatus', 'No confident prediction', 'warning');
    const meta = document.getElementById('demoSignMeta');
    if (meta) meta.textContent = 'No sign was accepted for this frame window. Hold a supported sign clearly and try again.';
    experimentalMessage('experimentalPrediction', `NO CONFIDENT SIGN · raw score ${percent}% · ${validationStatus} · model ${result.model_version}. The score is not calibrated accuracy.`, 'warning');
  } else {
    lastExperimentalPrediction = result;
    document.getElementById('detectedWord').textContent = result.label;
    document.getElementById('signMeta').textContent = `Raw model score ${percent}% · not calibrated`;
    const meta = document.getElementById('demoSignMeta');
    if (meta) meta.textContent = '';
    experimentalMessage('experimentalPrediction', `PREDICTED ${result.label} · confidence score ${percent}% · not abstained · ${validationStatus} · model ${result.model_version}. The score is not calibrated accuracy.`);
  }
}

function validateExperimentalPrediction(result, scope, artifactId) {
  const activeModel = experimentalStatus?.active_model;
  const supportedLabels = (activeModel?.supports || []).map((word) => word.label);
  const valid = Number.isFinite(result.confidence) &&
    result.confidence >= 0 && result.confidence <= 1 &&
    typeof result.abstained === 'boolean' &&
    result.scope === scope &&
    result.artifact_id === artifactId &&
    result.model_version === artifactId &&
    result.artifact_id === artifactId &&
    result.entry_id === (
      result.abstained
        ? null
        : activeModel?.supports.find((word) => word.label === result.label)?.entry_id
    ) &&
    result.current_storage_version === experimentalStatus?.storage_version &&
    result.current_vocabulary_version === experimentalStatus?.vocabulary_version &&
    (result.abstained
      ? result.label === null
      : (typeof result.label === 'string' && supportedLabels.includes(result.label)));
  if (!valid) {
    if (sequenceContext?.source === 'experimental') {
      markSequenceStale('The classifier response no longer matches the current model or vocabulary.');
    }
    throw new Error('The model or saved vocabulary changed, or the classifier returned an unsupported prediction. Refresh state and clear the sequence before continuing.');
  }
}

async function runExperimentalSequence(runId) {
  let nextWindowSerial = Math.max(30, experimentalFrameSerial);
  let windowIndex = 0;
  let windowsSinceAccepted = 0;
  let qualityRejections = 0;
  let requestFailures = 0;
  const activeCameraRun = cameraRunId;
  while (sequenceCollecting && runId === sequenceRunId &&
      activeCameraRun === cameraRunId && experimentalStream && experimentalFramePumpActive) {
    const context = sequenceContext;
    if (!context || context.source !== 'experimental' || sequenceStale) return;
    if (sequenceSignature(
      experimentalStatus,
      'experimental',
      context.scope,
      context.artifactId
    ) !== context.signature) {
      markSequenceStale('The selected model or saved vocabulary changed.');
      return;
    }

    while (experimentalFrameSerial < nextWindowSerial &&
        sequenceCollecting && runId === sequenceRunId &&
        activeCameraRun === cameraRunId && experimentalStream && experimentalFramePumpActive) {
      await new Promise((resolve) => setTimeout(resolve, 50));
    }
    if (!sequenceCollecting || runId !== sequenceRunId || activeCameraRun !== cameraRunId) return;
    if (!experimentalStream || !experimentalFramePumpActive) {
      pauseAiSequence(false);
      aiSentenceMessage('aiSentenceStatus', 'Sequence capture paused because the camera frame stream stopped. Restart the camera to continue.', 'error');
      return;
    }
    if (experimentalFrameBuffer.length < 30) continue;
    const frames = experimentalFrameBuffer.slice(-30);
    const frameTimes = experimentalFrameTimes.slice(-30);
    const windowStartedAt = frameTimes[0] ?? performance.now();
    const frameCollectionMs = frameTimes.length > 1
      ? frameTimes[frameTimes.length - 1] - frameTimes[0]
      : null;
    const requestedAtFrame = experimentalFrameSerial;
    lastSequenceWindowSerial = requestedAtFrame;
    windowIndex += 1;
    experimentalBusy = true;
    setLiveStatus('recognitionStatus', 'Analyzing sign…', 'working');
    updateFrameProgress();
    updateSequenceControls();
    const requestStartedAt = performance.now();
    try {
      const result = await experimentalApi('/api/experimental/predict', {
        method: 'POST',
        body: JSON.stringify({
          scope: context.scope,
          artifact_id: null,
          frames
        })
      });
      const requestFinishedAt = performance.now();
      if (!sequenceCollecting || runId !== sequenceRunId || activeCameraRun !== cameraRunId) return;
      validateExperimentalPrediction(result, context.scope, context.artifactId);
      const confident = !result.abstained && result.label &&
        result.confidence >= SENTENCE_DEMO_MIN_CONFIDENCE;
      windowsSinceAccepted = confident ? 0 : windowsSinceAccepted + 1;
      renderExperimentalPrediction(result);
      setLiveStatus(
        'recognitionStatus',
        result.abstained || !result.label || result.confidence < SENTENCE_DEMO_MIN_CONFIDENCE
          ? 'No confident prediction'
          : 'Recognized',
        result.abstained || !result.label || result.confidence < SENTENCE_DEMO_MIN_CONFIDENCE
          ? 'warning'
          : 'good'
      );
      processSentenceDemoPrediction(result, 'experimental');
      if (result.abstained || !result.label) {
        sequenceAccumulator.observe({ label: null, uncertain: true }, 2);
        aiSentenceMessage('aiSentenceStatus', 'No sign accepted for this window. A pause or missing hand is ignored; continue when ready.', 'warning');
      } else {
        const accepted = collectStablePrediction({
          label: result.label,
          confidence: result.confidence,
          uncertain: false
        }, 2, 'experimental');
        if (!accepted) {
          aiSentenceMessage('aiSentenceStatus', `Checking that ${result.label} remains stable across another complete window…`);
        } else {
          aiSentenceMessage('aiSentenceStatus', `${result.label} added after two consistent windows. Review or correct it before finishing.`);
        }
      }
      recordRecognitionTiming({
        windowIndex,
        frameCollectionMs,
        clientRequestMs: requestFinishedAt - requestStartedAt,
        windowToResultMs: requestFinishedAt - windowStartedAt,
        result,
        windowsSinceAccepted,
        qualityRejections,
        requestFailures
      });
    } catch (error) {
      if (!sequenceCollecting || runId !== sequenceRunId || activeCameraRun !== cameraRunId) return;
      if (error.status === 409) throw error;
      showPredictionFailure(error);
      const requestFinishedAt = performance.now();
      const handsMatch = String(error.message || '').match(/detected in (\d+)\/30/i);
      if (handsMatch) {
        qualityRejections += 1;
        const handFrames = Number(handsMatch[1]);
        processSentenceDemoPrediction({
          abstained: true,
          hand_quality_rejected: true,
          sequence_quality: { frames: 30, hand_frames: handFrames }
        }, 'experimental');
        sequenceAccumulator.observe({ label: null, uncertain: true }, 2);
      } else {
        requestFailures += 1;
      }
      recordRecognitionTiming({
        windowIndex,
        frameCollectionMs,
        clientRequestMs: requestFinishedAt - requestStartedAt,
        windowToResultMs: requestFinishedAt - windowStartedAt,
        error,
        windowsSinceAccepted,
        qualityRejections,
        requestFailures
      });
    } finally {
      experimentalBusy = false;
      updateFrameProgress();
      updateSequenceControls();
      if (cameraActive &&
          cameraRecognizerSource === 'experimental' &&
          cameraRunId !== activeCameraRun &&
          !sequenceCollecting &&
          !sequenceStale) {
        startAiSequence();
      }
    }
    nextWindowSerial = requestedAtFrame + 10;
  }
}

async function testExperimentalSign() {
  if (!experimentalStream || experimentalBusy) return;
  const button = document.getElementById('testExperimentalSign');
  experimentalBusy = true;
  button.disabled = true;
  updateSequenceControls();
  document.getElementById('addExperimentalPrediction').disabled = true;
  lastExperimentalPrediction = null;
  document.getElementById('addExperimentalPrediction').disabled = true;
  document.getElementById('addExperimentalUncertain').disabled = true;
  try {
    const frames = await captureExperimentalWindow((count) => {
      experimentalMessage('experimentalPrediction', `Capturing sign sequence: ${count}/30 frames…`);
    });
    const scope = document.getElementById('experimentalScope').value;
    const artifactId = document.getElementById('candidateSelect').value;
    if (!artifactId) throw new Error('Select an experimental candidate before testing.');
    experimentalMessage('experimentalPrediction', 'Checking 30-frame landmarks and running the selected candidate…');
    const result = await experimentalApi('/api/experimental/predict', {
      method: 'POST',
      body: JSON.stringify({
        scope,
        artifact_id: artifactId,
        frames
      })
    });
    validateExperimentalPrediction(result, scope, artifactId);
    renderExperimentalPrediction(result);
  } catch (error) {
    experimentalMessage('experimentalPrediction', `Could not test this sign: ${error.message}`, 'error');
    if (error.status === 409) await loadExperimentalVocabulary({ quiet: true });
  } finally {
    experimentalBusy = false;
    updateSequenceControls();
    updateCandidateControls();
  }
}

async function addExperimentalPrediction() {
  if (!lastExperimentalPrediction || lastExperimentalPrediction.abstained) return;
  if (document.getElementById('sentenceRecognizer').value !== 'experimental') {
    experimentalMessage('experimentalPrediction', 'Select the experimental recognizer before adding this prediction.', 'warning');
    return;
  }
  try {
    await experimentalApi('/control', {
      method: 'POST',
      body: JSON.stringify({ action: 'add_word', word: lastExperimentalPrediction.label })
    });
    appendAiSign(
      lastExperimentalPrediction.label,
      lastExperimentalPrediction.confidence,
      false,
      'experimental'
    );
    experimentalMessage('experimentalPrediction', `${lastExperimentalPrediction.label} added to the sentence.`, 'good');
    lastExperimentalPrediction = null;
    updateCandidateControls();
  } catch (error) {
    experimentalMessage('experimentalPrediction', `Could not add the prediction to the sentence: ${error.message}`, 'error');
  }
}

function addExperimentalUncertain() {
  if (!lastExperimentalPrediction || !lastExperimentalPrediction.abstained) return;
  if (!appendAiSign(
    null,
    lastExperimentalPrediction.confidence,
    true,
    'experimental'
  )) return;
  experimentalMessage('experimentalPrediction', 'Abstention added only to the AI review sequence; it was not inserted as a sentence word.', 'warning');
  lastExperimentalPrediction = null;
  updateCandidateControls();
}

// ── NAVBAR ──
function initNavbar() {
  const h = document.getElementById('hamburger');
  const m = document.getElementById('mobileNav');
  h.addEventListener('click', () => {
    const isOpen = m.classList.toggle('open');
    h.setAttribute('aria-expanded', String(isOpen));
    h.setAttribute('aria-label', isOpen ? 'Close navigation' : 'Open navigation');
  });
  m.querySelectorAll('a').forEach(a => a.addEventListener('click', () => {
    m.classList.remove('open');
    h.setAttribute('aria-expanded', 'false');
    h.setAttribute('aria-label', 'Open navigation');
  }));
}

function initSmoothScroll() {
  document.querySelectorAll('a[href^="#"]').forEach(a => {
    a.addEventListener('click', function(e) {
      e.preventDefault();
      const t = document.querySelector(this.getAttribute('href'));
      if (t) t.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  });
}

// ── CAMERA TOGGLE ──

function syncCameraControls() {
  const advancedButton = document.getElementById('advancedCameraToggle');
  if (!advancedButton) return;
  const selected = document.getElementById('sentenceRecognizer')?.value;
  const label = selected === 'browser_knn' ? 'Hello fallback' : 'active sequence model';
  advancedButton.textContent = cameraActive
    ? 'Stop camera'
    : `Start ${label}`;
  advancedButton.disabled = cameraTransition;
}

async function toggleCamera(useAdvancedRecognizer = false) {
  const recognizer = window.knnRecognizer;

  if (cameraTransition) return;
  cameraTransition = true;
  const button = document.getElementById('btnCam');
  if (button) button.disabled = true;
  syncCameraControls();

  try {
    if (cameraActive) {
      setLiveStatus('cameraStatus', 'Stopping camera', 'working');
      stopExperimentalCamera();
      if (recognizer && !recognizer.stopCamera()) return;
      if (mainCameraStream) {
        mainCameraStream.getTracks().forEach((track) => track.stop());
        mainCameraStream = null;
      }
      cameraRecognizerSource = null;
      if (sequenceCollecting && sequenceContext?.source === 'browser_knn') {
        pauseAiSequence(false);
        aiSentenceMessage('aiSentenceStatus', 'Browser camera stopped; sequence capture paused. Reviewed signs are preserved.');
      }

      const video = document.getElementById('knnVideo');
      const image = document.getElementById('videoStream');

      if (video) video.style.display = 'none';
      if (image) image.style.display = 'none';
      const placeholder = document.getElementById('cameraPlaceholder');
      if (placeholder) placeholder.style.display = 'flex';

      stopUI();
      setLiveStatus('cameraStatus', 'Camera off', 'off');
      setLiveStatus('handStatus', 'Waiting for hands', 'idle');
      setLiveStatus('recognitionStatus', 'Ready when you are', 'idle');
      const progressHint = document.getElementById('frameProgressHint');
      if (progressHint) progressHint.textContent = 'Sign to begin';
      updateFrameProgress();
      const demoMeta = document.getElementById('demoSignMeta');
      if (demoMeta) demoMeta.textContent = 'Waiting for a clear sign.';
    } else {
      setLiveStatus('cameraStatus', 'Starting camera', 'working');
      setLiveStatus('handStatus', 'Waiting for hands', 'idle');
      setLiveStatus('recognitionStatus', 'Collecting frames', 'working');
      if (experimentalStream) throw new Error('Stop the experimental camera before starting the browser KNN camera.');
      const statusResponse = await fetch(`${API}/status`);
      if (!statusResponse.ok) {
        throw new Error(`Could not check the Python camera status (HTTP ${statusResponse.status}).`);
      }
      const backendStatus = await statusResponse.json();
      if (backendStatus.camera_active) {
        throw new Error('Stop the Python webcam before starting the browser webcam.');
      }

      const video = document.getElementById('knnVideo');
      const image = document.getElementById('videoStream');
      const recognizerSelect = document.getElementById('sentenceRecognizer');
      const source = useAdvancedRecognizer ? recognizerSelect.value : 'experimental';
      if (!useAdvancedRecognizer) recognizerSelect.value = 'experimental';
      if (source !== 'experimental' && !recognizer) {
        aiSentenceMessage('aiSentenceStatus', 'The legacy Browser KNN fallback did not load. Select the active sequence model or check the browser console.', 'error');
        setLiveStatus('recognitionStatus', 'Fallback unavailable', 'error');
        const meta = document.getElementById('demoSignMeta');
        if (meta) meta.textContent = 'No compatible active model or Hello fallback is available. Check the model setup before signing.';
        return;
      }
      if (source === 'experimental') {
        mainCameraStream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } }
        });
        const stream = mainCameraStream;
        stream.getTracks().forEach((track) => {
          track.addEventListener('ended', () => {
            if (mainCameraStream !== stream) return;
            stopExperimentalCamera();
            mainCameraStream = null;
            cameraRecognizerSource = null;
            if (cameraActive) {
              stopUI();
              setLiveStatus('cameraStatus', 'Camera error', 'error');
              setLiveStatus('handStatus', 'Waiting for hands', 'idle');
              setLiveStatus('recognitionStatus', 'Unable to continue', 'error');
            }
            aiSentenceMessage('aiSentenceStatus', 'The camera stream ended. Start the camera again to continue.', 'error');
          }, { once: true });
        });
        video.srcObject = mainCameraStream;
        await video.play();
        experimentalStream = mainCameraStream;
        startExperimentalFramePump();
      } else {
        if (!await recognizer.startCamera()) {
          setLiveStatus('cameraStatus', 'Camera error', 'error');
          setLiveStatus('recognitionStatus', 'Unable to start', 'error');
          const meta = document.getElementById('demoSignMeta');
          if (meta) meta.textContent = 'The Hello fallback camera could not start. Choose a compatible active sequence model or check browser camera access.';
          return;
        }
        experimentalStream = video.srcObject;
        startExperimentalFramePump();
      }

      if (video?.srcObject) {
        if (image) image.style.display = 'none';

        video.style.display = 'block';
        video.style.transform = 'scaleX(-1)';
        const placeholder = document.getElementById('cameraPlaceholder');
        if (placeholder) placeholder.style.display = 'none';
        cameraRecognizerSource = source;
        startUI(true);
        setLiveStatus('cameraStatus', 'Camera ready', 'ready');
        setLiveStatus('handStatus', 'Waiting for hands', 'idle');
        setLiveStatus('recognitionStatus', 'Collecting frames', 'working');
        updateFrameProgress();
        startAiSequence();
      }
    }
  } catch (error) {
    console.error('[Camera] Could not toggle browser camera:', error);
    setLiveStatus('cameraStatus', 'Camera error', 'error');
    setLiveStatus('recognitionStatus', 'Unable to start', 'error');
    const meta = document.getElementById('demoSignMeta');
    if (meta) meta.textContent = 'Camera could not start. Check camera permission and that no other camera is in use.';
    window.dispatchEvent(new CustomEvent('knn-status', {
      detail: `Camera error: ${error.message}`
    }));
  } finally {
    cameraTransition = false;
    if (button) button.disabled = false;
    syncCameraControls();
  }
}


function startUI(useBrowserCamera = false) {
  cameraActive = true;
  const btn      = document.getElementById('btnCam');
  const camLabel = document.getElementById('camLabel');
  const camIcon  = document.getElementById('camIcon');
  const overlay  = document.getElementById('feedOverlayTop');

  btn.classList.add('active');
  camLabel.textContent = 'Stop Camera';
  camIcon.innerHTML    = '<rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>';
  overlay.style.display = 'flex';
  document.getElementById('feedFrame')?.classList.add('active');
  const placeholder = document.getElementById('cameraPlaceholder');
  if (placeholder) placeholder.style.display = 'none';
  setLiveStatus('cameraStatus', 'Camera ready', 'ready');
  syncCameraControls();

  if (!useBrowserCamera) {
  pollTimer = setInterval(pollStatus, 300);
}
}

function stopUI() {
  cameraActive = false;
  clearInterval(pollTimer); pollTimer = null;

  const btn      = document.getElementById('btnCam');
  const camLabel = document.getElementById('camLabel');
  const camIcon  = document.getElementById('camIcon');
  const overlay  = document.getElementById('feedOverlayTop');

  btn.classList.remove('active');
  camLabel.textContent = 'Start Camera';
  camIcon.innerHTML    = '<path d="M8 5v14l11-7z"/>';
  overlay.style.display = 'none';
  document.getElementById('feedFrame')?.classList.remove('active');
  const placeholder = document.getElementById('cameraPlaceholder');
  if (placeholder) placeholder.style.display = 'flex';
  syncCameraControls();

  setDetected('—', null);
  updateBars(0, 0);
}

// ── POLL STATUS ──
async function pollStatus() {
  try {
    const res = await fetch(`${API}/status`);
    if (!res.ok) return;
    updateUI(await res.json());
  } catch (_) {}
}

// ── UPDATE UI ──
function updateUI(d) {
  if (currentTranslatorMode === 'sentence_demo') {
    updateSentenceDemoFromStatus(d);
  }
  if (!d.hands) {
    setDetected('—', null);
    setMeta('');
  } else if (d.current_word) {
    setDetected(d.current_word, d.current_conf);
    if (d.top_vote_word && d.top_vote_count > 0) {
      setMeta(`Vote: ${d.top_vote_word} (${d.top_vote_count} / 6)`);
    } else {
      setMeta('');
    }
  } else {
    setDetected('...', null);
    setMeta('');
  }

  updateBars(d.buf_pct, d.vote_pct);

  setBuildingText(
    d.sentence_words && d.sentence_words.length
      ? d.sentence_words.join(' ')
      : '—'
  );

  showFinal(d.completed_sentence || '');
  setOriginalGloss(d.completed_gloss_words || []);

  const sel = document.getElementById('langSelect');
  if (sel && d.lang_code) sel.value = d.lang_code;
}

function setDetected(word, conf) {
  const el = document.getElementById('detectedWord');
  if (el) el.textContent = word;
  const badge = document.getElementById('signMeta');
  if (badge) badge.textContent = conf !== null && conf !== undefined ? `${conf}% confidence` : '';
}
function setMeta(text) {
  const el = document.getElementById('signMeta');
  if (el) el.textContent = text;
}
function setBuildingText(text) {
  const el = document.getElementById('buildingText');
  if (el) el.textContent = text;
}
function showFinal(text) {
  const card = document.getElementById('finalCard');
  const el   = document.getElementById('finalText');
  if (!card || !el) return;
  if (text) {
    el.textContent      = text;
    card.style.display  = 'block';
  } else {
    card.style.display  = 'none';
    el.textContent      = '';
  }
}
function setOriginalGloss(words) {
  const element = document.getElementById('originalGloss');
  if (element) {
    element.textContent = words.length ? `Original signs: ${words.join(' ')}` : '';
  }
}
function updateBars(bufPct, votePct) {
  const b = document.getElementById('bufferBar');
  const v = document.getElementById('voteBar');
  if (b) b.style.width = `${Math.min(bufPct  * 100, 100)}%`;
  if (v) v.style.width = `${Math.min(votePct * 100, 100)}%`;
}

// ── CONTROLS ──
async function sendControl(action, extra = {}) {
  try {
    const res = await fetch(`${API}/control`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ action, ...extra }),
    });
    if (!res.ok) return;
    const d = await res.json();
    if (d.sentence_words !== undefined)
      setBuildingText(d.sentence_words.length ? d.sentence_words.join(' ') : '—');
    if (d.completed_sentence !== undefined)
      showFinal(d.completed_sentence);
    if (d.completed_gloss_words !== undefined)
      setOriginalGloss(d.completed_gloss_words);
    const sel = document.getElementById('langSelect');
    if (sel && d.lang_code) sel.value = d.lang_code;
  } catch (e) { console.error(e); }
}

function finalizeSentence() { sendControl('finalize'); }
function clearSentence()    {
  sendControl('clear');
  setBuildingText('—');
  showFinal('');
  setOriginalGloss([]);
}
function undoWord()         { sendControl('undo'); }
function speakOutput()      { sendControl('speak'); }
function changeLanguage() {
  const sel = document.getElementById('langSelect');
  if (sel) sendControl('language', { lang: sel.value });
}

// ── KEYBOARD SHORTCUTS ──
const LANG_KEYS = { '1':'en', '2':'hi', '3':'bn', '4':'ta', '5':'te', '6':'mr', '7':'gu' };

document.addEventListener('keydown', (e) => {
  // Don't trigger if user is typing in an input/select
  if (['INPUT','SELECT','TEXTAREA'].includes(document.activeElement.tagName)) return;

  switch (e.key) {
    case 'Enter':
      e.preventDefault();
      finalizeSentence();
      break;
    case 'b':
    case 'B':
      undoWord();
      break;
    case 'c':
    case 'C':
      clearSentence();
      break;
    case 's':
    case 'S':
      speakOutput();
      break;
    default:
      if (LANG_KEYS[e.key]) {
        const sel = document.getElementById('langSelect');
        if (sel) {
          sel.value = LANG_KEYS[e.key];
          changeLanguage();
        }
      }
  }
});

/* ── Browser KNN recognizer integration ── */

window.addEventListener('knn-status', (event) => {
  console.log('[KNN]', event.detail);
  if (sequenceCollecting &&
      sequenceContext?.source === 'browser_knn' &&
      String(event.detail).startsWith('Prediction error:')) {
    pauseAiSequence(false);
    aiSentenceMessage('aiSentenceStatus', `Browser KNN sequence capture paused: ${event.detail}`, 'error');
  }
});

window.addEventListener('knn-confidence', (event) => {
  const { word, confidence } = event.detail;
  if (typeof word !== 'string' || !Number.isFinite(confidence) ||
      confidence < 0 || confidence > 1 || !cameraActive) return;
  if (document.getElementById('sentenceRecognizer')?.value === 'experimental') return;

  const wordElement = document.getElementById('detectedWord');
  const metaElement = document.getElementById('signMeta');

  if (wordElement) {
    wordElement.textContent = word === 'Hello'
      ? word
      : 'Unsupported · Hello fallback only';
  }
  setLiveStatus('handStatus', 'Hand detected', 'ready');
  setLiveStatus('recognitionStatus', 'Analyzing sign', 'working');
  if (metaElement) {
    metaElement.textContent = word === 'Hello'
      ? `Hello fallback · raw score ${(confidence * 100).toFixed(1)}% · uncalibrated`
      : 'Other Browser KNN classes are ignored by the translator';
  }
});

window.addEventListener('knn-release', () => {
  if (sequenceContext?.source === 'browser_knn') {
    sequenceAccumulator.release();
    if (sequenceCollecting) {
      aiSentenceMessage('aiSentenceStatus', 'Sign released. A repeated sign can now be collected after the next stable detection.');
    }
  }
});

window.addEventListener('knn-word', async (event) => {
  const { word, confidence } = event.detail;
  if (!cameraActive) return;
  const supportedLabels = backendRecognizerInfo?.browser_knn?.vocabulary || [];
  if (typeof word !== 'string' ||
      !supportedLabels.includes(word) ||
      !window.knnRecognizer?.words?.includes(word) ||
      word === 'OTHER' ||
      !Number.isFinite(confidence) || confidence < 0 || confidence > 1) {
    console.error('[KNN] Ignoring invalid recognized word event:', event.detail);
    return;
  }
  if (document.getElementById('sentenceRecognizer')?.value === 'experimental') {
    return;
  }
  if (word !== 'Hello') {
    if (sequenceCollecting &&
        document.getElementById('sentenceRecognizer')?.value === 'browser_knn') {
      sequenceAccumulator.observe({ label: null, uncertain: true }, 1);
    }
    setDetected('Unsupported · Hello fallback only', null);
    setMeta('Other Browser KNN classes are ignored by the translator');
    setLiveStatus('recognitionStatus', 'Unsupported sign', 'warning');
    return;
  }

  setLiveStatus('recognitionStatus', 'Recognized', 'good');
  processSentenceDemoPrediction({ label: word, confidence, abstained: false }, 'browser_knn');
  if (sequenceCollecting &&
      document.getElementById('sentenceRecognizer')?.value === 'browser_knn') {
    collectStablePrediction({ label: word, confidence, uncertain: false }, 1, 'browser_knn');
    return;
  }
  try {
    const response = await fetch(`${API}/control`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'add_word', word })
    });

    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const data = await response.json();

    if (typeof updateUI === 'function') updateUI({
      ...data,
      hands: true,
      current_word: word,
      current_conf: confidence * 100
    });
  } catch (error) {
    console.error('[KNN] Could not add recognized word:', error);
    window.dispatchEvent(new CustomEvent('knn-status', {
      detail: `Could not add "${word}" to the sentence: ${error.message}`
    }));
  }
});


function trainKNN(classIndex) {
  if (!window.knnRecognizer) {
    console.error('KNN recognizer bundle is not loaded.');
    return;
  }

  return window.knnRecognizer.trainClass(classIndex);
}

function startKNNPrediction() {
  if (!window.knnRecognizer) {
    console.error('KNN recognizer bundle is not loaded.');
    return;
  }

  window.knnRecognizer.startPrediction();
}

function saveKNNTrainingData() {
  if (window.knnRecognizer) window.knnRecognizer.saveTrainingData();
}

function exportKNNTrainingData() {
  if (window.knnRecognizer) window.knnRecognizer.exportTrainingData();
}

function importKNNTrainingData(input) {
  const file = input.files && input.files[0];
  if (!file) return;
  if (!confirm('Importing replaces the currently loaded and saved training examples. Continue?')) {
    input.value = '';
    return;
  }
  window.knnRecognizer.importTrainingData(file).finally(() => {
    input.value = '';
  });
}

function clearKNNTrainingData() {
  if (!window.knnRecognizer) return;
  if (confirm('Permanently clear the saved and loaded KNN training examples?')) {
    window.knnRecognizer.clearTrainingData();
  }
}

window.addEventListener('knn-status', (event) => {
  const status = document.getElementById('knnStatus');
  if (status) status.textContent = event.detail;
});

window.addEventListener('beforeunload', () => {
  stopExperimentalCamera();
  window.knnRecognizer?.stopCamera();
});


/* ═══════════════════════════════════════════════════════════════════════════
   Sentence Demo Mode Module
   Maps verified recognition triggers to predefined demonstration sentences
   ═══════════════════════════════════════════════════════════════════════════ */

let currentTranslatorMode = 'normal'; // 'normal' | 'sentence_demo'
let sentenceDemoMappings = {};
let sentenceDemoValidationReport = null;
let sentenceDemoAutoSpeak = true;
let sentenceDemoCurrentSentence = '';
let sentenceDemoCurrentSign = '';
let sentenceDemoActiveHoldSign = null;
let sentenceDemoLastTriggerTime = 0;
let sentenceDemoPendingSign = null;
let sentenceDemoPendingCount = 0;
let sentenceDemoReleaseCount = 0;
const SENTENCE_DEMO_MIN_CONFIDENCE = 0.55;
const SENTENCE_DEMO_STABILITY_WINDOWS = 1; // Trigger after the first confident prediction
const SENTENCE_DEMO_COOLDOWN_MS = 3000;    // 3.0s cooldown between triggers
const SENTENCE_DEMO_RELEASE_WINDOWS = 2;   // Hands absent for 2 windows releases the sign hold

function initSentenceDemo() {
  loadSentenceDemoConfig();
}

function setTranslatorMode(mode) {
  currentTranslatorMode = mode;
  const isDemo = mode === 'sentence_demo';

  const btnNormal = document.getElementById('btnModeNormal');
  const btnDemo = document.getElementById('btnModeSentenceDemo');
  const indicator = document.getElementById('demoModeIndicatorText');
  const banner = document.getElementById('demoDisclosureBanner');
  if (btnNormal) btnNormal.classList.toggle('active', !isDemo);
  if (btnDemo) btnDemo.classList.toggle('active', isDemo);
  if (banner) banner.style.display = isDemo ? 'flex' : 'none';

  if (indicator) {
    indicator.textContent = isDemo
      ? 'Sentence Demo Mode · Active'
      : 'Standard Translation Mode';
  }

  fetch(`${API}/control`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: 'set_sentence_demo_mode', enabled: isDemo })
  }).catch(() => {});

  if (isDemo) {
    loadSentenceDemoConfig();
  }
}

async function loadSentenceDemoConfig(forceReload = false) {
  try {
    const url = forceReload ? `${API}/api/sentence-demo/reload` : `${API}/api/sentence-demo/config`;
    const res = await fetch(url, { method: forceReload ? 'POST' : 'GET' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    sentenceDemoMappings = data.mappings || {};
    sentenceDemoValidationReport = data.validation || null;
    renderSentenceDemoMappings(data);
  } catch (error) {
    console.error('[Sentence Demo] Failed to load configuration:', error);
    const container = document.getElementById('demoMappingsList');
    if (container) {
      container.innerHTML = `<p class="empty-mappings-text" style="color:#B00020;">Could not load mappings: ${error.message}</p>`;
    }
  }
}

function refreshDemoMappings() {
  loadSentenceDemoConfig(true);
}

function findSentenceMapping(label) {
  if (!label || typeof label !== 'string') return null;
  const cleanLabel = label.trim().toLowerCase();
  for (const [sign, sentence] of Object.entries(sentenceDemoMappings)) {
    if (sign.trim().toLowerCase() === cleanLabel) {
      return { sign, sentence };
    }
  }
  return null;
}

function processSentenceDemoPrediction(result, source) {
  const detectedSignEl = document.getElementById('detectedWord');
  const signMetaEl = document.getElementById('demoSignMeta');
  const cooldownDot = document.getElementById('demoCooldownDot');
  const cooldownText = document.getElementById('demoCooldownText');

  if (!result || result.abstained || !result.label || (result.confidence !== undefined && result.confidence < SENTENCE_DEMO_MIN_CONFIDENCE)) {
    if (result?.abstained) {
      const handFrames = result.sequence_quality?.hand_frames;
      const handQualityRejected = result.hand_quality_rejected === true;
      const status = handQualityRejected
        ? (handFrames === 0 ? 'Hands not detected' : 'Hands not visible clearly enough')
        : 'No confident prediction';
      setLiveStatus(
        'recognitionStatus',
        status,
        'warning'
      );
    }
    sentenceDemoReleaseCount += 1;
    if (sentenceDemoReleaseCount >= SENTENCE_DEMO_RELEASE_WINDOWS) {
      sentenceDemoActiveHoldSign = null;
      sentenceDemoPendingSign = null;
      sentenceDemoPendingCount = 0;
      if (cooldownDot) cooldownDot.className = 'cooldown-dot ready';
      if (cooldownText) cooldownText.textContent = 'Ready for sign';
    }
    return;
  }

  sentenceDemoReleaseCount = 0;
  const label = result.label.trim();
  const confPct = result.confidence !== undefined ? Math.round(result.confidence * 100) : 100;

  // Stability filtering: require SENTENCE_DEMO_STABILITY_WINDOWS consecutive matching windows
  if (label === sentenceDemoPendingSign) {
    sentenceDemoPendingCount += 1;
  } else {
    sentenceDemoPendingSign = label;
    sentenceDemoPendingCount = 1;
  }

  if (detectedSignEl) detectedSignEl.textContent = label;

  if (sentenceDemoPendingCount < SENTENCE_DEMO_STABILITY_WINDOWS) {
    if (signMetaEl) {
      signMetaEl.textContent = `Stabilizing "${label}" (${sentenceDemoPendingCount}/${SENTENCE_DEMO_STABILITY_WINDOWS})…`;
    }
    return;
  }

  setLiveStatus('recognitionStatus', 'Recognized', 'good');
  if (signMetaEl) signMetaEl.textContent = `Model score ${confPct}% · raw, not calibrated accuracy.`;

  // Look up mapped sentence
  const matched = findSentenceMapping(label);
  const sentenceDisplay = document.getElementById('demoSentenceDisplay');
  const triggerInfo = document.getElementById('demoTriggerInfo');

  if (!matched) {
    if (sentenceDisplay) {
      sentenceDisplay.textContent = `Recognized sign "${label}" has no configured sentence mapping.`;
    }
    if (triggerInfo) {
      triggerInfo.textContent = `No demonstration sentence is configured for ${label}.`;
    }
    return;
  }

  // Duplicate prevention & cooldown check
  const now = Date.now();
  if (sentenceDemoActiveHoldSign === matched.sign && (now - sentenceDemoLastTriggerTime < SENTENCE_DEMO_COOLDOWN_MS)) {
    if (cooldownDot) cooldownDot.className = 'cooldown-dot held';
    if (cooldownText) cooldownText.textContent = `Sign "${matched.sign}" held in view · Cooldown active (release sign to re-trigger)`;
    return;
  }

  // Trigger!
  sentenceDemoActiveHoldSign = matched.sign;
  sentenceDemoLastTriggerTime = now;
  sentenceDemoCurrentSentence = matched.sentence;
  sentenceDemoCurrentSign = matched.sign;

  if (sentenceDisplay) {
    sentenceDisplay.textContent = `“${matched.sentence}”`;
  }
  if (triggerInfo) {
    triggerInfo.textContent = `Triggered by sign: ${matched.sign} (${confPct}% score)`;
  }
  if (cooldownDot) cooldownDot.className = 'cooldown-dot held';
  if (cooldownText) cooldownText.textContent = `Sign "${matched.sign}" triggered · Release sign to re-trigger`;

  if (sentenceDemoAutoSpeak) {
    speakDemoSentence(matched.sentence);
  }
}

function updateSentenceDemoFromStatus(d) {
  if (!d) return;
  const detectedSignEl = document.getElementById('detectedWord');
  const signMetaEl = document.getElementById('demoSignMeta');
  const sentenceDisplay = document.getElementById('demoSentenceDisplay');
  const triggerInfo = document.getElementById('demoTriggerInfo');

  if (d.current_word) {
    if (detectedSignEl) detectedSignEl.textContent = d.current_word;
    if (signMetaEl) signMetaEl.textContent = `${d.current_conf}% model score`;
  } else if (!d.hands) {
    if (detectedSignEl) detectedSignEl.textContent = '—';
    if (signMetaEl) signMetaEl.textContent = 'Waiting for sign…';
  }

  if (d.sentence_demo_sentence && d.sentence_demo_sentence !== sentenceDemoCurrentSentence) {
    sentenceDemoCurrentSentence = d.sentence_demo_sentence;
    sentenceDemoCurrentSign = d.sentence_demo_trigger;
    if (sentenceDisplay) sentenceDisplay.textContent = `“${d.sentence_demo_sentence}”`;
    if (triggerInfo) triggerInfo.textContent = `Triggered by sign: ${d.sentence_demo_trigger}`;
  }
}

function speakCurrentDemoSentence() {
  speakDemoSentence(sentenceDemoCurrentSentence);
}

function initQuickPhrases() {
  const categoriesElement = document.getElementById('quickPhraseCategories');
  const phrasesElement = document.getElementById('quickPhraseList');
  const selectedElement = document.getElementById('quickPhraseSelected');
  const clearButton = document.getElementById('quickPhraseClear');
  const speakButton = document.getElementById('quickPhraseSpeak');
  if (!categoriesElement || !phrasesElement || !selectedElement || !clearButton || !speakButton) return;

  categoriesElement.replaceChildren(...QUICK_PHRASE_CATEGORIES.map((category) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'quick-phrase-category';
    button.textContent = category.name;
    button.setAttribute('aria-pressed', String(category.name === activeQuickPhraseCategory));
    button.addEventListener('click', () => {
      stopQuickPhraseSpeech();
      activeQuickPhraseCategory = category.name;
      selectedQuickPhrase = '';
      selectedElement.textContent = 'Choose a phrase to see it here.';
      clearButton.disabled = true;
      speakButton.disabled = true;
      setQuickPhraseStatus('Choose a phrase, then select Speak Phrase.', 'ready');
      renderQuickPhraseCategories();
      renderQuickPhraseSuggestions();
    });
    return button;
  }));

  clearButton.addEventListener('click', clearQuickPhrase);
  speakButton.addEventListener('click', speakQuickPhrase);
  renderQuickPhraseCategories();
  renderQuickPhraseSuggestions();
  setQuickPhraseStatus('Choose a phrase, then select Speak Phrase.', 'ready');
}

function renderQuickPhraseCategories() {
  const container = document.getElementById('quickPhraseCategories');
  if (!container) return;
  container.querySelectorAll('.quick-phrase-category').forEach((button) => {
    button.setAttribute('aria-pressed', String(button.textContent === activeQuickPhraseCategory));
  });
}

function renderQuickPhraseSuggestions() {
  const container = document.getElementById('quickPhraseList');
  if (!container) return;
  const category = QUICK_PHRASE_CATEGORIES.find((item) => item.name === activeQuickPhraseCategory);
  if (!category) return;
  container.replaceChildren(...category.phrases.map((phrase) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'quick-phrase-option';
    button.textContent = phrase;
    button.setAttribute('aria-pressed', String(phrase === selectedQuickPhrase));
    button.addEventListener('click', () => selectQuickPhrase(phrase));
    return button;
  }));
}

function selectQuickPhrase(phrase) {
  selectedQuickPhrase = phrase;
  const selectedElement = document.getElementById('quickPhraseSelected');
  const clearButton = document.getElementById('quickPhraseClear');
  const speakButton = document.getElementById('quickPhraseSpeak');
  if (selectedElement) selectedElement.textContent = phrase;
  if (clearButton) clearButton.disabled = false;
  if (speakButton) speakButton.disabled = false;
  renderQuickPhraseSuggestions();
  speakQuickPhrase();
}

function clearQuickPhrase() {
  stopQuickPhraseSpeech();
  selectedQuickPhrase = '';
  const selectedElement = document.getElementById('quickPhraseSelected');
  const clearButton = document.getElementById('quickPhraseClear');
  const speakButton = document.getElementById('quickPhraseSpeak');
  if (selectedElement) selectedElement.textContent = 'Choose a phrase to see it here.';
  if (clearButton) clearButton.disabled = true;
  if (speakButton) speakButton.disabled = true;
  setQuickPhraseStatus('Select a phrase to prepare it for speech.', 'ready');
  renderQuickPhraseSuggestions();
}

function stopQuickPhraseSpeech() {
  quickPhraseSpeechRunId += 1;
  if (quickPhraseSpeaking && 'speechSynthesis' in window) window.speechSynthesis.cancel();
  quickPhraseSpeaking = false;
}

function speakQuickPhrase() {
  if (!selectedQuickPhrase) {
    setQuickPhraseStatus('Select a phrase before speaking.', 'warning');
    return;
  }
  quickPhraseSpeechRunId += 1;
  quickPhraseSpeaking = true;
  speakDemoSentence(selectedQuickPhrase, 'quickPhraseStatus', true, quickPhraseSpeechRunId);
}

function setQuickPhraseStatus(message, state) {
  const status = document.getElementById('quickPhraseStatus');
  if (!status) return;
  status.textContent = message;
  status.dataset.state = state;
}

function speakDemoSentence(sentence, statusElementId = 'speechStatus', isQuickPhrase = false, quickPhraseRunId = 0) {
  const text = (sentence || sentenceDemoCurrentSentence || '').trim();
  if (!text) return;

  const speechStatus = document.getElementById(statusElementId);
  const updateStatus = (message, state) => {
    if (isQuickPhrase && quickPhraseRunId !== quickPhraseSpeechRunId) return;
    if (!speechStatus) return;
    speechStatus.textContent = message;
    if (isQuickPhrase) {
      speechStatus.dataset.state = state;
      quickPhraseSpeaking = state === 'speaking';
    }
  };
  updateStatus('Starting speech…', 'speaking');
  if ('speechSynthesis' in window) {
    try {
      const utterance = new SpeechSynthesisUtterance(text);
      const languages = {
        en: 'en-IN', hi: 'hi-IN', bn: 'bn-IN', ta: 'ta-IN',
        te: 'te-IN', mr: 'mr-IN', gu: 'gu-IN'
      };
      const langSel = document.getElementById('demoLangSelect') || document.getElementById('langSelect');
      const lang = langSel ? langSel.value : 'en';
      utterance.lang = languages[lang] || 'en-IN';
      utterance.onstart = () => {
        updateStatus(isQuickPhrase ? 'Speaking selected phrase' : 'Speaking sentence', 'speaking');
      };
      utterance.onend = () => {
        updateStatus('Speech finished', 'finished');
      };
      utterance.onerror = () => {
        updateStatus('Speech is unavailable. Check the browser audio settings.', 'error');
      };
      window.speechSynthesis.cancel();
      window.speechSynthesis.speak(utterance);
      return;
    } catch (e) {
      console.warn('[Sentence Demo] Speech synthesis error:', e);
    }
  }

  const langSel = document.getElementById('demoLangSelect') || document.getElementById('langSelect');
  const lang_code = langSel ? langSel.value : 'en';
  fetch(`${API}/api/sentence-demo/speak`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sentence: text, lang_code })
    })
    .then((response) => {
      if (!response.ok) throw new Error(`Speech request failed (${response.status}).`);
      updateStatus('Speech sent to system audio', 'finished');
    })
    .catch((error) => {
      console.error('[Sentence Demo] Speech request failed:', error);
      updateStatus('Speech is unavailable. Check system audio and text-to-speech setup.', 'error');
    });
}

function clearDemoSentence() {
  sentenceDemoCurrentSentence = '';
  sentenceDemoCurrentSign = '';
  sentenceDemoActiveHoldSign = null;
  sentenceDemoPendingSign = null;
  sentenceDemoPendingCount = 0;
  sentenceDemoReleaseCount = 0;

  const sentenceDisplay = document.getElementById('demoSentenceDisplay');
  if (sentenceDisplay) sentenceDisplay.textContent = 'Perform a configured sign to trigger its sentence.';
  const triggerInfo = document.getElementById('demoTriggerInfo');
  if (triggerInfo) triggerInfo.textContent = 'No sign triggered yet.';
  const detectedSignEl = document.getElementById('detectedWord');
  if (detectedSignEl) detectedSignEl.textContent = '—';
  const signMetaEl = document.getElementById('demoSignMeta');
  if (signMetaEl) signMetaEl.textContent = 'Waiting for sign…';
  const cooldownDot = document.getElementById('demoCooldownDot');
  if (cooldownDot) cooldownDot.className = 'cooldown-dot ready';
  const cooldownText = document.getElementById('demoCooldownText');
  if (cooldownText) cooldownText.textContent = 'Ready for sign';

  fetch(`${API}/control`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: 'clear_sentence_demo' })
  }).catch(() => {});
}

function toggleAutoSpeak(checked) {
  sentenceDemoAutoSpeak = Boolean(checked);
}

function toggleMappingEditor(show) {
  const form = document.getElementById('demoMappingEditorForm');
  const btn = document.getElementById('btnToggleMappingEditor');
  if (!form) return;
  const isShown = show !== undefined ? show : form.style.display === 'none';
  form.style.display = isShown ? 'block' : 'none';
  if (btn) btn.textContent = isShown ? '✕ Close Form' : '+ Add / Edit';
}

function renderSentenceDemoMappings(data) {
  const container = document.getElementById('demoMappingsList');
  const countEl = document.getElementById('demoMappingCount');
  if (!container) return;

  const mappings = data.mappings || {};
  const entries = data.validation?.entries || [];
  const count = Object.keys(mappings).length;
  if (countEl) countEl.textContent = count;

  if (count === 0) {
    container.innerHTML = '<p class="empty-mappings-text">No mappings configured yet. Click "+ Add / Edit" to create one.</p>';
    return;
  }

  container.innerHTML = '';
  const entriesMap = {};
  entries.forEach((e) => {
    entriesMap[e.sign.toLowerCase()] = e;
  });

  Object.entries(mappings).forEach(([sign, sentence]) => {
    const item = document.createElement('div');
    item.className = 'demo-mapping-item';
    const entryInfo = entriesMap[sign.toLowerCase()];
    const isSupported = entryInfo ? entryInfo.supported : true;

    const signWrap = document.createElement('div');
    signWrap.className = 'mapping-sign-wrap';
    const signTag = document.createElement('span');
    signTag.className = 'mapping-sign-tag';
    signTag.textContent = sign;

    const badge = document.createElement('span');
    badge.className = `mapping-badge ${isSupported ? 'supported' : 'unsupported'}`;
    badge.textContent = isSupported ? '✓ Active Model' : '⚠ Unsupported';
    badge.title = entryInfo?.reason || (isSupported ? 'Recognized by active sequence model' : 'Not recognized by active model');
    signWrap.append(signTag, badge);

    const sentenceSpan = document.createElement('span');
    sentenceSpan.className = 'mapping-sentence-text';
    sentenceSpan.textContent = sentence;
    sentenceSpan.title = sentence;

    item.append(signWrap, sentenceSpan);
    container.appendChild(item);
  });
}

async function saveNewMapping() {
  const signInput = document.getElementById('newMappingSign');
  const sentenceInput = document.getElementById('newMappingSentence');
  const feedback = document.getElementById('editorFeedback');

  const sign = (signInput?.value || '').trim();
  const sentence = (sentenceInput?.value || '').trim();

  if (!sign || !sentence) {
    if (feedback) {
      feedback.textContent = 'Both sign label and sentence are required.';
      feedback.className = 'editor-feedback error';
    }
    return;
  }

  try {
    const updatedMappings = { ...sentenceDemoMappings, [sign]: sentence };
    const res = await fetch(`${API}/api/sentence-demo/config`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mappings: updatedMappings })
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || `HTTP ${res.status}`);
    }
    const data = await res.json();
    sentenceDemoMappings = data.mappings || updatedMappings;
    renderSentenceDemoMappings(data);
    if (signInput) signInput.value = '';
    if (sentenceInput) sentenceInput.value = '';
    if (feedback) {
      feedback.textContent = `Saved mapping for "${sign}"!`;
      feedback.className = 'editor-feedback good';
      setTimeout(() => { feedback.textContent = ''; }, 3000);
    }
  } catch (err) {
    if (feedback) {
      feedback.textContent = err.message;
      feedback.className = 'editor-feedback error';
    }
  }
}
