// ═══════ ISL Translator — Starry Night Frontend ═══════

const API = window.location.origin;
let cameraActive = false;
let pollTimer    = null;
let cameraTransition = false;
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
let experimentalRefreshTimer = null;
let backendRecognizerInfo = null;
let vocabularyRefreshId = 0;
let experimentalFramePumpActive = false;
let experimentalFrameBuffer = [];
let experimentalFrameSerial = 0;
const sequenceAccumulator = new window.ISLSequenceAccumulator.SequenceAccumulator();

document.addEventListener('DOMContentLoaded', () => {
  initNavbar();
  initSmoothScroll();
  initDots();
  document.getElementById('videoStream').src = `${API}/video_feed`;
  if (window.knnRecognizer) window.knnRecognizer.initializeTrainingData();
  initAiSentenceBuilder();
  initExperimentalStudio();
  window.addEventListener('focus', refreshVocabularyOnReturn);
  document.addEventListener('visibilitychange', refreshVocabularyOnReturn);
});

function aiSentenceMessage(id, message, kind = '') {
  const element = document.getElementById(id);
  if (!element) return;
  element.textContent = message;
  element.dataset.kind = kind;
}

function initAiSentenceBuilder() {
  const recognizer = document.getElementById('sentenceRecognizer');
  if (!recognizer) return;
  recognizer.addEventListener('change', () => {
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
    updateSequenceControls();
  });
  document.getElementById('startAiSequence').addEventListener('click', startAiSequence);
  document.getElementById('pauseAiSequence').addEventListener('click', pauseAiSequence);
  document.getElementById('undoAiSign').addEventListener('click', undoAiSign);
  document.getElementById('clearAiSigns').addEventListener('click', clearAiSignSequence);
  document.getElementById('finishAiSentence').addEventListener('click', finishAiSentence);
  document.getElementById('acceptAiSentence').addEventListener('click', acceptAiSentence);
  document.getElementById('speakAiSentence').addEventListener('click', speakAiSentence);
  updateSequenceControls();
  refreshAiProviderStatus();
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
  updateSequenceControls();
  renderAiSignSequence();
}

async function startAiSequence() {
  if (sequenceCollecting || sequenceStale) return;
  const source = document.getElementById('sentenceRecognizer').value;
  const requestedScope = document.getElementById('experimentalScope').value;
  aiSentenceMessage('aiSentenceStatus', 'Checking the latest recognizer and vocabulary state…');
  await loadExperimentalVocabulary({ quiet: true });
  if (!experimentalStatus) return;
  if (experimentalStatus.scope !== requestedScope) {
    aiSentenceMessage('aiSentenceStatus', 'The selected vocabulary scope changed while refreshing. Try again.', 'warning');
    return;
  }

  if (source === 'browser_knn') {
    const labels = backendRecognizerInfo?.browser_knn?.vocabulary || [];
    const localLabels = window.knnRecognizer?.words?.slice(0, -1) || [];
    if (!window.knnRecognizer || JSON.stringify(labels) !== JSON.stringify(localLabels)) {
      aiSentenceMessage('aiSentenceStatus', 'The browser KNN labels do not match the backend. Refresh the page before collecting a sequence.', 'error');
      return;
    }
    if (!cameraActive) {
      aiSentenceMessage('aiSentenceStatus', 'Start the Browser KNN camera first, then start sequence capture.', 'warning');
      return;
    }
    if (!window.knnRecognizer.isRecognizing()) {
      window.knnRecognizer.startPrediction();
      if (!window.knnRecognizer.isRecognizing()) {
        aiSentenceMessage('aiSentenceStatus', 'Recognition did not start. Check the KNN status and training-example counts.', 'error');
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
    aiSentenceMessage('aiSentenceStatus', activeModel?.unusable_reason || 'No current validated sequence model is active. Use Browser KNN or prepare a model in Developer Studio.', 'warning');
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
  scope.addEventListener('change', () => loadExperimentalVocabulary());
  loadExperimentalVocabulary();
  experimentalRefreshTimer = setInterval(() => {
    if (document.visibilityState === 'visible') {
      loadExperimentalVocabulary({ quiet: true });
    }
  }, 20000);
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
    throw error;
  }
  return result;
}

async function loadExperimentalVocabulary({ quiet = false } = {}) {
  const scope = document.getElementById('experimentalScope').value;
  const refreshId = ++vocabularyRefreshId;
  const previousStatus = experimentalStatus;
  if (!quiet) experimentalMessage('recognizerStateStatus', 'Refreshing vocabulary and model state from Flask…');
  try {
    const data = await experimentalApi(`/api/vocabulary?scope=${encodeURIComponent(scope)}`);
    if (refreshId !== vocabularyRefreshId) return;
    experimentalStatus = data;
    backendRecognizerInfo = data.recognizers || null;
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
    experimentalMessage('recognizerStateStatus', `Could not load vocabulary: ${error.message}`, 'error');
    document.getElementById('sentenceRecognizer').replaceChildren(
      new Option('Recognizer status unavailable', 'browser_knn')
    );
    if (sequenceCollecting || sequenceContext || previousStatus) {
      markSequenceStale('The backend state could not be verified.');
    }
  }
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
      ? `${knn?.label || 'Browser KNN'} · ${knnLabels.length} fixed signs`
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
  if ([...select.options].some((option) => option.value === previous && !option.disabled)) {
    select.value = previous;
  } else {
    select.value = 'browser_knn';
  }

  const labels = knnMatches ? knnLabels.join(', ') : 'KNN labels unavailable';
  const active = experimental?.active_model;
  const modelText = active?.usable
    ? `Active validated sequence model ${active.artifact_id} · ${active.supports.length} of ${data.words.length} saved words supported.`
    : `No validated active sequence model. ${active?.unusable_reason || 'Candidate testing and activation are available in Developer Studio.'}`;
  aiSentenceMessage(
    'recognizerStateStatus',
    `Backend vocabulary refreshed · Browser KNN is fixed to ${labels}. ${modelText} New saved examples require explicit training, evaluation, and activation.`
  );
}

function refreshVocabularyOnReturn() {
  if (document.visibilityState === 'hidden') return;
  loadExperimentalVocabulary({ quiet: true });
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
    document.getElementById('startExperimentalCamera').disabled = true;
    document.getElementById('stopExperimentalCamera').disabled = false;
    updateCandidateControls();
    experimentalMessage('experimentalPrediction', 'Camera ready. Select a current candidate, hold one sign clearly, then capture a 30-frame test. Candidates without four sessions per word are unvalidated.');
  } catch (error) {
    stopExperimentalCamera();
    experimentalMessage('experimentalPrediction', `Could not start experimental camera: ${error.message}`, 'error');
  } finally {
    cameraTransition = false;
  }
}

function stopExperimentalCamera() {
  if (sequenceContext?.source === 'experimental' && sequenceCollecting) {
    pauseAiSequence(false);
    aiSentenceMessage('aiSentenceStatus', 'Camera stopped; sequence capture paused. Reviewed signs are preserved.');
  }
  experimentalStream = null;
  experimentalFramePumpActive = false;
  experimentalFrameBuffer = [];
  experimentalFrameSerial = 0;
  lastExperimentalPrediction = null;
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
  experimentalFrameSerial = 0;
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
        if (experimentalFrameBuffer.length > 30) experimentalFrameBuffer.shift();
        experimentalFrameSerial += 1;
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
  if (result.abstained || !result.label) {
    lastExperimentalPrediction = result;
    document.getElementById('detectedWord').textContent = 'Unsure / unsupported';
    document.getElementById('signMeta').textContent = `Experimental · confidence score ${percent}% · abstained`;
    experimentalMessage('experimentalPrediction', `ABSTAINED · no word accepted · confidence score ${percent}% · ${validationStatus} · model ${result.model_version}. The score is not calibrated accuracy.`, 'warning');
  } else {
    lastExperimentalPrediction = result;
    document.getElementById('detectedWord').textContent = result.label;
    document.getElementById('signMeta').textContent = `Experimental · confidence score ${percent}% · not abstained`;
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
  let nextWindowSerial = experimentalFrameSerial + 30;
  while (sequenceCollecting && runId === sequenceRunId &&
      experimentalStream && experimentalFramePumpActive) {
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
        experimentalStream && experimentalFramePumpActive) {
      await new Promise((resolve) => setTimeout(resolve, 50));
    }
    if (!sequenceCollecting || runId !== sequenceRunId) return;
    if (!experimentalStream || !experimentalFramePumpActive) {
      pauseAiSequence(false);
      aiSentenceMessage('aiSentenceStatus', 'Sequence capture paused because the camera frame stream stopped. Restart the camera to continue.', 'error');
      return;
    }
    if (experimentalFrameBuffer.length < 30) continue;
    const frames = experimentalFrameBuffer.slice(-30);
    const requestedAtFrame = experimentalFrameSerial;
    experimentalBusy = true;
    updateSequenceControls();
    try {
      const result = await experimentalApi('/api/experimental/predict', {
        method: 'POST',
        body: JSON.stringify({
          scope: context.scope,
          artifact_id: null,
          frames
        })
      });
      if (!sequenceCollecting || runId !== sequenceRunId) return;
      validateExperimentalPrediction(result, context.scope, context.artifactId);
      renderExperimentalPrediction(result);
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
    } finally {
      experimentalBusy = false;
      updateSequenceControls();
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
  h.addEventListener('click', () => m.classList.toggle('open'));
  m.querySelectorAll('a').forEach(a => a.addEventListener('click', () => m.classList.remove('open')));
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

// ── TRICOLOR DOTS ANIMATION ──
function initDots() {
  const canvas = document.getElementById('dotsCanvas');
  const ctx    = canvas.getContext('2d');

  // Indian flag tricolour + navy (Ashoka Chakra)
  const COLORS = [
    { r: 255, g: 153, b:  51, a: 0.55 },   // Saffron  #FF9933
    { r: 255, g: 255, b: 255, a: 0.35 },   // White    #FFFFFF
    { r:  19, g: 136, b:   8, a: 0.55 },   // Green    #138808
    { r:   0, g:   0, b: 128, a: 0.40 },   // Navy     #000080 (chakra)
  ];

  const NUM_DOTS    = 110;
  const CONNECT_DIST = 140;   // px — max distance to draw a connecting line
  const SPEED       = 0.35;   // base speed

  let W, H, dots = [];

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }

  function makeDot() {
    const c = COLORS[Math.floor(Math.random() * COLORS.length)];
    const angle = Math.random() * Math.PI * 2;
    const speed = SPEED * (0.4 + Math.random() * 0.6);
    return {
      x:  Math.random() * W,
      y:  Math.random() * H,
      vx: Math.cos(angle) * speed,
      vy: Math.sin(angle) * speed,
      r:  1.2 + Math.random() * 2,       // dot radius
      c,
      pulse: Math.random() * Math.PI * 2,
      pulseSpeed: 0.012 + Math.random() * 0.016,
    };
  }

  function init() {
    resize();
    dots = Array.from({ length: NUM_DOTS }, makeDot);
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);

    // Update positions
    dots.forEach(d => {
      d.x  += d.vx;
      d.y  += d.vy;
      d.pulse += d.pulseSpeed;

      // Wrap edges
      if (d.x < -10) d.x = W + 10;
      if (d.x > W + 10) d.x = -10;
      if (d.y < -10) d.y = H + 10;
      if (d.y > H + 10) d.y = -10;
    });

    // Draw connecting lines between nearby dots
    for (let i = 0; i < dots.length; i++) {
      for (let j = i + 1; j < dots.length; j++) {
        const a = dots[i], b = dots[j];
        const dx = a.x - b.x, dy = a.y - b.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < CONNECT_DIST) {
          const fade = 1 - dist / CONNECT_DIST;
          // Blend the two dot colours for the line
          const cr = Math.round((a.c.r + b.c.r) / 2);
          const cg = Math.round((a.c.g + b.c.g) / 2);
          const cb = Math.round((a.c.b + b.c.b) / 2);
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.strokeStyle = `rgba(${cr},${cg},${cb},${fade * 0.18})`;
          ctx.lineWidth   = 0.75;
          ctx.stroke();
        }
      }
    }

    // Draw dots
    dots.forEach(d => {
      const pulse  = Math.sin(d.pulse);
      const radius = d.r * (1 + pulse * 0.25);
      const alpha  = d.c.a * (0.75 + pulse * 0.25);
      const { r, g, b } = d.c;

      // Soft glow
      const grd = ctx.createRadialGradient(d.x, d.y, 0, d.x, d.y, radius * 4);
      grd.addColorStop(0,   `rgba(${r},${g},${b},${alpha * 0.5})`);
      grd.addColorStop(0.4, `rgba(${r},${g},${b},${alpha * 0.15})`);
      grd.addColorStop(1,   `rgba(${r},${g},${b},0)`);
      ctx.beginPath();
      ctx.arc(d.x, d.y, radius * 4, 0, Math.PI * 2);
      ctx.fillStyle = grd;
      ctx.fill();

      // Core dot
      ctx.beginPath();
      ctx.arc(d.x, d.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${r},${g},${b},${alpha})`;
      ctx.fill();
    });

    requestAnimationFrame(draw);
  }

  init();
  draw();
  window.addEventListener('resize', () => { resize(); });
}

// ── CAMERA TOGGLE ──

async function toggleCamera() {
  const recognizer = window.knnRecognizer;

  if (!recognizer) {
    console.error('KNN recognizer bundle is not loaded.');
    alert('KNN recognizer failed to load. Check the browser console.');
    return;
  }

  if (cameraTransition) return;
  cameraTransition = true;
  const button = document.getElementById('btnCam');
  if (button) button.disabled = true;

  try {
    if (cameraActive) {
      stopExperimentalCamera();
      if (!recognizer.stopCamera()) return;
      if (sequenceCollecting && sequenceContext?.source === 'browser_knn') {
        pauseAiSequence(false);
        aiSentenceMessage('aiSentenceStatus', 'Browser camera stopped; sequence capture paused. Reviewed signs are preserved.');
      }

      const video = document.getElementById('knnVideo');
      const image = document.getElementById('videoStream');

      if (video) video.style.display = 'none';
      if (image) image.style.display = 'block';

      stopUI();
    } else {
      if (experimentalStream) throw new Error('Stop the experimental camera before starting the browser KNN camera.');
      const statusResponse = await fetch(`${API}/status`);
      if (!statusResponse.ok) {
        throw new Error(`Could not check the Python camera status (HTTP ${statusResponse.status}).`);
      }
      const backendStatus = await statusResponse.json();
      if (backendStatus.camera_active) {
        throw new Error('Stop the Python webcam before starting the browser webcam.');
      }

      if (!await recognizer.startCamera()) return;

      const video = document.getElementById('knnVideo');
      const image = document.getElementById('videoStream');

      if (video?.srcObject) {
        if (image) image.style.display = 'none';

        video.style.display = 'block';
        video.style.transform = 'scaleX(-1)';
        experimentalStream = video.srcObject;
        startExperimentalFramePump();

        startUI(true);
      }
    }
  } catch (error) {
    console.error('[KNN] Could not toggle browser camera:', error);
    window.dispatchEvent(new CustomEvent('knn-status', {
      detail: `Camera error: ${error.message}`
    }));
  } finally {
    cameraTransition = false;
    if (button) button.disabled = false;
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
      confidence < 0 || confidence > 1) return;

  const wordElement = document.getElementById('detectedWord');
  const metaElement = document.getElementById('signMeta');

  if (wordElement) {
    wordElement.textContent = word === 'OTHER' ? 'Unsure / unsupported' : word;
  }
  if (metaElement) {
    metaElement.textContent = word === 'OTHER'
      ? 'Browser KNN · OTHER class · not a sign'
      : `Browser KNN · raw score ${(confidence * 100).toFixed(1)}% · uncalibrated`;
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
