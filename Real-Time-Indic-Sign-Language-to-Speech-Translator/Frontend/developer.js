(() => {
  "use strict";

  const API_HOSTS = new Set(["localhost", "127.0.0.1", "[::1]"]);
  const API_HOST = API_HOSTS.has(window.location.hostname)
    ? window.location.hostname
    : "localhost";
  const API = `http://${API_HOST}:5000`;
  const FRAME_COUNT = 30;
  const PREPROCESSING_VERSION = "isl-holistic-30x378-v1";
  const video = document.getElementById("cameraPreview");
  const canvas = document.getElementById("captureCanvas");
  const context = canvas.getContext("2d", { willReadFrequently: false });
  const pending = [];
  const selectedSavedSamples = new Set();
  let stream = null;
  let activeDraft = null;
  let capturing = false;
  let startingSession = false;
  let sessionSelectionUncertain = false;
  let leaseHeartbeat = null;

  const byId = (id) => document.getElementById(id);

  async function api(path, options = {}) {
    const url = `${API}${path}`;
    const method = (options.method || "GET").toUpperCase();
    let response;
    try {
      response = await fetch(url, {
        ...options,
        headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      });
    } catch (error) {
      console.error("[Developer Studio] Request could not reach the API", {
        method,
        url,
        error: error.message,
      });
      throw error;
    }
    const result = await response.json().catch(() => ({}));
    if (!response.ok) {
      console.warn("[Developer Studio] API request returned an error", {
        method,
        url,
        status: response.status,
      });
      throw new Error(result.error || `Request failed (${response.status})`);
    }
    return result;
  }

  function setMessage(id, text, kind = "") {
    const element = byId(id);
    element.textContent = text;
    element.dataset.kind = kind;
  }

  function updateActiveDraftDisplay() {
    if (!activeDraft) return;
    byId("activeDraftMeta").textContent =
      `${activeDraft.scope} scope · label ID ${activeDraft.label_id} · ` +
      `${activeDraft.preprocessing_version} · ${activeDraft.sample_count} saved · ` +
      `${activeDraft.session_count || 0} recorded session(s) · ` +
      `${activeDraft.training_ready ? "training ready" : `${Math.max(0, activeDraft.minimum_training_sessions - (activeDraft.session_count || 0))} more session(s) for training`} · ` +
      `${activeDraft.evaluation_ready ? "evaluation ready" : `${Math.max(0, activeDraft.minimum_evaluation_sessions - (activeDraft.session_count || 0))} more session(s) for evaluation`}`;
    const sessionId = activeDraft.session_id;
    byId("activeSessionMeta").textContent = sessionId
      ? `Current session ${sessionId.slice(0, 12)} · ${activeDraft.session_sample_count || 0} saved example(s) in this session`
      : "No capture session is available. Resume this draft to create one.";
    byId("newSession").disabled = !sessionId || capturing || startingSession || pending.length > 0 || sessionSelectionUncertain;
    byId("captureButton").disabled = !stream || capturing || startingSession || sessionSelectionUncertain || !sessionId;
    byId("saveDraft").disabled = pending.length === 0 || sessionSelectionUncertain;
  }

  function setActiveDraft(draft, sessionId) {
    const sameDraft = activeDraft?.entry_id === draft.entry_id &&
      activeDraft?.scope === draft.scope;
    const nextSession = sessionId || draft.session_id;
    if (startingSession && !sameDraft) {
      setMessage("draftMessage", "Wait for the new session request to finish before switching drafts.", "warning");
      return false;
    }
    if (pending.length && (!sameDraft || activeDraft?.session_id !== nextSession)) {
      setMessage(
        "draftMessage",
        "Save or discard pending examples before switching drafts or sessions.",
        "warning"
      );
      return false;
    }
    sessionSelectionUncertain = false;
    activeDraft = { ...draft, session_id: nextSession };
    byId("activeDraftTitle").textContent = draft.label;
    updateActiveDraftDisplay();
    refreshDrafts().catch((error) => setMessage("draftMessage", error.message, "error"));
    refreshSavedSamples().catch((error) => setMessage("draftMessage", error.message, "error"));
    return true;
  }

  async function refreshDrafts() {
    const scopes = ["shared", "personal"];
    const all = [];
    for (const scope of scopes) {
      const result = await api(`/developer/api/drafts?scope=${scope}`);
      all.push(...result.drafts);
    }
    const list = byId("draftList");
    list.replaceChildren();
    if (!all.length) {
      list.className = "draft-list empty-state";
      list.textContent = "No saved drafts yet.";
      return;
    }
    list.className = "draft-list";
    for (const draft of all) {
      const card = document.createElement("article");
      card.className = "draft-card";
      card.dataset.active = String(activeDraft?.entry_id === draft.entry_id &&
        activeDraft?.scope === draft.scope);
      const openDraft = document.createElement("button");
      openDraft.type = "button";
      openDraft.className = "draft-card-open";
      openDraft.innerHTML = `<span class="draft-card-head"><span class="draft-name"></span><span class="scope-label"></span></span><span class="draft-meta"></span>`;
      openDraft.querySelector(".draft-name").textContent = draft.label;
      openDraft.querySelector(".scope-label").textContent = draft.scope;
      openDraft.querySelector(".draft-meta").textContent =
        `${draft.sample_count} saved · ${draft.session_count} recorded session(s) · ` +
        `${draft.session_count}/${draft.minimum_training_sessions} training sessions`;
      openDraft.addEventListener("click", () => {
        if (draft.session_id) {
          setActiveDraft(draft, draft.session_id);
        } else {
          resumeDraft(draft).catch((error) =>
            setMessage("draftMessage", error.message, "error")
          );
        }
      });
      const deleteButton = document.createElement("button");
      deleteButton.type = "button";
      deleteButton.className = "button button-danger small-button delete-word-button";
      deleteButton.textContent = "Delete";
      deleteButton.disabled = capturing;
      deleteButton.setAttribute("aria-label", `Delete ${draft.label} (${draft.scope})`);
      deleteButton.addEventListener("click", () => {
        deleteDraft(draft, deleteButton).catch((error) =>
          setMessage("draftMessage", error.message, "error")
        );
      });
      card.append(openDraft, deleteButton);
      list.append(card);
    }
  }

  async function deleteDraft(draft, button) {
    if (capturing) {
      setMessage("draftMessage", "Wait for capture to finish before deleting a word.", "warning");
      return;
    }
    let latestDraft;
    try {
      const result = await api(`/developer/api/drafts?scope=${draft.scope}`);
      latestDraft = result.drafts.find((item) => item.entry_id === draft.entry_id);
    } catch (error) {
      setMessage("draftMessage", error.message, "error");
      return;
    }
    if (!latestDraft) {
      setMessage("draftMessage", "This vocabulary word is no longer in the selected scope.", "error");
      return;
    }

    const modelReferences = latestDraft.model_references || [];
    const knownArtifacts = modelReferences
      .filter((item) => !item.unreadable)
      .map((item) => item.artifact_id);
    const unreadableArtifacts = modelReferences.filter((item) => item.unreadable);
    const modelWarning = modelReferences.length
      ? `\n\nWarning: Existing model artifacts will remain unchanged and may still ` +
        `recognize this word${knownArtifacts.length ? ` (references: ${knownArtifacts.join(", ")})` : ""}.` +
        (unreadableArtifacts.length
          ? ` ${unreadableArtifacts.length} artifact label map(s) could not be inspected.`
          : "")
      : "";
    const hasPending = pending.some((item) =>
      item.entry_id === draft.entry_id && item.scope === draft.scope
    );
    const pendingWarning = hasPending
      ? " Any unsaved captures for this word in this page will also be discarded."
      : "";
    const confirmed = window.confirm(
      `Delete the vocabulary word “${latestDraft.label}” (${latestDraft.scope})?\n\n` +
      "This permanently removes the word's draft record, capture sessions, saved examples, " +
      "and their stored sequence files." + pendingWarning + modelWarning
    );
    if (!confirmed) return;

    button.disabled = true;
    try {
      const result = await api(
        `/developer/api/drafts/${encodeURIComponent(draft.entry_id)}`,
        {
          method: "DELETE",
          body: JSON.stringify({ scope: draft.scope, confirmed: true }),
        }
      );
      for (let index = pending.length - 1; index >= 0; index -= 1) {
        if (pending[index].entry_id === draft.entry_id &&
            pending[index].scope === draft.scope) {
          pending.splice(index, 1);
        }
      }
      renderPending();
      if (activeDraft?.entry_id === draft.entry_id && activeDraft?.scope === draft.scope) {
        activeDraft = null;
        byId("activeDraftTitle").textContent = "No draft selected";
        byId("activeDraftMeta").textContent = "Create a draft or select one from the list.";
        byId("activeSessionMeta").textContent = "Select a draft to see its recording session.";
        byId("newSession").disabled = true;
        byId("captureButton").disabled = true;
      }
      await refreshDrafts();
      await refreshSavedSamples();
      const modelNote = result.model_references.length
        ? " Existing model artifacts were left unchanged."
        : "";
      setMessage(
        "draftMessage",
        `Deleted “${result.deleted.label}” (${draft.scope}), including ` +
          `${result.deleted.deleted_samples} saved examples and ` +
          `${result.deleted.deleted_sessions} sessions.${modelNote}`,
        "good"
      );
    } catch (error) {
      button.disabled = false;
      setMessage("draftMessage", error.message, "error");
    }
  }

  async function resumeDraft(draft) {
    const result = await api("/developer/api/drafts", {
      method: "POST",
      body: JSON.stringify({ label: draft.label, scope: draft.scope }),
    });
    setActiveDraft(result.draft, result.session_id);
  }

  async function startNewSession() {
    if (!activeDraft) throw new Error("Select a vocabulary draft first.");
    if (capturing) throw new Error("Wait for the current capture to finish first.");
    if (pending.length) {
      throw new Error("Save or discard pending examples before starting a new session.");
    }
    const previous = activeDraft;
    startingSession = true;
    updateActiveDraftDisplay();
    try {
      const result = await api(
        `/developer/api/drafts/${encodeURIComponent(previous.entry_id)}/sessions`,
        {
          method: "POST",
          body: JSON.stringify({ scope: previous.scope }),
        }
      );
      if (!result.session_id || result.session_id === previous.session_id) {
        throw new Error("The server did not confirm a new capture session. Refresh and select the draft before capturing.");
      }
      setActiveDraft(result.draft, result.session_id);
      setMessage(
        "draftMessage",
        `Started a new session for ${previous.label}. Existing examples were not changed.`,
        "good"
      );
    } catch (error) {
      // A lost response can follow a successful server write; require an explicit refresh/reselection.
      sessionSelectionUncertain = true;
      updateActiveDraftDisplay();
      throw new Error(`${error.message} Refresh drafts and select the word before capturing to confirm its session.`);
    } finally {
      startingSession = false;
      updateActiveDraftDisplay();
    }
  }

  async function refreshSavedSamples() {
    const requestedDraft = activeDraft;
    const list = byId("savedSamples");
    selectedSavedSamples.clear();
    byId("deleteSavedSamples").disabled = true;
    list.replaceChildren();
    if (!requestedDraft) {
      list.className = "sample-list empty-state";
      list.textContent = "Saved examples appear here after reopening a draft.";
      return;
    }
    const result = await api(
      `/developer/api/drafts/${encodeURIComponent(requestedDraft.entry_id)}/samples?scope=${requestedDraft.scope}`
    );
    if (activeDraft !== requestedDraft) return;
    activeDraft.session_sample_count = result.samples.filter(
      (sample) => sample.session_id === activeDraft.session_id
    ).length;
    if (!result.samples.length) {
      list.className = "sample-list empty-state";
      list.textContent = "No examples have been saved to this draft yet.";
      updateActiveDraftDisplay();
      return;
    }
    list.className = "sample-list";
    result.samples.forEach((sample, index) => {
      const card = document.createElement("article");
      card.className = "sample-card";
      const head = document.createElement("div");
      head.className = "sample-card-head";
      const title = document.createElement("span");
      title.className = "sample-title";
      title.textContent = `Saved example ${index + 1} · ${sample.sample_id.slice(0, 12)}`;
      const scope = document.createElement("span");
      scope.className = "scope-label";
      scope.textContent = sample.hand_frames === FRAME_COUNT ? "quality OK" : "review";
      const selectLabel = document.createElement("label");
      selectLabel.className = "sample-select";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.value = sample.sample_id;
      checkbox.setAttribute("aria-label", `Select saved example ${index + 1} for deletion`);
      checkbox.addEventListener("change", () => {
        if (checkbox.checked) selectedSavedSamples.add(sample.sample_id);
        else selectedSavedSamples.delete(sample.sample_id);
        byId("deleteSavedSamples").disabled = selectedSavedSamples.size === 0;
      });
      selectLabel.append(checkbox, document.createTextNode("Delete"));
      head.append(title, scope, selectLabel);
      const meta = document.createElement("div");
      meta.className = "sample-meta";
      meta.textContent = `${sample.hand_frames}/${sample.frames} hand-visible frames · ${sample.created_at} · SHA-256 ${sample.sha256.slice(0, 12)}…`;
      card.append(head, meta);
      list.append(card);
    });
    updateActiveDraftDisplay();
  }

  function renderPending() {
    const container = byId("pendingSamples");
    container.replaceChildren();
    byId("pendingCount").textContent = String(pending.length);
    byId("saveDraft").disabled = pending.length === 0 || !activeDraft;
    if (!pending.length) {
      container.className = "sample-list empty-state";
      container.textContent = "Capture a sequence to review it here.";
      updateActiveDraftDisplay();
      return;
    }
    container.className = "sample-list";
    pending.forEach((sample, index) => {
      const card = document.createElement("article");
      card.className = "sample-card";
      const head = document.createElement("div");
      head.className = "sample-card-head";
      const title = document.createElement("span");
      title.className = "sample-title";
      title.textContent = `Pending example ${index + 1}`;
      const discard = document.createElement("button");
      discard.type = "button";
      discard.className = "button button-quiet small-button";
      discard.textContent = "Discard / re-record";
      discard.addEventListener("click", () => {
        pending.splice(index, 1);
        renderPending();
      });
      head.append(title, discard);
      const meta = document.createElement("div");
      meta.className = "sample-meta";
      meta.textContent = `${sample.quality.hand_frames}/${sample.quality.frames} frames have hand landmarks · ${sample.quality.preprocessing_version}`;
      const strip = document.createElement("div");
      strip.className = "frame-strip";
      sample.previewFrames.forEach((frame) => {
        const image = document.createElement("img");
        image.src = frame;
        image.alt = "Temporary webcam frame from reviewed capture";
        strip.append(image);
      });
      card.append(head, meta, strip);
      container.append(card);
    });
    updateActiveDraftDisplay();
  }

  async function enableCamera() {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("This browser does not support webcam access. Use localhost in a modern browser.");
    }
    await api("/developer/api/capture/lease", {
      method: "POST",
      body: JSON.stringify({ active: true }),
    });
    leaseHeartbeat = window.setInterval(() => {
      api("/developer/api/capture/lease", {
        method: "POST",
        body: JSON.stringify({ active: true }),
      }).catch((error) => setMessage("captureMessage", error.message, "error"));
    }, 20_000);
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
      });
      video.srcObject = stream;
      await video.play();
      byId("videoPlaceholder").hidden = true;
      byId("cameraState").textContent = "Camera on";
      byId("cameraState").dataset.state = "on";
      byId("startCamera").disabled = true;
      byId("stopCamera").disabled = false;
      byId("captureButton").disabled = !activeDraft;
      setMessage("captureMessage", "Camera ready. Frame samples are sent locally for landmark extraction.");
    } catch (error) {
      window.clearInterval(leaseHeartbeat);
      leaseHeartbeat = null;
      if (stream) stream.getTracks().forEach((track) => track.stop());
      stream = null;
      await api("/developer/api/capture/lease", {
        method: "POST",
        body: JSON.stringify({ active: false }),
      });
      throw error;
    }
  }

  function stopCamera() {
    if (capturing) return;
    if (stream) stream.getTracks().forEach((track) => track.stop());
    if (leaseHeartbeat) window.clearInterval(leaseHeartbeat);
    leaseHeartbeat = null;
    stream = null;
    video.srcObject = null;
    byId("videoPlaceholder").hidden = false;
    byId("cameraState").textContent = "Camera off";
    byId("cameraState").dataset.state = "off";
    byId("startCamera").disabled = false;
    byId("stopCamera").disabled = true;
    byId("captureButton").disabled = true;
    api("/developer/api/capture/lease", {
      method: "POST",
      body: JSON.stringify({ active: false }),
    }).catch((error) => setMessage("captureMessage", error.message, "error"));
    setMessage("captureMessage", "Camera stopped.");
  }

  function waitForVideoFrame() {
    return new Promise((resolve) => {
      if (video.requestVideoFrameCallback) {
        video.requestVideoFrameCallback(() => resolve());
      } else {
        requestAnimationFrame(() => resolve());
      }
    });
  }

  async function captureSequence() {
    if (sessionSelectionUncertain) {
      throw new Error("Refresh drafts and select the word again to confirm the active session.");
    }
    if (!activeDraft) throw new Error("Create or select a draft before capture.");
    if (!stream || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      throw new Error("Enable the webcam before starting capture.");
    }
    capturing = true;
    updateActiveDraftDisplay();
    byId("captureButton").disabled = true;
    byId("startCamera").disabled = true;
    byId("stopCamera").disabled = true;
    byId("captureProgress").value = 0;
    const frames = [];
    const previewFrames = [];
    canvas.width = 320;
    canvas.height = 240;
    const previewIndices = new Set([0, 7, 14, 21, 29]);
    try {
      for (let index = 0; index < FRAME_COUNT; index += 1) {
        await waitForVideoFrame();
        context.save();
        context.translate(canvas.width, 0);
        context.scale(-1, 1);
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        context.restore();
        const frame = canvas.toDataURL("image/jpeg", 0.62);
        frames.push(frame);
        if (previewIndices.has(index)) previewFrames.push(frame);
        byId("captureProgress").value = index + 1;
        byId("progressCount").textContent = `${index + 1} / ${FRAME_COUNT}`;
        byId("progressText").textContent = "Collecting complete sequence";
        await new Promise((resolve) => setTimeout(resolve, 90));
      }
      byId("progressText").textContent = "Checking landmarks…";
      const result = await api("/developer/api/capture/process", {
        method: "POST",
        body: JSON.stringify({ frames }),
      });
      pending.push({
        entry_id: activeDraft.entry_id,
        scope: activeDraft.scope,
        sequence: result.sequence,
        quality: result.quality,
        previewFrames,
      });
      renderPending();
      setMessage(
        "captureMessage",
        `Sequence passed: hands visible in ${result.quality.hand_frames}/${result.quality.frames} frames.`,
        "good"
      );
    } finally {
      capturing = false;
      updateActiveDraftDisplay();
      byId("startCamera").disabled = Boolean(stream);
      byId("stopCamera").disabled = !stream;
    }
  }

  async function saveDraftSamples() {
    if (sessionSelectionUncertain) {
      throw new Error("Refresh drafts and select the word again to confirm the active session.");
    }
    if (!activeDraft) throw new Error("Select a draft first.");
    while (pending.length) {
      const sample = pending[0];
      if (sample.entry_id !== activeDraft.entry_id ||
          sample.scope !== activeDraft.scope) {
        throw new Error("Pending sequence is tied to a different draft.");
      }
      await api(
        `/developer/api/drafts/${encodeURIComponent(activeDraft.entry_id)}/samples`,
        {
          method: "POST",
          body: JSON.stringify({
            scope: activeDraft.scope,
            session_id: activeDraft.session_id,
            preprocessing_version: PREPROCESSING_VERSION,
            sequence: sample.sequence,
          }),
        }
      );
      pending.shift();
      renderPending();
    }
    await refreshDrafts();
    const response = await api(`/developer/api/drafts?scope=${activeDraft.scope}`);
    const refreshed = response.drafts.find((draft) => draft.entry_id === activeDraft.entry_id);
    if (refreshed) activeDraft = { ...refreshed, session_id: activeDraft.session_id };
    updateActiveDraftDisplay();
    await refreshSavedSamples();
    setMessage("draftMessage", "Accepted examples saved to this local draft.", "good");
    await refreshTrainingLab().catch(() => {});
  }

  async function deleteSelectedSamples() {
    if (!activeDraft || selectedSavedSamples.size === 0) return;
    const sampleIds = [...selectedSavedSamples];
    const confirmed = window.confirm(
      `Permanently delete ${sampleIds.length} selected example(s) from ` +
      `"${activeDraft.label}" (${activeDraft.scope})? This cannot be undone. ` +
      "The draft, sessions, and other examples will remain."
    );
    if (!confirmed) return;

    try {
      const result = await api(
        `/developer/api/drafts/${encodeURIComponent(activeDraft.entry_id)}/samples`,
        {
          method: "DELETE",
          body: JSON.stringify({ scope: activeDraft.scope, sample_ids: sampleIds }),
        }
      );
      activeDraft.sample_count = result.sample_count;
      updateActiveDraftDisplay();
      await refreshDrafts();
      await refreshSavedSamples();
      await refreshTrainingLab().catch(() => {});
      setMessage(
        "draftMessage",
        `Permanently deleted ${result.deleted_count} selected example(s). The draft and remaining examples are unchanged.`,
        "good"
      );
    } catch (error) {
      setMessage("draftMessage", error.message, "error");
    }
  }

  function updateTrainingCandidateControls() {
    const select = byId("trainingCandidate");
    const candidates = JSON.parse(select.dataset.candidates || "[]");
    const candidate = candidates.find((item) => item.artifact_id === select.value);
    byId("activateModelCandidate").disabled = !candidate ||
      candidate.active || !candidate.evaluation_valid ||
      !candidate.data_current || !candidate.vocabulary_matches ||
      !candidate.preprocessing_compatible || Boolean(candidate.invalid);
  }

  async function refreshTrainingLab() {
    const scope = byId("trainingScope").value;
    setMessage("trainingLabStatus", "Checking the latest saved vocabulary and model state…");
    byId("trainModelCandidate").disabled = true;
    try {
      const data = await api(`/api/vocabulary?scope=${encodeURIComponent(scope)}`);
      const list = byId("trainingVocabulary");
      list.replaceChildren();
      for (const word of data.words || []) {
        const item = document.createElement("article");
        item.className = "sample-row";
        const title = document.createElement("strong");
        title.textContent = `${word.label} · ${word.sample_count} example(s) · ${word.session_count} session(s)`;
        const state = document.createElement("span");
        state.textContent = word.recognizable_by_active_model
          ? "Included in current active model"
          : "Not included in current active model";
        item.append(title, state);
        list.append(item);
      }

      const active = data.active_model;
      const activeMessage = active?.usable
        ? `Active ${scope} model ${active.artifact_id}: ${active.supports.length}/${data.words.length} current labels; data and complete validation report match.`
        : `No usable active ${scope} model. ${active?.unusable_reason || "Train and validate a candidate before activation."}`;
      setMessage("trainingLabStatus", `${data.words.length} saved vocabulary word(s). ${activeMessage}`, active?.usable ? "good" : "warning");

      const select = byId("trainingCandidate");
      const previous = select.value;
      const candidates = data.candidates || [];
      select.replaceChildren();
      if (!candidates.length) select.append(new Option("No candidates yet", ""));
      for (const candidate of candidates) {
        const evaluation = candidate.validation &&
          Number.isFinite(candidate.validation.accuracy)
          ? `${(candidate.validation.accuracy * 100).toFixed(1)}% held-out accuracy`
          : "no held-out evaluation";
        const state = [
          candidate.active ? "active" : "",
          candidate.evaluation_valid ? "validated" : "UNVALIDATED",
          candidate.data_current ? "data current" : "data stale",
          candidate.vocabulary_matches ? "labels current" : "labels stale",
          evaluation,
        ].filter(Boolean).join(" · ");
        select.append(new Option(`${candidate.artifact_id} (${state})`, candidate.artifact_id));
      }
      select.disabled = candidates.length === 0;
      if (candidates.some((candidate) => candidate.artifact_id === previous)) {
        select.value = previous;
      }
      select.dataset.candidates = JSON.stringify(candidates);
      const canTrain = data.words.length >= 2 &&
        data.words.every((word) => word.training_ready);
      byId("trainModelCandidate").disabled = !canTrain;
      const selected = candidates.find((candidate) => candidate.artifact_id === select.value);
      byId("trainingCandidateStatus").textContent = selected
        ? `Label map: ${selected.supported_label_count || selected.label_map?.length || 0}/${data.words.length} · ` +
          `${selected.evaluation_valid ? "complete session-separated validation report present" : "activation blocked: complete session-separated validation is missing"}`
        : "Candidates are never activated automatically.";
      updateTrainingCandidateControls();
      return data;
    } catch (error) {
      setMessage("trainingLabStatus", `Could not refresh model state: ${error.message}`, "error");
      throw error;
    }
  }

  async function trainModelCandidate() {
    const button = byId("trainModelCandidate");
    button.disabled = true;
    setMessage("trainingCandidateStatus", "Training and evaluating a separate candidate from saved sequences. The active model will not change…");
    try {
      const result = await api("/api/experimental/train", {
        method: "POST",
        body: JSON.stringify({ scope: byId("trainingScope").value }),
      });
      setMessage(
        "trainingCandidateStatus",
        result.candidate.evaluation_ready
          ? `Candidate ${result.candidate.artifact_id} trained with session-separated evaluation. Review its report and activate explicitly if appropriate.`
          : `Candidate ${result.candidate.artifact_id} trained but is UNVALIDATED. More independent sessions are required before activation.`,
        result.candidate.evaluation_ready ? "good" : "warning"
      );
      await refreshTrainingLab();
      byId("trainingCandidate").value = result.candidate.artifact_id;
      updateTrainingCandidateControls();
    } catch (error) {
      setMessage("trainingCandidateStatus", `Candidate training failed: ${error.message}`, "error");
      await refreshTrainingLab().catch(() => {});
    }
  }

  async function activateModelCandidate() {
    const artifactId = byId("trainingCandidate").value;
    if (!artifactId) return;
    const button = byId("activateModelCandidate");
    button.disabled = true;
    try {
      const result = await api(`/api/experimental/activate/${encodeURIComponent(artifactId)}`, {
        method: "POST",
        body: JSON.stringify({ scope: byId("trainingScope").value }),
      });
      setMessage("trainingCandidateStatus", `Activated ${result.artifact_id}. The main translator will refresh the active model state automatically.`, "good");
      await refreshTrainingLab();
    } catch (error) {
      setMessage("trainingCandidateStatus", `Candidate was not activated: ${error.message}`, "error");
      await refreshTrainingLab().catch(() => {});
    }
  }

  byId("draftForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (pending.length) {
      setMessage(
        "draftMessage",
        "Save or discard pending examples before creating or switching drafts.",
        "warning"
      );
      return;
    }
    setMessage("draftMessage", "Creating or opening draft…");
    try {
      const result = await api("/developer/api/drafts", {
        method: "POST",
        body: JSON.stringify({
          label: byId("wordLabel").value,
          scope: byId("scopeSelect").value,
        }),
      });
      if (!setActiveDraft(result.draft, result.session_id)) return;
      setMessage("draftMessage", `Draft ready: ${result.draft.label} (${result.draft.scope}).`, "good");
    } catch (error) {
      setMessage("draftMessage", error.message, "error");
    }
  });

  byId("startCamera").addEventListener("click", () => {
    enableCamera().catch((error) => setMessage("captureMessage", error.message, "error"));
  });
  byId("stopCamera").addEventListener("click", stopCamera);
  byId("captureButton").addEventListener("click", () => {
    captureSequence().catch((error) => setMessage("captureMessage", error.message, "error"));
  });
  byId("saveDraft").addEventListener("click", () => {
    saveDraftSamples().catch((error) => setMessage("draftMessage", error.message, "error"));
  });
  byId("newSession").addEventListener("click", () => {
    startNewSession().catch((error) => setMessage("draftMessage", error.message, "error"));
  });
  byId("deleteSavedSamples").addEventListener("click", () => {
    deleteSelectedSamples().catch((error) => setMessage("draftMessage", error.message, "error"));
  });
  byId("trainingScope").addEventListener("change", () => {
    refreshTrainingLab().catch(() => {});
  });
  byId("refreshTrainingLab").addEventListener("click", () => {
    refreshTrainingLab().catch(() => {});
  });
  byId("trainingCandidate").addEventListener("change", updateTrainingCandidateControls);
  byId("trainModelCandidate").addEventListener("click", trainModelCandidate);
  byId("activateModelCandidate").addEventListener("click", activateModelCandidate);
  byId("refreshDrafts").addEventListener("click", () => {
    refreshDrafts().catch((error) => setMessage("draftMessage", error.message, "error"));
  });
  window.addEventListener("beforeunload", () => {
    if (stream) stream.getTracks().forEach((track) => track.stop());
    if (leaseHeartbeat) window.clearInterval(leaseHeartbeat);
    navigator.sendBeacon(
      `${API}/developer/api/capture/lease`,
      new Blob([JSON.stringify({ active: false })], { type: "application/json" })
    );
  });

  refreshDrafts().catch((error) => setMessage("draftMessage", error.message, "error"));
  refreshTrainingLab().catch(() => {});
})();
