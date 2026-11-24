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
  const builtinPending = [];
  const selectedSavedSamples = new Set();
  let stream = null;
  let activeDraft = null;
  let activeBuiltinSession = null;
  let sentenceMappingRequestId = 0;
  let builtinMappingRequestId = 0;
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

  async function loadBuiltinLabels() {
    try {
      await refreshBuiltinTrainingData();
      setMessage("builtinMappingStatus", "Choose an existing label to view or edit its sentence mapping.");
    } catch (error) {
      byId("builtinMappingLabel").replaceChildren(new Option("Built-in labels unavailable", ""));
      byId("builtinCaptureLabel").replaceChildren(new Option("Built-in labels unavailable", ""));
      setMessage("builtinMappingStatus", error.message, "error");
      setMessage("builtinCaptureStatus", error.message, "error");
    }
  }

  async function refreshBuiltinTrainingData() {
    const result = await api("/developer/api/builtin-training-data");
    const labels = result.labels || [];
    for (const [id, prompt] of [
      ["builtinMappingLabel", "Select a built-in label…"],
      ["builtinCaptureLabel", "Select a built-in label…"],
    ]) {
      const select = byId(id);
      const previous = select.value;
      select.replaceChildren(new Option(prompt, ""));
      for (const item of labels) select.add(new Option(item.label, item.label));
      if (labels.some((item) => item.label === previous)) select.value = previous;
    }
    renderBuiltinLabelCounts(labels);
    updateBuiltinCaptureControls();
    return labels;
  }

  function renderBuiltinLabelCounts(labels) {
    const container = byId("builtinSampleCounts");
    container.replaceChildren();
    container.className = "sample-list";
    for (const item of labels) {
      const row = document.createElement("article");
      row.className = "builtin-label-row";
      const label = document.createElement("strong");
      label.className = "sample-title";
      label.textContent = item.label;
      const counts = document.createElement("span");
      counts.className = "sample-meta";
      counts.textContent = `${item.sample_count} sample(s) · ${item.session_count} session(s) with saved samples · ${item.capture_session_count} total session(s) started`;
      const eligibility = document.createElement("span");
      eligibility.className = "builtin-training-eligibility";
      eligibility.textContent = item.session_count < 4
        ? `${item.session_count}/4 sessions · need ${4 - item.session_count} more`
        : `${item.session_count}/4 sessions · eligible`;
      const actions = document.createElement("div");
      actions.className = "builtin-label-actions";
      const train = document.createElement("button");
      train.type = "button";
      train.className = "button button-secondary small-button";
      train.textContent = "Train";
      train.disabled = item.session_count < 4;
      train.title = item.session_count < 4
        ? `Disabled: needs ${4 - item.session_count} more independent session(s) with saved samples.`
        : "Train a separate candidate with this label and all other eligible shared vocabulary data.";
      train.addEventListener("click", () => {
        trainBuiltinLabel(item, train).catch((error) =>
          setMessage("builtinTrainingStatus", error.message, "error")
        );
      });
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "button button-danger small-button";
      remove.textContent = "Delete";
      remove.disabled = item.sample_count === 0 && item.capture_session_count === 0;
      remove.title = remove.disabled ? "No saved samples or recording sessions for this label." :
        `Delete only ${item.label} built-in samples and sessions.`;
      remove.addEventListener("click", () => {
        deleteBuiltinLabelData(item).catch((error) =>
          setMessage("builtinTrainingStatus", error.message, "error")
        );
      });
      actions.append(train, remove);
      row.append(label, counts, eligibility, actions);
      container.append(row);
    }
    if (!labels.length) {
      container.className = "sample-list empty-state";
      container.textContent = "No built-in vocabulary labels were found.";
    }
  }

  async function trainBuiltinLabel(item, button) {
    if (item.session_count < 4) return;
    button.disabled = true;
    setMessage(
      "builtinTrainingStatus",
      `Building the combined dataset for ${item.label} → checking independent sessions → training and evaluating a separate candidate. The active model will not change…`
    );
    try {
      const result = await api("/api/experimental/train", {
        method: "POST",
        body: JSON.stringify({ scope: "shared", target_label: item.label }),
      });
      const candidate = result.candidate;
      const excluded = candidate.data?.excluded_labels || [];
      const withheld = excluded.length
        ? ` Excluded for insufficient data: ${excluded.map((entry) => `${entry.label} (${entry.session_count}/${entry.minimum_sessions} sessions)`).join(", ")}.`
        : "";
      setMessage(
        "builtinTrainingStatus",
        candidate.evaluation_ready
          ? `${item.label} candidate ${candidate.artifact_id} trained and evaluated across separate recording sessions. Review the report; activation remains explicit.${withheld}`
          : `${item.label} candidate ${candidate.artifact_id} was saved, but evaluation is insufficient for activation: ${candidate.insufficiencies.join(" ")}${withheld}`,
        candidate.evaluation_ready ? "good" : "warning"
      );
      await refreshBuiltinTrainingData();
      await refreshTrainingLab();
      byId("trainingCandidate").value = candidate.artifact_id;
      updateTrainingCandidateControls();
    } catch (error) {
      setMessage("builtinTrainingStatus", `Training failed: ${error.message}`, "error");
      await refreshTrainingLab().catch(() => {});
    } finally {
      button.disabled = false;
      await refreshBuiltinTrainingData().catch(() => {});
    }
  }

  async function deleteBuiltinLabelData(item) {
    const confirmed = window.confirm(
      `Delete built-in training data for “${item.label}”? This permanently removes ` +
      `${item.sample_count} saved sample(s) and ${item.capture_session_count} recording session(s) ` +
      `from the dedicated built-in store. The vocabulary label, sentence mapping, drafts, legacy dataset, ` +
      `and other labels' samples will remain.`
    );
    if (!confirmed) return;
    setMessage("builtinTrainingStatus", `Deleting only ${item.label} built-in samples…`);
    try {
      const result = await api("/developer/api/builtin-training-data", {
        method: "DELETE",
        body: JSON.stringify({ label: item.label, confirmed: true }),
      });
      await refreshBuiltinTrainingData();
      setMessage(
        "builtinTrainingStatus",
        `Deleted ${result.deleted_samples} sample(s) and ${result.deleted_sessions} recording session(s) for ${result.label}. Other labels and datasets were not changed.`,
        "good"
      );
    } catch (error) {
      setMessage("builtinTrainingStatus", error.message, "error");
    }
  }

  function updateBuiltinCaptureControls() {
    const label = byId("builtinCaptureLabel").value;
    if (activeBuiltinSession && activeBuiltinSession.label !== label) {
      activeBuiltinSession = null;
      byId("builtinSessionInfo").textContent = label
        ? "Start a new recording session for this label."
        : "Select a label to begin.";
    }
    byId("startBuiltinSession").disabled = !label || capturing;
    byId("captureBuiltinSequence").disabled = !stream || capturing ||
      !activeBuiltinSession || activeBuiltinSession.label !== label;
    byId("saveBuiltinSamples").disabled = capturing || builtinPending.length === 0;
  }

  async function startBuiltinCaptureSession() {
    const label = byId("builtinCaptureLabel").value;
    if (!label) throw new Error("Select a built-in vocabulary label first.");
    const session = await api("/developer/api/builtin-training-data/sessions", {
      method: "POST",
      body: JSON.stringify({ label }),
    });
    activeBuiltinSession = session;
    byId("builtinSessionInfo").textContent =
      `Session ${session.session_id.slice(0, 12)} · ${session.label}`;
    await refreshBuiltinTrainingData();
    setMessage("builtinCaptureStatus", "New session ready. Captured sequences will be reviewed before saving.");
    updateBuiltinCaptureControls();
  }

  async function loadBuiltinMapping(label) {
    const requestId = ++builtinMappingRequestId;
    const textarea = byId("builtinMappingSentence");
    const button = byId("saveBuiltinMapping");
    if (!label) {
      textarea.value = "";
      textarea.disabled = true;
      button.disabled = true;
      setMessage("builtinMappingStatus", "Choose an existing label to view or edit its sentence mapping.");
      return;
    }
    textarea.disabled = true;
    button.disabled = true;
    setMessage("builtinMappingStatus", "Loading mapping…");
    try {
      const result = await api(`/developer/api/builtin-sentence-mapping?label=${encodeURIComponent(label)}`);
      if (requestId !== builtinMappingRequestId || byId("builtinMappingLabel").value !== label) return;
      textarea.value = result.mapping?.sentence || "";
      textarea.disabled = false;
      button.disabled = false;
      setMessage("builtinMappingStatus", result.message, result.mapping_saved ? "good" : "");
    } catch (error) {
      if (requestId === builtinMappingRequestId) setMessage("builtinMappingStatus", error.message, "error");
    }
  }

  async function saveBuiltinMapping() {
    const label = byId("builtinMappingLabel").value;
    const sentence = byId("builtinMappingSentence").value.trim();
    if (!label) {
      setMessage("builtinMappingStatus", "Choose a built-in label first.", "error");
      return;
    }
    if (!sentence) {
      setMessage("builtinMappingStatus", "Enter a non-empty predefined sentence.", "error");
      byId("builtinMappingSentence").focus();
      return;
    }
    const requestId = ++builtinMappingRequestId;
    const button = byId("saveBuiltinMapping");
    button.disabled = true;
    setMessage("builtinMappingStatus", "Saving built-in sentence mapping…");
    try {
      const result = await api("/developer/api/builtin-sentence-mapping", {
        method: "PUT",
        body: JSON.stringify({ label, sentence }),
      });
      if (requestId !== builtinMappingRequestId || byId("builtinMappingLabel").value !== label) return;
      byId("builtinMappingSentence").value = result.mapping?.sentence || "";
      setMessage("builtinMappingStatus", result.message, "good");
    } catch (error) {
      if (requestId === builtinMappingRequestId) setMessage("builtinMappingStatus", error.message, "error");
    } finally {
      if (requestId === builtinMappingRequestId) button.disabled = false;
    }
  }

  function updateActiveDraftDisplay() {
    updateBuiltinCaptureControls();
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
    loadSentenceMapping(activeDraft);
    updateActiveDraftDisplay();
    refreshDrafts().catch((error) => setMessage("draftMessage", error.message, "error"));
    refreshSavedSamples().catch((error) => setMessage("draftMessage", error.message, "error"));
    return true;
  }

  async function loadSentenceMapping(draft) {
    const requestId = ++sentenceMappingRequestId;
    const textarea = byId("activeSentenceMapping");
    const button = byId("saveSentenceMapping");
    textarea.disabled = true;
    button.disabled = true;
    setMessage("sentenceMappingStatus", "Loading sentence mapping…");
    try {
      const result = await api(
        `/developer/api/drafts/${encodeURIComponent(draft.entry_id)}/sentence-mapping?scope=${encodeURIComponent(draft.scope)}`
      );
      if (requestId !== sentenceMappingRequestId || activeDraft?.entry_id !== draft.entry_id || activeDraft?.scope !== draft.scope) return;
      textarea.value = result.mapping?.sentence || "";
      textarea.disabled = false;
      button.disabled = false;
      setMessage("sentenceMappingStatus", result.message, result.state === "available" ? "good" : result.state === "pending_activation" ? "warning" : "");
    } catch (error) {
      if (requestId !== sentenceMappingRequestId) return;
      setMessage("sentenceMappingStatus", error.message, "error");
    }
  }

  async function saveSentenceMapping(draft, sentence) {
    const requestId = ++sentenceMappingRequestId;
    const textarea = byId("activeSentenceMapping");
    const button = byId("saveSentenceMapping");
    button.disabled = true;
    setMessage("sentenceMappingStatus", "Saving sentence mapping…");
    try {
      const result = await api(
        `/developer/api/drafts/${encodeURIComponent(draft.entry_id)}/sentence-mapping`,
        { method: "PUT", body: JSON.stringify({ scope: draft.scope, sentence }) }
      );
      if (requestId !== sentenceMappingRequestId || activeDraft?.entry_id !== draft.entry_id || activeDraft?.scope !== draft.scope) return;
      textarea.value = result.mapping?.sentence || "";
      setMessage("sentenceMappingStatus", result.message, result.state === "available" ? "good" : "warning");
      return true;
    } catch (error) {
      if (requestId === sentenceMappingRequestId) setMessage("sentenceMappingStatus", error.message, "error");
      return false;
    } finally {
      if (requestId === sentenceMappingRequestId) button.disabled = false;
    }
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
      updateBuiltinCaptureControls();
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
    updateBuiltinCaptureControls();
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

  async function collectWebcamSequence() {
    if (!stream || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      throw new Error("Enable the webcam before starting capture.");
    }
    capturing = true;
    updateActiveDraftDisplay();
    byId("captureButton").disabled = true;
    byId("captureBuiltinSequence").disabled = true;
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
      return { result, previewFrames };
    } finally {
      capturing = false;
      updateActiveDraftDisplay();
      byId("startCamera").disabled = Boolean(stream);
      byId("stopCamera").disabled = !stream;
    }
  }

  async function captureSequence() {
    if (sessionSelectionUncertain) {
      throw new Error("Refresh drafts and select the word again to confirm the active session.");
    }
    if (!activeDraft) throw new Error("Create or select a draft before capture.");
    const draft = activeDraft;
    const { result, previewFrames } = await collectWebcamSequence();
    pending.push({
      entry_id: draft.entry_id,
      scope: draft.scope,
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
  }

  async function captureBuiltinSequence() {
    const label = byId("builtinCaptureLabel").value;
    const session = activeBuiltinSession;
    if (!label || !session || session.label !== label) {
      throw new Error("Select a built-in label and start a recording session first.");
    }
    const { result, previewFrames } = await collectWebcamSequence();
    builtinPending.push({
      label: session.label,
      session_id: session.session_id,
      sequence: result.sequence,
      quality: result.quality,
      previewFrames,
    });
    renderBuiltinPending();
    setMessage(
      "builtinCaptureStatus",
      `Sequence reviewed: ${result.quality.hand_frames}/${result.quality.frames} frames contain hand landmarks. Save it to append it to ${session.label}.`,
      "good"
    );
  }

  function renderBuiltinPending() {
    const container = byId("builtinPendingSamples");
    container.replaceChildren();
    byId("builtinPendingCount").textContent = `${builtinPending.length} pending`;
    byId("saveBuiltinSamples").disabled = capturing || builtinPending.length === 0;
    if (!builtinPending.length) {
      container.className = "sample-list empty-state";
      container.textContent = "Captured built-in sequences appear here for review before saving.";
      updateBuiltinCaptureControls();
      return;
    }
    container.className = "sample-list";
    builtinPending.forEach((sample, index) => {
      const row = document.createElement("article");
      row.className = "sample-card";
      const text = document.createElement("span");
      text.className = "sample-meta";
      text.textContent = `${sample.label} · session ${sample.session_id.slice(0, 12)} · ${sample.quality.hand_frames}/${sample.quality.frames} hand-visible frames · ${sample.quality.preprocessing_version}`;
      const discard = document.createElement("button");
      discard.type = "button";
      discard.className = "button button-quiet small-button";
      discard.textContent = "Discard pending";
      discard.addEventListener("click", () => {
        builtinPending.splice(index, 1);
        renderBuiltinPending();
      });
      row.append(text, discard);
      container.append(row);
    });
    updateBuiltinCaptureControls();
  }

  async function saveBuiltinSamples() {
    if (!builtinPending.length) return;
    const button = byId("saveBuiltinSamples");
    button.disabled = true;
    setMessage("builtinCaptureStatus", "Saving reviewed sequences to the separate built-in dataset…");
    try {
      while (builtinPending.length) {
        const sample = builtinPending[0];
        await api(
          `/developer/api/builtin-training-data/sessions/${encodeURIComponent(sample.session_id)}/samples`,
          {
            method: "POST",
            body: JSON.stringify({
              sequence: sample.sequence,
              preprocessing_version: PREPROCESSING_VERSION,
            }),
          }
        );
        builtinPending.shift();
        renderBuiltinPending();
      }
      const labels = await refreshBuiltinTrainingData();
      const selected = labels.find((item) => item.label === byId("builtinCaptureLabel").value);
      setMessage(
        "builtinCaptureStatus",
        `Samples appended. ${selected?.label || "Built-in label"}: ${selected?.sample_count || 0} saved sample(s) across ${selected?.session_count || 0} recording session(s). Training has not started.`,
        "good"
      );
    } catch (error) {
      setMessage("builtinCaptureStatus", error.message, "error");
    } finally {
      button.disabled = builtinPending.length === 0;
      updateBuiltinCaptureControls();
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
        ? `Training vocabulary (${selected.supported_label_count || selected.label_map?.length || 0} classes): ` +
          `${(selected.label_map || []).map((item) => item.label).join(", ")} · ` +
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

  async function exportDataset() {
    const scope = byId("trainingScope").value;
    if (scope === "personal" && !window.confirm(
      "This ZIP contains your private personal examples. Download the personal dataset?"
    )) {
      return;
    }

    const button = byId("exportDataset");
    button.disabled = true;
    setMessage("exportDatasetStatus", `Preparing the ${scope} dataset export…`);
    try {
      const response = await fetch(
        `${API}/developer/api/dataset/export?scope=${encodeURIComponent(scope)}`
      );
      if (!response.ok) {
        const result = await response.json().catch(() => ({}));
        throw new Error(result.error || `Export failed (${response.status})`);
      }

      const archive = await response.blob();
      const url = URL.createObjectURL(archive);
      const link = document.createElement("a");
      link.href = url;
      link.download = `isl-${scope}-dataset.zip`;
      document.body.append(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 0);
      setMessage(
        "exportDatasetStatus",
        `Downloaded ${scope} dataset (${archive.size.toLocaleString()} bytes).`,
        "good"
      );
    } catch (error) {
      setMessage("exportDatasetStatus", error.message, "error");
    } finally {
      button.disabled = false;
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
      if (activeDraft) await loadSentenceMapping(activeDraft);
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
      const sentence = byId("newWordSentence").value.trim();
      const result = await api("/developer/api/drafts", {
        method: "POST",
        body: JSON.stringify({
          label: byId("wordLabel").value,
          scope: byId("scopeSelect").value,
        }),
      });
      if (!setActiveDraft(result.draft, result.session_id)) return;
      setMessage("draftMessage", `Draft ready: ${result.draft.label} (${result.draft.scope}).`, "good");
      if (sentence) {
        byId("activeSentenceMapping").value = sentence;
        const saved = await saveSentenceMapping(result.draft, sentence);
        if (saved) byId("newWordSentence").value = "";
        else setMessage("draftMessage", "Draft created. The sentence mapping was not saved; retry it in the Sentence Mapping editor.", "warning");
      }
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
  byId("builtinCaptureLabel").addEventListener("change", updateBuiltinCaptureControls);
  byId("startBuiltinSession").addEventListener("click", () => {
    startBuiltinCaptureSession().catch((error) => setMessage("builtinCaptureStatus", error.message, "error"));
  });
  byId("captureBuiltinSequence").addEventListener("click", () => {
    captureBuiltinSequence().catch((error) => setMessage("builtinCaptureStatus", error.message, "error"));
  });
  byId("saveBuiltinSamples").addEventListener("click", saveBuiltinSamples);
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
  byId("exportDataset").addEventListener("click", exportDataset);
  byId("trainModelCandidate").addEventListener("click", trainModelCandidate);
  byId("activateModelCandidate").addEventListener("click", activateModelCandidate);
  byId("saveSentenceMapping").addEventListener("click", () => {
    if (!activeDraft) return;
    const sentence = byId("activeSentenceMapping").value.trim();
    if (!sentence) {
      setMessage("sentenceMappingStatus", "Enter a non-empty predefined sentence.", "error");
      byId("activeSentenceMapping").focus();
      return;
    }
    saveSentenceMapping(activeDraft, sentence);
  });
  byId("builtinMappingLabel").addEventListener("change", (event) => {
    loadBuiltinMapping(event.currentTarget.value);
  });
  byId("saveBuiltinMapping").addEventListener("click", saveBuiltinMapping);
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
  loadBuiltinLabels();
})();
