# Experimental expandable-vocabulary classifier

This engine is a developer-only experiment. It is not connected to the live
recognizer, the KNN demo, or the built-in Keras model.

## Classifier

The CPU-only NumPy softmax classifier uses temporal pooling over the versioned
`isl-holistic-30x378-v1` 30-frame sequences. It computes five ordered temporal
bin means and four adjacent-bin differences (3,402 features). This retains
coarse movement and temporal order while avoiding a new runtime dependency or
large recurrent model. It is a lightweight isolated-sign baseline, not
unrestricted continuous ISL translation.

Training normalizes features using training data only, then fits a regularized
linear softmax model with deterministic Adam updates. The fixed 0.55
abstention threshold is a conservative engineering default, not a calibrated
unknown-sign detector; inspect unknown/background results before interpreting
it as rejection performance.

## Data and evaluation requirements

- Each class needs at least two independent capture sessions to participate in
  training.
- Session-aware validation requires at least four independent sessions for
  each class, reserving at least two whole sessions for training and two for
  validation. With two or three sessions, all sessions remain in training and
  no validation metric is produced. Below four sessions, reports explicitly
  mark evaluation as insufficient; no training sequence is reused as validation.
- A class with fewer than two sessions blocks model fitting rather than being
  silently omitted.
- False-accept and abstention rates are measured only when a separate
  unknown/background sequence set is provided. That set must use independently
  recorded session IDs that do not overlap with vocabulary sessions.
- Results are experimental even when these minimum counts are met; they do not
  establish generalization to other signers or cameras.

## Commands

Run from the project root with the project's Python environment:

```powershell
venv\Scripts\python.exe scripts\expandable_classifier.py train --scope shared
venv\Scripts\python.exe scripts\expandable_classifier.py train --scope personal
venv\Scripts\python.exe scripts\expandable_classifier.py list --scope shared
venv\Scripts\python.exe scripts\expandable_classifier.py evaluate --scope shared --artifact-id <artifact-id>
venv\Scripts\python.exe scripts\expandable_classifier.py activate --scope shared --artifact-id <older-artifact-id>
venv\Scripts\python.exe scripts\expandable_classifier.py infer --scope shared --sequence <sequence.npy> --preprocessing-version isl-holistic-30x378-v1
```

Training creates a new immutable version under
`data\vocabulary_drafts\<scope>\models\<artifact-id>\`; it does not overwrite
older artifacts. `--promote` updates only that scope's `active.json` pointer.
Rollback is selecting an earlier artifact with `activate`; artifacts are not
deleted. Shared and personal examples, labels, and model directories are
separate. No export or publishing operation is provided.

Samples saved from the Phase 2 review interface are the developer-accepted
training examples. Training reads only those persisted samples for the
requested scope; pending captures are never included.

## Unknown/background manifest

Optional unknown evaluation data is supplied as a JSON manifest adjacent to
its `.npy` sequence files:

```json
{
  "samples": [
    {
      "file": "unknown-session-01-seq-01.npy",
      "session_id": "unknown-session-01",
      "class": "unknown",
      "preprocessing_version": "isl-holistic-30x378-v1"
    }
  ]
}
```

Each file must contain exactly one finite `(30, 378)` array. Use
`"class": "background"` for non-sign/background examples. Keep the manifest and
arrays separate from vocabulary examples and provide independently recorded
sessions. The report shows the observed false-accept and abstention rates at
the fixed threshold and the number of independent sessions; it flags sets
with fewer than two sessions as insufficient.

Training artifacts include the label map, preprocessing and feature
configuration, random seed, split assignments, validation metrics, and a
SHA-256 checksum. Loading checks the artifact schema, scope, input contract,
parameter shapes, and checksum.
