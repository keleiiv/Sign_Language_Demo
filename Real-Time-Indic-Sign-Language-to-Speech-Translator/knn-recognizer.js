
import { KNNImageClassifier } from 'deeplearn-knn-image-classifier';
import * as dl from 'deeplearn';

const IMAGE_SIZE = 227;
const TOPK = 10;
const TRAINING_EXAMPLES = 30;
const MIN_EXAMPLES_PER_CLASS = 5;
const CONFIRMATION_FRAMES = 3;
const RELEASE_FRAMES = 12;
const TRAINING_DB_NAME = 'isl-knn-training';
const TRAINING_DB_VERSION = 1;
const TRAINING_STORE_NAME = 'datasets';
const TRAINING_RECORD_KEY = 'current';
const TRAINING_FORMAT = 'isl-knn-training';
const MAX_EXAMPLES_PER_CLASS = 300;
const MAX_IMAGE_BYTES = 4 * 1024 * 1024;
const MAX_DATASET_BYTES = 56 * 1024 * 1024;
const MAX_ARCHIVE_BYTES = 80 * 1024 * 1024;

const SIGN_WORDS = ['Hello', 'Thank you', 'Sorry', 'Yes', 'No'];
const OTHER_CLASS_INDEX = SIGN_WORDS.length;
const WORDS = SIGN_WORDS.concat('OTHER');

let video;
let canvas;
let ctx;
let knn;
let stream;
let running = false;
let recognizing = false;
let training = false;
let predicting = false;
let lastPrediction = -1;
let candidatePrediction = -1;
let candidateFrames = 0;
let unknownFrames = 0;
let trainingExamples = SIGN_WORDS.map(() => []);
let trainingDatabasePromise = null;
let initializationPromise = null;
let trainingDataInitialized = false;
let classifierPromise = null;
let savedDataBlocked = false;
let storageUnavailable = false;

const MIN_CONFIDENCE = 0.90;

function emitStatus(message) {
  window.dispatchEvent(
    new CustomEvent('knn-status', { detail: message })
  );
}

function errorMessage(error) {
  return error instanceof Error ? error.message : String(error);
}

function openTrainingDatabase() {
  if (trainingDatabasePromise) return trainingDatabasePromise;
  if (!window.indexedDB) {
    return Promise.reject(new Error('IndexedDB is unavailable in this browser.'));
  }

  trainingDatabasePromise = new Promise((resolve, reject) => {
    const request = window.indexedDB.open(TRAINING_DB_NAME, TRAINING_DB_VERSION);
    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(TRAINING_STORE_NAME)) {
        database.createObjectStore(TRAINING_STORE_NAME);
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => {
      trainingDatabasePromise = null;
      reject(request.error || new Error('Could not open training-data storage.'));
    };
    request.onblocked = () => {
      trainingDatabasePromise = null;
      reject(new Error('Training-data storage is blocked by another page.'));
    };
  });
  return trainingDatabasePromise;
}

function readTrainingRecord() {
  return openTrainingDatabase().then(database => new Promise((resolve, reject) => {
    const transaction = database.transaction(TRAINING_STORE_NAME, 'readonly');
    const request = transaction.objectStore(TRAINING_STORE_NAME).get(TRAINING_RECORD_KEY);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error('Could not read saved training data.'));
    transaction.onabort = () => reject(transaction.error || new Error('Reading saved training data was aborted.'));
  }));
}

function writeTrainingRecord(record) {
  return openTrainingDatabase().then(database => new Promise((resolve, reject) => {
    const transaction = database.transaction(TRAINING_STORE_NAME, 'readwrite');
    transaction.objectStore(TRAINING_STORE_NAME).put(record, TRAINING_RECORD_KEY);
    transaction.oncomplete = resolve;
    transaction.onerror = () => reject(transaction.error || new Error('Could not save training data.'));
    transaction.onabort = () => reject(transaction.error || new Error('Saving training data was aborted.'));
  }));
}

function deleteTrainingRecord() {
  return openTrainingDatabase().then(database => new Promise((resolve, reject) => {
    const transaction = database.transaction(TRAINING_STORE_NAME, 'readwrite');
    transaction.objectStore(TRAINING_STORE_NAME).delete(TRAINING_RECORD_KEY);
    transaction.oncomplete = resolve;
    transaction.onerror = () => reject(transaction.error || new Error('Could not clear saved training data.'));
    transaction.onabort = () => reject(transaction.error || new Error('Clearing saved training data was aborted.'));
  }));
}

function makeTrainingRecord(examples = trainingExamples) {
  return {
    format: TRAINING_FORMAT,
    version: TRAINING_DB_VERSION,
    imageSize: IMAGE_SIZE,
    labels: SIGN_WORDS.slice(),
    classes: SIGN_WORDS.map((label, classIndex) => ({
      label,
      examples: examples[classIndex].slice()
    }))
  };
}

function validateTrainingRecord(record) {
  if (!record || record.format !== TRAINING_FORMAT ||
      record.version !== TRAINING_DB_VERSION || record.imageSize !== IMAGE_SIZE ||
      !Array.isArray(record.labels) ||
      record.labels.length !== SIGN_WORDS.length ||
      record.labels.some((label, index) => label !== SIGN_WORDS[index]) ||
      !Array.isArray(record.classes) ||
      record.classes.length !== SIGN_WORDS.length) {
    throw new Error('Training data format or sign labels are incompatible.');
  }

  let totalExamples = 0;
  let totalBytes = 0;
  const examples = record.classes.map((entry, classIndex) => {
    if (!entry || entry.label !== SIGN_WORDS[classIndex] ||
        !Array.isArray(entry.examples) ||
        entry.examples.length > MAX_EXAMPLES_PER_CLASS) {
      throw new Error(`Training data for ${SIGN_WORDS[classIndex]} is invalid.`);
    }
    entry.examples.forEach((image) => {
      if (!image || typeof image.slice !== 'function' ||
          image.type !== 'image/png' || !Number.isInteger(image.size) ||
          image.size <= 0 || image.size > MAX_IMAGE_BYTES) {
        throw new Error(`A saved ${SIGN_WORDS[classIndex]} example is invalid.`);
      }
      totalBytes += image.size;
    });
    totalExamples += entry.examples.length;
    return entry.examples.slice();
  });

  if (totalExamples === 0) {
    throw new Error('Training data contains no examples.');
  }
  if (totalBytes > MAX_DATASET_BYTES) {
    throw new Error('Training data exceeds the supported 56 MB total size.');
  }
  return examples;
}

function canvasToBlob(canvas) {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (!blob) {
        reject(new Error('Could not encode a webcam training example.'));
      } else if (blob.type !== 'image/png' || blob.size > MAX_IMAGE_BYTES) {
        reject(new Error('The webcam training example has an unsupported size or format.'));
      } else {
        resolve(blob);
      }
    }, 'image/png');
  });
}

async function withDecodedImage(blob, callback) {
  if (typeof window.createImageBitmap === 'function') {
    const bitmap = await window.createImageBitmap(blob);
    try {
      if (bitmap.width !== IMAGE_SIZE || bitmap.height !== IMAGE_SIZE) {
        throw new Error('A training image has incompatible dimensions.');
      }
      return callback(bitmap);
    } finally {
      bitmap.close();
    }
  }

  const objectUrl = window.URL.createObjectURL(blob);
  try {
    const image = await new Promise((resolve, reject) => {
      const element = new Image();
      element.onload = () => resolve(element);
      element.onerror = () => reject(new Error('A training image could not be decoded.'));
      element.src = objectUrl;
    });
    if (image.naturalWidth !== IMAGE_SIZE || image.naturalHeight !== IMAGE_SIZE) {
      throw new Error('A training image has incompatible dimensions.');
    }
    return callback(image);
  } finally {
    window.URL.revokeObjectURL(objectUrl);
  }
}

async function addImageBlob(classifier, blob, classIndex) {
  await withDecodedImage(blob, (source) => {
    const image = dl.fromPixels(source);
    try {
      classifier.addImage(image, classIndex);
    } finally {
      image.dispose();
    }
  });
}

async function buildClassifier(examples) {
  const classifier = new KNNImageClassifier(WORDS.length, TOPK);
  try {
    emitStatus('Loading browser recognizer model...');
    await classifier.load();
    for (let classIndex = 0; classIndex < examples.length; classIndex++) {
      for (const blob of examples[classIndex]) {
        await addImageBlob(classifier, blob, classIndex);
      }
    }

    const counts = classifier.getClassExampleCount();
    examples.forEach((classExamples, classIndex) => {
      if (counts[classIndex] !== classExamples.length) {
        throw new Error(`Could not restore all ${SIGN_WORDS[classIndex]} examples.`);
      }
    });
    return classifier;
  } catch (error) {
    classifier.dispose();
    throw error;
  }
}

async function initializeTrainingData() {
  if (trainingDataInitialized) return !savedDataBlocked;
  if (initializationPromise) return initializationPromise;
  initializationPromise = (async () => {
    emitStatus('Loading saved training data...');
    let record;
    try {
      record = await readTrainingRecord();
    } catch (error) {
      savedDataBlocked = false;
      storageUnavailable = true;
      emitStatus(`Could not access saved training data: ${errorMessage(error)}. Training can continue for this session; export examples for safekeeping.`);
      return false;
    }

    if (!record) {
      savedDataBlocked = false;
      storageUnavailable = false;
      emitStatus('No saved training data found.');
      return false;
    }
    try {
      const examples = validateTrainingRecord(record);
      const classifier = await buildClassifier(examples);
      knn = classifier;
      trainingExamples = examples;
      savedDataBlocked = false;
      storageUnavailable = false;
      const loadedSigns = examples.filter(classExamples => classExamples.length > 0).length;
      const summary = SIGN_WORDS.map((label, index) =>
        `${label}: ${examples[index].length}`
      ).join(', ');
      emitStatus(`${loadedSigns} signs loaded. ${summary}.`);
      return true;
    } catch (error) {
      savedDataBlocked = true;
      emitStatus(`Saved training data could not be loaded and was preserved: ${errorMessage(error)}. Import valid data or clear it explicitly.`);
      return false;
    }
  })();
  try {
    return await initializationPromise;
  } finally {
    trainingDataInitialized = true;
    initializationPromise = null;
  }
}

async function ensureClassifier() {
  if (knn) return knn;
  if (savedDataBlocked) {
    throw new Error('Saved training data needs attention. Import valid data or clear it explicitly.');
  }
  if (!classifierPromise) {
    classifierPromise = buildClassifier(trainingExamples).then((classifier) => {
      knn = classifier;
      return classifier;
    }).finally(() => {
      classifierPromise = null;
    });
  }
  return classifierPromise;
}

async function persistTrainingExamples(examples = trainingExamples) {
  const record = makeTrainingRecord(examples);
  validateTrainingRecord(record);
  const total = record.classes.reduce((sum, entry) => sum + entry.examples.length, 0);
  if (total === 0) throw new Error('There are no training examples to save.');
  await writeTrainingRecord(record);
}

function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = reader.result;
      if (typeof dataUrl !== 'string') {
        reject(new Error('Could not prepare a training image for export.'));
        return;
      }
      resolve(dataUrl.slice(dataUrl.indexOf(',') + 1));
    };
    reader.onerror = () => reject(reader.error || new Error('Could not read a training image.'));
    reader.readAsDataURL(blob);
  });
}

function decodeBase64Image(data, label) {
  if (typeof data !== 'string' || data.length === 0 ||
      !/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(data)) {
    throw new Error(`Training data for ${label} contains invalid image data.`);
  }
  const binary = window.atob(data);
  if (binary.length === 0 || binary.length > MAX_IMAGE_BYTES) {
    throw new Error(`A ${label} training image exceeds the supported size.`);
  }
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Blob([bytes], { type: 'image/png' });
}

async function parseTrainingArchive(file) {
  if (!file || file.size <= 0 || file.size > MAX_ARCHIVE_BYTES) {
    throw new Error('Choose a non-empty training-data JSON file smaller than 80 MB.');
  }
  let archive;
  try {
    archive = JSON.parse(await file.text());
  } catch (error) {
    throw new Error(`Training-data file is not valid JSON: ${errorMessage(error)}`);
  }
  if (!archive || archive.format !== TRAINING_FORMAT ||
      archive.version !== TRAINING_DB_VERSION || archive.imageSize !== IMAGE_SIZE ||
      !Array.isArray(archive.labels) ||
      archive.labels.length !== SIGN_WORDS.length ||
      archive.labels.some((label, index) => label !== SIGN_WORDS[index]) ||
      !Array.isArray(archive.classes) ||
      archive.classes.length !== SIGN_WORDS.length) {
    throw new Error('Training-data file version, image size, or sign labels are incompatible.');
  }

  let totalExamples = 0;
  let totalBytes = 0;
  const examples = archive.classes.map((entry, classIndex) => {
    if (!entry || entry.label !== SIGN_WORDS[classIndex] ||
        !Array.isArray(entry.examples) ||
        entry.examples.length > MAX_EXAMPLES_PER_CLASS) {
      throw new Error(`Training data for ${SIGN_WORDS[classIndex]} is invalid.`);
    }
    totalExamples += entry.examples.length;
    return entry.examples.map((example) => {
      if (!example || example.type !== 'image/png') {
        throw new Error(`A ${SIGN_WORDS[classIndex]} example is not a PNG image.`);
      }
      const blob = decodeBase64Image(example.data, SIGN_WORDS[classIndex]);
      totalBytes += blob.size;
      if (totalBytes > MAX_DATASET_BYTES) {
        throw new Error('Training data exceeds the supported 56 MB total size.');
      }
      return blob;
    });
  });
  if (totalExamples === 0) throw new Error('Training-data file contains no examples.');

  for (const classExamples of examples) {
    for (const blob of classExamples) {
      await withDecodedImage(blob, () => {});
    }
  }
  return examples;
}

async function saveTrainingData() {
  if (training) {
    emitStatus('Wait for training to finish before saving.');
    return;
  }
  await initializeTrainingData();
  if (savedDataBlocked) {
    emitStatus('Saved training data is incompatible. Import valid data or clear it explicitly before saving.');
    return;
  }
  try {
    await persistTrainingExamples();
    storageUnavailable = false;
    emitStatus('Training data saved in this browser.');
  } catch (error) {
    storageUnavailable = true;
    emitStatus(`Could not save training data: ${errorMessage(error)}`);
  }
}

async function exportTrainingData() {
  if (training) {
    emitStatus('Wait for training to finish before exporting.');
    return;
  }
  const total = trainingExamples.reduce((sum, examples) => sum + examples.length, 0);
  if (!total) {
    emitStatus('There are no training examples to export.');
    return;
  }
  const totalBytes = trainingExamples.reduce(
    (sum, examples) => sum + examples.reduce((classSum, blob) => classSum + blob.size, 0),
    0
  );
  if (totalBytes > MAX_DATASET_BYTES) {
    emitStatus('Training data exceeds the supported 56 MB total size and cannot be exported.');
    return;
  }
  try {
    const record = makeTrainingRecord();
    const archive = {
      format: record.format,
      version: record.version,
      imageSize: record.imageSize,
      labels: record.labels,
      classes: await Promise.all(record.classes.map(async (entry) => ({
        label: entry.label,
        examples: await Promise.all(entry.examples.map(async blob => ({
          type: blob.type,
          data: await blobToBase64(blob)
        })))
      })))
    };
    const file = new Blob([JSON.stringify(archive)], { type: 'application/json' });
    if (file.size > MAX_ARCHIVE_BYTES) {
      throw new Error('Export exceeds the supported 80 MB file size.');
    }
    const url = window.URL.createObjectURL(file);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'isl-knn-training.json';
    link.click();
    setTimeout(() => window.URL.revokeObjectURL(url), 1000);
    emitStatus(`Exported ${total} training examples.`);
  } catch (error) {
    emitStatus(`Could not export training data: ${errorMessage(error)}`);
  }
}

async function importTrainingData(file) {
  if (training || recognizing) {
    emitStatus('Stop recognition and wait for training before importing data.');
    return;
  }
  try {
    await initializeTrainingData();
    const examples = await parseTrainingArchive(file);
    const classifier = await buildClassifier(examples);
    try {
      await persistTrainingExamples(examples);
      savedDataBlocked = false;
      storageUnavailable = false;
      emitStatus('Training data imported and saved in this browser.');
    } catch (error) {
      storageUnavailable = true;
      emitStatus(`Training data imported for this session, but could not be saved in this browser: ${errorMessage(error)}`);
    }
    savedDataBlocked = false;
    if (knn) knn.dispose();
    knn = classifier;
    trainingExamples = examples;
    trainingDataInitialized = true;
  } catch (error) {
    emitStatus(`Could not import training data; existing saved and loaded data were left unchanged: ${errorMessage(error)}`);
  }
}

async function clearTrainingData() {
  if (training || recognizing) {
    emitStatus('Stop recognition and wait for training before clearing data.');
    return;
  }
  try {
    await deleteTrainingRecord();
    if (knn) {
      for (let classIndex = 0; classIndex < WORDS.length; classIndex++) {
        knn.clearClass(classIndex);
      }
    }
    trainingExamples = SIGN_WORDS.map(() => []);
    savedDataBlocked = false;
    storageUnavailable = false;
    trainingDataInitialized = true;
    emitStatus('Saved and loaded training data cleared.');
  } catch (error) {
    emitStatus(`Could not clear training data; the existing data was preserved: ${errorMessage(error)}`);
  }
}

async function startCamera() {
  if (running) return true;

  video = document.getElementById('knnVideo');
  canvas = document.createElement('canvas');
  canvas.width = IMAGE_SIZE;
  canvas.height = IMAGE_SIZE;
  ctx = canvas.getContext('2d');

  if (!video) {
    emitStatus('Missing webcam video element.');
    return false;
  }

  try {
    await initializeTrainingData();
    await ensureClassifier();
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user' },
      audio: false
    });

    video.srcObject = stream;
    await video.play();

    running = true;
    emitStatus(storageUnavailable
      ? 'Camera ready. Persistent storage is unavailable; export examples to keep a copy.'
      : 'Camera ready. Train each sign before predicting.');
    return true;
  } catch (error) {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      stream = null;
    }
    if (video) video.srcObject = null;
    emitStatus(`Camera or model error: ${errorMessage(error)}`);
    return false;
  }
}

async function trainClass(classIndex) {
  if (!running || !knn || !video || video.readyState < 2) {
    emitStatus('Start the camera and wait for the video first.');
    return;
  }

  if (!Number.isInteger(classIndex) ||
      classIndex < 0 || classIndex >= OTHER_CLASS_INDEX) {
    emitStatus('Invalid training class.');
    return;
  }
  if (recognizing) {
    emitStatus('Stop recognition before collecting more examples.');
    return;
  }
  if (training) {
    emitStatus('Wait for the current training session to finish.');
    return;
  }

  training = true;

  const classifier = knn;
  const previousExamples = trainingExamples[classIndex];
  const initialCount = classifier.getClassExampleCount()[classIndex];
  if (initialCount + TRAINING_EXAMPLES > MAX_EXAMPLES_PER_CLASS) {
    training = false;
    emitStatus(`A maximum of ${MAX_EXAMPLES_PER_CLASS} examples per sign is supported. Clear or export training data before collecting more.`);
    return;
  }

  const collectedExamples = [];
  try {
    for (let i = 0; i < TRAINING_EXAMPLES && running; i++) {
      if (video.readyState < 2) {
        throw new Error('The webcam video is not ready.');
      }
      ctx.drawImage(video, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
      const image = dl.fromPixels(canvas);
      try {
        classifier.addImage(image, classIndex);
      } finally {
        image.dispose();
      }
      collectedExamples.push(await canvasToBlob(canvas));
      const collected = classifier.getClassExampleCount()[classIndex] - initialCount;
      if (collected !== i + 1) {
        throw new Error(`Expected ${i + 1} collected examples, but the classifier stored ${collected}.`);
      }
      emitStatus(`Training ${WORDS[classIndex]}: ${collected}/${TRAINING_EXAMPLES} examples...`);
      await new Promise(resolve => setTimeout(resolve, 100));
    }

    const collected = classifier.getClassExampleCount()[classIndex] - initialCount;
    if (collected !== TRAINING_EXAMPLES ||
        collectedExamples.length !== TRAINING_EXAMPLES) {
      throw new Error(`Training ended with ${collected} of ${TRAINING_EXAMPLES} examples.`);
    }

    const nextExamples = trainingExamples.map(examples => examples.slice());
    nextExamples[classIndex] = previousExamples.concat(collectedExamples);
    const totalBytes = nextExamples.reduce(
      (sum, examples) => sum + examples.reduce((classSum, blob) => classSum + blob.size, 0),
      0
    );
    if (totalBytes > MAX_DATASET_BYTES) {
      throw new Error('Training data exceeds the supported 56 MB total size.');
    }
    trainingExamples = nextExamples;
    try {
      await persistTrainingExamples();
      storageUnavailable = false;
      emitStatus(`Collected ${collected} examples for ${WORDS[classIndex]}; training data saved.`);
    } catch (error) {
      storageUnavailable = true;
      emitStatus(`Collected ${collected} examples for ${WORDS[classIndex]}, but could not save them: ${errorMessage(error)}. Use Export Data to keep a copy.`);
    }
  } catch (error) {
    let rollbackMessage = '';
    try {
      classifier.clearClass(classIndex);
      for (const blob of previousExamples) {
        await addImageBlob(classifier, blob, classIndex);
      }
    } catch (rollbackError) {
      classifier.dispose();
      knn = null;
      rollbackMessage = ` Rollback failed: ${errorMessage(rollbackError)}. Restart the recognizer before continuing.`;
    }
    emitStatus(`Training error: ${errorMessage(error)}.${rollbackMessage}`);
  } finally {
    training = false;
  }
}

function getTrainingCounts() {
  return knn ? knn.getClassExampleCount() : [];
}

async function predictLoop() {
  if (!running || !recognizing || predicting || !knn || video.readyState < 2) {
    if (running && recognizing) requestAnimationFrame(predictLoop);
    return;
  }

  predicting = true;
  let image;

  try {
    ctx.drawImage(video, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
    image = dl.fromPixels(canvas);

    const result = await knn.predictClass(image);
    if (!result || !Array.isArray(result.confidences)) {
      throw new Error('The classifier returned an invalid prediction.');
    }
    const index = result.classIndex;
    const rawConfidence = result.confidences[index];
    const confidence = Number.isFinite(rawConfidence) &&
      rawConfidence >= 0 && rawConfidence <= 1
      ? rawConfidence
      : 0;
    const validPrediction = Number.isInteger(index) &&
      index >= 0 && index < WORDS.length &&
      confidence >= MIN_CONFIDENCE;

    if (validPrediction && index < OTHER_CLASS_INDEX) {
      unknownFrames = 0;
      if (candidatePrediction === index) {
        candidateFrames++;
      } else {
        candidatePrediction = index;
        candidateFrames = 1;
      }

      if (candidateFrames >= CONFIRMATION_FRAMES && index !== lastPrediction) {
        lastPrediction = index;
        window.dispatchEvent(new CustomEvent('knn-word', {
          detail: { word: WORDS[index], confidence }
        }));
      }
    } else {
      candidatePrediction = -1;
      candidateFrames = 0;
      unknownFrames++;
      if (unknownFrames >= RELEASE_FRAMES && lastPrediction !== -1) {
        const releasedWord = WORDS[lastPrediction];
        lastPrediction = -1;
        window.dispatchEvent(new CustomEvent('knn-release', {
          detail: { word: releasedWord }
        }));
      }
    }

    if (Number.isInteger(index) && index >= 0 && index < WORDS.length) {
      window.dispatchEvent(new CustomEvent('knn-confidence', {
        detail: { word: WORDS[index], confidence }
      }));
    }
  } catch (error) {
    emitStatus(`Prediction error: ${errorMessage(error)}`);
    recognizing = false;
  } finally {
    if (image) image.dispose();
    predicting = false;
  }

  if (running && recognizing) requestAnimationFrame(predictLoop);
}

function startPrediction() {
  if (!running || !knn) {
    emitStatus('Start the camera first.');
    return;
  }

  if (training) {
    emitStatus('Wait for training to finish before starting recognition.');
    return;
  }

  const counts = getTrainingCounts();
  const missingClasses = SIGN_WORDS
    .filter((_, index) =>
      !Number.isInteger(counts[index]) ||
      counts[index] < MIN_EXAMPLES_PER_CLASS
    );
  if (missingClasses.length) {
    const details = missingClasses.map((word) => {
      const index = WORDS.indexOf(word);
      return `${word}: ${counts[index] || 0}/${MIN_EXAMPLES_PER_CLASS}`;
    });
    emitStatus(`Collect at least ${MIN_EXAMPLES_PER_CLASS} examples per sign first (${details.join(', ')}).`);
    return;
  }

  if (recognizing) {
    emitStatus('Recognition is already running.');
    return;
  }

  lastPrediction = -1;
  candidatePrediction = -1;
  candidateFrames = 0;
  unknownFrames = 0;
  recognizing = true;
  predictLoop();
  emitStatus('Prediction running.');
}

function stopCamera() {
  if (training) {
    emitStatus('Wait for training to finish before stopping the camera.');
    return false;
  }

  running = false;
  recognizing = false;

  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
  }

  if (video) {
    video.srcObject = null;
  }

  lastPrediction = -1;
  candidatePrediction = -1;
  candidateFrames = 0;
  unknownFrames = 0;
  emitStatus('Camera stopped.');
  return true;
}

window.knnRecognizer = {
  initializeTrainingData,
  startCamera,
  trainClass,
  startPrediction,
  stopCamera,
  saveTrainingData,
  exportTrainingData,
  importTrainingData,
  clearTrainingData,
  words: WORDS,
  labels: SIGN_WORDS.slice(),
  isRecognizing: () => recognizing
};
