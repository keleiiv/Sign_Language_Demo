const assert = require('assert');
const fs = require('node:fs');
const vm = require('node:vm');

// Test suite for Sentence Demo Mode frontend logic
console.log('Running Frontend Sentence Demo Mode logic tests...');

class SentenceDemoLogic {
  constructor(mappings = {}) {
    this.mappings = mappings;
    this.activeHoldSign = null;
    this.lastTriggerTime = 0;
    this.pendingSign = null;
    this.pendingCount = 0;
    this.releaseCount = 0;
    this.minConfidence = 0.55;
    this.stabilityWindows = 1;
    this.cooldownMs = 3000;
    this.releaseWindows = 2;

    this.triggeredEvents = [];
  }

  findMapping(label) {
    if (!label || typeof label !== 'string') return null;
    const clean = label.trim().toLowerCase();
    for (const [sign, sentence] of Object.entries(this.mappings)) {
      if (sign.trim().toLowerCase() === clean) {
        return { sign, sentence };
      }
    }
    return null;
  }

  processPrediction(result, now = Date.now()) {
    if (!result || result.abstained || !result.label || (result.confidence !== undefined && result.confidence < this.minConfidence)) {
      this.releaseCount += 1;
      if (this.releaseCount >= this.releaseWindows) {
        this.activeHoldSign = null;
        this.pendingSign = null;
        this.pendingCount = 0;
      }
      return { status: 'abstained_or_released', activeHoldSign: this.activeHoldSign };
    }

    this.releaseCount = 0;
    const label = result.label.trim();

    // Stability filtering
    if (label === this.pendingSign) {
      this.pendingCount += 1;
    } else {
      this.pendingSign = label;
      this.pendingCount = 1;
    }

    if (this.pendingCount < this.stabilityWindows) {
      return { status: 'stabilizing', label, count: this.pendingCount };
    }

    const matched = this.findMapping(label);
    if (!matched) {
      return { status: 'unmapped', label };
    }

    // Duplicate hold & cooldown check
    if (this.activeHoldSign === matched.sign && (now - this.lastTriggerTime < this.cooldownMs)) {
      return { status: 'held_in_cooldown', sign: matched.sign, sentence: matched.sentence };
    }

    // Trigger!
    this.activeHoldSign = matched.sign;
    this.lastTriggerTime = now;
    const event = { sign: matched.sign, sentence: matched.sentence, timestamp: now };
    this.triggeredEvents.push(event);
    return { status: 'triggered', ...event };
  }
}

// Setup test mappings
const testMappings = {
  help: 'Could you please help me?',
  water: 'May I please have a glass of water?',
  sorry: 'I apologize for the misunderstanding.'
};

const demo = new SentenceDemoLogic(testMappings);
let t = 1000;

// Test 1: First confident prediction triggers immediately
const r1 = demo.processPrediction({ label: 'help', confidence: 0.90 }, t);
assert.strictEqual(r1.status, 'triggered', 'First confident window should trigger');
assert.strictEqual(demo.triggeredEvents.length, 1);
assert.strictEqual(demo.triggeredEvents[0].sentence, 'Could you please help me?');

// Test 2: Repeated prediction stays held during cooldown
t += 100;
const r2 = demo.processPrediction({ label: 'help', confidence: 0.88 }, t);
assert.strictEqual(r2.status, 'held_in_cooldown');
assert.strictEqual(demo.triggeredEvents.length, 1);

// Test 3: Sign remains held during cooldown -> duplicate prevented
t += 500;
const r3 = demo.processPrediction({ label: 'help', confidence: 0.85 }, t);
assert.strictEqual(r3.status, 'held_in_cooldown');
assert.strictEqual(demo.triggeredEvents.length, 1, 'No duplicate event while held');

// Test 4: Low confidence is rejected
t += 500;
const r4 = demo.processPrediction({ label: 'water', confidence: 0.30 }, t);
assert.strictEqual(r4.status, 'abstained_or_released');

// Test 5: Hands lowered for 2 windows releases the hold
t += 500;
demo.processPrediction({ abstained: true }, t);
t += 100;
demo.processPrediction({ abstained: true }, t);
assert.strictEqual(demo.activeHoldSign, null, 'Active hold sign must be released after 2 empty windows');

// Test 6: Performing the same sign again after release triggers again
t += 500;
const r6 = demo.processPrediction({ label: 'help', confidence: 0.87 }, t);
assert.strictEqual(r6.status, 'triggered');
assert.strictEqual(demo.triggeredEvents.length, 2);

// Test 7: Switching to a different mapped sign triggers that new sentence
t += 500;
const r7 = demo.processPrediction({ label: 'water', confidence: 0.92 }, t);
assert.strictEqual(r7.status, 'triggered');
assert.strictEqual(r7.sentence, 'May I please have a glass of water?');
assert.strictEqual(demo.triggeredEvents.length, 3);

// Test 8: Unmapped sign does not trigger a sentence
t += 500;
const r8 = demo.processPrediction({ label: 'unmapped_sign', confidence: 0.95 }, t);
assert.strictEqual(r8.status, 'unmapped');
assert.strictEqual(demo.triggeredEvents.length, 3, 'No sentence triggered for unmapped sign');

const appElements = new Map();
const appContext = {
  document: {
    addEventListener() {},
    getElementById(id) {
      if (!appElements.has(id)) {
        appElements.set(id, { textContent: '', dataset: {}, className: '' });
      }
      return appElements.get(id);
    }
  },
  window: {
    location: { origin: 'http://localhost' },
    addEventListener() {},
    ISLSequenceAccumulator: require('../Frontend/sequence-accumulator.js')
  },
  console: { info() {}, error() {} },
  setInterval() {}
};
const translatorScript = fs.readFileSync(
  require.resolve('../Frontend/app.js'),
  'utf8'
);
vm.runInNewContext(
  `${translatorScript}\nprocessSentenceDemoPrediction({ abstained: true }, 'experimental');`,
  appContext
);
assert.strictEqual(
  appElements.get('recognitionStatus').textContent,
  'No confident prediction',
  'An abstained model response must not throw or masquerade as an API failure'
);

console.log('All Frontend Sentence Demo Mode logic tests passed successfully!');
