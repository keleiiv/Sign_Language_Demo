const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

class MockElement {
  constructor(id = '', className = '') {
    this.id = id;
    this.className = className;
    this.children = [];
    this.listeners = {};
    this.attributes = {};
    this.dataset = {};
    this.textContent = '';
    this.disabled = false;
    this.value = '';
  }

  addEventListener(type, callback) { this.listeners[type] = callback; }
  setAttribute(name, value) { this.attributes[name] = value; }
  getAttribute(name) { return this.attributes[name]; }
  replaceChildren(...children) { this.children = children; }
  querySelectorAll(selector) {
    const className = selector.slice(1);
    return this.children.filter((child) => child.className.split(/\s+/).includes(className));
  }
  click() {
    if (!this.disabled && this.listeners.click) this.listeners.click();
  }
}

const elements = new Map();
for (const id of [
  'quickPhraseCategories', 'quickPhraseList', 'quickPhraseSelected',
  'quickPhraseClear', 'quickPhraseSpeak', 'quickPhraseStatus', 'langSelect',
  'speechStatus', 'detectedWord', 'signMeta', 'demoSentenceDisplay'
]) elements.set(id, new MockElement(id));
elements.get('langSelect').value = 'hi';
elements.get('speechStatus').textContent = 'Existing sentence speech status';
elements.get('detectedWord').textContent = 'HELLO';
elements.get('signMeta').textContent = 'Confidence: 0.91';
elements.get('demoSentenceDisplay').textContent = 'Recognized sentence stays here.';

const utterances = [];
let activeUtterance = null;
class MockUtterance {
  constructor(text) { this.text = text; this.lang = ''; }
}
const document = {
  addEventListener() {},
  getElementById(id) { return elements.get(id) || null; },
  createElement() { return new MockElement(); }
};
const window = {
  location: { origin: 'http://localhost' },
  ISLSequenceAccumulator: { SequenceAccumulator: class {} },
  speechSynthesis: {
    cancel() {
      const previous = activeUtterance;
      activeUtterance = null;
      if (previous && previous.onerror) previous.onerror({ error: 'canceled' });
    },
    speak(utterance) {
      utterances.push(utterance);
      activeUtterance = utterance;
    }
  },
  addEventListener() {}
};
const sandbox = { window, document, SpeechSynthesisUtterance: MockUtterance, console, fetch() {} };
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(__dirname, '..', 'Frontend', 'app.js'), 'utf8'), sandbox);
assert.equal(vm.runInContext('QUICK_PHRASE_CATEGORIES.reduce((total, item) => total + item.phrases.length, 0)', sandbox), 25);

const byId = (id) => elements.get(id);
sandbox.initQuickPhrases();
assert.equal(byId('quickPhraseCategories').children.length, 5);
assert.equal(byId('quickPhraseList').children.length, 6);

byId('quickPhraseList').children[0].click();
assert.equal(byId('quickPhraseSelected').textContent, 'Hello!');
assert.equal(byId('quickPhraseSpeak').disabled, false);
assert.equal(utterances.length, 1, 'selecting a phrase must speak immediately');
assert.equal(utterances[0].text, 'Hello!');
assert.equal(utterances[0].lang, 'hi-IN');
assert.equal(byId('quickPhraseStatus').textContent, 'Starting speech…');
assert.equal(byId('detectedWord').textContent, 'HELLO');
assert.equal(byId('signMeta').textContent, 'Confidence: 0.91');
assert.equal(byId('demoSentenceDisplay').textContent, 'Recognized sentence stays here.');
assert.equal(vm.runInContext('aiSignSequence.length', sandbox), 0, 'manual phrases must not enter the AI sign sequence');

utterances[0].onstart();
assert.equal(byId('quickPhraseStatus').textContent, 'Speaking selected phrase');
byId('quickPhraseList').children[1].click();
assert.equal(utterances.length, 2, 'selecting another phrase must replace the active utterance');
assert.equal(utterances[1].text, 'Good morning!');
assert.equal(activeUtterance, utterances[1]);
assert.equal(byId('quickPhraseSelected').textContent, 'Good morning!');
utterances[0].onend();
assert.equal(byId('quickPhraseStatus').textContent, 'Starting speech…', 'stale events from the canceled phrase must not overwrite the new status');
utterances[1].onstart();
assert.equal(byId('quickPhraseStatus').textContent, 'Speaking selected phrase');
activeUtterance = null;
utterances[1].onend();
assert.equal(byId('quickPhraseStatus').textContent, 'Speech finished');
byId('quickPhraseSpeak').click();
assert.equal(utterances.length, 3, 'Speak Phrase remains available to replay the selection');
assert.equal(utterances[2].text, 'Good morning!');
assert.equal(utterances[2].lang, 'hi-IN', 'quick phrases must use the shared speech language selector');
assert.equal(byId('speechStatus').textContent, 'Existing sentence speech status');

byId('quickPhraseCategories').children.find((button) => button.textContent === 'Basic Needs').click();
assert.equal(byId('quickPhraseList').children.length, 5);
assert.equal(byId('quickPhraseSelected').textContent, 'Choose a phrase to see it here.');
assert.equal(byId('quickPhraseSpeak').disabled, true);

byId('quickPhraseList').children[0].click();
assert.equal(byId('quickPhraseSelected').textContent, 'I need some water.');
byId('quickPhraseClear').click();
assert.equal(byId('quickPhraseSelected').textContent, 'Choose a phrase to see it here.');
assert.equal(byId('quickPhraseClear').disabled, true);
console.log('Frontend Quick Phrases interaction tests passed.');
