const assert = require('node:assert/strict');
const { SequenceAccumulator } = require('../Frontend/sequence-accumulator.js');

const experimental = new SequenceAccumulator();
assert.equal(experimental.observe({ label: 'Food' }, 2), null);
assert.equal(experimental.observe({ label: 'Where' }, 2), null);
assert.deepEqual(experimental.observe({ label: 'Where' }, 2), { label: 'Where' });
assert.equal(experimental.observe({ label: 'Where' }, 2), null);
assert.equal(experimental.observe({ label: null, uncertain: true }, 2), null);
assert.equal(experimental.observe({ label: 'Where' }, 2), null);
assert.equal(experimental.observe({ label: 'Where' }, 2), null);
assert.equal(experimental.observe({ label: null, uncertain: true }, 2), null);
assert.equal(experimental.observe({ label: null, uncertain: true }, 2), null);
assert.deepEqual(experimental.observe({ label: 'Where' }, 2), null);
assert.deepEqual(experimental.observe({ label: 'Where' }, 2), { label: 'Where' });

const browserKnn = new SequenceAccumulator();
assert.deepEqual(browserKnn.observe({ label: 'Hello' }), { label: 'Hello' });
assert.equal(browserKnn.observe({ label: 'Hello' }), null);
browserKnn.release();
assert.deepEqual(browserKnn.observe({ label: 'Hello' }), { label: 'Hello' });

console.log('Frontend sequence accumulator checks passed.');
