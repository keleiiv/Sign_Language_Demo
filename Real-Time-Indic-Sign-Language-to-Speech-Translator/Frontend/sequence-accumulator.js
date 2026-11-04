(function attachSequenceAccumulator(root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.ISLSequenceAccumulator = api;
}(typeof window === 'undefined' ? globalThis : window, function createApi() {
  class SequenceAccumulator {
    constructor(releaseAfter = 2) {
      this.releaseAfter = releaseAfter;
      this.pendingLabel = null;
      this.pendingCount = 0;
      this.lastEmittedLabel = null;
      this.releaseCount = 0;
    }

    observe(prediction, confirmations = 1) {
      if (!prediction || typeof prediction.label !== 'string' ||
          !prediction.label.trim()) {
        this.pendingLabel = null;
        this.pendingCount = 0;
        this.releaseCount += 1;
        if (this.releaseCount >= this.releaseAfter) this.release();
        return null;
      }

      this.releaseCount = 0;
      const label = prediction.label.trim();
      if (label === this.pendingLabel) {
        this.pendingCount += 1;
      } else {
        this.pendingLabel = label;
        this.pendingCount = 1;
      }
      if (this.pendingCount < confirmations ||
          label === this.lastEmittedLabel) return null;

      this.lastEmittedLabel = label;
      this.pendingLabel = null;
      this.pendingCount = 0;
      return { ...prediction, label };
    }

    release() {
      this.pendingLabel = null;
      this.pendingCount = 0;
      this.lastEmittedLabel = null;
      this.releaseCount = 0;
    }

    reset() {
      this.release();
    }
  }

  return { SequenceAccumulator };
}));
