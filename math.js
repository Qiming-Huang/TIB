/* Exact scalar bottleneck equations. The full model fits the residual through an encoder. */
(function (root) {
  "use strict";
  const sigmoid = value => 1 / (1 + Math.exp(-value));
  function mechanism({ beta, alpha, logVariance, maskLogit, residual }) {
    const variance = Math.exp(logVariance);
    const h = sigmoid(maskLogit);
    const mean = h; // Standardized original feature f = 1.
    const std = Math.sqrt(variance) * (1 - h);
    const posteriorVariance = std * std;
    const clampedVariance = Math.max(posteriorVariance, 1e-8);
    const kl = 0.5 * (mean * mean + clampedVariance - Math.log(clampedVariance) - 1);
    const weight = Math.pow(variance, alpha);
    const fittingContribution = weight * residual / variance;
    const normalization = weight * 0.5 * logVariance;
    const likelihood = weight * (0.5 * logVariance + residual / variance);
    const compression = beta * kl;
    return { variance, h, mean, std, posteriorVariance, kl, weight, fittingContribution, normalization, likelihood, compression,
      total: likelihood + compression, fittingCoefficient: Math.pow(variance, alpha - 1) };
  }
  function samples(mean, std, count, seed) {
    // Mulberry32 gives reproducible noise, including for seed 0.
    let state = Math.trunc(seed) >>> 0;
    function uniform() {
      state = (state + 0x6D2B79F5) >>> 0;
      let value = Math.imul(state ^ state >>> 15, 1 | state);
      value ^= value + Math.imul(value ^ value >>> 7, 61 | value);
      return ((value ^ value >>> 14) >>> 0) / 4294967296;
    }
    return Array.from({ length: count }, () => {
      const gaussian = Math.sqrt(-2 * Math.log(Math.max(uniform(), 1e-12))) * Math.cos(2 * Math.PI * uniform());
      return mean + std * gaussian;
    });
  }
  function learningRate(initial, finalRatio, step, steps, schedule) {
    if (schedule === "constant" || steps === 1) return initial;
    const progress = step / (steps - 1);
    const remaining = schedule === "cosine" ? (1 + Math.cos(Math.PI * progress)) / 2 : 1 - progress;
    return initial * (finalRatio + (1 - finalRatio) * remaining);
  }
  const api = { sigmoid, mechanism, samples, learningRate };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.TIBMath = api;
})(typeof window !== "undefined" ? window : globalThis);
