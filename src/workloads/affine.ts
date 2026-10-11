import type { Accuracy, Inputs, Parameters, Preset } from '../benchmark/types.ts';

export const K_VALUES = [0, 4, 16, 64, 256];
export const R_VALUES = [1, 10, 100];
export const MAX_N = 2 ** 24;
export const GENERATOR_VERSION = 'xorshift32-v1-interleaved-f32';

export function random32(seed: number): () => number {
  let state = seed >>> 0;
  if (state === 0) throw new Error('Seed must be a nonzero uint32.');
  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 2 ** 32;
  };
}

export function validateParameters(p: Parameters, seed: number): void {
  if (!Number.isInteger(p.n) || p.n < 1 || p.n > MAX_N) throw new Error(`N must be an integer from 1 to ${MAX_N}.`);
  if (!K_VALUES.includes(p.k)) throw new Error('K must be 0, 4, 16, 64, or 256.');
  if (!R_VALUES.includes(p.r)) throw new Error('R must be 1, 10, or 100.');
  if (!Number.isInteger(seed) || seed < 1 || seed > 0xffffffff) throw new Error('Seed must be a nonzero uint32.');
}

export function configurations(preset: Preset, p: Parameters): Parameters[] {
  if (preset === 'quick') return [256, 4096, 65536, 1048576].map(n => ({ n, k: 0, r: 10 }));
  if (preset === 'size') {
    const sizes = [...Array.from({ length: 17 }, (_, i) => 2 ** (i + 8)), 1023, 65537, 1048573];
    return sizes.sort((a, b) => a - b).map(n => ({ ...p, n }));
  }
  if (preset === 'intensity') return K_VALUES.map(k => ({ ...p, k }));
  return [{ ...p }];
}

function generate(parameters: number | Parameters, seed: number): Inputs {
  const n = typeof parameters === 'number' ? parameters : parameters.n;
  const random = random32(seed);
  const x = new Float32Array(n);
  const y = new Float32Array(n);
  for (let i = 0; i < n; i++) { x[i] = random(); y[i] = random(); }
  return { x, y };
}

function validate(inputs: Inputs, output: Float32Array, p: Parameters): Accuracy {
  if (output.length !== p.n || inputs.x.length !== p.n || inputs.y.length !== p.n) {
    return { validation_passed: false, max_abs_error: null, relative_l2_error: null };
  }
  let passed = true;
  let maxError = 0;
  let squaredError = 0;
  let squaredReference = 0;
  for (let i = 0; i < p.n; i++) {
    let reference = 0.5 * inputs.x[i] + inputs.y[i];
    for (let step = 0; step < p.k; step++) reference = (255 * reference + 0.5) / 256;
    const error = Math.abs(output[i] - reference);
    if (!Number.isFinite(error)) return { validation_passed: false, max_abs_error: null, relative_l2_error: null };
    if (error > 1e-5 + 1e-4 * Math.abs(reference)) passed = false;
    maxError = Math.max(maxError, error);
    squaredError += error * error;
    squaredReference += reference * reference;
  }
  return {
    validation_passed: passed, max_abs_error: maxError,
    relative_l2_error: squaredReference === 0 ? (squaredError === 0 ? 0 : null) : Math.sqrt(squaredError / squaredReference),
  };
}

export const affine = {
  id: 'W1' as const, generate, validate, validateParameters,
  outputLength: (p: Parameters) => p.n,
  logicalBytes: (p: Parameters) => 12 * p.n,
};
