import { random32, R_VALUES } from './affine.ts';
import type { Accuracy, MatrixInputs, MatrixParameters, Preset, Scenario, Workload } from '../benchmark/types.ts';

export const MAX_DIMENSION = 2048;
export const GENERATOR_VERSION = 'xorshift32-v1-matmul-b-then-a-row-major-f32';
export const SQUARE_SIZES = [32, 64, 127, 128, 255, 256, 512, 513, 1024, 2048];

export function validateParameters(p: MatrixParameters, seed: number): void {
  for (const [name, value] of [['M', p.m], ['N', p.n], ['K', p.k]] as const) {
    if (!Number.isInteger(value) || value < 1 || value > MAX_DIMENSION) throw new Error(`${name} must be an integer from 1 to ${MAX_DIMENSION}.`);
  }
  if (!R_VALUES.includes(p.r)) throw new Error('R must be 1, 10, or 100.');
  if (!Number.isInteger(seed) || seed < 1 || seed > 0xffffffff) throw new Error('Seed must be a nonzero uint32.');
}

export function configurations(preset: Preset, p: MatrixParameters): MatrixParameters[] {
  if (preset === 'quick') return [32, 64, 128].map(n => ({ m: n, n, k: n, r: 1 }));
  if (preset === 'size') return SQUARE_SIZES.map(n => ({ m: n, n, k: n, r: p.r }));
  if (preset === 'shape') return [[256, 256, 256], [64, 1024, 256], [1024, 64, 256]].map(([m, n, k]) => ({ m, n, k, r: p.r }));
  if (preset !== 'custom') throw new Error('Unknown matrix preset.');
  return [{ ...p }];
}

// Separate double-precision reference, evaluated once before any measurements.
// Traverse rows of B and accumulate columns, independently of the measured dot-product loops.
export function reference(a: Float32Array[], b: Float32Array, p: MatrixParameters): Float64Array {
  const result = new Float64Array(p.r * p.m * p.n);
  for (let batch = 0; batch < p.r; batch++) {
    for (let row = 0; row < p.m; row++) {
      const offset = (batch * p.m + row) * p.n;
      for (let k = 0; k < p.k; k++) {
        const value = a[batch][row * p.k + k];
        for (let col = 0; col < p.n; col++) result[offset + col] += value * b[k * p.n + col];
      }
    }
  }
  return result;
}

function generate(p: MatrixParameters, seed: number): MatrixInputs {
  validateParameters(p, seed);
  const random = random32(seed);
  const fill = (length: number): Float32Array => {
    const values = new Float32Array(length);
    for (let i = 0; i < length; i++) values[i] = random() - 0.5;
    return values;
  };
  const b = fill(p.k * p.n);
  const a = Array.from({ length: p.r }, () => fill(p.m * p.k));
  return { a, b, reference: reference(a, b, p) };
}

function validate(inputs: MatrixInputs, output: Float32Array, p: MatrixParameters, scenario: Scenario = 'warm'): Accuracy {
  const length = p.m * p.n * (scenario === 'reuse' ? p.r : 1);
  const failed: Accuracy = { validation_passed: false, max_abs_error: null, relative_l2_error: null };
  if (output.length !== length || inputs.a.length !== p.r || inputs.a.some(a => a.length !== p.m * p.k) ||
      inputs.b.length !== p.k * p.n || inputs.reference.length !== p.r * p.m * p.n) return failed;
  let passed = true;
  let maxError = 0;
  let squaredError = 0;
  let squaredReference = 0;
  for (let i = 0; i < length; i++) {
    const expected = inputs.reference[i];
    const error = Math.abs(output[i] - expected);
    if (!Number.isFinite(error)) return failed;
    if (error > 1e-5 + 1e-4 * Math.abs(expected)) passed = false;
    maxError = Math.max(maxError, error);
    squaredError += error * error;
    squaredReference += expected * expected;
  }
  return { validation_passed: passed, max_abs_error: maxError,
    relative_l2_error: squaredReference === 0 ? (squaredError === 0 ? 0 : null) : Math.sqrt(squaredError / squaredReference) };
}

export const matmul: Workload<MatrixParameters, MatrixInputs> = {
  id: 'W2', generate, validate, validateParameters,
  outputLength: (p, scenario) => p.m * p.n * (scenario === 'reuse' ? p.r : 1),
  logicalBytes: p => 4 * (p.m * p.k + p.k * p.n + p.m * p.n),
};
