import { MAX_N, random32, R_VALUES } from './affine.ts';
import type { Accuracy, Preset, ReductionInputs, ReductionParameters, Workload } from '../benchmark/types.ts';

export const B_VALUES = [1, 16, 256];
export const GENERATOR_VERSION = 'xorshift32-v1-reduction-batch-major-f32';
export const REDUCTION_WORKGROUP_SIZE = 64;

// Includes an execution stage for N=1 so every call writes its final output.
export function reductionStages(n: number): number[] {
  if (!Number.isInteger(n) || n < 1 || n > MAX_N) throw new Error('Invalid reduction length.');
  const lengths: number[] = [];
  do { n = Math.ceil(n / REDUCTION_WORKGROUP_SIZE); lengths.push(n); } while (n > 1);
  return lengths;
}

export function validateParameters(p: ReductionParameters, seed: number): void {
  if (!Number.isInteger(p.n) || p.n < 1 || p.n > MAX_N) throw new Error(`N must be an integer from 1 to ${MAX_N}.`);
  if (!B_VALUES.includes(p.b)) throw new Error('B must be 1, 16, or 256.');
  if (!R_VALUES.includes(p.r)) throw new Error('R must be 1, 10, or 100.');
  if (!Number.isInteger(seed) || seed < 1 || seed > 0xffffffff) throw new Error('Seed must be a nonzero uint32.');
}

export function configurations(preset: Preset, p: ReductionParameters): ReductionParameters[] {
  if (preset === 'quick') return [256, 4096, 65536, 1048576].map(n => ({ n, b: 1, r: 10 }));
  if (preset === 'size') return [...Array.from({ length: 17 }, (_, i) => 2 ** (i + 8)), 1023, 65537, 1048573]
    .sort((a, b) => a - b).map(n => ({ n, b: 1, r: p.r }));
  if (preset === 'batch') return B_VALUES.flatMap(b => [256, 4096, 65536].map(n => ({ n, b, r: p.r })));
  if (preset !== 'custom') throw new Error('Unknown reduction preset.');
  return [{ ...p }];
}

// Independent double accumulation, never calling a measured backend kernel.
export function reference(x: Float32Array, p: ReductionParameters): Float64Array {
  const sums = new Float64Array(p.b);
  for (let batch = 0; batch < p.b; batch++) {
    for (let i = batch * p.n; i < (batch + 1) * p.n; i++) sums[batch] += x[i];
  }
  return sums;
}

function generate(p: ReductionParameters, seed: number): ReductionInputs {
  validateParameters(p, seed);
  const random = random32(seed);
  const x = new Float32Array(p.n * p.b);
  for (let i = 0; i < x.length; i++) x[i] = random();
  return { x, reference: reference(x, p) };
}

function validate(inputs: ReductionInputs, output: Float32Array, p: ReductionParameters): Accuracy {
  const failed: Accuracy = { validation_passed: false, max_abs_error: null, relative_l2_error: null };
  if (inputs.x.length !== p.n * p.b || inputs.reference.length !== p.b || output.length !== p.b) return failed;
  let passed = true, maxError = 0, squaredError = 0, squaredReference = 0;
  for (let i = 0; i < p.b; i++) {
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

export const reduction: Workload<ReductionParameters, ReductionInputs> = {
  id: 'W3', generate, validate, validateParameters,
  outputLength: p => p.b,
  logicalBytes: p => 4 * (p.n * p.b + p.b),
};
