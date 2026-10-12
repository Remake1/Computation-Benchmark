import type { AnyInputs, AnyParameters, Backend, Inputs, MatrixParameters, Parameters, WorkloadId } from './types.ts';
import { affine, K_VALUES } from '../workloads/affine.ts';
import { matmul, reference } from '../workloads/matmul.ts';
import { nextTask } from './runner.ts';

export interface SelfTestResult { backend: string; status: 'passed' | 'failed' | 'unsupported' | 'cancelled'; cases: number; reason: string | null }
export async function selfTest(backends: Backend<AnyParameters, AnyInputs>[], signal: AbortSignal, progress: (message: string) => void,
  workload: WorkloadId = 'W1'): Promise<SelfTestResult[]> {
  if (workload === 'W2') return matrixSelfTest(backends, signal, progress);
  const results: SelfTestResult[] = [];
  for (const backend of backends) {
    let cases = 0;
    let initialized = false;
    try {
      await backend.initialize(); initialized = true;
      for (const n of [1, 3, 4, 5, 63, 64, 65, 1023, 65537, 1048573]) {
        for (const k of n > 1023 ? [0] : K_VALUES) {
          if (signal.aborted) break;
          progress(`Self-test ${backend.id}: N=${n}, K=${k}`);
          const p: Parameters = { n, k, r: 10 };
          const inputs: Inputs = n === 1 ? { x: new Float32Array([1]), y: new Float32Array([0.25]) }
            : n === 3 ? { x: new Float32Array(3), y: new Float32Array(3) } : affine.generate(n, 12345);
          const output = new Float32Array(n);
          try {
            await backend.prepare(p); await backend.stage(inputs);
            await backend.warm(inputs, output);
            if (n === 1 && k === 0 && output[0] !== 0.75) throw new Error('Hand-checkable affine case failed.');
            if (!affine.validate(inputs, output, p).validation_passed) throw new Error(`Warm correctness failed: N=${n}, K=${k}`);
            await backend.resident(p.r); await backend.readOutput(output);
            if (!affine.validate(inputs, output, p).validation_passed) throw new Error(`Resident correctness failed: N=${n}, K=${k}`);
            cases += 2;
          } finally { backend.dispose(); }
          await nextTask();
        }
        if (signal.aborted) break;
      }
      results.push({ backend: backend.id, status: signal.aborted ? 'cancelled' : 'passed', cases, reason: null });
    } catch (error) { results.push({ backend: backend.id, status: initialized ? 'failed' : 'unsupported', cases, reason: String(error) }); }
    finally { backend.dispose(); }
  }
  return results;
}

async function matrixSelfTest(backends: Backend<AnyParameters, AnyInputs>[], signal: AbortSignal,
  progress: (message: string) => void): Promise<SelfTestResult[]> {
  const results: SelfTestResult[] = [];
  const configurations: MatrixParameters[] = [
    { m: 1, n: 1, k: 1, r: 100 },
    { m: 2, n: 2, k: 3, r: 10 }, // Hand-checkable multiplication.
    { m: 3, n: 5, k: 7, r: 10 }, // Zero matrix.
    { m: 3, n: 3, k: 3, r: 10 }, // Identity B.
    ...[63, 64, 65].map(n => ({ m: 1, n, k: 7, r: 10 })),
    { m: 7, n: 13, k: 5, r: 10 },
    { m: 4, n: 17, k: 9, r: 10 },
    { m: 17, n: 4, k: 9, r: 10 },
    ...[31, 32, 33].map(n => ({ m: n, n, k: n, r: 1 })), // SIMD block boundaries.
    { m: 127, n: 127, k: 127, r: 1 },
    { m: 4, n: 65, k: 2048, r: 1 },
  ];
  for (const backend of backends) {
    let cases = 0;
    let initialized = false;
    try {
      await backend.initialize(); initialized = true;
      for (const p of configurations) {
        if (signal.aborted) break;
        progress(`W2 self-test ${backend.id}: M=${p.m}, N=${p.n}, K=${p.k}, R=${p.r}`);
        const inputs = matmul.generate(p, 12345);
        if (p.m === 2) {
          inputs.a[0].set([1, 2, 3, 4, 5, 6]); inputs.b.set([7, 8, 9, 10, 11, 12]);
        } else if (p.m === 3 && p.n === 5) {
          for (const a of inputs.a) a.fill(0);
        } else if (p.m === 3 && p.n === 3) {
          inputs.b.set([1, 0, 0, 0, 1, 0, 0, 0, 1]);
        }
        inputs.reference = reference(inputs.a, inputs.b, p);
        const output = new Float32Array(matmul.outputLength(p, 'reuse'));
        const warm = output.subarray(0, matmul.outputLength(p, 'warm'));
        try {
          await backend.prepare(p); await backend.stage(inputs);
          await backend.warm(inputs, warm);
          if (p.m === 2 && !warm.every((value, i) => value === [58, 64, 139, 154][i])) throw new Error('Hand-checkable matrix case failed.');
          if (!matmul.validate(inputs, warm, p, 'warm').validation_passed) throw new Error('Warm matrix correctness failed.');
          await backend.reuse!(inputs, output);
          if (!matmul.validate(inputs, output, p, 'reuse').validation_passed) throw new Error('B-reuse correctness failed.');
          cases += 2;
        } finally { backend.dispose(); }
        await nextTask();
      }
      results.push({ backend: backend.id, status: signal.aborted ? 'cancelled' : 'passed', cases, reason: null });
    } catch (error) { results.push({ backend: backend.id, status: initialized ? 'failed' : 'unsupported', cases, reason: String(error) }); }
    finally { backend.dispose(); }
  }
  return results;
}
