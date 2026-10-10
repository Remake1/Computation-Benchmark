import type { Backend, Inputs, Parameters } from './types.ts';
import { affine, K_VALUES } from '../workloads/affine.ts';
import { nextTask } from './runner.ts';

export interface SelfTestResult { backend: string; status: 'passed' | 'failed' | 'unsupported' | 'cancelled'; cases: number; reason: string | null }
export async function selfTest(backends: Backend[], signal: AbortSignal, progress: (message: string) => void): Promise<SelfTestResult[]> {
  const results: SelfTestResult[] = [];
  for (const backend of backends) {
    let cases = 0;
    let initialized = false;
    try {
      await backend.initialize(); initialized = true;
      for (const n of [1, 3, 63, 64, 65, 1023, 65537, 1048573]) {
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
