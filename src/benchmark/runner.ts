import { random32, validateParameters } from '../workloads/affine.ts';
import type { Accuracy, Backend, BackendId, Parameters, RunOptions, SampleRecord, Scenario, Status, Workload } from './types.ts';

export interface RunnerHooks {
  record(record: SampleRecord): void;
  progress(message: string): void;
  ready?(backends: Backend[]): void;
  now?: () => number;
  yield?: () => Promise<void>;
}
const unavailableAccuracy = { validation_passed: null, max_abs_error: null, relative_l2_error: null };
export const nextTask = (): Promise<void> => new Promise(resolve => setTimeout(resolve, 0));

export function estimateMemory(n: number, backends: Backend[]): number {
  // Shared canonical inputs/output; resident CPU output; Wasm growth headroom;
  // four GPU buffers plus allowance for writeBuffer staging; fixed overhead.
  return 16 * 2 ** 20 + n * 12 + backends.reduce((bytes, backend) => {
    if (backend.id === 'wasm-scalar') return bytes + Math.max(n * 16, Number(backend.metadata.heap_bytes) || 0);
    return bytes + n * (backend.id === 'webgpu-baseline' ? 24 : 4);
  }, 0);
}

export function shuffle<T>(values: T[], random: () => number): T[] {
  const result = [...values];
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}

export function timerResolution(): number {
  let previous = performance.now();
  let smallest = Infinity;
  for (let i = 0; i < 10000; i++) {
    const current = performance.now();
    const delta = current - previous;
    if (delta > 0) smallest = Math.min(smallest, delta);
    previous = current;
  }
  return Number.isFinite(smallest) ? smallest : 1;
}

export function validateRun(options: RunOptions, backends: Backend[]): void {
  if (!backends.length || !options.scenarios.length || !options.configurations.length) throw new Error('Select at least one backend, scenario, and configuration.');
  if (new Set(backends.map(b => b.id)).size !== backends.length || new Set(options.scenarios).size !== options.scenarios.length) throw new Error('Duplicate backend or scenario.');
  if (options.scenarios.some(s => s !== 'warm' && s !== 'resident')) throw new Error('Unknown timing scenario.');
  if (!Number.isInteger(options.samples) || options.samples < 1 || options.samples > 10000) throw new Error('Samples must be an integer from 1 to 10000.');
  if (!Number.isInteger(options.warmups) || options.warmups < 0) throw new Error('Invalid warmup count.');
  if (!Number.isFinite(options.budgetMs) || options.budgetMs <= 0 || !Number.isFinite(options.memoryBudgetBytes) || options.memoryBudgetBytes <= 0) throw new Error('Budgets must be positive.');
  if (!Number.isFinite(options.timerResolutionMs) || options.timerResolutionMs <= 0) throw new Error('Invalid timer resolution.');
  random32(options.schedulingSeed);
  for (const p of options.configurations) validateParameters(p, options.seed);
}

export async function runBenchmark(workload: Workload, backends: Backend[], options: RunOptions,
  signal: AbortSignal, hooks: RunnerHooks): Promise<void> {
  validateRun(options, backends);
  const now = hooks.now ?? (() => performance.now());
  const yieldTask = hooks.yield ?? nextTask;
  const random = random32(options.schedulingSeed);
  const unavailable = new Map<BackendId, string>();
  const staged = new Set<BackendId>();
  let sampleId = 0;
  const emit = (backend: Backend, p: Parameters, scenario: Scenario, status: Status,
    reason: string | null = null, elapsed: number | null = null, accuracy: Accuracy | null = null, index: number | null = null): void => {
    const remote = backend.id === 'wasm-scalar' || backend.id === 'webgpu-baseline';
    const upload = remote && scenario === 'warm' ? p.n * 8 : 0;
    const download = remote && scenario === 'warm' ? p.n * 4 : 0;
    hooks.record({
      record_type: 'sample', experiment_id: options.experimentId, environment_id: options.experimentId,
      session_id: options.sessionId, timestamp: new Date().toISOString(), sample_id: sampleId++, sample_index: index,
      seed: options.seed, scheduling_seed: options.schedulingSeed, workload: workload.id, parameters: { ...p }, backend: backend.id,
      implementation_tier: 'baseline', timing_scenario: scenario, repetitions: scenario === 'warm' ? 1 : p.r,
      elapsed_ms: elapsed, gpu_elapsed_ms: null, input_bytes: p.n * 8, output_bytes: p.n * 4,
      upload_bytes: elapsed === null ? null : upload, download_bytes: elapsed === null ? null : download,
      staged_bytes: elapsed === null ? null : upload + download,
      resident_setup_upload_bytes: remote && staged.has(backend.id) ? p.n * 8 : 0,
      validation_readback_bytes: scenario === 'resident' && elapsed !== null ? p.n * 4 : 0,
      allocation_policy: 'preallocated per configuration', warmup_policy: { iterations: options.warmups },
      cache_policy: 'warm backend; browser/driver caches uncontrolled', worker_count: 0,
      compiler_flags: backend.metadata.compiler_flags ?? null, source_revision: options.sourceRevision ?? backend.metadata.source_revision ?? null,
      tuning: backend.id === 'webgpu-baseline' ? { workgroup_size: 64, elements_per_invocation: 1 } : {},
      timer_resolution_ms: options.timerResolutionMs,
      near_timer_granularity: elapsed !== null && elapsed < 10 * options.timerResolutionMs,
      ...(accuracy ?? unavailableAccuracy), status, skip_reason: reason,
    });
  };
  try {
    for (const backend of backends) {
      try { await backend.initialize(); }
      catch (error) { unavailable.set(backend.id, String(error)); }
    }
    hooks.ready?.(backends);
    for (const p of options.configurations) {
      staged.clear();
      hooks.progress(`Preparing N=${p.n}, K=${p.k}, R=${p.r}`);
      const active = backends.filter(backend => !unavailable.has(backend.id));
      const stopped = new Set<BackendId>();
      const budgets = new Map<BackendId, number>(active.map(backend => [backend.id, 0]));
      const finish = (backend: Backend, status: Status, reason: string): void => {
        for (const scenario of options.scenarios) emit(backend, p, scenario, status, reason);
        stopped.add(backend.id);
      };
      for (const backend of backends) {
        const reason = unavailable.get(backend.id);
        if (reason) finish(backend, 'unsupported', reason);
      }
      if (signal.aborted) {
        for (const backend of active) finish(backend, 'cancelled', 'Cancelled before configuration.');
        continue;
      }
      if (estimateMemory(p.n, active) > options.memoryBudgetBytes) {
        for (const backend of active) finish(backend, 'skipped', 'Estimated combined memory exceeds configured budget.');
        continue;
      }
      let inputs;
      let output: Float32Array;
      try { inputs = workload.generate(p.n, options.seed); output = new Float32Array(p.n); }
      catch (error) {
        for (const backend of active) finish(backend, 'error', `Canonical allocation failed: ${error}`);
        continue;
      }
      try {
        for (const backend of active) {
          if (signal.aborted) { finish(backend, 'cancelled', 'Cancelled during preparation.'); continue; }
          try {
            await backend.prepare(p);
            await backend.stage(inputs);
            staged.add(backend.id);
            // Five executions per backend/configuration, cycling selected scenarios.
            for (let i = 0; i < options.warmups; i++) {
              const scenario = options.scenarios[i % options.scenarios.length];
              if (signal.aborted) break;
              if (scenario === 'warm') await backend.warm(inputs, output);
              else { await backend.resident(p.r); await backend.readOutput(output); }
              const accuracy = workload.validate(inputs, output, p);
              if (!accuracy.validation_passed) {
                emit(backend, p, scenario, 'validation-failed', 'Warmup validation failed.', null, accuracy);
                finish(backend, 'validation-failed', 'Stopped after warmup validation failure.');
                break;
              }
              await yieldTask();
            }
          } catch (error) { finish(backend, 'error', String(error)); backend.dispose(); }
        }
        for (let index = 0; index < options.samples; index++) {
          if (stopped.size === backends.length) break;
          for (const backend of shuffle(active, random)) {
            if (stopped.has(backend.id)) continue;
            for (const scenario of options.scenarios) {
              if (signal.aborted) { finish(backend, 'cancelled', 'Cancelled; partial samples retained.'); break; }
              if (budgets.get(backend.id)! >= options.budgetMs) { finish(backend, 'budget-exhausted', 'Sampling budget reached; partial samples retained.'); break; }
              hooks.progress(`N=${p.n}, K=${p.k}, ${backend.id}, ${scenario}: sample ${index + 1}/${options.samples}`);
              await yieldTask();
              if (signal.aborted) { finish(backend, 'cancelled', 'Cancelled; partial samples retained.'); break; }
              const start = now();
              try {
                const elapsed = scenario === 'warm' ? await backend.warm(inputs, output) : await backend.resident(p.r);
                if (!Number.isFinite(elapsed) || elapsed < 0) throw new Error('Backend returned an invalid duration.');
                if (scenario === 'resident') await backend.readOutput(output);
                const accuracy = workload.validate(inputs, output, p);
                emit(backend, p, scenario, accuracy.validation_passed ? 'ok' : 'validation-failed',
                  accuracy.validation_passed ? null : 'Complete output failed accuracy contract.', elapsed, accuracy, index);
                if (!accuracy.validation_passed) { finish(backend, 'validation-failed', 'Stopped after validation failure.'); break; }
              } catch (error) { finish(backend, 'error', String(error)); break; }
              finally { budgets.set(backend.id, budgets.get(backend.id)! + now() - start); }
            }
          }
        }
      } finally { for (const backend of active) backend.dispose(); }
    }
  } finally { for (const backend of backends) backend.dispose(); }
}
