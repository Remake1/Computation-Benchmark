import test from 'node:test';
import assert from 'node:assert/strict';
import { runBenchmark, estimateMemory, shuffle } from '../src/benchmark/runner.ts';
import { summarize, jsonLines } from '../src/benchmark/results.ts';
import { affine, random32 } from '../src/workloads/affine.ts';
import { affineNumber } from '../src/backends/javascript.ts';
import type { Backend, BackendId, Inputs, Parameters, RunOptions, SampleRecord } from '../src/benchmark/types.ts';

class FakeBackend implements Backend {
  readonly metadata = {};
  readonly id: BackendId;
  events: string[] = [];
  failInitialization = false;
  failPrepare = false;
  corrupt = false;
  inputs!: Inputs;
  output = new Float32Array(0);
  p!: Parameters;
  constructor(id: BackendId = 'js-number') { this.id = id; }
  async initialize() { this.events.push('initialize'); if (this.failInitialization) throw new Error('unavailable'); }
  prepare(p: Parameters) { this.events.push('prepare'); this.p = p; if (this.failPrepare) throw new Error('allocation failed'); this.output = new Float32Array(p.n); }
  stage(inputs: Inputs) { this.events.push('stage'); this.inputs = inputs; }
  warm(inputs: Inputs, out: Float32Array) { this.events.push('warm'); affineNumber(inputs.x, inputs.y, out, this.p.k); if (this.corrupt) out[0] = NaN; return 10; }
  resident(r: number) { this.events.push(`resident:${r}`); affineNumber(this.inputs.x, this.inputs.y, this.output, this.p.k); return 20; }
  readOutput(out: Float32Array) { this.events.push('read'); out.set(this.output); }
  dispose() { this.events.push('dispose'); }
}
const defaults = (): RunOptions => ({ experimentId: 'run', sessionId: 'session', seed: 12345, schedulingSeed: 42,
  samples: 3, warmups: 5, budgetMs: 30000, memoryBudgetBytes: 2 ** 30, timerResolutionMs: 0.1,
  scenarios: ['warm', 'resident'], configurations: [{ n: 65, k: 0, r: 10 }] });
async function run(backends: Backend[], overrides: Partial<RunOptions> = {}, onRecord?: (record: SampleRecord) => void) {
  const records: SampleRecord[] = [];
  let clock = 0;
  await runBenchmark(affine, backends, { ...defaults(), ...overrides }, new AbortController().signal, {
    record: record => { records.push(record); onRecord?.(record); }, progress: () => {}, now: () => clock++, yield: async () => {},
  });
  return records;
}

test('runner warms separately, stages before measurement, and excludes readback from resident elapsed', async () => {
  const backend = new FakeBackend('wasm-scalar');
  const records = await run([backend]);
  assert.equal(records.length, 6);
  assert.ok(records.every(record => record.status === 'ok' && record.validation_passed));
  assert.deepEqual(backend.events.slice(0, 3), ['initialize', 'prepare', 'stage']);
  assert.equal(backend.events.filter(event => event === 'warm').length, 6);
  assert.equal(backend.events.filter(event => event === 'resident:10').length, 5);
  assert.equal(backend.events.filter(event => event === 'read').length, 5);
  const warm = records.find(record => record.timing_scenario === 'warm')!;
  const resident = records.find(record => record.timing_scenario === 'resident')!;
  assert.equal(warm.elapsed_ms, 10);
  assert.equal(warm.staged_bytes, 65 * 12);
  assert.equal(resident.elapsed_ms, 20);
  assert.equal(resident.repetitions, 10);
  assert.equal(resident.staged_bytes, 0);
  assert.equal(resident.validation_readback_bytes, 65 * 4);
  assert.equal(resident.gpu_elapsed_ms, null);
  assert.equal(backend.events.at(-1), 'dispose');
});

test('unsupported and allocation-failing backends do not prevent other measurements', async () => {
  const unsupported = new FakeBackend('webgpu-baseline'); unsupported.failInitialization = true;
  const broken = new FakeBackend('wasm-scalar'); broken.failPrepare = true;
  const good = new FakeBackend();
  const records = await run([unsupported, broken, good]);
  assert.equal(records.filter(r => r.status === 'unsupported').length, 2);
  assert.ok(records.filter(r => r.status === 'unsupported').every(r => r.validation_passed === null && r.staged_bytes === null));
  assert.equal(records.filter(r => r.status === 'error').length, 2);
  assert.equal(records.filter(r => r.status === 'ok').length, 6);
  assert.ok(broken.events.includes('dispose'));
});

test('validation failure stops subsequent samples and never contributes to rankings', async () => {
  const backend = new FakeBackend(); backend.corrupt = true;
  const records = await run([backend], { warmups: 0 });
  assert.equal(backend.events.filter(event => event === 'warm').length, 1);
  assert.ok(records.every(r => r.status === 'validation-failed'));
  assert.equal(summarize(records)[0].count, 0);
});

test('a failing warmup exports its accuracy and stops before any timed sample', async () => {
  const backend = new FakeBackend(); backend.corrupt = true;
  const records = await run([backend]);
  assert.ok(records.every(record => record.elapsed_ms === null));
  assert.ok(records.some(record => record.validation_passed === false));
  assert.equal(backend.events.filter(event => event === 'warm').length, 1);
});

test('memory skips occur before generating or allocating configuration inputs', async () => {
  const backend = new FakeBackend();
  const records = await run([backend], { memoryBudgetBytes: 1 });
  assert.ok(records.every(r => r.status === 'skipped'));
  assert.ok(!backend.events.includes('prepare'));
  assert.ok(estimateMemory(2 ** 24, [new FakeBackend('wasm-scalar'), new FakeBackend('webgpu-baseline')]) > 2 ** 29);
});

test('sampling budget preserves completed samples and explicitly records truncation', async () => {
  const records = await run([new FakeBackend()], { budgetMs: 1 });
  assert.equal(records.filter(r => r.status === 'ok').length, 1);
  assert.equal(records.filter(r => r.status === 'budget-exhausted').length, 2);
});

test('cancellation retains completed samples and identifies remaining configurations', async () => {
  const records: SampleRecord[] = [];
  const controller = new AbortController();
  const backend = new FakeBackend();
  await runBenchmark(affine, [backend], { ...defaults(), configurations: [{ n: 65, k: 0, r: 10 }, { n: 64, k: 4, r: 1 }] }, controller.signal, {
    progress: () => {}, yield: async () => {}, record: r => { records.push(r); if (r.status === 'ok') controller.abort(); },
  });
  assert.equal(records.filter(r => r.status === 'ok').length, 1);
  assert.equal(records.filter(r => r.status === 'cancelled').length, 4);
  assert.ok(backend.events.includes('dispose'));
});

test('matched backend order is reproducible and independent of input generation', () => {
  const first = random32(42); const second = random32(42);
  const a = Array.from({ length: 5 }, () => shuffle([1, 2, 3, 4], first));
  const b = Array.from({ length: 5 }, () => shuffle([1, 2, 3, 4], second));
  assert.deepEqual(a, b);
  assert.ok(new Set(a.map(order => order.join(','))).size > 1);
});

test('summary and JSONL use exactly the retained valid raw samples', async () => {
  const records = await run([new FakeBackend(), new FakeBackend('wasm-scalar')]);
  const rows = summarize(records);
  assert.equal(rows.length, 4);
  const row = rows.find(row => row.backend === 'wasm-scalar' && row.scenario === 'resident')!;
  assert.equal(row.count, 3);
  assert.equal(row.median, 20);
  assert.equal(row.iqr, 0);
  assert.equal(row.perOperation, 2);
  assert.equal(row.numberSpeedup, 1);
  assert.equal(row.bandwidthGBs, 12 * 65 * 10 / (20 * 1e6));
  const exported = jsonLines({ record_type: 'environment', session_id: 'session' }, records).trim().split('\n').map(line => JSON.parse(line));
  assert.equal(exported.length, records.length + 1);
  assert.deepEqual(exported.slice(1), records);
});

test('invalid run options fail before backend initialization', async () => {
  const backend = new FakeBackend();
  await assert.rejects(() => run([backend], { samples: 0 }));
  await assert.rejects(() => run([backend], { scenarios: [] }));
  assert.deepEqual(backend.events, []);
});
