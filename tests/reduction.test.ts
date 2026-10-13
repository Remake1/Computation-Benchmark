import test from 'node:test';
import assert from 'node:assert/strict';
import { reduction, reference, configurations, validateParameters, reductionStages } from '../src/workloads/reduction.ts';
import { random32 } from '../src/workloads/affine.ts';
import { JavaScriptBackend, reductionNumber, reductionF32 } from '../src/backends/javascript.ts';
import { dispatchWorkgroupGrid } from '../src/backends/dispatch.ts';
import { estimateMemory, runBenchmark } from '../src/benchmark/runner.ts';
import { summarize, jsonLines } from '../src/benchmark/results.ts';
import { ConfigurationSkip } from '../src/benchmark/types.ts';
import type { Backend, BackendId, ReductionInputs, ReductionParameters, RunOptions, SampleRecord } from '../src/benchmark/types.ts';

test('reduction generation is deterministic, batch-major, nonnegative float32 with cached double sums', () => {
  const p = { n: 5, b: 16, r: 10 };
  const inputs = reduction.generate(p, 12345), random = random32(12345);
  assert.deepEqual(inputs, reduction.generate(p, 12345));
  assert.notDeepEqual(inputs.x, reduction.generate(p, 42).x);
  assert.equal(inputs.x.length, p.n * p.b);
  for (const value of inputs.x) {
    assert.equal(value, Math.fround(random()));
    assert.ok(value >= 0 && value <= 1);
  }
  assert.deepEqual(inputs.reference, reference(inputs.x, p));
  assert.equal(reduction.outputLength(p, 'warm'), 16);
  assert.equal(reduction.outputLength(p, 'resident'), 16);
});

test('reduction presets cover single vectors, batches, and tails and reject invalid parameters', () => {
  const p = { n: 65, b: 16, r: 100 };
  assert.deepEqual(configurations('custom', p), [p]);
  assert.deepEqual(configurations('quick', p), [256, 4096, 65536, 1048576].map(n => ({ n, b: 1, r: 10 })));
  const sizes = configurations('size', p);
  assert.equal(sizes.length, 20);
  assert.ok([256, 2 ** 24, 1023, 65537, 1048573].every(n => sizes.some(p => p.n === n)));
  assert.ok(sizes.every(p => p.b === 1 && p.r === 100));
  const batches = configurations('batch', p);
  assert.equal(batches.length, 9);
  for (const b of [1, 16, 256]) for (const n of [256, 4096, 65536]) assert.ok(batches.some(p => p.b === b && p.n === n));
  for (const changes of [{ n: 0 }, { n: 1.5 }, { n: 2 ** 24 + 1 }, { b: 2 }, { r: 2 }]) {
    assert.throws(() => validateParameters({ ...p, ...changes }, 1));
  }
  for (const seed of [0, -1, 1.5, 2 ** 32]) assert.throws(() => validateParameters(p, seed));
  assert.throws(() => configurations('shape', p));
});

test('JS kernels sum every batch independently, including lane and stage boundaries', () => {
  for (const kernel of [reductionNumber, reductionF32]) {
    for (const n of [1, 3, 4, 5, 63, 64, 65, 4096, 4097, 65537, 2 ** 20]) {
      const p = { n, b: 16, r: 10 };
      const inputs = reduction.generate(p, 12345), output = new Float32Array(p.b).fill(NaN);
      kernel(inputs.x, output, p);
      assert.equal(reduction.validate(inputs, output, p).validation_passed, true, `N=${n}`);
      assert.ok(output.every(Number.isFinite));
      inputs.x.fill(0); inputs.reference.fill(0);
      kernel(inputs.x, output, p);
      assert.deepEqual(output, new Float32Array(p.b));
    }
    const p = { n: 3, b: 16, r: 1 }, x = new Float32Array(48);
    for (let batch = 0; batch < 16; batch++) x.set([batch, 1, 2], batch * 3);
    const output = new Float32Array(16);
    kernel(x, output, p);
    assert.deepEqual([...output], Array.from({ length: 16 }, (_, i) => i + 3));
  }
});

test('reduction validation rejects incomplete, corrupt, and nonfinite sums', () => {
  const p = { n: 3, b: 16, r: 1 }, inputs = reduction.generate(p, 42);
  const output = Float32Array.from(inputs.reference);
  assert.equal(reduction.validate(inputs, output, p).validation_passed, true);
  assert.equal(reduction.validate(inputs, output.subarray(0, 15), p).validation_passed, false);
  assert.equal(reduction.validate({ ...inputs, x: inputs.x.subarray(1) }, output, p).validation_passed, false);
  assert.equal(reduction.validate({ ...inputs, reference: inputs.reference.subarray(1) }, output, p).validation_passed, false);
  for (const value of [NaN, Infinity, 999]) {
    const corrupt = output.slice(); corrupt[15] = value;
    assert.equal(reduction.validate(inputs, corrupt, p).validation_passed, false);
  }
});

test('stage planning includes N=1, progresses at tree boundaries, and flattens large batched dispatches', () => {
  assert.deepEqual(reductionStages(1), [1]);
  assert.deepEqual(reductionStages(64), [1]);
  assert.deepEqual(reductionStages(65), [2, 1]);
  assert.deepEqual(reductionStages(4096), [64, 1]);
  assert.deepEqual(reductionStages(4097), [65, 2, 1]);
  assert.deepEqual(reductionStages(2 ** 24), [262144, 4096, 64, 1]);
  const chunks = reductionStages(65536)[0], groups = chunks * 256;
  const grid = dispatchWorkgroupGrid(groups, 65535);
  assert.equal(grid.y, 5);
  assert.ok(grid.x * grid.y >= groups);
  // Extra grid groups are excluded as entire batches; chunk never crosses a vector.
  const last = groups - 1;
  assert.equal(Math.floor(last / chunks), 255);
  assert.equal(last % chunks, chunks - 1);
  assert.throws(() => dispatchWorkgroupGrid(groups, 2));
  assert.throws(() => reductionStages(0));
});

test('JS backend resident calls restart from original vectors and can switch workloads', async () => {
  for (const f32 of [false, true]) {
    const backend = new JavaScriptBackend(f32);
    const p = { n: 65, b: 16, r: 100 }, inputs = reduction.generate(p, 42), output = new Float32Array(p.b);
    backend.prepare(p); backend.stage(inputs); backend.warm(inputs, output);
    const expected = output.slice();
    backend.resident(p.r); backend.readOutput(output);
    assert.deepEqual(output, expected);
    inputs.x.fill(0); backend.warm(inputs, output);
    assert.ok(output.every(value => value === 0));
    backend.prepare({ n: 3, k: 0, r: 1 });
    backend.warm({ x: new Float32Array([1, 2, 3]), y: new Float32Array(3) }, new Float32Array(3));
    backend.prepare({ m: 1, n: 1, k: 1, r: 1 });
    const matrix = new Float32Array(1);
    backend.warm({ a: [new Float32Array([2])], b: new Float32Array([3]), reference: new Float64Array([6]) }, matrix);
    assert.equal(matrix[0], 6);
    backend.prepare(p); backend.stage(inputs); backend.resident(10); backend.readOutput(output);
    assert.ok(output.every(value => value === 0));
    backend.dispose(); backend.dispose();
  }
});

class FakeReductionBackend implements Backend<ReductionParameters, ReductionInputs> {
  readonly metadata: Record<string, unknown> = {};
  readonly implementations = { W3: { tier: 'baseline' as const, tuning: { summation_order: 'test sequential' } } };
  events: string[] = [];
  unsupported = false; skipped = false; corrupt = false;
  private p!: ReductionParameters;
  private x = new Float32Array(0);
  private output = new Float32Array(0);
  readonly id: BackendId;
  constructor(id: BackendId = 'wasm-scalar') { this.id = id; }
  async initialize() { if (this.unsupported) throw new Error('unavailable'); }
  prepare(p: ReductionParameters) {
    this.events.push('prepare'); if (this.skipped) throw new ConfigurationSkip('device limit');
    this.p = p; this.output = new Float32Array(p.b);
  }
  stage(inputs: ReductionInputs) { this.events.push('stage'); this.x = inputs.x.slice(); }
  warm(inputs: ReductionInputs, output: Float32Array) {
    this.events.push('warm'); this.stage(inputs); reductionF32(this.x, output, this.p);
    if (this.corrupt) output[output.length - 1] = NaN;
    return 10;
  }
  resident(r: number) {
    this.events.push(`resident:${r}`);
    for (let i = 0; i < r; i++) reductionF32(this.x, this.output, this.p);
    return 20;
  }
  readOutput(output: Float32Array) { this.events.push('read'); output.set(this.output); }
  dispose() { this.events.push('dispose'); }
}
const defaults = (): RunOptions<ReductionParameters> => ({
  experimentId: 'w3', sessionId: 'session', seed: 42, schedulingSeed: 1, samples: 2, warmups: 2,
  budgetMs: 30000, memoryBudgetBytes: 2 ** 30, timerResolutionMs: 0.1,
  scenarios: ['warm', 'resident'], configurations: [{ n: 65, b: 16, r: 10 }],
});
async function run(backends: Backend<ReductionParameters, ReductionInputs>[], overrides: Partial<RunOptions<ReductionParameters>> = {}) {
  const records: SampleRecord[] = []; let clock = 0;
  await runBenchmark(reduction, backends, { ...defaults(), ...overrides }, new AbortController().signal, {
    record: record => records.push(record), progress: () => {}, yield: async () => {}, now: () => clock++,
  });
  return records;
}

test('W3 records final-only transfers, excludes resident readback, and exports GPU stage count', async () => {
  const backend = new FakeReductionBackend('webgpu-baseline'), records = await run([backend]);
  assert.equal(records.length, 4);
  assert.ok(records.every(record => record.status === 'ok' && record.validation_passed));
  const warm = records[0], resident = records[1];
  assert.deepEqual(warm.parameters, { n: 65, b: 16, r: 10 });
  assert.equal(warm.input_bytes, 4 * 65 * 16); assert.equal(warm.output_bytes, 4 * 16);
  assert.equal(warm.upload_bytes, warm.input_bytes); assert.equal(warm.download_bytes, 4 * 16);
  assert.equal(warm.staged_bytes, reduction.logicalBytes({ n: 65, b: 16, r: 10 }));
  assert.equal(warm.tuning.reduction_stages, 2);
  assert.equal(resident.elapsed_ms, 20); assert.equal(resident.repetitions, 10);
  assert.equal(resident.staged_bytes, 0); assert.equal(resident.validation_readback_bytes, 4 * 16);
  assert.equal(resident.resident_setup_upload_bytes, warm.input_bytes);
  assert.equal(backend.events.filter(event => event === 'read').length, 3);
  const exported = jsonLines({ workload: 'W3', schema_version: 3 }, records).trim().split('\n').map(line => JSON.parse(line));
  assert.equal(exported[0].schema_version, 3); assert.deepEqual(exported.slice(1), records);
});

test('W3 summaries include B in grouping/matching and compute repeated element throughput', async () => {
  const records = await run([new FakeReductionBackend('js-number'), new FakeReductionBackend('wasm-scalar')], {
    configurations: [{ n: 65, b: 1, r: 10 }, { n: 65, b: 16, r: 10 }],
  });
  const rows = summarize(records);
  assert.equal(rows.length, 8);
  for (const row of rows) {
    assert.ok('b' in row.parameters);
    assert.equal(row.elementsPerSecond, 65 * row.parameters.b * (row.scenario === 'warm' ? 1 : 10) * 1000 / row.median!);
    assert.equal(row.bandwidthGBs, null); assert.equal(row.gflops, null);
    assert.equal(row.numberSpeedup, 1);
  }
});

test('W3 memory counts references, partials, staging, and both retained Wasm heaps before generation', async () => {
  const p = { n: 65537, b: 16, r: 10 }, gpu = new FakeReductionBackend('webgpu-baseline');
  const scalar = new FakeReductionBackend(), simd = new FakeReductionBackend('wasm-simd');
  scalar.metadata.heap_bytes = 2 ** 27; simd.metadata.heap_bytes = 2 ** 28;
  const input = 4 * p.n * p.b, output = 4 * p.b, stages = reductionStages(p.n);
  const gpuBytes = 2 * input + stages.reduce((sum, n) => sum + 4 * p.b * n, 0) + output + 16 * stages.length;
  assert.equal(estimateMemory(p, [scalar, simd, gpu]), 16 * 2 ** 20 + input + 3 * output + 2 ** 27 + 2 ** 28 + gpuBytes);
  let generated = false;
  const backend = new FakeReductionBackend(), records: SampleRecord[] = [];
  await runBenchmark({ ...reduction, generate(p, seed) { generated = true; return reduction.generate(p, seed); } }, [backend],
    { ...defaults(), memoryBudgetBytes: 1 }, new AbortController().signal,
    { record: record => records.push(record), progress: () => {} });
  assert.equal(generated, false); assert.ok(!backend.events.includes('prepare'));
  assert.ok(records.every(record => record.status === 'skipped'));
});

test('W3 isolates unavailable/skipped/corrupt backends, rejects reuse, and retains budget/cancellation records', async () => {
  const unsupported = new FakeReductionBackend('webgpu-baseline'); unsupported.unsupported = true;
  const skipped = new FakeReductionBackend('wasm-simd'); skipped.skipped = true;
  const corrupt = new FakeReductionBackend(); corrupt.corrupt = true;
  const records = await run([unsupported, skipped, corrupt, new FakeReductionBackend('js-number')]);
  assert.ok(records.some(record => record.status === 'unsupported'));
  assert.ok(records.some(record => record.status === 'skipped'));
  assert.ok(records.some(record => record.status === 'validation-failed'));
  assert.equal(records.filter(record => record.status === 'ok').length, 4);
  assert.equal(summarize(records).find(row => row.backend === corrupt.id)!.count, 0);
  await assert.rejects(() => run([new FakeReductionBackend()], { scenarios: ['reuse'] }));
  const budget = await run([new FakeReductionBackend()], { budgetMs: 1 });
  assert.equal(budget.filter(record => record.status === 'ok').length, 1);
  assert.equal(budget.filter(record => record.status === 'budget-exhausted').length, 2);
  const controller = new AbortController(), cancelled: SampleRecord[] = [];
  await runBenchmark(reduction, [new FakeReductionBackend()], defaults(), controller.signal, {
    record: record => { cancelled.push(record); if (record.status === 'ok') controller.abort(); }, progress: () => {}, yield: async () => {},
  });
  assert.equal(cancelled.filter(record => record.status === 'ok').length, 1);
  assert.equal(cancelled.filter(record => record.status === 'cancelled').length, 2);
});
