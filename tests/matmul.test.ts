import test from 'node:test';
import assert from 'node:assert/strict';
import { matmul, configurations, GENERATOR_VERSION, reference, SQUARE_SIZES, validateParameters } from '../src/workloads/matmul.ts';
import { random32 } from '../src/workloads/affine.ts';
import { JavaScriptBackend, matmulF32, matmulNumber } from '../src/backends/javascript.ts';
import { runBenchmark, estimateMemory } from '../src/benchmark/runner.ts';
import { jsonLines, summarize } from '../src/benchmark/results.ts';
import { ConfigurationSkip } from '../src/benchmark/types.ts';
import type { Backend, BackendId, MatrixInputs, MatrixParameters, RunOptions, SampleRecord } from '../src/benchmark/types.ts';

test('matrix inputs are deterministic float32 values with B generated before all different A inputs', () => {
  const p = { m: 2, n: 3, k: 4, r: 10 };
  const inputs = matmul.generate(p, 12345);
  assert.deepEqual(inputs, matmul.generate(p, 12345));
  assert.notDeepEqual(inputs, matmul.generate(p, 42));
  assert.notDeepEqual(inputs.a[0], inputs.a[1]);
  assert.match(GENERATOR_VERSION, /b-then-a/);
  const random = random32(12345);
  for (const values of [inputs.b, ...inputs.a]) {
    for (const value of values) {
      assert.equal(value, Math.fround(random() - 0.5));
      assert.ok(value >= -0.5 && value <= 0.5);
    }
  }
  assert.equal(inputs.reference.length, p.r * p.m * p.n);
});

test('matrix parameters and presets include all specified dimensions, shapes, and tails', () => {
  const p = { m: 17, n: 65, k: 7, r: 100 };
  validateParameters(p, 0xffffffff);
  for (const field of ['m', 'n', 'k'] as const) {
    for (const value of [0, -1, 0.5, NaN, Infinity, 2049]) assert.throws(() => validateParameters({ ...p, [field]: value }, 1));
  }
  assert.throws(() => validateParameters({ ...p, r: 2 }, 1));
  for (const seed of [0, -1, 0.5, NaN, 2 ** 32]) assert.throws(() => validateParameters(p, seed));
  assert.deepEqual(configurations('quick', p), [32, 64, 128].map(n => ({ m: n, n, k: n, r: 1 })));
  assert.deepEqual(configurations('size', p), SQUARE_SIZES.map(n => ({ m: n, n, k: n, r: 100 })));
  assert.deepEqual(configurations('shape', p).map(p => [p.m, p.n, p.k, p.r]), [[256, 256, 256, 100], [64, 1024, 256, 100], [1024, 64, 256, 100]]);
  assert.deepEqual(configurations('custom', p), [p]);
  assert.throws(() => configurations('intensity', p));
});

test('both matrix JS kernels pass hand, zero, identity, rectangular, and dispatch-tail cases', () => {
  for (const kernel of [matmulNumber, matmulF32]) {
    const p = { m: 2, n: 2, k: 3, r: 1 };
    const a = new Float32Array([1, 2, 3, 4, 5, 6]);
    const b = new Float32Array([7, 8, 9, 10, 11, 12]);
    const out = new Float32Array(4).fill(NaN);
    kernel(a, b, out, p);
    assert.deepEqual([...out], [58, 64, 139, 154]);
    assert.deepEqual([...reference([a], b, p)], [58, 64, 139, 154]);
    for (const dimensions of [[1, 1, 1], [1, 63, 7], [1, 64, 7], [1, 65, 7], [7, 13, 5], [4, 17, 9], [17, 4, 9], [127, 127, 127], [4, 65, 2048]]) {
      const [m, n, k] = dimensions;
      const p = { m, n, k, r: 1 };
      const inputs = matmul.generate(p, 12345);
      const output = new Float32Array(m * n);
      kernel(inputs.a[0], inputs.b, output, p);
      assert.equal(matmul.validate(inputs, output, p).validation_passed, true);
    }
    const square = { m: 3, n: 3, k: 3, r: 1 };
    const inputs = matmul.generate(square, 1);
    inputs.b.set([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    inputs.reference = reference(inputs.a, inputs.b, square);
    const output = new Float32Array(9);
    kernel(inputs.a[0], inputs.b, output, square);
    assert.deepEqual(output, inputs.a[0]);
    inputs.a[0].fill(0); inputs.reference = reference(inputs.a, inputs.b, square);
    kernel(inputs.a[0], inputs.b, output, square);
    assert.deepEqual(matmul.validate(inputs, output, square), { validation_passed: true, max_abs_error: 0, relative_l2_error: 0 });
  }
});

test('matrix reuse validates all outputs for R=1,10,100 and rejects corrupt or incomplete batches', async () => {
  for (const r of [1, 10, 100]) {
    const p = { m: 2, n: 3, k: 5, r };
    const inputs = matmul.generate(p, 42);
    for (const f32 of [false, true]) {
      const backend = new JavaScriptBackend(f32);
      const output = new Float32Array(matmul.outputLength(p, 'reuse'));
      backend.prepare(p); backend.stage(inputs); backend.reuse(inputs, output);
      assert.equal(matmul.validate(inputs, output, p, 'reuse').validation_passed, true);
      for (const value of [999, NaN, Infinity]) {
        output[output.length - 1] = value;
        assert.equal(matmul.validate(inputs, output, p, 'reuse').validation_passed, false);
      }
      assert.equal(matmul.validate(inputs, output.subarray(0, output.length - 1), p, 'reuse').validation_passed, false);
      backend.dispose();
    }
  }
});

class FakeMatrixBackend implements Backend<MatrixParameters, MatrixInputs> {
  readonly metadata = {};
  events: string[] = [];
  private js = new JavaScriptBackend();
  corrupt = false;
  skip = false;
  unsupported = false;
  readonly id: BackendId;
  constructor(id: BackendId = 'js-number') { this.id = id; }
  async initialize() { this.events.push('initialize'); if (this.unsupported) throw new Error('Unavailable'); }
  prepare(p: MatrixParameters) { this.events.push('prepare'); if (this.skip) throw new ConfigurationSkip('GPU limit'); this.js.prepare(p); }
  stage(inputs: MatrixInputs) { this.events.push('stage-B'); this.js.stage(inputs); }
  warm(inputs: MatrixInputs, output: Float32Array) { this.events.push('warm'); this.js.warm(inputs, output); return 10; }
  reuse(inputs: MatrixInputs, output: Float32Array) {
    this.events.push('reuse'); this.js.reuse(inputs, output);
    if (this.corrupt) output[output.length - 1] = NaN;
    return 20;
  }
  resident() { throw new Error('W1 only'); }
  readOutput() { throw new Error('Reuse output must already be in JS'); }
  dispose() { this.events.push('dispose'); this.js.dispose(); }
}
const defaults = (): RunOptions<MatrixParameters> => ({
  experimentId: 'w2-run', sessionId: 'session', seed: 12345, schedulingSeed: 42,
  samples: 3, warmups: 5, budgetMs: 30000, memoryBudgetBytes: 2 ** 30, timerResolutionMs: 0.1,
  scenarios: ['warm', 'reuse'], configurations: [{ m: 2, n: 3, k: 5, r: 10 }],
});
async function run(backends: FakeMatrixBackend[], overrides: Partial<RunOptions<MatrixParameters>> = {}) {
  const records: SampleRecord[] = [];
  let clock = 0;
  await runBenchmark(matmul, backends, { ...defaults(), ...overrides }, new AbortController().signal, {
    record: record => records.push(record), progress: () => {}, now: () => clock++, yield: async () => {},
  });
  return records;
}

test('W2 runner stages B once, includes every reuse transfer, and returns all C outputs before validation', async () => {
  const backend = new FakeMatrixBackend('wasm-scalar');
  const records = await run([backend]);
  assert.equal(records.length, 6);
  assert.ok(records.every(record => record.status === 'ok' && record.validation_passed));
  assert.deepEqual(backend.events.slice(0, 3), ['initialize', 'prepare', 'stage-B']);
  assert.equal(backend.events.filter(event => event === 'stage-B').length, 1);
  assert.equal(backend.events.filter(event => event === 'warm').length, 6);
  assert.equal(backend.events.filter(event => event === 'reuse').length, 5);
  const warm = records.find(record => record.timing_scenario === 'warm')!;
  const reuse = records.find(record => record.timing_scenario === 'reuse')!;
  assert.equal(warm.upload_bytes, 4 * (2 * 5 + 5 * 3));
  assert.equal(warm.download_bytes, 4 * 2 * 3);
  assert.equal(reuse.elapsed_ms, 20);
  assert.equal(reuse.repetitions, 10);
  assert.equal(reuse.input_bytes, 4 * (10 * 2 * 5 + 5 * 3));
  assert.equal(reuse.output_bytes, 4 * 10 * 2 * 3);
  assert.equal(reuse.upload_bytes, 4 * 10 * 2 * 5);
  assert.equal(reuse.download_bytes, 4 * 10 * 2 * 3);
  assert.equal(reuse.staged_bytes, reuse.upload_bytes! + reuse.download_bytes!);
  assert.equal(reuse.resident_setup_upload_bytes, 4 * 5 * 3);
  assert.equal(reuse.validation_readback_bytes, 0);
  assert.equal(reuse.gpu_elapsed_ms, null);
  const js = (await run([new FakeMatrixBackend()])).find(record => record.timing_scenario === 'reuse')!;
  assert.equal(js.staged_bytes, 0);
});

test('W2 memory includes batch inputs, outputs, references, GPU staging/readback and retained Wasm heap', async () => {
  const p = { m: 2, n: 3, k: 5, r: 10 };
  const a = 40, b = 60, c = 24;
  const canonical = 16 * 2 ** 20 + 10 * a + b + 3 * 10 * c;
  assert.equal(estimateMemory(p, []), canonical);
  assert.equal(estimateMemory(p, [{ id: 'webgpu-baseline', metadata: {} }]), canonical + a + b + c + 10 * c + 10 * a + b + 16);
  assert.equal(estimateMemory(p, [{ id: 'wasm-scalar', metadata: { heap_bytes: 2 ** 25 } }]), canonical + 2 ** 25);
  let generated = false;
  const backend = new FakeMatrixBackend();
  const records: SampleRecord[] = [];
  await runBenchmark({ ...matmul, generate: () => { generated = true; throw new Error('Must not allocate'); } }, [backend],
    { ...defaults(), memoryBudgetBytes: 1 }, new AbortController().signal,
    { record: record => records.push(record), progress: () => {} });
  assert.equal(generated, false);
  assert.ok(records.every(record => record.status === 'skipped'));
  assert.ok(!backend.events.includes('prepare'));
});

test('W2 records unsupported/skipped backends and rejects incorrect reuse output without stopping healthy backends', async () => {
  const skipped = new FakeMatrixBackend('webgpu-baseline'); skipped.skip = true;
  const unsupported = new FakeMatrixBackend('wasm-scalar'); unsupported.unsupported = true;
  const records = await run([skipped, unsupported, new FakeMatrixBackend()]);
  assert.equal(records.filter(record => record.status === 'skipped').length, 2);
  assert.equal(records.filter(record => record.status === 'unsupported').length, 2);
  assert.equal(records.filter(record => record.status === 'ok').length, 6);
  const corrupt = new FakeMatrixBackend(); corrupt.corrupt = true;
  const failed = await run([corrupt], { warmups: 0, scenarios: ['reuse'] });
  assert.ok(failed.every(record => record.status === 'validation-failed'));
  assert.equal(summarize(failed)[0].count, 0);
  assert.equal(corrupt.events.filter(event => event === 'reuse').length, 1);
  await assert.rejects(() => run([new FakeMatrixBackend()], { scenarios: ['resident'] }));
});

test('W2 summaries compute GFLOP/s and separate workloads, M dimensions, and timing scenarios', async () => {
  const records = await run([new FakeMatrixBackend(), new FakeMatrixBackend('wasm-scalar')]);
  const warm = records.find(record => record.backend === 'js-number' && record.timing_scenario === 'warm')!;
  const otherShape = { ...warm, parameters: { m: 4, n: 3, k: 5, r: 10 }, elapsed_ms: 100 };
  const w1 = { ...warm, workload: 'W1', parameters: { n: 3, k: 0, r: 10 }, elapsed_ms: 999 };
  const rows = summarize([...records, otherShape, w1]);
  assert.equal(rows.length, 6);
  const reuse = rows.find(row => row.backend === 'wasm-scalar' && row.scenario === 'reuse')!;
  assert.equal(reuse.gflops, 2 * 2 * 3 * 5 * 10 / (20 * 1e6));
  assert.equal(reuse.bandwidthGBs, null);
  assert.equal(reuse.perOperation, 2);
  assert.equal(reuse.numberSpeedup, 1);
  assert.equal(rows.find(row => row.workload === 'W1')!.gflops, null);
  const exported = jsonLines({ record_type: 'environment', generator_version: GENERATOR_VERSION }, records)
    .trim().split('\n').map(line => JSON.parse(line));
  assert.deepEqual(exported.slice(1), records);
  assert.equal(exported[1].workload, 'W2');
  assert.deepEqual(exported[1].parameters, defaults().configurations[0]);
});

test('W2 cancellation and budgets preserve completed batches and report remaining work', async () => {
  const budget = await run([new FakeMatrixBackend()], { scenarios: ['reuse'], budgetMs: 1 });
  assert.equal(budget.filter(record => record.status === 'ok').length, 1);
  assert.equal(budget.filter(record => record.status === 'budget-exhausted').length, 1);
  const records: SampleRecord[] = [];
  const controller = new AbortController();
  await runBenchmark(matmul, [new FakeMatrixBackend()], defaults(), controller.signal, {
    record: record => { records.push(record); if (record.status === 'ok') controller.abort(); },
    progress: () => {}, yield: async () => {},
  });
  assert.equal(records.filter(record => record.status === 'ok').length, 1);
  assert.equal(records.filter(record => record.status === 'cancelled').length, 2);
});
