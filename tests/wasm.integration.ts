import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { WasmBackend } from '../src/backends/wasm.ts';
import { affine, K_VALUES } from '../src/workloads/affine.ts';
import { matmul, reference } from '../src/workloads/matmul.ts';
import { reduction, reference as reductionReference } from '../src/workloads/reduction.ts';
import { affineF32, matmulF32 } from '../src/backends/javascript.ts';
import { selfTest } from '../src/benchmark/self-test.ts';
import { runBenchmark } from '../src/benchmark/runner.ts';
import type { SampleRecord } from '../src/benchmark/types.ts';

// Use the generated Emscripten glue and real Wasm exports. Supply local bytes
// through its instantiation hook so these tests require no browser or network.
class LocalWasmBackend extends WasmBackend {
  protected async load(): Promise<void> {
    const simd = this.id === 'wasm-simd';
    const stem = simd ? 'affine-simd' : 'affine';
    const base = new URL('../public/wasm/', import.meta.url);
    const { default: factory } = await import(new URL(`${stem}.mjs`, base).href);
    const compiled = await WebAssembly.compile(await readFile(new URL(`${stem}.wasm`, base)));
    this.module = await factory({
      instantiateWasm(imports: WebAssembly.Imports, receive: (instance: WebAssembly.Instance) => void) {
        receive(new WebAssembly.Instance(compiled, imports));
      },
    });
    this.metadata = JSON.parse(await readFile(new URL(`${simd ? 'build-simd' : 'build'}.json`, base), 'utf8'));
    this.metadata.heap_bytes = this.module.HEAPF32.byteLength;
  }
  heap(): ArrayBufferLike { return this.module.HEAPF32.buffer; }
}

for (const simd of [false, true]) {
  const name = simd ? 'SIMD' : 'scalar';
  test(`compiled ${name} passes all complete browser correctness suites`, async () => {
    const backend = new LocalWasmBackend(simd);
    for (const workload of ['W1', 'W2', 'W3'] as const) {
      const results = await selfTest([backend], new AbortController().signal, () => {}, workload);
      assert.equal(results[0].status, 'passed', results[0].reason ?? '');
      assert.ok(results[0].cases > 20);
    }
  });

  test(`compiled ${name} W3 handles lanes/batches, overwrites output, and preserves resident input`, async () => {
    const backend = new LocalWasmBackend(simd);
    await backend.initialize();
    for (const n of [1, 3, 4, 5, 63, 64, 65, 4096, 4097, 65537]) {
      const p = { n, b: 16, r: 10 }, inputs = reduction.generate(p, 42), output = new Float32Array(p.b);
      // Independently model the documented f32 order, including horizontal sum and tail.
      const expected = new Float32Array(p.b);
      for (let batch = 0; batch < p.b; batch++) {
        const lanes = [0, 0, 0, 0]; let i = 0, sum = 0;
        if (simd) {
          for (; i + 4 <= n; i += 4) for (let lane = 0; lane < 4; lane++) {
            lanes[lane] = Math.fround(lanes[lane] + inputs.x[batch * n + i + lane]);
          }
          sum = Math.fround(Math.fround(Math.fround(lanes[0] + lanes[1]) + lanes[2]) + lanes[3]);
        }
        for (; i < n; i++) sum = Math.fround(sum + inputs.x[batch * n + i]);
        expected[batch] = sum;
      }
      backend.prepare(p); backend.stage(inputs); backend.warm(inputs, output);
      assert.deepEqual(output, expected, `N=${n}`);
      for (const r of [1, 10, 100]) {
        backend.resident(r); backend.readOutput(output);
        assert.deepEqual(output, expected);
      }
      inputs.x.fill(0);
      backend.resident(10); backend.readOutput(output);
      assert.deepEqual(output, expected, 'canonical changes must not affect already staged Wasm input');
      inputs.reference = reductionReference(inputs.x, p);
      backend.warm(inputs, output);
      assert.ok(output.every(value => value === 0));
      backend.resident(10); backend.readOutput(output);
      assert.ok(output.every(value => value === 0));
      backend.dispose(); backend.dispose();
    }
  });

  test(`compiled ${name} W1 matches f32 boundaries at every SIMD tail and intensity`, async () => {
    const backend = new LocalWasmBackend(simd);
    await backend.initialize();
    for (const n of [1, 2, 3, 4, 5, 6, 7, 8, 9, 1023, 65537]) {
      const inputs = affine.generate(n, 42);
      for (const k of K_VALUES) {
        const p = { n, k, r: 10 };
        const output = new Float32Array(n);
        const expected = new Float32Array(n);
        affineF32(inputs.x, inputs.y, expected, k);
        backend.prepare(p); backend.warm(inputs, output);
        assert.deepEqual(output, expected, `N=${n}, K=${k}`);
        backend.resident(p.r); backend.readOutput(output);
        assert.deepEqual(output, expected);
        backend.dispose();
      }
    }
  });

  test(`compiled ${name} W2 preserves accumulation across block boundaries and overwrites reused C`, async () => {
    const backend = new LocalWasmBackend(simd);
    await backend.initialize();
    const shapes = [[1, 2, 1], [2, 3, 5], [3, 4, 7], [5, 6, 9],
      [31, 31, 31], [32, 32, 32], [33, 33, 33], [35, 67, 65],
      [127, 127, 127], [255, 255, 255], [3, 513, 513], [513, 3, 33], [4, 65, 2048]];
    for (const [m, n, k] of shapes) {
      const p = { m, n, k, r: 1 };
      const inputs = matmul.generate(p, 42);
      const output = new Float32Array(m * n);
      const expected = new Float32Array(m * n);
      matmulF32(inputs.a[0], inputs.b, expected, p);
      backend.prepare(p); backend.warm(inputs, output);
      assert.deepEqual(output, expected, `M=${m}, N=${n}, K=${k}`);
      backend.warm(inputs, output);
      assert.deepEqual(output, expected, 'C must start at zero on every execution');
      inputs.a[0].fill(0);
      backend.warm(inputs, output);
      assert.ok(output.every(value => value === 0));
      backend.dispose();
    }
    for (const r of [1, 10, 100]) {
      const p = { m: 3, n: 35, k: 33, r };
      const inputs = matmul.generate(p, 12345);
      backend.prepare(p); backend.stage(inputs);
      const output = new Float32Array(r * p.m * p.n);
      backend.reuse(inputs, output);
      assert.equal(matmul.validate(inputs, output, p, 'reuse').validation_passed, true);
      // Mutating canonical B after staging must not change a resident B batch.
      inputs.b.fill(0);
      backend.reuse(inputs, output);
      assert.equal(matmul.validate(inputs, output, p, 'reuse').validation_passed, true);
      backend.stage(inputs);
      inputs.reference = reference(inputs.a, inputs.b, p);
      backend.reuse(inputs, output);
      assert.ok(output.every(value => value === 0));
      backend.dispose();
    }
  });

  test(`compiled ${name} refreshes views after heap growth and can switch workloads`, async () => {
    const backend = new LocalWasmBackend(simd);
    await backend.initialize();
    const before = backend.heap();
    const p = { n: 2 ** 21 + 1, k: 0, r: 1 };
    const inputs = affine.generate(p, 42);
    const output = new Float32Array(p.n);
    backend.prepare(p);
    assert.notEqual(backend.heap(), before);
    backend.warm(inputs, output);
    assert.equal(affine.validate(inputs, output, p).validation_passed, true);
    const matrix = { m: 2, n: 5, k: 3, r: 10 };
    const matrixInputs = matmul.generate(matrix, 1);
    backend.prepare(matrix); backend.stage(matrixInputs);
    const batch = new Float32Array(matrix.r * matrix.m * matrix.n);
    backend.reuse(matrixInputs, batch);
    assert.equal(matmul.validate(matrixInputs, batch, matrix, 'reuse').validation_passed, true);
    const reductionParameters = { n: 2 ** 20 + 1, b: 16, r: 1 };
    const reductionInputs = reduction.generate(reductionParameters, 42), sums = new Float32Array(16);
    const beforeReduction = backend.heap();
    backend.prepare(reductionParameters);
    assert.notEqual(backend.heap(), beforeReduction);
    backend.stage(reductionInputs); backend.resident(1); backend.readOutput(sums);
    assert.equal(reduction.validate(reductionInputs, sums, reductionParameters).validation_passed, true);
    backend.prepare(p); backend.warm(inputs, output);
    assert.equal(affine.validate(inputs, output, p).validation_passed, true);
    backend.dispose(); backend.dispose();
    assert.equal(backend.metadata.heap_bytes, backend.heap().byteLength);
  });
}

test('real SIMD runs export optimized tier, tuning, compiler flags, and all timed transfers', async () => {
  const backend = new LocalWasmBackend(true);
  for (const workload of [affine, matmul]) {
    const matrix = workload.id === 'W2';
    const p = matrix ? { m: 3, n: 35, k: 33, r: 10 } : { n: 7, k: 16, r: 10 };
    const records: SampleRecord[] = [];
    await runBenchmark(workload, [backend], {
      experimentId: 'simd', sessionId: 'session', seed: 42, schedulingSeed: 1,
      samples: 2, warmups: 2, budgetMs: 30000, memoryBudgetBytes: 2 ** 30, timerResolutionMs: 0.1,
      scenarios: matrix ? ['warm', 'reuse'] : ['warm', 'resident'], configurations: [p],
    }, new AbortController().signal, { record: record => records.push(record), progress: () => {}, yield: async () => {} });
    assert.equal(records.length, 4);
    assert.ok(records.every(record => record.status === 'ok' && record.validation_passed));
    assert.ok(records.every(record => record.implementation_tier === 'optimized' && record.tuning.simd_lanes === 4));
    assert.ok(records.every(record => (record.compiler_flags as string[]).includes('-msimd128')));
    const warm = records.find(record => record.timing_scenario === 'warm')!;
    assert.equal(warm.staged_bytes, workload.logicalBytes(p));
    if (matrix) {
      const reuse = records.find(record => record.timing_scenario === 'reuse')!;
      assert.equal(reuse.tuning.block_size, 32);
      assert.equal(reuse.upload_bytes, 4 * 3 * p.k * p.r);
      assert.equal(reuse.download_bytes, 4 * 3 * p.n * p.r);
      assert.equal(reuse.resident_setup_upload_bytes, 4 * p.k * p.n);
    } else {
      const resident = records.find(record => record.timing_scenario === 'resident')!;
      assert.equal(resident.staged_bytes, 0);
      assert.equal(resident.validation_readback_bytes, 4 * p.n);
    }
  }
});

test('real W3 Wasm exports tier, reduction order, and final-only transfers', async () => {
  for (const simd of [false, true]) {
    const backend = new LocalWasmBackend(simd), p = { n: 65, b: 16, r: 10 }, records: SampleRecord[] = [];
    await runBenchmark(reduction, [backend], {
      experimentId: 'w3-wasm', sessionId: 'session', seed: 42, schedulingSeed: 1,
      samples: 2, warmups: 2, budgetMs: 30000, memoryBudgetBytes: 2 ** 30, timerResolutionMs: 0.1,
      scenarios: ['warm', 'resident'], configurations: [p],
    }, new AbortController().signal, { record: record => records.push(record), progress: () => {}, yield: async () => {} });
    assert.equal(records.length, 4);
    assert.ok(records.every(record => record.status === 'ok' && record.validation_passed));
    assert.ok(records.every(record => record.implementation_tier === (simd ? 'optimized' : 'baseline')));
    assert.ok(records.every(record => typeof record.tuning.summation_order === 'string'));
    const warm = records[0], resident = records[1];
    assert.equal(warm.upload_bytes, 4 * p.n * p.b); assert.equal(warm.download_bytes, 4 * p.b);
    assert.equal(resident.staged_bytes, 0); assert.equal(resident.validation_readback_bytes, 4 * p.b);
    assert.equal(resident.resident_setup_upload_bytes, 4 * p.n * p.b);
    assert.equal(resident.tuning.reduction_stages, null);
  }
});
