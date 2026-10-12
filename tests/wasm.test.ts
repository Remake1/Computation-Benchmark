import test from 'node:test';
import assert from 'node:assert/strict';
import { WasmBackend, supportsWasmSimd } from '../src/backends/wasm.ts';
import { estimateMemory, runBenchmark } from '../src/benchmark/runner.ts';
import { affine } from '../src/workloads/affine.ts';
import { JavaScriptBackend } from '../src/backends/javascript.ts';
import { summarize } from '../src/benchmark/results.ts';
import type { SampleRecord } from '../src/benchmark/types.ts';

test('SIMD capability detection records unsupported without blocking scalar or JS', async t => {
  assert.equal(supportsWasmSimd(), true);
  t.mock.method(WebAssembly, 'validate', () => false);
  assert.equal(supportsWasmSimd(), false);
  const backend = new WasmBackend(true);
  await assert.rejects(backend.initialize(), /SIMD is unavailable/);
  const records: SampleRecord[] = [];
  await runBenchmark(affine, [backend, new JavaScriptBackend()], {
    experimentId: 'simd', sessionId: 'session', seed: 42, schedulingSeed: 1,
    samples: 1, warmups: 0, budgetMs: 1000, memoryBudgetBytes: 2 ** 30, timerResolutionMs: 0.1,
    scenarios: ['warm', 'resident'], configurations: [{ n: 5, k: 0, r: 10 }],
  }, new AbortController().signal, { record: record => records.push(record), progress: () => {}, yield: async () => {} });
  const unsupported = records.filter(record => record.backend === 'wasm-simd');
  assert.equal(unsupported.length, 2);
  assert.ok(unsupported.every(record => record.status === 'unsupported' && record.implementation_tier === 'optimized'));
  assert.equal(records.filter(record => record.status === 'ok').length, 2);
  assert.equal(new WasmBackend().id, 'wasm-scalar');
});

test('memory budgets account for both independently retained Wasm heaps', () => {
  for (const p of [1023, { m: 33, n: 35, k: 31, r: 10 }]) {
    const scalar = { id: 'wasm-scalar' as const, metadata: { heap_bytes: 2 ** 25 } };
    const simd = { id: 'wasm-simd' as const, metadata: { heap_bytes: 2 ** 26 } };
    const canonical = estimateMemory(p, []);
    assert.equal(estimateMemory(p, [scalar]), canonical + 2 ** 25);
    assert.equal(estimateMemory(p, [simd]), canonical + 2 ** 26);
    assert.equal(estimateMemory(p, [scalar, simd]), canonical + 2 ** 25 + 2 ** 26);
  }
});

test('summaries keep baseline and optimized samples separate and compare JS only within a tier', async () => {
  const records: SampleRecord[] = [];
  await runBenchmark(affine, [new JavaScriptBackend()], {
    experimentId: 'simd', sessionId: 'session', seed: 42, schedulingSeed: 1,
    samples: 1, warmups: 0, budgetMs: 1000, memoryBudgetBytes: 2 ** 30, timerResolutionMs: 0.1,
    scenarios: ['warm'], configurations: [{ n: 5, k: 0, r: 10 }],
  }, new AbortController().signal, { record: record => records.push(record), progress: () => {}, yield: async () => {} });
  const js = { ...records[0], elapsed_ms: 10 };
  const simd = { ...js, backend: 'wasm-simd' as const, implementation_tier: 'optimized' as const, elapsed_ms: 1 };
  const scalar = { ...js, backend: 'wasm-scalar' as const, elapsed_ms: 5 };
  const rows = summarize([js, scalar, simd, { ...simd, implementation_tier: 'baseline', elapsed_ms: 2 }]);
  assert.equal(rows.length, 4);
  const optimized = rows.find(row => row.tier === 'optimized')!;
  assert.equal(optimized.count, 1);
  assert.equal(optimized.numberSpeedup, null);
  assert.equal(optimized.f32Speedup, null);
  assert.equal(optimized.scalarSpeedup, 5);
  assert.equal(summarize([js, { ...scalar, status: 'validation-failed' }, simd])
    .find(row => row.tier === 'optimized')!.scalarSpeedup, null);
  assert.equal(rows.find(row => row.backend === 'wasm-simd' && row.tier === 'baseline')!.numberSpeedup, 5);
});
