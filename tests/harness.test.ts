import test from 'node:test';
import assert from 'node:assert/strict';
import { JavaScriptBackend } from '../src/backends/javascript.ts';
import { runBenchmark } from '../src/benchmark/runner.ts';
import { jsonLines } from '../src/benchmark/results.ts';
import type { SampleRecord } from '../src/benchmark/types.ts';
import { createEnvironment, readExperiment, selectedBackends } from '../src/harness/experiment.ts';
import { getWorkload, WORKLOADS } from '../src/harness/workloads.ts';

const context = { sessionId: 'session', experimentId: 'experiment', timerResolutionMs: 0.1 };
const browser = {
  user_agent: 'test browser',
  platform: 'test platform',
  language: 'en-US',
  hardware_concurrency: 8,
  device_memory_gib: null,
  secure_context: true,
  cross_origin_isolated: false,
  execution_context: 'main thread',
  visibility_at_start: 'visible' as const,
};

function formData(values: Record<string, string> = {}, scenarios = ['warm', 'resident']): FormData {
  const data = new FormData();
  for (const [name, value] of Object.entries({
    preset: 'custom', n: '65', k: '4', r: '10', seed: '12345', samples: '3',
    budget: '1.5', memory: '128', notes: 'Test notes\nwith a second line.', ...values,
  })) {
    data.set(name, value);
  }
  data.append('backend', 'js-number');
  for (const scenario of scenarios) data.append('scenario', scenario);
  return data;
}

test('harness parses W1 settings without adding matrix dimensions and converts budget units', () => {
  const backend = new JavaScriptBackend();
  const experiment = readExperiment(formData(), WORKLOADS.W1, [backend], context);
  assert.deepEqual(experiment.backends, [backend]);
  assert.equal(experiment.notes, 'Test notes\nwith a second line.');
  assert.deepEqual(experiment.options, {
    ...context,
    seed: 12345,
    schedulingSeed: (12345 ^ 0x9e3779b9) >>> 0,
    samples: 3,
    warmups: 5,
    budgetMs: 1500,
    memoryBudgetBytes: 128 * 2 ** 20,
    sourceRevision: null,
    scenarios: ['warm', 'resident'],
    configurations: [{ n: 65, k: 4, r: 10 }],
  });
});

test('harness uses preset configurations and preserves a nonzero scheduling seed', () => {
  const quick = readExperiment(formData({ preset: 'quick' }), WORKLOADS.W1, [], context);
  assert.deepEqual(quick.options.configurations, [256, 4096, 65536, 1048576].map(n => ({ n, k: 0, r: 10 })));
  const experiment = readExperiment(formData({ seed: String(0x9e3779b9) }), WORKLOADS.W1, [], context);
  assert.equal(experiment.options.schedulingSeed, 1);
});

test('harness maps W2 custom and shape presets to matrix parameters and reuse scenarios', () => {
  const custom = readExperiment(formData({ m: '2', n: '3', k: '5' }, ['warm', 'reuse']), WORKLOADS.W2, [], context);
  assert.deepEqual(custom.options.configurations, [{ m: 2, n: 3, k: 5, r: 10 }]);
  assert.deepEqual(custom.options.scenarios, ['warm', 'reuse']);
  const shape = readExperiment(formData({ preset: 'shape', m: '128' }, ['reuse']), WORKLOADS.W2, [], context);
  assert.deepEqual(shape.options.configurations, [
    { m: 256, n: 256, k: 256, r: 10 },
    { m: 64, n: 1024, k: 256, r: 10 },
    { m: 1024, n: 64, k: 256, r: 10 },
  ]);
});

test('harness rejects mismatched workload choices and invalid custom parameters', () => {
  assert.throws(() => getWorkload('unknown'), /Unknown workload/);
  assert.throws(() => readExperiment(formData({ preset: 'shape' }), WORKLOADS.W1, [], context), /Unknown preset/);
  assert.throws(() => readExperiment(formData({}, ['reuse']), WORKLOADS.W1, [], context), /Unknown timing scenario/);
  assert.throws(() => readExperiment(formData({ m: '128' }), WORKLOADS.W2, [], context), /Unknown timing scenario/);
  assert.throws(() => readExperiment(formData({ n: '0' }), WORKLOADS.W1, [], context), /N must be/);
  assert.throws(() => readExperiment(formData({ m: '2049' }, ['reuse']), WORKLOADS.W2, [], context), /M must be/);
  const missing = formData();
  missing.delete('seed');
  assert.throws(() => readExperiment(missing, WORKLOADS.W1, [], context), /Missing form field: seed/);
});

test('backend selection retains registered order and does not duplicate selected backends', () => {
  const backends = [new JavaScriptBackend(), new JavaScriptBackend(true)];
  const data = formData();
  data.delete('backend');
  data.append('backend', 'js-f32');
  data.append('backend', 'js-number');
  data.append('backend', 'js-f32');
  assert.deepEqual(selectedBackends(data, backends), backends);
  data.delete('backend');
  assert.deepEqual(selectedBackends(data, backends), []);
});

for (const id of ['W1', 'W2'] as const) {
  test(`${id} harness settings produce valid benchmark records and the existing JSONL environment schema`, async () => {
    const definition = getWorkload(id);
    const backend = new JavaScriptBackend();
    const data = id === 'W2'
      ? formData({ m: '2', n: '3', k: '5' }, ['warm', 'reuse'])
      : formData();
    const experiment = readExperiment(data, definition, [backend], context);
    const records: SampleRecord[] = [];
    let environment: ReturnType<typeof createEnvironment>;
    await runBenchmark(definition.workload, experiment.backends, experiment.options, new AbortController().signal, {
      ready: () => { environment = createEnvironment(definition, experiment, [backend], browser); },
      record: record => { records.push(record); },
      progress: () => {},
      yield: async () => {},
    });

    const exported = jsonLines(environment!, records).trim().split('\n').map(line => JSON.parse(line));
    assert.equal(exported.length, 7);
    assert.ok(records.every(record => record.status === 'ok' && record.validation_passed));
    assert.deepEqual(exported.slice(1), records);
    assert.deepEqual(exported[0], {
      record_type: 'environment',
      schema_version: id === 'W2' ? 2 : 1,
      workload: id,
      experiment_id: context.experimentId,
      environment_id: context.experimentId,
      session_id: context.sessionId,
      timestamp: environment!.timestamp,
      source_revision: null,
      generator_version: id === 'W2' ? 'xorshift32-v1-matmul-b-then-a-row-major-f32' : 'xorshift32-v1-interleaved-f32',
      generator_seed: 12345,
      run_options: experiment.options,
      ...browser,
      notes: experiment.notes,
      backends: { 'js-number': backend.metadata },
      accuracy_contract: { atol: 1e-5, rtol: 1e-4 },
      staged_bytes_definition: 'Explicit API-level transfers/copies inside timing; not physical bus traffic.',
      resident_setup_definition: id === 'W2'
        ? 'One B staging per backend/configuration, shared by warmups and samples; not additive per record.'
        : 'One input staging per backend/configuration, shared by warmups and samples; not additive per record.',
    });
  });
}
