import type { AnyInputs, AnyParameters, Backend, RunOptions } from '../benchmark/types.ts';
import type { WorkloadDefinition } from './workloads.ts';

export type HarnessBackend = Backend<AnyParameters, AnyInputs>;

export interface Experiment {
  options: RunOptions<AnyParameters>;
  backends: HarnessBackend[];
  notes: string;
}

interface ExperimentContext {
  sessionId: string;
  experimentId: string;
  timerResolutionMs: number;
}

function numberField(data: FormData, name: string): number {
  const value = data.get(name);
  if (typeof value !== 'string') throw new Error(`Missing form field: ${name}.`);
  return Number(value);
}

export function selectedBackends(data: FormData, backends: HarnessBackend[]): HarnessBackend[] {
  const selected = new Set(data.getAll('backend'));
  return backends.filter(backend => selected.has(backend.id));
}

export function readExperiment(
  data: FormData,
  definition: WorkloadDefinition,
  backends: HarnessBackend[],
  context: ExperimentContext,
): Experiment {
  const preset = definition.presets.find(option => option.value === data.get('preset'));
  if (!preset) throw new Error('Unknown preset for this workload.');

  const scenarios = data.getAll('scenario').map(value => {
    const scenario = definition.scenarios.find(option => option.value === value);
    if (!scenario) throw new Error('Unknown timing scenario for this workload.');
    return scenario.value;
  });
  const parameters = {
    n: numberField(data, 'n'),
    k: numberField(data, 'k'),
    r: numberField(data, 'r'),
    ...(definition.fields.some(field => field.name === 'm') ? { m: numberField(data, 'm') } : {}),
  };
  const seed = numberField(data, 'seed');
  const configurations = definition.configurations(preset.value, parameters);
  for (const configuration of configurations) {
    definition.workload.validateParameters(configuration, seed);
  }

  return {
    backends: selectedBackends(data, backends),
    notes: String(data.get('notes') ?? ''),
    options: {
      experimentId: context.experimentId,
      sessionId: context.sessionId,
      seed,
      schedulingSeed: (seed ^ 0x9e3779b9) >>> 0 || 1,
      samples: numberField(data, 'samples'),
      warmups: 5,
      budgetMs: numberField(data, 'budget') * 1000,
      memoryBudgetBytes: numberField(data, 'memory') * 2 ** 20,
      timerResolutionMs: context.timerResolutionMs,
      sourceRevision: backends.find(backend => backend.id === 'wasm-scalar')?.metadata.source_revision as string | undefined ?? null,
      scenarios,
      configurations,
    },
  };
}

export function readBrowserEnvironment() {
  const visibleNavigator = navigator as Navigator & { deviceMemory?: number };
  return {
    user_agent: navigator.userAgent,
    platform: navigator.platform,
    language: navigator.language,
    hardware_concurrency: navigator.hardwareConcurrency,
    device_memory_gib: visibleNavigator.deviceMemory ?? null,
    secure_context: isSecureContext,
    cross_origin_isolated: crossOriginIsolated,
    execution_context: 'main thread',
    visibility_at_start: document.visibilityState,
  };
}

export function createEnvironment(
  definition: WorkloadDefinition,
  experiment: Experiment,
  backends: HarnessBackend[],
  browser = readBrowserEnvironment(),
) {
  const { options, notes } = experiment;
  return {
    record_type: 'environment',
    schema_version: definition.schemaVersion,
    workload: definition.workload.id,
    experiment_id: options.experimentId,
    environment_id: options.experimentId,
    session_id: options.sessionId,
    timestamp: new Date().toISOString(),
    source_revision: backends.find(backend => backend.id === 'wasm-scalar')?.metadata.source_revision ?? null,
    generator_version: definition.generatorVersion,
    generator_seed: options.seed,
    run_options: options,
    ...browser,
    notes,
    backends: Object.fromEntries(backends.map(backend => [backend.id, backend.metadata])),
    accuracy_contract: { atol: 1e-5, rtol: 1e-4 },
    staged_bytes_definition: 'Explicit API-level transfers/copies inside timing; not physical bus traffic.',
    resident_setup_definition: definition.residentSetupDefinition,
  };
}

export type EnvironmentRecord = ReturnType<typeof createEnvironment>;
