import {
  affine,
  configurations as affineConfigurations,
  GENERATOR_VERSION as AFFINE_GENERATOR_VERSION,
  K_VALUES,
  MAX_N,
} from '../workloads/affine.ts';
import {
  matmul,
  configurations as matrixConfigurations,
  GENERATOR_VERSION as MATRIX_GENERATOR_VERSION,
  MAX_DIMENSION,
} from '../workloads/matmul.ts';
import type {
  AnyInputs,
  AnyParameters,
  MatrixParameters,
  Preset,
  Scenario,
  Workload,
  WorkloadId,
} from '../benchmark/types.ts';

export type ParameterName = keyof MatrixParameters;

interface NumberField {
  name: Exclude<ParameterName, 'r'>;
  label: string;
  type: 'number';
  value: number;
  min: number;
  max: number;
}

interface SelectField {
  name: Exclude<ParameterName, 'r'>;
  label: string;
  type: 'select';
  values: readonly number[];
}

export type ParameterField = NumberField | SelectField;

export interface WorkloadDefinition {
  workload: Workload<AnyParameters, AnyInputs>;
  title: string;
  algorithmHtml: string;
  fields: readonly ParameterField[];
  presets: readonly { value: Preset; label: string; description: string }[];
  repetitionLabel: string;
  scenarios: readonly { value: Scenario; label: string }[];
  scenarioDescription: string;
  resultsDescription: string;
  throughput: { label: string; metric: 'bandwidthGBs' | 'gflops' };
  schemaVersion: number;
  generatorVersion: string;
  residentSetupDefinition: string;
  configurations(preset: Preset, parameters: AnyParameters): AnyParameters[];
}

const warmScenario = { value: 'warm', label: 'Warm end-to-end (JS-visible output)' } as const;

export const WORKLOADS: Record<WorkloadId, WorkloadDefinition> = {
  W1: {
    workload: affine,
    title: 'W1: vector affine transform',
    algorithmHtml: '<code>t = 0.5*x + y</code>; for K steps, <code>t = 0.99609375*t + 0.001953125</code>.',
    fields: [
      { name: 'n', label: 'N (elements)', type: 'number', value: 65536, min: 1, max: MAX_N },
      { name: 'k', label: 'K (recurrence steps)', type: 'select', values: K_VALUES },
    ],
    presets: [
      { value: 'quick', label: 'Quick check', description: 'Quick: N=256,4096,65536,1048576; K=0; R=10. N/K/R controls apply to other presets.' },
      { value: 'size', label: 'Size sweep', description: 'Size sweep: 2^8 through 2^24 plus three tail cases; uses selected K and R.' },
      { value: 'intensity', label: 'Intensity sweep', description: 'Intensity sweep: K=0,4,16,64,256; uses selected N and R.' },
      { value: 'custom', label: 'Custom case', description: 'Custom: one case with the selected N, K, and R.' },
    ],
    repetitionLabel: 'R (resident repetitions)',
    scenarios: [warmScenario, { value: 'resident', label: 'Resident batch (output stays in backend)' }],
    scenarioDescription: 'Resident repetitions overwrite the output from the same inputs. Final readback and validation are outside timing.',
    resultsDescription: 'K=0 is the original affine transform; K>0 is the intensity variant. Warm timings are individual request latency; resident timings are batch throughput. Bandwidth is logical, not physical memory traffic. Speedup = matching JS median / backend median.',
    throughput: { label: 'Logical GB/s', metric: 'bandwidthGBs' },
    schemaVersion: 1,
    generatorVersion: AFFINE_GENERATOR_VERSION,
    residentSetupDefinition: 'One input staging per backend/configuration, shared by warmups and samples; not additive per record.',
    configurations: affineConfigurations,
  },
  W2: {
    workload: matmul,
    title: 'W2: dense matrix multiplication',
    algorithmHtml: '<code>C[M,N] = A[M,K] × B[K,N]</code>; row-major float32 inputs in [-0.5, 0.5], direct row/column/K loops.',
    fields: [
      { name: 'm', label: 'M (rows of A)', type: 'number', value: 128, min: 1, max: MAX_DIMENSION },
      { name: 'n', label: 'N (columns of B)', type: 'number', value: 128, min: 1, max: MAX_DIMENSION },
      { name: 'k', label: 'K (shared dimension)', type: 'number', value: 128, min: 1, max: MAX_DIMENSION },
    ],
    presets: [
      { value: 'quick', label: 'Quick check', description: 'Quick: square sizes 32,64,128; R=1. Dimension/R controls apply to other presets.' },
      { value: 'size', label: 'Size sweep', description: 'Square size sweep: 32,64,127,128,255,256,512,513,1024,2048; uses selected R.' },
      { value: 'shape', label: 'Shape sweep', description: 'Shape sweep (M,N,K): (256,256,256), (64,1024,256), (1024,64,256); uses selected R.' },
      { value: 'custom', label: 'Custom case', description: 'Custom: one case with the selected M, N, K, and R.' },
    ],
    repetitionLabel: 'R (B-reuse batch size)',
    scenarios: [warmScenario, { value: 'reuse', label: 'B-reuse batch (all outputs returned to JS)' }],
    scenarioDescription: 'Reuse stages B once outside timing. Each of R different A inputs and every C readback are included in batch timing; validation is outside timing.',
    resultsDescription: 'Warm timings are individual request latency; reuse timings include R requests with fixed B. Approximate GFLOP/s = 2*M*N*K*repetitions / time. Speedup = matching JS median / backend median.',
    throughput: { label: 'Approx. GFLOP/s', metric: 'gflops' },
    schemaVersion: 2,
    generatorVersion: MATRIX_GENERATOR_VERSION,
    residentSetupDefinition: 'One B staging per backend/configuration, shared by warmups and samples; not additive per record.',
    configurations(preset, parameters) {
      if (!('m' in parameters)) throw new Error('Matrix dimensions require W2.');
      return matrixConfigurations(preset, parameters);
    },
  },
};

export function getWorkload(value: string): WorkloadDefinition {
  const definition = Object.values(WORKLOADS).find(entry => entry.workload.id === value);
  if (!definition) throw new Error(`Unknown workload: ${value}.`);
  return definition;
}
