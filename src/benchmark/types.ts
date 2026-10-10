export type BackendId = 'js-number' | 'js-f32' | 'wasm-scalar' | 'webgpu-baseline';
export type Scenario = 'warm' | 'resident';
export type Preset = 'quick' | 'size' | 'intensity' | 'custom';
export interface Parameters { n: number; k: number; r: number }
export interface Inputs { x: Float32Array; y: Float32Array }
export interface Accuracy {
  validation_passed: boolean;
  max_abs_error: number | null;
  relative_l2_error: number | null;
}
export interface Backend {
  readonly id: BackendId;
  readonly metadata: Record<string, unknown>;
  initialize(): Promise<void>;
  prepare(parameters: Parameters): void | Promise<void>;
  stage(inputs: Inputs): void | Promise<void>;
  warm(inputs: Inputs, output: Float32Array): number | Promise<number>;
  resident(repetitions: number): number | Promise<number>;
  readOutput(output: Float32Array): void | Promise<void>;
  dispose(): void;
}
export interface Workload {
  id: string;
  generate(n: number, seed: number): Inputs;
  validate(inputs: Inputs, output: Float32Array, parameters: Parameters): Accuracy;
  logicalBytes(parameters: Parameters): number;
}
export interface RunOptions {
  experimentId: string;
  sessionId: string;
  seed: number;
  schedulingSeed: number;
  samples: number;
  warmups: number;
  budgetMs: number;
  memoryBudgetBytes: number;
  timerResolutionMs: number;
  sourceRevision?: string | null;
  scenarios: Scenario[];
  configurations: Parameters[];
}
export type Status = 'ok' | 'unsupported' | 'skipped' | 'error' | 'validation-failed' | 'cancelled' | 'budget-exhausted';
export interface SampleRecord extends Omit<Accuracy, 'validation_passed'> {
  validation_passed: boolean | null;
  record_type: 'sample';
  experiment_id: string;
  environment_id: string;
  session_id: string;
  timestamp: string;
  sample_id: number;
  sample_index: number | null;
  seed: number;
  scheduling_seed: number;
  workload: string;
  parameters: Parameters;
  backend: BackendId;
  implementation_tier: 'baseline';
  timing_scenario: Scenario;
  repetitions: number;
  elapsed_ms: number | null;
  gpu_elapsed_ms: null;
  input_bytes: number;
  output_bytes: number;
  staged_bytes: number | null;
  upload_bytes: number | null;
  download_bytes: number | null;
  resident_setup_upload_bytes: number;
  validation_readback_bytes: number;
  allocation_policy: string;
  warmup_policy: { iterations: number };
  cache_policy: string;
  worker_count: 0;
  compiler_flags: unknown;
  source_revision: unknown;
  tuning: Record<string, unknown>;
  timer_resolution_ms: number;
  near_timer_granularity: boolean;
  status: Status;
  skip_reason: string | null;
}
