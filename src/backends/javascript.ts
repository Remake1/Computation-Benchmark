import type { Backend, BackendId, Inputs, Parameters } from '../benchmark/types.ts';

export function affineNumber(x: Float32Array, y: Float32Array, out: Float32Array, k: number): void {
  for (let i = 0; i < out.length; i++) {
    let t = 0.5 * x[i] + y[i];
    for (let step = 0; step < k; step++) t = 0.99609375 * t + 0.001953125;
    out[i] = t;
  }
}
export function affineF32(x: Float32Array, y: Float32Array, out: Float32Array, k: number): void {
  for (let i = 0; i < out.length; i++) {
    let t = Math.fround(Math.fround(0.5 * x[i]) + y[i]);
    for (let step = 0; step < k; step++) t = Math.fround(Math.fround(0.99609375 * t) + 0.001953125);
    out[i] = t;
  }
}

export class JavaScriptBackend implements Backend {
  readonly id: BackendId;
  readonly metadata = { arithmetic: '', workgroup_size: null };
  private kernel: typeof affineNumber;
  private inputs!: Inputs;
  private output = new Float32Array(0);
  private k = 0;
  constructor(f32 = false) {
    this.id = f32 ? 'js-f32' : 'js-number';
    this.kernel = f32 ? affineF32 : affineNumber;
    this.metadata.arithmetic = f32 ? 'fround after each multiply/add' : 'Number intermediates, f32 store';
  }
  async initialize(): Promise<void> {}
  prepare(p: Parameters): void { this.k = p.k; this.output = new Float32Array(p.n); }
  stage(inputs: Inputs): void { this.inputs = inputs; }
  warm(inputs: Inputs, output: Float32Array): number {
    const start = performance.now();
    this.kernel(inputs.x, inputs.y, output, this.k);
    return performance.now() - start;
  }
  resident(r: number): number {
    const start = performance.now();
    for (let i = 0; i < r; i++) this.kernel(this.inputs.x, this.inputs.y, this.output, this.k);
    return performance.now() - start;
  }
  readOutput(output: Float32Array): void { output.set(this.output); }
  dispose(): void { this.output = new Float32Array(0); this.inputs = { x: new Float32Array(0), y: new Float32Array(0) }; }
}
