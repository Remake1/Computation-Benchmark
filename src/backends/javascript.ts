import type { AnyInputs, AnyParameters, Backend, BackendId, MatrixParameters } from '../benchmark/types.ts';

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

export function matmulNumber(a: Float32Array, b: Float32Array, out: Float32Array, p: MatrixParameters): void {
  for (let row = 0; row < p.m; row++) {
    for (let col = 0; col < p.n; col++) {
      let sum = 0;
      for (let k = 0; k < p.k; k++) sum += a[row * p.k + k] * b[k * p.n + col];
      out[row * p.n + col] = sum;
    }
  }
}
export function matmulF32(a: Float32Array, b: Float32Array, out: Float32Array, p: MatrixParameters): void {
  for (let row = 0; row < p.m; row++) {
    for (let col = 0; col < p.n; col++) {
      let sum = 0;
      for (let k = 0; k < p.k; k++) sum = Math.fround(sum + Math.fround(a[row * p.k + k] * b[k * p.n + col]));
      out[row * p.n + col] = sum;
    }
  }
}

export class JavaScriptBackend implements Backend<AnyParameters, AnyInputs> {
  readonly id: BackendId;
  readonly metadata = { arithmetic: '', workgroup_size: null };
  private kernel: typeof affineNumber;
  private matrixKernel: typeof matmulNumber;
  private inputs!: AnyInputs;
  private output = new Float32Array(0);
  private k = 0;
  private matrix?: MatrixParameters;
  constructor(f32 = false) {
    this.id = f32 ? 'js-f32' : 'js-number';
    this.kernel = f32 ? affineF32 : affineNumber;
    this.matrixKernel = f32 ? matmulF32 : matmulNumber;
    this.metadata.arithmetic = f32 ? 'fround after each multiply/add' : 'Number intermediates, f32 store';
  }
  async initialize(): Promise<void> {}
  prepare(p: AnyParameters): void {
    this.k = p.k; this.matrix = 'm' in p ? p : undefined;
    this.output = new Float32Array(this.matrix ? 0 : p.n);
  }
  stage(inputs: AnyInputs): void { this.inputs = inputs; }
  warm(inputs: AnyInputs, output: Float32Array): number {
    const start = performance.now();
    if ('a' in inputs) this.matrixKernel(inputs.a[0], inputs.b, output, this.matrix!);
    else this.kernel(inputs.x, inputs.y, output, this.k);
    return performance.now() - start;
  }
  reuse(inputs: AnyInputs, output: Float32Array): number {
    if (!('a' in inputs) || !this.matrix) throw new Error('B reuse requires W2.');
    const size = this.matrix.m * this.matrix.n;
    const views = inputs.a.map((_, i) => output.subarray(i * size, (i + 1) * size));
    const start = performance.now();
    for (let i = 0; i < inputs.a.length; i++) this.matrixKernel(inputs.a[i], inputs.b, views[i], this.matrix);
    return performance.now() - start;
  }
  resident(r: number): number {
    if (!('x' in this.inputs)) throw new Error('Resident mode requires W1.');
    const start = performance.now();
    for (let i = 0; i < r; i++) this.kernel(this.inputs.x, this.inputs.y, this.output, this.k);
    return performance.now() - start;
  }
  readOutput(output: Float32Array): void { output.set(this.output); }
  dispose(): void { this.output = new Float32Array(0); this.inputs = { x: new Float32Array(0), y: new Float32Array(0) }; }
}
