import type { AnyInputs, AnyParameters, Backend, BackendId, MatrixParameters, ReductionParameters } from '../benchmark/types.ts';

export function reductionNumber(x: Float32Array, out: Float32Array, p: ReductionParameters): void {
  for (let batch = 0; batch < p.b; batch++) {
    let sum = 0;
    for (let i = 0; i < p.n; i++) sum += x[batch * p.n + i];
    out[batch] = sum;
  }
}
export function reductionF32(x: Float32Array, out: Float32Array, p: ReductionParameters): void {
  for (let batch = 0; batch < p.b; batch++) {
    let sum = 0;
    for (let i = 0; i < p.n; i++) sum = Math.fround(sum + x[batch * p.n + i]);
    out[batch] = sum;
  }
}

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
  private reductionKernel: typeof reductionNumber;
  readonly implementations = { W3: { tier: 'baseline' as const, tuning: { summation_order: 'sequential ascending index per vector' } } };
  private inputs!: AnyInputs;
  private output = new Float32Array(0);
  private k = 0;
  private matrix?: MatrixParameters;
  private reduction?: ReductionParameters;
  constructor(f32 = false) {
    this.id = f32 ? 'js-f32' : 'js-number';
    this.kernel = f32 ? affineF32 : affineNumber;
    this.matrixKernel = f32 ? matmulF32 : matmulNumber;
    this.reductionKernel = f32 ? reductionF32 : reductionNumber;
    this.metadata.arithmetic = f32 ? 'fround after each multiply/add' : 'Number intermediates, f32 store';
  }
  async initialize(): Promise<void> {}
  prepare(p: AnyParameters): void {
    this.k = 'k' in p ? p.k : 0; this.matrix = 'm' in p ? p : undefined;
    this.reduction = 'b' in p ? p : undefined;
    this.output = new Float32Array(this.matrix ? 0 : this.reduction ? this.reduction.b : p.n);
  }
  stage(inputs: AnyInputs): void { this.inputs = inputs; }
  warm(inputs: AnyInputs, output: Float32Array): number {
    const start = performance.now();
    if ('a' in inputs) this.matrixKernel(inputs.a[0], inputs.b, output, this.matrix!);
    else if ('y' in inputs) this.kernel(inputs.x, inputs.y, output, this.k);
    else this.reductionKernel(inputs.x, output, this.reduction!);
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
    if (!('x' in this.inputs)) throw new Error('Resident mode requires W1 or W3.');
    const start = performance.now();
    for (let i = 0; i < r; i++) {
      if ('y' in this.inputs) this.kernel(this.inputs.x, this.inputs.y, this.output, this.k);
      else this.reductionKernel(this.inputs.x, this.output, this.reduction!);
    }
    return performance.now() - start;
  }
  readOutput(output: Float32Array): void { output.set(this.output); }
  dispose(): void { this.output = new Float32Array(0); this.inputs = { x: new Float32Array(0), y: new Float32Array(0) }; }
}
