import type { AnyInputs, AnyParameters, Backend, Implementation, WorkloadId } from '../benchmark/types.ts';

// A minimal module with a v128.const in a function body; validate without execution.
export function supportsWasmSimd(): boolean {
  return typeof WebAssembly !== 'undefined' && WebAssembly.validate(new Uint8Array([
    0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123,
    3, 2, 1, 0, 10, 22, 1, 20, 0, 253, 12,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11,
  ]));
}

interface WasmModule {
  HEAPF32: Float32Array;
  _malloc(bytes: number): number;
  _free(ptr: number): void;
  _affine(x: number, y: number, out: number, n: number, k: number): void;
  _matmul(a: number, b: number, out: number, m: number, n: number, k: number): void;
  _reduce_sum(x: number, out: number, n: number, b: number): void;
}
export class WasmBackend implements Backend<AnyParameters, AnyInputs> {
  readonly id: 'wasm-scalar' | 'wasm-simd';
  readonly implementations: Partial<Record<WorkloadId, Implementation>>;
  metadata: Record<string, unknown> = {};
  protected module!: WasmModule;
  private initialization?: Promise<void>;
  private pointers: number[] = [];
  private p!: AnyParameters;
  private x!: Float32Array;
  private y!: Float32Array;
  private output!: Float32Array;
  constructor(simd = false) {
    this.id = simd ? 'wasm-simd' : 'wasm-scalar';
    this.implementations = simd ? {
      W1: { tier: 'optimized', tuning: { simd_lanes: 4, scalar_tail: true,
        arithmetic: 'f32 multiply then add; no fusion', recurrence: 'runtime K, in registers' } },
      W2: { tier: 'optimized', tuning: { simd_lanes: 4, block_size: 32, scalar_tail: true,
        loop_order: 'row-block/column-block/K-block/row/K/column',
        summation_order: 'ascending K per output, unchanged from scalar', arithmetic: 'f32 multiply then add; no fusion' } },
      W3: { tier: 'optimized', tuning: { simd_lanes: 4, scalar_tail: true,
        summation_order: 'four strided f32 lane accumulators; combine lanes left-to-right, then scalar tail' } },
    } : { W3: { tier: 'baseline', tuning: { summation_order: 'sequential ascending index per vector' } } };
  }
  initialize(): Promise<void> {
    return this.initialization ??= this.load();
  }
  protected async load(): Promise<void> {
    if (typeof WebAssembly === 'undefined') throw new Error('WebAssembly is unavailable.');
    if (this.id === 'wasm-simd' && !supportsWasmSimd()) throw new Error('WebAssembly SIMD is unavailable.');
    const base = new URL(`${import.meta.env.BASE_URL}wasm/`, location.href).href;
    const simd = this.id === 'wasm-simd';
    const response = await fetch(`${base}${simd ? 'build-simd' : 'build'}.json`);
    if (!response.ok) throw new Error('Wasm assets are missing. Run pnpm build:wasm.');
    this.metadata = await response.json();
    const moduleUrl = `${base}${simd ? 'affine-simd' : 'affine'}.mjs`;
    const { default: factory } = await import(/* @vite-ignore */ moduleUrl);
    this.module = await factory({ locateFile: (name: string) => `${base}${name}` });
    this.metadata.heap_bytes = this.module.HEAPF32.byteLength;
    this.metadata.implementations = this.implementations;
  }
  prepare(p: AnyParameters): void {
    this.dispose();
    this.p = p;
    try {
      const lengths = 'm' in p ? [p.m * p.k, p.k * p.n, p.m * p.n] : 'b' in p ? [p.n * p.b, p.b] : [p.n, p.n, p.n];
      for (const length of lengths) {
        const ptr = this.module._malloc(length * 4);
        if (!ptr) throw new Error('Wasm allocation failed.');
        this.pointers.push(ptr);
      }
      // All allocations may grow memory. Acquire views only afterward.
      const heap = this.module.HEAPF32;
      const views = this.pointers.map((ptr, i) => new Float32Array(heap.buffer, ptr, lengths[i]));
      this.x = views[0];
      this.y = 'b' in p ? new Float32Array(0) : views[1];
      this.output = views.at(-1)!;
    } catch (error) { this.dispose(); throw error; }
  }
  stage(inputs: AnyInputs): void {
    if ('a' in inputs) this.y.set(inputs.b);
    else { this.x.set(inputs.x); if ('y' in inputs) this.y.set(inputs.y); }
  }
  private execute(): void {
    if ('m' in this.p) this.module._matmul(this.pointers[0], this.pointers[1], this.pointers[2], this.p.m, this.p.n, this.p.k);
    else if ('b' in this.p) this.module._reduce_sum(this.pointers[0], this.pointers[1], this.p.n, this.p.b);
    else this.module._affine(this.pointers[0], this.pointers[1], this.pointers[2], this.p.n, this.p.k);
  }
  warm(inputs: AnyInputs, output: Float32Array): number {
    const start = performance.now();
    this.stage(inputs);
    if ('a' in inputs) this.x.set(inputs.a[0]);
    this.execute();
    output.set(this.output);
    return performance.now() - start;
  }
  reuse(inputs: AnyInputs, output: Float32Array): number {
    if (!('a' in inputs) || !('m' in this.p)) throw new Error('B reuse requires W2.');
    const size = this.p.m * this.p.n;
    const views = inputs.a.map((_, i) => output.subarray(i * size, (i + 1) * size));
    const start = performance.now();
    for (let i = 0; i < inputs.a.length; i++) {
      this.x.set(inputs.a[i]); this.execute(); views[i].set(this.output);
    }
    return performance.now() - start;
  }
  resident(r: number): number {
    const start = performance.now();
    for (let i = 0; i < r; i++) this.execute();
    return performance.now() - start;
  }
  readOutput(output: Float32Array): void { output.set(this.output); }
  dispose(): void {
    for (const ptr of this.pointers) this.module._free(ptr);
    this.pointers = [];
    this.x = this.y = this.output = new Float32Array(0);
    if (this.module) this.metadata.heap_bytes = this.module.HEAPF32.byteLength;
  }
}
