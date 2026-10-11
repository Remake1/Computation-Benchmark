import type { AnyInputs, AnyParameters, Backend } from '../benchmark/types.ts';

interface WasmModule {
  HEAPF32: Float32Array;
  _malloc(bytes: number): number;
  _free(ptr: number): void;
  _affine(x: number, y: number, out: number, n: number, k: number): void;
  _matmul(a: number, b: number, out: number, m: number, n: number, k: number): void;
}
export class WasmBackend implements Backend<AnyParameters, AnyInputs> {
  readonly id = 'wasm-scalar' as const;
  metadata: Record<string, unknown> = {};
  private module!: WasmModule;
  private initialization?: Promise<void>;
  private pointers: number[] = [];
  private p!: AnyParameters;
  private x!: Float32Array;
  private y!: Float32Array;
  private output!: Float32Array;
  initialize(): Promise<void> {
    return this.initialization ??= this.load();
  }
  private async load(): Promise<void> {
    if (typeof WebAssembly === 'undefined') throw new Error('WebAssembly is unavailable.');
    const base = new URL(`${import.meta.env.BASE_URL}wasm/`, location.href).href;
    const response = await fetch(`${base}build.json`);
    if (!response.ok) throw new Error('Wasm assets are missing. Run pnpm build:wasm.');
    this.metadata = await response.json();
    const moduleUrl = `${base}affine.mjs`;
    const { default: factory } = await import(/* @vite-ignore */ moduleUrl);
    this.module = await factory({ locateFile: (name: string) => `${base}${name}` });
  }
  prepare(p: AnyParameters): void {
    this.dispose();
    this.p = p;
    try {
      const lengths = 'm' in p ? [p.m * p.k, p.k * p.n, p.m * p.n] : [p.n, p.n, p.n];
      for (const length of lengths) {
        const ptr = this.module._malloc(length * 4);
        if (!ptr) throw new Error('Wasm allocation failed.');
        this.pointers.push(ptr);
      }
      // All allocations may grow memory. Acquire views only afterward.
      const heap = this.module.HEAPF32;
      [this.x, this.y, this.output] = this.pointers.map((ptr, i) => new Float32Array(heap.buffer, ptr, lengths[i]));
    } catch (error) { this.dispose(); throw error; }
  }
  stage(inputs: AnyInputs): void {
    if ('a' in inputs) this.y.set(inputs.b);
    else { this.x.set(inputs.x); this.y.set(inputs.y); }
  }
  private execute(): void {
    if ('m' in this.p) this.module._matmul(this.pointers[0], this.pointers[1], this.pointers[2], this.p.m, this.p.n, this.p.k);
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
