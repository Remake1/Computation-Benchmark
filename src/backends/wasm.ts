import type { Backend, Inputs, Parameters } from '../benchmark/types.ts';

interface WasmModule {
  HEAPF32: Float32Array;
  _malloc(bytes: number): number;
  _free(ptr: number): void;
  _affine(x: number, y: number, out: number, n: number, k: number): void;
}
export class WasmBackend implements Backend {
  readonly id = 'wasm-scalar' as const;
  metadata: Record<string, unknown> = {};
  private module!: WasmModule;
  private initialization?: Promise<void>;
  private pointers: number[] = [];
  private p!: Parameters;
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
  prepare(p: Parameters): void {
    this.dispose();
    this.p = p;
    try {
      for (let i = 0; i < 3; i++) {
        const ptr = this.module._malloc(p.n * 4);
        if (!ptr) throw new Error('Wasm allocation failed.');
        this.pointers.push(ptr);
      }
      // All allocations may grow memory. Acquire views only afterward.
      const heap = this.module.HEAPF32;
      [this.x, this.y, this.output] = this.pointers.map(ptr => new Float32Array(heap.buffer, ptr, p.n));
    } catch (error) { this.dispose(); throw error; }
  }
  stage(inputs: Inputs): void { this.x.set(inputs.x); this.y.set(inputs.y); }
  private execute(): void { this.module._affine(this.pointers[0], this.pointers[1], this.pointers[2], this.p.n, this.p.k); }
  warm(inputs: Inputs, output: Float32Array): number {
    const start = performance.now();
    this.stage(inputs);
    this.execute();
    output.set(this.output);
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
