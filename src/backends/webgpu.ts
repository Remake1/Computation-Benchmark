import shader from './affine.wgsl?raw';
import matrixShader from './matmul.wgsl?raw';
import { dispatchGrid, WORKGROUP_SIZE } from './dispatch.ts';
import { ConfigurationSkip } from '../benchmark/types.ts';
import type { AnyInputs, AnyParameters, Backend } from '../benchmark/types.ts';

export class WebGPUBackend implements Backend<AnyParameters, AnyInputs> {
  readonly id = 'webgpu-baseline' as const;
  metadata: Record<string, unknown> = { workgroup_size: WORKGROUP_SIZE };
  private initialization?: Promise<void>;
  private device!: GPUDevice;
  private pipeline!: GPUComputePipeline;
  private matrixPipeline!: GPUComputePipeline;
  private matrix = false;
  private buffers: GPUBuffer[] = [];
  private bindGroup!: GPUBindGroup;
  private grid = { x: 1, y: 1 };
  private bytes = 0;
  private failure: string | null = null;
  initialize(): Promise<void> { return this.initialization ??= this.load(); }
  private async load(): Promise<void> {
    if (!navigator.gpu) throw new Error('WebGPU is unavailable. Use a supported browser on localhost or HTTPS.');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No WebGPU adapter is available.');
    this.device = await adapter.requestDevice();
    this.device.lost.then(info => { this.failure = `GPU device lost: ${info.reason} ${info.message}`; });
    this.device.addEventListener('uncapturederror', event => { this.failure = event.error.message; });
    const info = adapter.info;
    this.metadata = {
      workgroup_size: WORKGROUP_SIZE,
      adapter: { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description },
      features: [...this.device.features],
      limits: Object.fromEntries(['maxBufferSize', 'maxStorageBufferBindingSize', 'maxComputeWorkgroupsPerDimension',
        'maxComputeInvocationsPerWorkgroup', 'maxComputeWorkgroupSizeX'].map(key => [key, this.device.limits[key as keyof GPUSupportedLimits]])),
    };
    const module = this.device.createShaderModule({ code: shader });
    const compilation = await module.getCompilationInfo();
    const errors = compilation.messages.filter(message => message.type === 'error');
    if (errors.length) throw new Error(errors.map(error => error.message).join('\n'));
    this.pipeline = await this.device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
  }
  private async initializeMatrixPipeline(): Promise<void> {
    if (this.matrixPipeline) return;
    const module = this.device.createShaderModule({ code: matrixShader });
    const compilation = await module.getCompilationInfo();
    const errors = compilation.messages.filter(message => message.type === 'error');
    if (errors.length) throw new Error(errors.map(error => error.message).join('\n'));
    this.matrixPipeline = await this.device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
  }
  private check(): void { if (this.failure) throw new Error(this.failure); }
  async prepare(p: AnyParameters): Promise<void> {
    this.dispose();
    this.check();
    const limits = this.device.limits;
    this.matrix = 'm' in p;
    const sizes = 'm' in p ? [p.m * p.k * 4, p.k * p.n * 4, p.m * p.n * 4, p.r * p.m * p.n * 4, 16]
      : [p.n * 4, p.n * 4, p.n * 4, p.n * 4, 16];
    this.bytes = sizes[2];
    if (sizes.some(size => size > limits.maxBufferSize) || sizes.slice(0, 3).some(size => size > limits.maxStorageBufferBindingSize)) {
      throw new ConfigurationSkip('Input/output exceeds GPU buffer/storage-binding limits.');
    }
    if (WORKGROUP_SIZE > limits.maxComputeWorkgroupSizeX || WORKGROUP_SIZE > limits.maxComputeInvocationsPerWorkgroup) throw new Error('GPU cannot support the baseline workgroup size.');
    try { this.grid = dispatchGrid(this.bytes / 4, limits.maxComputeWorkgroupsPerDimension); }
    catch (error) { throw new ConfigurationSkip(String(error)); }
    if (this.matrix) await this.initializeMatrixPipeline();
    const pipeline = this.matrix ? this.matrixPipeline : this.pipeline;
    this.device.pushErrorScope('out-of-memory');
    this.device.pushErrorScope('validation');
    let caught: unknown;
    try {
      const usages = [GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST];
      for (let i = 0; i < usages.length; i++) this.buffers.push(this.device.createBuffer({ size: sizes[i], usage: usages[i] }));
      this.bindGroup = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
        entries: [0, 1, 2, 4].map((index, binding) => ({ binding, resource: { buffer: this.buffers[index] } })) });
      const parameters = 'm' in p ? [p.m, p.n, p.k, this.grid.x] : [p.n, p.k, this.grid.x, 0];
      this.device.queue.writeBuffer(this.buffers[4], 0, new Uint32Array(parameters));
    } catch (error) { caught = error; }
    const validation = await this.device.popErrorScope();
    const allocation = await this.device.popErrorScope();
    if (caught || validation || allocation) { this.dispose(); throw caught ?? new Error((validation ?? allocation)!.message); }
  }
  async stage(inputs: AnyInputs): Promise<void> {
    if ('a' in inputs) {
      this.check();
      this.device.queue.writeBuffer(this.buffers[1], 0, inputs.b as Float32Array<ArrayBuffer>);
    } else this.upload(inputs);
    await this.device.queue.onSubmittedWorkDone();
    this.check();
  }
  private upload(inputs: AnyInputs): void {
    this.check();
    this.device.queue.writeBuffer(this.buffers[0], 0, ('a' in inputs ? inputs.a[0] : inputs.x) as Float32Array<ArrayBuffer>);
    this.device.queue.writeBuffer(this.buffers[1], 0, ('a' in inputs ? inputs.b : inputs.y) as Float32Array<ArrayBuffer>);
  }
  private encode(r: number, readback: boolean, offset = 0): GPUCommandBuffer {
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.matrix ? this.matrixPipeline : this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    for (let i = 0; i < r; i++) pass.dispatchWorkgroups(this.grid.x, this.grid.y);
    pass.end();
    if (readback) encoder.copyBufferToBuffer(this.buffers[2], 0, this.buffers[3], offset, this.bytes);
    return encoder.finish();
  }
  private async mapOutput(output: Float32Array): Promise<void> {
    const readback = this.buffers[3];
    await readback.mapAsync(GPUMapMode.READ);
    try { output.set(new Float32Array(readback.getMappedRange(0, output.byteLength))); }
    finally { readback.unmap(); }
    this.check();
  }
  async warm(inputs: AnyInputs, output: Float32Array): Promise<number> {
    this.check();
    const start = performance.now();
    this.upload(inputs);
    this.device.queue.submit([this.encode(1, true)]);
    await this.mapOutput(output);
    return performance.now() - start;
  }
  async reuse(inputs: AnyInputs, output: Float32Array): Promise<number> {
    if (!('a' in inputs) || !this.matrix) throw new Error('B reuse requires W2.');
    this.check();
    const start = performance.now();
    for (let i = 0; i < inputs.a.length; i++) {
      this.device.queue.writeBuffer(this.buffers[0], 0, inputs.a[i] as Float32Array<ArrayBuffer>);
      // Submit each dispatch before overwriting A; queue operations preserve this order.
      this.device.queue.submit([this.encode(1, true, i * this.bytes)]);
    }
    await this.mapOutput(output);
    return performance.now() - start;
  }
  async resident(r: number): Promise<number> {
    this.check();
    const start = performance.now();
    this.device.queue.submit([this.encode(r, false)]);
    await this.device.queue.onSubmittedWorkDone();
    const elapsed = performance.now() - start;
    this.check();
    return elapsed;
  }
  async readOutput(output: Float32Array): Promise<void> {
    this.check();
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.buffers[2], 0, this.buffers[3], 0, this.bytes);
    this.device.queue.submit([encoder.finish()]);
    await this.mapOutput(output);
  }
  dispose(): void { for (const buffer of this.buffers) buffer.destroy(); this.buffers = []; }
}
