import shader from './affine.wgsl?raw';
import { dispatchGrid, WORKGROUP_SIZE } from './dispatch.ts';
import type { Backend, Inputs, Parameters } from '../benchmark/types.ts';

export class WebGPUBackend implements Backend {
  readonly id = 'webgpu-baseline' as const;
  metadata: Record<string, unknown> = { workgroup_size: WORKGROUP_SIZE };
  private initialization?: Promise<void>;
  private device!: GPUDevice;
  private pipeline!: GPUComputePipeline;
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
  private check(): void { if (this.failure) throw new Error(this.failure); }
  async prepare(p: Parameters): Promise<void> {
    this.dispose();
    this.check();
    const limits = this.device.limits;
    this.bytes = p.n * 4;
    if (this.bytes > limits.maxBufferSize || this.bytes > limits.maxStorageBufferBindingSize) throw new Error('Input exceeds GPU buffer/storage-binding limits.');
    if (WORKGROUP_SIZE > limits.maxComputeWorkgroupSizeX || WORKGROUP_SIZE > limits.maxComputeInvocationsPerWorkgroup) throw new Error('GPU cannot support the baseline workgroup size.');
    this.grid = dispatchGrid(p.n, limits.maxComputeWorkgroupsPerDimension);
    this.device.pushErrorScope('out-of-memory');
    this.device.pushErrorScope('validation');
    let caught: unknown;
    try {
      const usages = [GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST];
      for (let i = 0; i < usages.length; i++) this.buffers.push(this.device.createBuffer({ size: i === 4 ? 16 : this.bytes, usage: usages[i] }));
      this.bindGroup = this.device.createBindGroup({ layout: this.pipeline.getBindGroupLayout(0),
        entries: [0, 1, 2, 4].map((index, binding) => ({ binding, resource: { buffer: this.buffers[index] } })) });
      this.device.queue.writeBuffer(this.buffers[4], 0, new Uint32Array([p.n, p.k, this.grid.x, 0]));
    } catch (error) { caught = error; }
    const validation = await this.device.popErrorScope();
    const allocation = await this.device.popErrorScope();
    if (caught || validation || allocation) { this.dispose(); throw caught ?? new Error((validation ?? allocation)!.message); }
  }
  async stage(inputs: Inputs): Promise<void> {
    this.upload(inputs);
    await this.device.queue.onSubmittedWorkDone();
    this.check();
  }
  private upload(inputs: Inputs): void {
    this.check();
    this.device.queue.writeBuffer(this.buffers[0], 0, inputs.x as Float32Array<ArrayBuffer>);
    this.device.queue.writeBuffer(this.buffers[1], 0, inputs.y as Float32Array<ArrayBuffer>);
  }
  private encode(r: number, readback: boolean): GPUCommandBuffer {
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    for (let i = 0; i < r; i++) pass.dispatchWorkgroups(this.grid.x, this.grid.y);
    pass.end();
    if (readback) encoder.copyBufferToBuffer(this.buffers[2], 0, this.buffers[3], 0, this.bytes);
    return encoder.finish();
  }
  private async mapOutput(output: Float32Array): Promise<void> {
    const readback = this.buffers[3];
    await readback.mapAsync(GPUMapMode.READ);
    try { output.set(new Float32Array(readback.getMappedRange())); }
    finally { readback.unmap(); }
    this.check();
  }
  async warm(inputs: Inputs, output: Float32Array): Promise<number> {
    this.check();
    const start = performance.now();
    this.upload(inputs);
    this.device.queue.submit([this.encode(1, true)]);
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
