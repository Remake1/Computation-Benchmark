import shader from './affine.wgsl?raw';
import matrixShader from './matmul.wgsl?raw';
import reductionShader from './reduction.wgsl?raw';
import { dispatchGrid, dispatchWorkgroupGrid, WORKGROUP_SIZE } from './dispatch.ts';
import { reductionStages, REDUCTION_WORKGROUP_SIZE } from '../workloads/reduction.ts';
import { ConfigurationSkip } from '../benchmark/types.ts';
import type { AnyInputs, AnyParameters, Backend, ReductionParameters } from '../benchmark/types.ts';

export class WebGPUBackend implements Backend<AnyParameters, AnyInputs> {
  readonly id = 'webgpu-baseline' as const;
  metadata: Record<string, unknown> = { workgroup_size: WORKGROUP_SIZE };
  private initialization?: Promise<void>;
  private device!: GPUDevice;
  private pipeline!: GPUComputePipeline;
  private matrixPipeline!: GPUComputePipeline;
  private reductionPipeline!: GPUComputePipeline;
  private reduction = false;
  private reductionPasses: { bindGroup: GPUBindGroup; grid: { x: number; y: number } }[] = [];
  private finalReduction!: GPUBuffer;
  private readback!: GPUBuffer;
  readonly implementations = { W3: { tier: 'baseline' as const, tuning: {
    workgroup_size: REDUCTION_WORKGROUP_SIZE, elements_per_invocation: 1,
    summation_order: '64-element chunk trees, repeated stages per vector',
  } } };
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
        'maxComputeInvocationsPerWorkgroup', 'maxComputeWorkgroupSizeX', 'maxComputeWorkgroupStorageSize'].map(key => [key, this.device.limits[key as keyof GPUSupportedLimits]])),
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
    this.reduction = 'b' in p;
    if ('b' in p) { await this.prepareReduction(p); return; }
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
      this.readback = this.buffers[3];
      this.bindGroup = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
        entries: [0, 1, 2, 4].map((index, binding) => ({ binding, resource: { buffer: this.buffers[index] } })) });
      const parameters = 'm' in p ? [p.m, p.n, p.k, this.grid.x] : [p.n, p.k, this.grid.x, 0];
      this.device.queue.writeBuffer(this.buffers[4], 0, new Uint32Array(parameters));
    } catch (error) { caught = error; }
    const validation = await this.device.popErrorScope();
    const allocation = await this.device.popErrorScope();
    if (caught || validation || allocation) { this.dispose(); throw caught ?? new Error((validation ?? allocation)!.message); }
  }
  private async prepareReduction(p: ReductionParameters): Promise<void> {
    const limits = this.device.limits;
    const lengths = reductionStages(p.n);
    const sizes = [p.n * p.b * 4, ...lengths.map(n => n * p.b * 4)];
    if (sizes.some(size => size > limits.maxBufferSize || size > limits.maxStorageBufferBindingSize)) {
      throw new ConfigurationSkip('Reduction input/partials exceed GPU buffer/storage-binding limits.');
    }
    if (REDUCTION_WORKGROUP_SIZE > limits.maxComputeWorkgroupSizeX || REDUCTION_WORKGROUP_SIZE > limits.maxComputeInvocationsPerWorkgroup ||
        REDUCTION_WORKGROUP_SIZE * 4 > limits.maxComputeWorkgroupStorageSize) {
      throw new ConfigurationSkip('GPU cannot support the reduction workgroup.');
    }
    let grids: { x: number; y: number }[];
    try { grids = lengths.map(n => dispatchWorkgroupGrid(n * p.b, limits.maxComputeWorkgroupsPerDimension)); }
    catch (error) { throw new ConfigurationSkip(String(error)); }
    if (!this.reductionPipeline) {
      const module = this.device.createShaderModule({ code: reductionShader });
      const compilation = await module.getCompilationInfo();
      const errors = compilation.messages.filter(message => message.type === 'error');
      if (errors.length) throw new Error(errors.map(error => error.message).join('\n'));
      this.reductionPipeline = await this.device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    }
    this.bytes = p.b * 4;
    this.device.pushErrorScope('out-of-memory');
    this.device.pushErrorScope('validation');
    let caught: unknown;
    try {
      const input = this.device.createBuffer({ size: sizes[0], usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      this.buffers.push(input);
      let source = input;
      let sourceLength = p.n;
      for (let i = 0; i < lengths.length; i++) {
        const output = this.device.createBuffer({ size: sizes[i + 1], usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        this.buffers.push(output);
        const uniform = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.buffers.push(uniform);
        this.device.queue.writeBuffer(uniform, 0, new Uint32Array([sourceLength, p.b, lengths[i], grids[i].x]));
        const bindGroup = this.device.createBindGroup({ layout: this.reductionPipeline.getBindGroupLayout(0), entries: [
          { binding: 0, resource: { buffer: source } },
          { binding: 1, resource: { buffer: output } },
          { binding: 2, resource: { buffer: uniform } },
        ] });
        this.reductionPasses.push({ bindGroup, grid: grids[i] });
        source = output; sourceLength = lengths[i];
      }
      this.finalReduction = source;
      this.readback = this.device.createBuffer({ size: this.bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      this.buffers.push(this.readback);
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
    if ('a' in inputs || 'y' in inputs) this.device.queue.writeBuffer(this.buffers[1], 0, ('a' in inputs ? inputs.b : inputs.y) as Float32Array<ArrayBuffer>);
  }
  private encode(r: number, readback: boolean, offset = 0): GPUCommandBuffer {
    const encoder = this.device.createCommandEncoder();
    if (this.reduction) {
      for (let i = 0; i < r; i++) {
        for (const stage of this.reductionPasses) {
          const pass = encoder.beginComputePass();
          pass.setPipeline(this.reductionPipeline); pass.setBindGroup(0, stage.bindGroup);
          pass.dispatchWorkgroups(stage.grid.x, stage.grid.y); pass.end();
        }
      }
      if (readback) encoder.copyBufferToBuffer(this.finalReduction, 0, this.readback, offset, this.bytes);
      return encoder.finish();
    }
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.matrix ? this.matrixPipeline : this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    for (let i = 0; i < r; i++) pass.dispatchWorkgroups(this.grid.x, this.grid.y);
    pass.end();
    if (readback) encoder.copyBufferToBuffer(this.buffers[2], 0, this.readback, offset, this.bytes);
    return encoder.finish();
  }
  private async mapOutput(output: Float32Array): Promise<void> {
    const readback = this.readback;
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
    encoder.copyBufferToBuffer(this.reduction ? this.finalReduction : this.buffers[2], 0, this.readback, 0, this.bytes);
    this.device.queue.submit([encoder.finish()]);
    await this.mapOutput(output);
  }
  dispose(): void {
    for (const buffer of this.buffers) buffer.destroy();
    this.buffers = []; this.reductionPasses = []; this.reduction = false;
  }
}
