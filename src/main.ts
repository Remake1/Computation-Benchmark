import './style.css';
import { JavaScriptBackend } from './backends/javascript.ts';
import { WasmBackend } from './backends/wasm.ts';
import { WebGPUBackend } from './backends/webgpu.ts';
import { createHarness } from './harness/controller.ts';

const app = document.querySelector<HTMLDivElement>('#app');
if (!app) throw new Error('Missing harness root: #app.');

const harness = createHarness(app, [
  new JavaScriptBackend(),
  new JavaScriptBackend(true),
  new WasmBackend(),
  new WasmBackend(true),
  new WebGPUBackend(),
]);

void harness.initialize();
