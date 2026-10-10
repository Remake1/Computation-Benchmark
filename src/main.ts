import './style.css';
import { JavaScriptBackend } from './backends/javascript.ts';
import { WasmBackend } from './backends/wasm.ts';
import { WebGPUBackend } from './backends/webgpu.ts';
import { runBenchmark, timerResolution } from './benchmark/runner.ts';
import { jsonLines, summarize } from './benchmark/results.ts';
import { selfTest } from './benchmark/self-test.ts';
import { affine, configurations, GENERATOR_VERSION, validateParameters } from './workloads/affine.ts';
import type { Backend, Preset, RunOptions, SampleRecord, Scenario } from './benchmark/types.ts';

const backends: Backend[] = [new JavaScriptBackend(), new JavaScriptBackend(true), new WasmBackend(), new WebGPUBackend()];
const sessionId = crypto.randomUUID();
let records: SampleRecord[] = [];
let environment: Record<string, unknown> | null = null;
let controller: AbortController | null = null;
const app = document.querySelector<HTMLDivElement>('#app')!;
app.innerHTML = `
  <h1>W1: vector affine transform</h1>
  <p>Baseline implementations: ordinary JS, float32-control JS, scalar C++/Wasm, and WebGPU.</p>
  <p><code>t = 0.5*x + y</code>; for K steps, <code>t = 0.99609375*t + 0.001953125</code>.</p>
  <details open><summary>Backend capabilities</summary><pre id="capabilities">Checking…</pre></details>
  <form id="controls">
    <fieldset><legend>Experiment</legend><div class="fields">
      <label>Preset <select name="preset"><option value="quick">Quick check</option><option value="size">Size sweep</option><option value="intensity">Intensity sweep</option><option value="custom">Custom case</option></select></label>
      <label>N (elements) <input name="n" type="number" min="1" max="16777216" step="1" value="65536" required></label>
      <label>K (recurrence steps) <select name="k"><option>0</option><option>4</option><option>16</option><option>64</option><option>256</option></select></label>
      <label>R (resident repetitions) <select name="r"><option>1</option><option selected>10</option><option>100</option></select></label>
      <label>Seed <input name="seed" type="number" min="1" max="4294967295" step="1" value="12345" required></label>
      <label>Samples <input name="samples" type="number" min="1" max="10000" step="1" value="5" required></label>
      <label>Sampling budget per backend/case (seconds) <input name="budget" type="number" min="0.001" step="any" value="30" required></label>
      <label>Combined memory budget (MiB) <input name="memory" type="number" min="1" step="1" value="1024" required></label>
    </div><p id="preset-description"></p></fieldset>
    <fieldset><legend>Backends</legend><div class="choices">
      ${backends.map(backend => `<label><input type="checkbox" name="backend" value="${backend.id}" checked> ${backend.id}</label>`).join('')}
    </div></fieldset>
    <fieldset><legend>Timing scenarios</legend><div class="choices">
      <label><input type="checkbox" name="scenario" value="warm" checked> Warm end-to-end (JS-visible output)</label>
      <label><input type="checkbox" name="scenario" value="resident" checked> Resident batch (output stays in backend)</label>
    </div><p>Resident repetitions overwrite the output from the same inputs. Final readback and validation are outside timing.</p></fieldset>
    <label>Hardware, browser version, power mode, and other notes <textarea name="notes" rows="3"></textarea></label>
    <div class="actions"><button id="run" type="submit" disabled>Run benchmark</button><button id="cancel" type="button" disabled>Cancel</button><button id="export" type="button" disabled>Export JSONL</button><button id="self-test" type="button" disabled>Run correctness self-test</button></div>
  </form>
  <p id="progress" role="status" aria-live="polite">Initializing backends…</p>
  <pre id="self-test-results"></pre>
  <h2>Results</h2>
  <p>K=0 is the original affine transform; K&gt;0 is the intensity variant. Warm timings are individual request latency; resident timings are batch throughput. Bandwidth is logical, not physical memory traffic. Speedup = matching JS median / backend median.</p>
  <p>Five warmups per backend/configuration. Keep this page foreground and close DevTools for measurements. Cancellation and budget checks happen between operations. Very short measurements are flagged; no crossover or cross-session confidence claims are made here.</p>
  <div class="table-scroll"><table><caption>Current experiment: one row per case, backend, and scenario</caption>
    <thead><tr>${['N', 'K', 'R', 'Backend', 'Scenario', 'Samples', 'Median ms', 'IQR ms', 'ms / operation', 'vs JS Number', 'vs JS f32', 'Logical GB/s', 'Validation / status'].map(label => `<th scope="col">${label}</th>`).join('')}</tr></thead>
    <tbody id="results"></tbody>
  </table></div>
  <p id="record-count">No samples yet.</p>
`;
const form = document.querySelector<HTMLFormElement>('#controls')!;
const runButton = document.querySelector<HTMLButtonElement>('#run')!;
const cancelButton = document.querySelector<HTMLButtonElement>('#cancel')!;
const exportButton = document.querySelector<HTMLButtonElement>('#export')!;
const selfTestButton = document.querySelector<HTMLButtonElement>('#self-test')!;
const progress = document.querySelector<HTMLParagraphElement>('#progress')!;
const tableBody = document.querySelector<HTMLTableSectionElement>('#results')!;
const number = (data: FormData, name: string): number => Number(data.get(name));
const selectedBackends = () => {
  const selected = new FormData(form).getAll('backend');
  return backends.filter(backend => selected.includes(backend.id));
};
function busy(value: boolean): void {
  runButton.disabled = value; selfTestButton.disabled = value; cancelButton.disabled = !value;
  for (const input of form.querySelectorAll<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>('input, select, textarea')) input.disabled = value;
  exportButton.disabled = value || !environment;
}
function render(): void {
  const fragment = document.createDocumentFragment();
  const format = (value: number | null): string => value === null ? '—' : value.toPrecision(5);
  for (const row of summarize(records)) {
    const tr = document.createElement('tr');
    const status = [row.statuses.includes('ok') ? 'passed' : '', ...row.statuses.filter(status => status !== 'ok'), row.nearTimer ? 'near timer granularity' : '', ...row.reasons].filter(Boolean).join('; ');
    const values = [row.parameters.n, row.parameters.k, row.parameters.r, row.backend, row.scenario, row.count,
      format(row.median), format(row.iqr), format(row.perOperation), format(row.numberSpeedup), format(row.f32Speedup), format(row.bandwidthGBs), status];
    for (const value of values) { const td = document.createElement('td'); td.textContent = String(value); tr.append(td); }
    fragment.append(tr);
  }
  tableBody.replaceChildren(fragment);
  document.querySelector('#record-count')!.textContent = `${records.filter(record => record.elapsed_ms !== null).length} timed samples; ${records.length} raw sample/status records.`;
}
function describePreset(): void {
  const preset = new FormData(form).get('preset');
  const descriptions = {
    quick: 'Quick: N=256,4096,65536,1048576; K=0; R=10. N/K/R controls apply to other presets.',
    size: 'Size sweep: 2^8 through 2^24 plus three tail cases; uses selected K and R.',
    intensity: 'Intensity sweep: K=0,4,16,64,256; uses selected N and R.',
    custom: 'Custom: one case with the selected N, K, and R.',
  };
  document.querySelector('#preset-description')!.textContent = descriptions[preset as Preset];
}
form.querySelector<HTMLSelectElement>('[name=preset]')!.addEventListener('change', () => {
  form.querySelector<HTMLInputElement>('[name=samples]')!.value = new FormData(form).get('preset') === 'quick' ? '5' : '30';
  describePreset();
});
describePreset();
cancelButton.addEventListener('click', () => { controller?.abort(); progress.textContent = 'Cancelling after the current operation…'; });
exportButton.addEventListener('click', () => {
  if (!environment) return;
  const url = URL.createObjectURL(new Blob([jsonLines(environment, records)], { type: 'application/x-ndjson' }));
  const link = document.createElement('a'); link.href = url; link.download = `w1-${environment.experiment_id}.jsonl`; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
form.addEventListener('submit', async event => {
  event.preventDefault();
  if (controller) return;
  const data = new FormData(form);
  const seed = number(data, 'seed');
  const p = { n: number(data, 'n'), k: number(data, 'k'), r: number(data, 'r') };
  const selected = selectedBackends();
  const cases = configurations(data.get('preset') as Preset, p);
  try {
    for (const parameters of cases) validateParameters(parameters, seed);
    const experimentId = crypto.randomUUID();
    const options: RunOptions = {
      experimentId, sessionId, seed, schedulingSeed: (seed ^ 0x9e3779b9) >>> 0 || 1,
      samples: number(data, 'samples'), warmups: 5, budgetMs: number(data, 'budget') * 1000,
      memoryBudgetBytes: number(data, 'memory') * 2 ** 20, timerResolutionMs: timerResolution(),
      sourceRevision: backends.find(backend => backend.id === 'wasm-scalar')?.metadata.source_revision as string | undefined ?? null,
      scenarios: data.getAll('scenario') as Scenario[], configurations: cases,
    };
    records = []; environment = null; render(); controller = new AbortController(); busy(true);
    const visibleNavigator = navigator as Navigator & { deviceMemory?: number };
    await runBenchmark(affine, selected, options, controller.signal, {
      ready: () => {
        environment = {
          record_type: 'environment', schema_version: 1, experiment_id: experimentId, environment_id: experimentId,
          session_id: sessionId, timestamp: new Date().toISOString(),
          source_revision: backends.find(backend => backend.id === 'wasm-scalar')?.metadata.source_revision ?? null,
          generator_version: GENERATOR_VERSION, generator_seed: seed, run_options: options,
          user_agent: navigator.userAgent, platform: navigator.platform, language: navigator.language,
          hardware_concurrency: navigator.hardwareConcurrency, device_memory_gib: visibleNavigator.deviceMemory ?? null,
          secure_context: isSecureContext, cross_origin_isolated: crossOriginIsolated,
          execution_context: 'main thread', visibility_at_start: document.visibilityState,
          notes: data.get('notes'), backends: Object.fromEntries(backends.map(backend => [backend.id, backend.metadata])),
          accuracy_contract: { atol: 1e-5, rtol: 1e-4 },
          staged_bytes_definition: 'Explicit API-level transfers/copies inside timing; not physical bus traffic.',
          resident_setup_definition: 'One input staging per backend/configuration, shared by warmups and samples; not additive per record.',
        };
      },
      record: record => { records.push(record); render(); },
      progress: message => { progress.textContent = message; },
    });
    progress.textContent = controller.signal.aborted ? 'Cancelled. Partial results are available to export.' : 'Finished. Raw samples and statuses are available to export.';
  } catch (error) { progress.textContent = String(error); }
  finally { controller = null; busy(false); }
});
selfTestButton.addEventListener('click', async () => {
  if (controller) return;
  const selected = selectedBackends();
  if (!selected.length) { progress.textContent = 'Select at least one backend.'; return; }
  controller = new AbortController(); busy(true);
  try {
    const results = await selfTest(selected, controller.signal, message => { progress.textContent = message; });
    document.querySelector('#self-test-results')!.textContent = results.map(result => `${result.backend}: ${result.status} (${result.cases} checks)${result.reason ? ` — ${result.reason}` : ''}`).join('\n');
    document.body.dataset.selftest = results.some(result => result.status === 'failed') ? 'failed' : 'finished';
    progress.textContent = 'Correctness self-test finished. These executions are not benchmark samples.';
  } catch (error) { progress.textContent = String(error); document.body.dataset.selftest = 'failed'; }
  finally { controller = null; busy(false); }
});
async function initialize(): Promise<void> {
  const capabilities = await Promise.allSettled(backends.map(backend => backend.initialize()));
  document.querySelector('#capabilities')!.textContent = capabilities.map((result, i) =>
    `${backends[i].id}: ${result.status === 'fulfilled' ? 'available' : `unavailable — ${result.reason}`}`).join('\n');
  progress.textContent = 'Ready. Select a preset and run. Unavailable selected backends will be recorded explicitly.';
  busy(false);
  if (new URLSearchParams(location.search).has('selftest')) selfTestButton.click();
}
void initialize();
