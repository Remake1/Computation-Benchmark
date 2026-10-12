import type { BackendId } from '../benchmark/types.ts';
import { WORKLOADS } from './workloads.ts';

export function harnessTemplate(backendIds: readonly BackendId[]): string {
  return `
  <h1 id="workload-title"></h1>
  <p>Baseline implementations: ordinary JS, float32-control JS, scalar C++/Wasm, and WebGPU.</p>
  <p id="algorithm"></p>
  <details open><summary>Backend capabilities</summary><pre id="capabilities">Checking…</pre></details>
  <form id="controls">
    <fieldset><legend>Experiment</legend><div id="experiment-fields" class="fields">
      <label>Workload <select name="workload">${Object.values(WORKLOADS).map(definition => `<option value="${definition.workload.id}">${definition.title}</option>`).join('')}</select></label>
      <label>Preset <select name="preset"></select></label>
      <label><span id="r-label"></span> <select name="r"><option>1</option><option selected>10</option><option>100</option></select></label>
      <label>Seed <input name="seed" type="number" min="1" max="4294967295" step="1" value="12345" required></label>
      <label>Samples <input name="samples" type="number" min="1" max="10000" step="1" value="5" required></label>
      <label>Sampling budget per backend/case (seconds) <input name="budget" type="number" min="0.001" step="any" value="30" required></label>
      <label>Combined memory budget (MiB) <input name="memory" type="number" min="1" step="1" value="1024" required></label>
    </div><p id="preset-description"></p></fieldset>
    <fieldset><legend>Backends</legend><div class="choices">
      ${backendIds.map(id => `<label><input type="checkbox" name="backend" value="${id}" checked> ${id}</label>`).join('')}
    </div></fieldset>
    <fieldset><legend>Timing scenarios</legend><div id="scenarios" class="choices"></div><p id="scenario-description"></p></fieldset>
    <label>Hardware, browser version, power mode, and other notes <textarea name="notes" rows="3"></textarea></label>
    <div class="actions"><button id="run" type="submit" disabled>Run benchmark</button><button id="cancel" type="button" disabled>Cancel</button><button id="export" type="button" disabled>Export JSONL</button><button id="self-test" type="button" disabled>Run correctness self-test</button></div>
  </form>
  <p id="progress" role="status" aria-live="polite">Initializing backends…</p>
  <pre id="self-test-results"></pre>
  <h2>Results</h2>
  <p id="results-description"></p>
  <p>Five warmups per backend/configuration. Keep this page foreground and close DevTools for measurements. Cancellation and budget checks happen between samples/batches. Very short measurements are flagged; no crossover or cross-session confidence claims are made here.</p>
  <div class="table-scroll"><table><caption>Current experiment: one row per case, backend, and scenario</caption>
    <thead><tr id="result-headings"></tr></thead>
    <tbody id="results"></tbody>
  </table></div>
  <p id="record-count">No samples yet.</p>
`;
}
