import type { BackendId } from '../benchmark/types.ts';
import type { SelfTestResult } from '../benchmark/self-test.ts';
import { createResultsView } from './results-view.ts';
import { harnessTemplate } from './template.ts';
import type { ParameterField, WorkloadDefinition } from './workloads.ts';

function requiredElement<T extends Element>(root: ParentNode, selector: string): T {
  const element = root.querySelector<T>(selector);
  if (!element) throw new Error(`Missing harness element: ${selector}.`);
  return element;
}

function requiredLabel(control: Element): HTMLLabelElement {
  const label = control.closest('label');
  if (!label) throw new Error('Missing harness label for preset.');
  return label;
}

function parameterFieldHtml(field: ParameterField): string {
  const control = field.type === 'number'
    ? `<input name="${field.name}" type="number" min="${field.min}" max="${field.max}" step="1" value="${field.value}" required>`
    : `<select name="${field.name}">${field.values.map(value => `<option>${value}</option>`).join('')}</select>`;
  return `<label data-workload-field>${field.label} ${control}</label>`;
}

export function createHarnessView(root: HTMLElement, backendIds: readonly BackendId[]) {
  root.innerHTML = harnessTemplate(backendIds);

  const document = root.ownerDocument;
  const form = requiredElement<HTMLFormElement>(root, '#controls');
  const runButton = requiredElement<HTMLButtonElement>(root, '#run');
  const cancelButton = requiredElement<HTMLButtonElement>(root, '#cancel');
  const exportButton = requiredElement<HTMLButtonElement>(root, '#export');
  const selfTestButton = requiredElement<HTMLButtonElement>(root, '#self-test');
  const workloadSelect = requiredElement<HTMLSelectElement>(form, '[name=workload]');
  const presetSelect = requiredElement<HTMLSelectElement>(form, '[name=preset]');
  const presetLabel = requiredLabel(presetSelect);
  const samplesInput = requiredElement<HTMLInputElement>(form, '[name=samples]');
  const title = requiredElement<HTMLElement>(root, '#workload-title');
  const algorithm = requiredElement<HTMLElement>(root, '#algorithm');
  const capabilities = requiredElement<HTMLElement>(root, '#capabilities');
  const presetDescription = requiredElement<HTMLElement>(root, '#preset-description');
  const repetitionLabel = requiredElement<HTMLElement>(root, '#r-label');
  const scenarios = requiredElement<HTMLElement>(root, '#scenarios');
  const scenarioDescription = requiredElement<HTMLElement>(root, '#scenario-description');
  const resultsDescription = requiredElement<HTMLElement>(root, '#results-description');
  const progress = requiredElement<HTMLElement>(root, '#progress');
  const selfTestResults = requiredElement<HTMLElement>(root, '#self-test-results');
  const results = createResultsView(
    requiredElement<HTMLTableRowElement>(root, '#result-headings'),
    requiredElement<HTMLTableSectionElement>(root, '#results'),
    requiredElement<HTMLElement>(root, '#record-count'),
  );

  function describePreset(definition: WorkloadDefinition): void {
    presetDescription.textContent = definition.presets.find(preset => preset.value === presetSelect.value)?.description ?? '';
  }

  function configureWorkload(definition: WorkloadDefinition): void {
    title.textContent = definition.title;
    document.title = `${definition.workload.id} browser computation benchmark`;
    algorithm.innerHTML = definition.algorithmHtml;
    presetSelect.innerHTML = definition.presets.map(preset => `<option value="${preset.value}">${preset.label}</option>`).join('');

    for (const field of form.querySelectorAll('[data-workload-field]')) field.remove();
    presetLabel.insertAdjacentHTML('afterend', definition.fields.map(parameterFieldHtml).join(''));
    repetitionLabel.textContent = definition.repetitionLabel;
    scenarios.innerHTML = definition.scenarios.map(scenario =>
      `<label><input type="checkbox" name="scenario" value="${scenario.value}" checked> ${scenario.label}</label>`,
    ).join('');
    scenarioDescription.textContent = definition.scenarioDescription;
    resultsDescription.textContent = definition.resultsDescription;
    results.configure(definition);
    samplesInput.value = '5';
    describePreset(definition);
  }

  function updatePreset(definition: WorkloadDefinition): void {
    samplesInput.value = presetSelect.value === 'quick' ? '5' : '30';
    describePreset(definition);
  }

  function setBusy(busy: boolean, canExport: boolean): void {
    runButton.disabled = busy;
    selfTestButton.disabled = busy;
    cancelButton.disabled = !busy;
    exportButton.disabled = busy || !canExport;
    for (const input of form.querySelectorAll<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>('input, select, textarea')) {
      input.disabled = busy;
    }
  }

  function setProgress(message: string): void {
    progress.textContent = message;
  }

  function showCapabilities(backendIds: readonly BackendId[], capabilitiesResults: PromiseSettledResult<void>[]): void {
    capabilities.textContent = capabilitiesResults.map((result, index) =>
      `${backendIds[index]}: ${result.status === 'fulfilled' ? 'available' : `unavailable — ${result.reason}`}`,
    ).join('\n');
  }

  function clearSelfTest(): void {
    selfTestResults.textContent = '';
    delete document.body.dataset.selftest;
  }

  function showSelfTest(results: SelfTestResult[]): void {
    selfTestResults.textContent = results.map(result =>
      `${result.backend}: ${result.status} (${result.cases} checks)${result.reason ? ` — ${result.reason}` : ''}`,
    ).join('\n');
    document.body.dataset.selftest = results.some(result => result.status === 'failed') ? 'failed' : 'finished';
  }

  function failSelfTest(): void {
    document.body.dataset.selftest = 'failed';
  }

  return {
    form,
    cancelButton,
    exportButton,
    selfTestButton,
    workloadSelect,
    presetSelect,
    results,
    configureWorkload,
    updatePreset,
    setBusy,
    setProgress,
    showCapabilities,
    clearSelfTest,
    showSelfTest,
    failSelfTest,
  };
}
