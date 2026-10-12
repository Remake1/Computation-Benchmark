import { jsonLines } from '../benchmark/results.ts';
import { runBenchmark, timerResolution } from '../benchmark/runner.ts';
import { selfTest } from '../benchmark/self-test.ts';
import type { SampleRecord } from '../benchmark/types.ts';
import { createEnvironment, readExperiment, selectedBackends } from './experiment.ts';
import type { EnvironmentRecord, Experiment, HarnessBackend } from './experiment.ts';
import { createHarnessView } from './view.ts';
import { getWorkload, WORKLOADS } from './workloads.ts';

const READY_MESSAGE = 'Ready. Select a preset and run. Unavailable selected backends will be recorded explicitly.';

function downloadResults(environment: EnvironmentRecord, records: SampleRecord[]): void {
  const blob = new Blob([jsonLines(environment, records)], { type: 'application/x-ndjson' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `${environment.workload.toLowerCase()}-${environment.experiment_id}.jsonl`;
  try {
    link.click();
  } finally {
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
}

export function createHarness(root: HTMLElement, backends: HarnessBackend[]) {
  const backendIds = backends.map(backend => backend.id);
  const view = createHarnessView(root, backendIds);
  const sessionId = crypto.randomUUID();
  let definition = WORKLOADS.W1;
  let records: SampleRecord[] = [];
  let environment: EnvironmentRecord | null = null;
  let controller: AbortController | null = null;
  let initialized = false;

  function updateControls(): void {
    view.setBusy(!initialized || controller !== null, environment !== null);
  }

  async function executeOperation(
    operation: (signal: AbortSignal) => Promise<void>,
    onFailure?: () => void,
  ): Promise<void> {
    controller = new AbortController();
    updateControls();
    try {
      await operation(controller.signal);
    } catch (error) {
      view.setProgress(String(error));
      onFailure?.();
    } finally {
      controller = null;
      updateControls();
    }
  }

  async function startBenchmark(): Promise<void> {
    if (!initialized || controller) return;

    const workloadDefinition = definition;
    let experiment: Experiment;
    try {
      experiment = readExperiment(new FormData(view.form), workloadDefinition, backends, {
        sessionId,
        experimentId: crypto.randomUUID(),
        timerResolutionMs: timerResolution(),
      });
    } catch (error) {
      view.setProgress(String(error));
      return;
    }

    records = [];
    environment = null;
    view.results.render(records);
    await executeOperation(async signal => {
      try {
        await runBenchmark(workloadDefinition.workload, experiment.backends, experiment.options, signal, {
          ready: () => { environment = createEnvironment(workloadDefinition, experiment, backends); },
          record: record => {
            records.push(record);
            view.results.schedule(records);
          },
          progress: view.setProgress,
        });
      } finally {
        // Flush even when frames are paused in a background tab or a run fails.
        view.results.render(records);
      }
      view.setProgress(signal.aborted
        ? 'Cancelled. Partial results are available to export.'
        : 'Finished. Raw samples and statuses are available to export.');
    });
  }

  async function startSelfTest(): Promise<void> {
    if (!initialized || controller) return;
    const selected = selectedBackends(new FormData(view.form), backends);
    if (!selected.length) {
      view.setProgress('Select at least one backend.');
      return;
    }

    const workloadId = definition.workload.id;
    await executeOperation(async signal => {
      const results = await selfTest(selected, signal, view.setProgress, workloadId);
      view.showSelfTest(results);
      view.setProgress('Correctness self-test finished. These executions are not benchmark samples.');
    }, view.failSelfTest);
  }

  function changeWorkload(): void {
    if (!initialized || controller) return;
    definition = getWorkload(view.workloadSelect.value);
    for (const backend of backends) backend.dispose();
    records = [];
    environment = null;
    view.clearSelfTest();
    view.configureWorkload(definition);
    view.results.render(records);
    updateControls();
    view.setProgress(READY_MESSAGE);
  }

  view.form.addEventListener('submit', event => {
    event.preventDefault();
    void startBenchmark();
  });
  view.selfTestButton.addEventListener('click', () => { void startSelfTest(); });
  view.presetSelect.addEventListener('change', () => { view.updatePreset(definition); });
  view.workloadSelect.addEventListener('change', changeWorkload);
  view.cancelButton.addEventListener('click', () => {
    controller?.abort();
    view.setProgress('Cancelling after the current operation…');
  });
  view.exportButton.addEventListener('click', () => {
    if (environment) downloadResults(environment, records);
  });
  view.configureWorkload(definition);

  async function initialize(): Promise<void> {
    updateControls();
    const capabilities = await Promise.allSettled(backends.map(backend => backend.initialize()));
    view.showCapabilities(backendIds, capabilities);
    initialized = true;
    updateControls();
    view.setProgress(READY_MESSAGE);
    if (new URLSearchParams(location.search).has('selftest')) await startSelfTest();
  }

  return { initialize };
}
