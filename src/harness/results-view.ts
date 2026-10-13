import { summarize } from '../benchmark/results.ts';
import type { Summary } from '../benchmark/results.ts';
import type { SampleRecord } from '../benchmark/types.ts';
import type { ParameterName, WorkloadDefinition } from './workloads.ts';

interface ResultColumn {
  label: string;
  value(row: Summary): string | number;
}

const iqrFormatter = new Intl.NumberFormat('en-US', { maximumFractionDigits: 3, useGrouping: false });

function formatNumber(value: number | null): string {
  return value === null ? '—' : value.toPrecision(5);
}

function formatStatus(row: Summary): string {
  return [
    row.statuses.includes('ok') ? 'passed' : '',
    ...row.statuses.filter(status => status !== 'ok'),
    row.nearTimer ? 'near timer granularity' : '',
    ...row.reasons,
  ].filter(Boolean).join('; ');
}

function resultColumns(definition: WorkloadDefinition): ResultColumn[] {
  const dimensions: ParameterName[] = [...definition.fields.map(field => field.name), 'r'];
  return [
    ...dimensions.map(name => ({
      label: name.toUpperCase(),
      value: (row: Summary) => {
        const p = row.parameters;
        if (name === 'm') return 'm' in p ? p.m : '—';
        if (name === 'b') return 'b' in p ? p.b : '—';
        if (name === 'k') return 'k' in p ? p.k : '—';
        return p[name];
      },
    })),
    { label: 'Backend', value: row => row.backend },
    { label: 'Tier', value: row => row.tier },
    { label: 'Scenario', value: row => row.scenario },
    { label: 'Samples', value: row => row.count },
    { label: 'Median ms', value: row => formatNumber(row.median) },
    { label: 'IQR ms', value: row => row.iqr === null ? '—' : iqrFormatter.format(row.iqr) },
    { label: 'ms / operation', value: row => formatNumber(row.perOperation) },
    { label: definition.throughput.label, value: row => formatNumber(row[definition.throughput.metric]) },
    ...(definition.workload.id === 'W3' ? [{ label: 'GPU stages', value: (row: Summary) => row.reductionStages ?? '—' }] : []),
    { label: 'Validation / status', value: formatStatus },
  ];
}

export function createResultsView(
  headings: HTMLTableRowElement,
  body: HTMLTableSectionElement,
  recordCount: HTMLElement,
) {
  const document = body.ownerDocument;
  let columns: ResultColumn[] = [];
  let pendingFrame: number | null = null;

  function configure(definition: WorkloadDefinition): void {
    columns = resultColumns(definition);
    const cells = columns.map(column => {
      const cell = document.createElement('th');
      cell.scope = 'col';
      cell.textContent = column.label;
      return cell;
    });
    headings.replaceChildren(...cells);
  }

  function render(records: SampleRecord[]): void {
    if (pendingFrame !== null) {
      cancelAnimationFrame(pendingFrame);
      pendingFrame = null;
    }

    const fragment = document.createDocumentFragment();
    for (const summary of summarize(records)) {
      const row = document.createElement('tr');
      for (const column of columns) {
        const cell = document.createElement('td');
        cell.textContent = String(column.value(summary));
        row.append(cell);
      }
      fragment.append(row);
    }
    body.replaceChildren(fragment);

    const timedSamples = records.filter(record => record.elapsed_ms !== null).length;
    recordCount.textContent = `${timedSamples} timed samples; ${records.length} raw sample/status records.`;
  }

  function schedule(records: SampleRecord[]): void {
    // Coalesce samples into one table update per frame, outside backend timing.
    if (pendingFrame === null) {
      pendingFrame = requestAnimationFrame(() => render(records));
    }
  }

  return { configure, render, schedule };
}
