import type { AnyParameters, BackendId, SampleRecord, Scenario, Status } from './types.ts';

export interface Summary {
  backend: BackendId;
  workload: string;
  parameters: AnyParameters;
  scenario: Scenario;
  count: number;
  median: number | null;
  iqr: number | null;
  perOperation: number | null;
  numberSpeedup: number | null;
  f32Speedup: number | null;
  bandwidthGBs: number | null;
  gflops: number | null;
  statuses: Status[];
  reasons: string[];
  nearTimer: boolean;
}
export function quantile(sorted: number[], q: number): number {
  const position = (sorted.length - 1) * q;
  const lower = Math.floor(position);
  return sorted[lower] + (sorted[Math.ceil(position)] - sorted[lower]) * (position - lower);
}
export function summarize(records: SampleRecord[]): Summary[] {
  const groups = new Map<string, SampleRecord[]>();
  for (const record of records) {
    const { n, k, r } = record.parameters;
    const key = `${record.workload}/${record.backend}/${'m' in record.parameters ? record.parameters.m : ''}/${n}/${k}/${r}/${record.timing_scenario}`;
    const group = groups.get(key) ?? [];
    group.push(record); groups.set(key, group);
  }
  const summaries = [...groups.values()].map(group => {
    const first = group[0];
    const times = group.filter(r => r.status === 'ok' && r.elapsed_ms !== null).map(r => r.elapsed_ms!).sort((a, b) => a - b);
    const median = times.length ? quantile(times, 0.5) : null;
    const repetitions = first.repetitions;
    return {
      workload: first.workload, backend: first.backend, parameters: first.parameters, scenario: first.timing_scenario, count: times.length,
      median, iqr: times.length ? quantile(times, 0.75) - quantile(times, 0.25) : null,
      perOperation: median === null ? null : median / repetitions,
      numberSpeedup: null, f32Speedup: null,
      bandwidthGBs: first.workload === 'W1' && first.parameters.k === 0 && median !== null && median > 0 ? 12 * first.parameters.n * repetitions / (median * 1e6) : null,
      gflops: first.workload === 'W2' && 'm' in first.parameters && median !== null && median > 0
        ? 2 * first.parameters.m * first.parameters.n * first.parameters.k * repetitions / (median * 1e6) : null,
      statuses: [...new Set(group.map(r => r.status))],
      reasons: [...new Set(group.flatMap(r => r.skip_reason ? [r.skip_reason] : []))],
      nearTimer: group.some(r => r.near_timer_granularity),
    } as Summary;
  });
  for (const row of summaries) {
    const matching = (backend: BackendId) => summaries.find(other => other.workload === row.workload && other.backend === backend && other.scenario === row.scenario &&
      ('m' in other.parameters ? other.parameters.m : null) === ('m' in row.parameters ? row.parameters.m : null) &&
      other.parameters.n === row.parameters.n && other.parameters.k === row.parameters.k && other.parameters.r === row.parameters.r);
    if (row.median !== null && row.median > 0 && !row.statuses.includes('validation-failed')) {
      const number = matching('js-number');
      const f32 = matching('js-f32');
      row.numberSpeedup = number?.median != null && number.median > 0 && !number.statuses.includes('validation-failed') ? number.median / row.median : null;
      row.f32Speedup = f32?.median != null && f32.median > 0 && !f32.statuses.includes('validation-failed') ? f32.median / row.median : null;
    }
  }
  return summaries;
}
export function jsonLines(environment: Record<string, unknown>, records: SampleRecord[]): string {
  return [environment, ...records].map(record => JSON.stringify(record)).join('\n') + '\n';
}
