export const WORKGROUP_SIZE = 64;
export function dispatchGrid(n: number, limit: number): { x: number; y: number } {
  if (!Number.isInteger(n) || n < 1 || !Number.isInteger(limit) || limit < 1) throw new Error('Invalid dispatch dimensions.');
  const groups = Math.ceil(n / WORKGROUP_SIZE);
  const x = Math.min(groups, limit);
  const y = Math.ceil(groups / x);
  if (y > limit || x * y * WORKGROUP_SIZE > 0xffffffff) throw new Error('Input exceeds GPU dispatch limits.');
  return { x, y };
}
