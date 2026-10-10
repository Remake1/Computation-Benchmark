import test from 'node:test';
import assert from 'node:assert/strict';
import { affine, configurations, K_VALUES, MAX_N, random32, validateParameters } from '../src/workloads/affine.ts';
import { affineF32, affineNumber } from '../src/backends/javascript.ts';
import { dispatchGrid } from '../src/backends/dispatch.ts';

test('xorshift32 has a fixed documented sequence and identical seeded inputs', () => {
  const random = random32(1);
  assert.equal(random() * 2 ** 32, 270369);
  assert.equal(random() * 2 ** 32, 67634689);
  assert.deepEqual(affine.generate(65, 12345), affine.generate(65, 12345));
  assert.notDeepEqual(affine.generate(65, 12345), affine.generate(65, 42));
  const inputs = affine.generate(1000, 1);
  assert.ok([...inputs.x, ...inputs.y].every(value => value >= 0 && value <= 1));
  assert.throws(() => random32(0));
});

test('parameters reject invalid sizes, intensity, repetitions, and seeds', () => {
  const p = { n: 65, k: 4, r: 10 };
  validateParameters(p, 0xffffffff);
  for (const n of [0, -1, 1.5, Infinity, MAX_N + 1]) assert.throws(() => validateParameters({ ...p, n }, 1));
  assert.throws(() => validateParameters({ ...p, k: 3 }, 1));
  assert.throws(() => validateParameters({ ...p, r: 2 }, 1));
  for (const seed of [0, -1, 1.5, 2 ** 32, NaN]) assert.throws(() => validateParameters(p, seed));
});

test('presets vary one experimental axis and include tail cases', () => {
  const p = { n: 65, k: 16, r: 100 };
  assert.deepEqual(configurations('quick', p).map(p => [p.n, p.k, p.r]), [[256, 0, 10], [4096, 0, 10], [65536, 0, 10], [1048576, 0, 10]]);
  const sizes = configurations('size', p);
  assert.equal(sizes.length, 20);
  assert.equal(sizes.at(-1)?.n, MAX_N);
  assert.ok(sizes.some(p => p.n === 1048573));
  assert.ok(sizes.every(p => p.k === 16 && p.r === 100));
  assert.deepEqual(configurations('intensity', p).map(p => p.k), K_VALUES);
  assert.deepEqual(configurations('custom', p), [p]);
});

test('JS variants match independent reference on hand, zero, and tail cases for every K', () => {
  for (const kernel of [affineNumber, affineF32]) {
    const hand = new Float32Array(1);
    kernel(new Float32Array([1]), new Float32Array([0.25]), hand, 0);
    assert.equal(hand[0], 0.75);
    for (const n of [1, 63, 64, 65, 1023, 65537]) {
      const inputs = affine.generate(n, 12345);
      for (const k of K_VALUES) {
        const output = new Float32Array(n);
        kernel(inputs.x, inputs.y, output, k);
        assert.equal(affine.validate(inputs, output, { n, k, r: 10 }).validation_passed, true);
      }
    }
    const zeros = { x: new Float32Array(3), y: new Float32Array(3) };
    const output = new Float32Array(3);
    kernel(zeros.x, zeros.y, output, 0);
    assert.deepEqual(affine.validate(zeros, output, { n: 3, k: 0, r: 1 }), { validation_passed: true, max_abs_error: 0, relative_l2_error: 0 });
  }
});

test('validation rejects corrupt, nonfinite, and incomplete outputs', () => {
  const inputs = { x: new Float32Array([1]), y: new Float32Array([0.25]) };
  const p = { n: 1, k: 0, r: 1 };
  for (const value of [0, Infinity, NaN]) assert.equal(affine.validate(inputs, new Float32Array([value]), p).validation_passed, false);
  assert.equal(affine.validate(inputs, new Float32Array(0), p).validation_passed, false);
});

test('two-dimensional dispatch covers largest size within limits without allocating buffers', () => {
  for (const n of [1, 63, 64, 65, 1023, 65537, 1048573, MAX_N]) {
    const grid = dispatchGrid(n, 65535);
    assert.ok(grid.x <= 65535 && grid.y <= 65535);
    assert.ok(grid.x * grid.y * 64 >= n);
    const last = n - 1;
    const group = Math.floor(last / 64);
    const x = group % grid.x;
    const y = Math.floor(group / grid.x);
    assert.equal((y * grid.x + x) * 64 + last % 64, last);
    assert.ok(y < grid.y);
  }
  assert.deepEqual(dispatchGrid(MAX_N, 65535), { x: 65535, y: 5 });
  assert.throws(() => dispatchGrid(MAX_N, 1));
});
