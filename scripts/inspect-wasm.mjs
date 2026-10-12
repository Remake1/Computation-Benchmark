import { spawnSync } from 'node:child_process';
import { existsSync, realpathSync } from 'node:fs';
import { delimiter, dirname, join, resolve } from 'node:path';

const compiler = process.env.EMXX || 'em++';
const compilerPath = /[/\\]/.test(compiler) ? resolve(compiler)
  : (process.env.PATH || '').split(delimiter).map(directory => join(directory, compiler)).find(existsSync);
const sdkDisassembler = compilerPath && join(dirname(realpathSync(compilerPath)), '../bin/wasm-dis');
const disassembler = process.env.WASM_DIS || (sdkDisassembler && existsSync(sdkDisassembler) ? sdkDisassembler : 'wasm-dis');
for (const simd of [false, true]) {
  const stem = simd ? 'affine-simd' : 'affine';
  const result = spawnSync(disassembler, [`public/wasm/${stem}.wasm`], { encoding: 'utf8', maxBuffer: 16 * 2 ** 20 });
  if (result.error || result.status !== 0) {
    console.error('Wasm disassembly failed. Activate emsdk or set WASM_DIS to the SDK wasm-dis binary.', result.error || result.stderr);
    process.exit(1);
  }
  const wat = result.stdout;
  if (!simd && /\b(?:v128|f32x4|i32x4|i8x16|i16x8|f64x2|i64x2)\b/.test(wat)) {
    throw new Error('Scalar build contains SIMD instructions.');
  }
  if (/relaxed/.test(wat)) throw new Error('Relaxed SIMD is forbidden in the primary build.');
  for (const kernel of ['affine', 'matmul']) {
    const start = wat.indexOf(` (func $${kernel} (`);
    if (start === -1) throw new Error(`Missing named ${kernel} kernel; build with --profiling-funcs.`);
    const end = wat.indexOf('\n (func ', start + 1);
    const body = wat.slice(start, end === -1 ? undefined : end);
    if (simd) {
      for (const instruction of ['v128.load', 'v128.store', 'f32x4.mul', 'f32x4.add']) {
        if (!body.includes(`(${instruction}`)) throw new Error(`${kernel} is missing ${instruction}.`);
      }
    }
    console.log(`${stem}: ${kernel} confirmed ${simd ? 'SIMD loads/stores and f32x4 multiply/add' : 'scalar'}`);
  }
}
