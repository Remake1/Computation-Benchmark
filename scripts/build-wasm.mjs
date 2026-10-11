import { spawnSync } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';

const compiler = process.env.EMXX || 'em++';
const version = spawnSync(compiler, ['--version'], { encoding: 'utf8' });
if (version.error || version.status !== 0) {
  console.error('Emscripten is unavailable. Activate emsdk_env.sh or set EMXX to em++. See README.md.');
  process.exit(1);
}
const flags = [
  'cpp/affine.cpp', '-o', 'public/wasm/affine.mjs', '--no-entry',
  '-O3', '-fno-vectorize', '-fno-slp-vectorize', '-ffp-contract=off',
  '-sMODULARIZE=1', '-sEXPORT_ES6=1', '-sENVIRONMENT=web',
  '-sFILESYSTEM=0', '-sALLOW_MEMORY_GROWTH=1',
  '-sEXPORTED_FUNCTIONS=["_affine","_matmul","_malloc","_free"]',
  '-sEXPORTED_RUNTIME_METHODS=["HEAPF32"]',
];
mkdirSync('public/wasm', { recursive: true });
const build = spawnSync(compiler, flags, { stdio: 'inherit' });
if (build.error || build.status !== 0) process.exit(build.status || 1);
const revision = spawnSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' });
const dirty = spawnSync('git', ['status', '--porcelain'], { encoding: 'utf8' });
writeFileSync('public/wasm/build.json', JSON.stringify({
  compiler_version: version.stdout.trim(), compiler_flags: flags,
  build_command: [compiler, ...flags],
  source_revision: revision.status === 0 ? revision.stdout.trim() : null,
  source_dirty: dirty.status === 0 ? dirty.stdout.length > 0 : null,
  built_at: new Date().toISOString(),
}, null, 2));
console.log('Built public/wasm/affine.mjs, affine.wasm, and build.json');
