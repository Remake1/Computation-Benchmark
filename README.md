# Browser computation benchmark

W1 and W2 compare baseline `js-number`, `js-f32`, scalar C++/Wasm, and WebGPU compute. Choose a workload in the harness. W1 measures the vector affine transform (`K=0`) and a separately labeled recurrence/intensity variant (`K>0`). W2 measures dense row-major float32 matrix multiplication, `C[M,N] = A[M,K] × B[K,N]`, with direct row/column/K loops and one GPU invocation per output element. These are baseline CPU implementations, not optimized matrix multiplication. The app uses plain HTML controls and tables, with layout-only CSS.

## Workloads and timings

Both workloads run on the main thread and use five warmups, randomized backend order, complete-output validation, explicit capability/status records, cancellation, memory limits, and per-backend/configuration sampling budgets. Cancellation and sampling-budget checks occur between samples/batches, so a large CPU multiplication can delay cancellation. Unsupported backends do not prevent available backends from running.

W1 retains its quick, size, intensity, and custom presets. Warm end-to-end includes input staging and output return to JS. Resident batches repeat the same inputs and keep output in the backend; final readback and validation are outside timing. Logical bandwidth is shown only for `K=0`.

W2 provides quick squares (32, 64, 128; R=1; five samples), a square size sweep (32, 64, 127, 128, 255, 256, 512, 513, 1024, 2048), equal-work shape tests `(256,256,256)`, `(64,1024,256)`, `(1024,64,256)`, and custom dimensions from 1 to 2048. Other presets use selected R and default to 30 samples. Dimensions default to 128; R supports 1, 10, and 100.

- **Warm end-to-end:** stage A and B, multiply once, and return the complete C matrix to JS, all inside timing.
- **B-reuse batch:** stage B once outside timing; multiply R different A inputs, including each A staging and every C readback inside timing. GPU uploads/dispatches/copies are queued in order and the combined batch output is mapped once. This differs from W1's transfer-free resident timing.

W2 uses the existing xorshift32 PRNG: generate B first, then R row-major A matrices, rounding each seeded value in `[-0.5,0.5]` to float32. Generation and an independent double-precision reference (row/K/column traversal) run once before measurements. JS Number stores float32 results; JS f32 rounds each product and accumulation. Complete outputs are checked against `abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)` after every warmup and sample. Failures stop that backend/configuration and remain in the export.

W2 reports approximate GFLOP/s as `2*M*N*K*repetitions / (elapsed_ms*1e6)`, latency, IQR, time per operation, and matching JS speedups. All R canonical A matrices, returned C matrices, double references, and backend/staging buffers are included in memory estimates. Large size/R combinations can be skipped under the memory budget or GPU limits; a budget is checked between batches rather than interrupting a kernel.

Exports are named `w1-…jsonl` or `w2-…jsonl`. W1 retains schema version 1; W2 uses version 2 with `{m,n,k,r}` parameters and timing scenarios `warm`/`reuse`. An environment record stores generator version/seed, accuracy contract, backend/compiler metadata, and run options. Sample records store raw durations and statuses, maximum absolute/relative L2 error, and explicit API-level upload/download counts. For W2 reuse, `resident_setup_upload_bytes` describes one B staging per configuration (not additive per sample); timed staging includes all A and C copies, and `validation_readback_bytes` is zero because readback is already timed. GPU timestamps remain null.

The correctness self-test follows the selected workload. W2 checks hand-calculated multiplication, zero/identity inputs, rectangular/tail dimensions, long dot products, and changing-A batches with fixed B. Use it before collecting browser measurements.

## Setup

Use Node 24 (the repository pins 24.16.0) and pnpm 11.5.2:

```sh
nvm use
corepack enable
corepack pnpm install
```

Install the official [Emscripten SDK](https://emscripten.org/docs/getting_started/downloads.html). This implementation was verified with **6.0.12**. Keep the SDK outside the repository, or in the ignored `.emsdk/` directory:

```sh
git clone https://github.com/emscripten-core/emsdk.git /tmp/w1-emsdk
cd /tmp/w1-emsdk
./emsdk install 6.0.12
./emsdk activate 6.0.12
source ./emsdk_env.sh
```

Return to this project, build the module, and serve:

```sh
pnpm build:wasm
pnpm dev
```

The SDK supplies its own Node executable; if it changes the active version, `nvm use` restores the project version while retaining the compiler on PATH. Alternatively, set `EMXX` to the absolute path to `em++`. Serving requires localhost or HTTPS for WebGPU. Unsupported backends remain selectable and are exported explicitly; the available CPU backends continue to work.

```sh
pnpm check
pnpm test
pnpm build
pnpm preview
```

Production `build` recompiles Wasm before TypeScript/Vite. Development requires `build:wasm` again after C++ changes. Generated `.mjs`, `.wasm`, and `build.json` assets live in ignored `public/wasm/` and are copied to the production output. Build metadata records the compiler version, exact argument list, source revision, and whether the working tree was dirty. Commit source before final research builds for an identifiable revision; preserve exported metadata with results.
