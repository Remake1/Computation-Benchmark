# Browser computation benchmark

W1 and W2 compare baseline `js-number`, `js-f32`, `wasm-scalar`, and `webgpu-baseline`, plus an optimized `wasm-simd` backend. Choose a workload in the harness. W1 measures the vector affine transform (`K=0`) and a separately labeled recurrence/intensity variant (`K>0`). W2 measures dense row-major float32 matrix multiplication, `C[M,N] = A[M,K] × B[K,N]`. Baselines use direct row/column/K loops and one GPU invocation per output element; Wasm SIMD uses CPU blocking and vectors across columns. The app uses plain HTML controls and tables, with layout-only CSS.

## Wasm SIMD

Both Wasm variants compile the same `cpp/affine.cpp` source, with SIMD code selected by `__wasm_simd128__`. W1 loads four x/y elements into `f32x4` vectors, performs each runtime K recurrence in registers, and handles the final 1–3 elements with a scalar loop. W2 uses fixed 32×32×32 row/column/K blocks, broadcasts A values, and multiplies/adds four contiguous B/C columns at a time. All dimensions can end in partial blocks; columns outside complete vectors use scalar updates. Every execution zeros C before accumulation, including changing-A reuse batches.

SIMD lanes represent independent outputs. W2 retains ascending K accumulation per output across blocks, with no horizontal reduction or reassociation. Both variants round each float32 multiply/add separately. The builds use `-O3 -fno-vectorize -fno-slp-vectorize -ffp-contract=off`, with `-mno-simd128` for scalar and `-msimd128` for explicit SIMD intrinsics. Neither uses fast-math or relaxed SIMD. Disabling automatic vectorization keeps the scalar control and scalar tails explicit. See the [Emscripten SIMD documentation](https://emscripten.org/docs/porting/simd.html).

The harness probes SIMD support before loading the SIMD module, reports unsupported capability independently, and retains separate heaps and build metadata for each variant. Warm W1/W2 and W2 B reuse include the same JS/linear-memory copies as scalar; W1 resident execution excludes staging/readback. Memory budgets include both retained Wasm heaps. Samples export `implementation_tier: "optimized"` for SIMD, four lanes and scalar-tail handling, and W2 block size/loop/summation order. The results table shows the tier, timings, throughput, and validation status. W2 SIMD combines blocking and SIMD gains. Block size 32 is an initial choice, not a measured tuning winner.

## Workloads and timings

Both workloads run on the main thread and use five warmups, randomized backend order, complete-output validation, explicit capability/status records, cancellation, memory limits, and per-backend/configuration sampling budgets. Cancellation and sampling-budget checks occur between samples/batches, so a large CPU multiplication can delay cancellation. Unsupported backends do not prevent available backends from running.

W1 retains its quick, size, intensity, and custom presets. Warm end-to-end includes input staging and output return to JS. Resident batches repeat the same inputs and keep output in the backend; final readback and validation are outside timing. Logical bandwidth is shown only for `K=0`.

W2 provides quick squares (32, 64, 128; R=1; five samples), a square size sweep (32, 64, 127, 128, 255, 256, 512, 513, 1024, 2048), equal-work shape tests `(256,256,256)`, `(64,1024,256)`, `(1024,64,256)`, and custom dimensions from 1 to 2048. Other presets use selected R and default to 30 samples. Dimensions default to 128; R supports 1, 10, and 100.

- **Warm end-to-end:** stage A and B, multiply once, and return the complete C matrix to JS, all inside timing.
- **B-reuse batch:** stage B once outside timing; multiply R different A inputs, including each A staging and every C readback inside timing. GPU uploads/dispatches/copies are queued in order and the combined batch output is mapped once. This differs from W1's transfer-free resident timing.

W2 uses the existing xorshift32 PRNG: generate B first, then R row-major A matrices, rounding each seeded value in `[-0.5,0.5]` to float32. Generation and an independent double-precision reference (row/K/column traversal) run once before measurements. JS Number stores float32 results; JS f32 rounds each product and accumulation. Complete outputs are checked against `abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)` after every warmup and sample. Failures stop that backend/configuration and remain in the export.

W2 reports approximate GFLOP/s as `2*M*N*K*repetitions / (elapsed_ms*1e6)`, latency, IQR, and time per operation. All R canonical A matrices, returned C matrices, double references, and backend/staging buffers are included in memory estimates. Large size/R combinations can be skipped under the memory budget or GPU limits; a budget is checked between batches rather than interrupting a kernel.

Exports are named `w1-…jsonl` or `w2-…jsonl`. W1 retains schema version 1; W2 uses version 2 with `{m,n,k,r}` parameters and timing scenarios `warm`/`reuse`. An environment record stores generator version/seed, accuracy contract, backend/compiler metadata, and run options. Sample records store raw durations and statuses, maximum absolute/relative L2 error, and explicit API-level upload/download counts. For W2 reuse, `resident_setup_upload_bytes` describes one B staging per configuration (not additive per sample); timed staging includes all A and C copies, and `validation_readback_bytes` is zero because readback is already timed. GPU timestamps remain null.

The correctness self-test follows the selected workload. W2 checks hand-calculated multiplication, zero/identity inputs, rectangular/tail dimensions, long dot products, and changing-A batches with fixed B. Use it before collecting browser measurements.

## Harness code

`src/main.ts` creates the backends and starts the harness. The modules in `src/harness/` separate run lifecycle and cancellation (`controller.ts`), form parsing and export metadata (`experiment.ts`), markup (`template.ts`), DOM updates (`view.ts`), and result columns/rendering (`results-view.ts`). `workloads.ts` centralizes each workload's controls, presets, descriptions, metrics, and export versions. Table headings and cells share column definitions; sample updates render at most once per animation frame, with a final flush on completion, cancellation, or failure. Benchmark execution and validation remain in `src/benchmark/`.

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
pnpm test:wasm
pnpm build
pnpm preview
```

Production `build` recompiles both Wasm variants before TypeScript/Vite. Development requires `build:wasm` again after C++ changes and a page reload to load new modules. Generated assets live in ignored `public/wasm/` and are copied to production: scalar `affine.mjs`, `affine.wasm`, `build.json`; SIMD `affine-simd.mjs`, `affine-simd.wasm`, `build-simd.json`. Each metadata file records its backend, compiler version, exact argument list, source revision, and whether the working tree was dirty. Commit source before final research builds for an identifiable revision; preserve exported metadata with results.

`pnpm test:wasm` rebuilds both modules, disassembles them, and runs the real Emscripten modules in Node using local bytes. It requires the SDK's `wasm-dis` (automatically located beside an SDK compiler; set `WASM_DIS` if needed). Inspection rejects SIMD in the scalar module and relaxed SIMD in either module, and checks both SIMD kernels for `v128.load/store` and `f32x4.mul/add`. Function names are retained with `--profiling-funcs` for inspection. Integration tests cover hand/zero/identity cases, four-lane and block tails, rectangular and long-K multiplication, f32 agreement, repeated output overwrite, R=1/10/100 B reuse, heap growth, switching workloads, and exported timing/tuning records. `pnpm test` runs the toolchain-independent harness and capability tests.
