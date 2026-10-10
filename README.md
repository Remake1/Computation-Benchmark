# Browser computation benchmark

W1 compares baseline `js-number`, `js-f32`, scalar C++/Wasm, and WebGPU compute. It measures the vector affine transform (`K=0`) and a separately labeled recurrence/intensity variant (`K>0`). The app uses plain HTML controls and tables, with layout-only CSS.

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
