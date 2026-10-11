#!/usr/bin/env bash
set -euo pipefail

if [ ! -d .emsdk ]; then
  git clone --depth 1 \
    https://github.com/emscripten-core/emsdk.git .emsdk
fi

./.emsdk/emsdk install 6.0.12
./.emsdk/emsdk activate 6.0.12
source ./.emsdk/emsdk_env.sh

pnpm build
