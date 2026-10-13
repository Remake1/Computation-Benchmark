#include <cstdint>
#ifdef __wasm_simd128__
#include <algorithm>
#include <wasm_simd128.h>
#endif

extern "C" void reduce_sum(const float* x, float* out,
                           std::uint32_t n, std::uint32_t b) {
  for (std::uint32_t batch = 0; batch < b; ++batch) {
    const float* input = x + batch * n;
    std::uint32_t i = 0;
    float sum = 0.0f;
#ifdef __wasm_simd128__
    v128_t lanes = wasm_f32x4_splat(0.0f);
    for (; i + 4 <= n; i += 4) lanes = wasm_f32x4_add(lanes, wasm_v128_load(input + i));
    // Explicit left-to-right lane combination, followed by the scalar tail.
    sum = wasm_f32x4_extract_lane(lanes, 0) + wasm_f32x4_extract_lane(lanes, 1);
    sum += wasm_f32x4_extract_lane(lanes, 2);
    sum += wasm_f32x4_extract_lane(lanes, 3);
#endif
    for (; i < n; ++i) sum += input[i];
    out[batch] = sum;
  }
}

extern "C" void affine(const float* x, const float* y, float* out,
                       std::uint32_t n, std::uint32_t k) {
  std::uint32_t i = 0;
#ifdef __wasm_simd128__
  const v128_t scale = wasm_f32x4_splat(0.5f);
  const v128_t decay = wasm_f32x4_splat(0.99609375f);
  const v128_t bias = wasm_f32x4_splat(0.001953125f);
  for (; i + 4 <= n; i += 4) {
    v128_t t = wasm_f32x4_add(wasm_f32x4_mul(scale, wasm_v128_load(x + i)),
                             wasm_v128_load(y + i));
    for (std::uint32_t step = 0; step < k; ++step) {
      t = wasm_f32x4_add(wasm_f32x4_mul(decay, t), bias);
    }
    wasm_v128_store(out + i, t);
  }
#endif
  for (; i < n; ++i) {
    float t = 0.5f * x[i] + y[i];
    for (std::uint32_t step = 0; step < k; ++step) {
      t = 0.99609375f * t + 0.001953125f;
    }
    out[i] = t;
  }
}

extern "C" void matmul(const float* a, const float* b, float* out,
                       std::uint32_t m, std::uint32_t n, std::uint32_t k) {
#ifdef __wasm_simd128__
  // 32x32x32 blocking; lanes are independent contiguous output columns.
  // Every output accumulates inner indices in ascending order, as in scalar.
  constexpr std::uint32_t block = 32;
  std::fill(out, out + m * n, 0.0f);
  for (std::uint32_t row0 = 0; row0 < m; row0 += block) {
    for (std::uint32_t col0 = 0; col0 < n; col0 += block) {
      const auto row_end = std::min(row0 + block, m);
      const auto col_end = std::min(col0 + block, n);
      for (std::uint32_t inner0 = 0; inner0 < k; inner0 += block) {
        const auto inner_end = std::min(inner0 + block, k);
        for (std::uint32_t row = row0; row < row_end; ++row) {
          for (std::uint32_t inner = inner0; inner < inner_end; ++inner) {
            const float value = a[row * k + inner];
            const v128_t av = wasm_f32x4_splat(value);
            std::uint32_t col = col0;
            for (; col + 4 <= col_end; col += 4) {
              float* c = out + row * n + col;
              const v128_t product = wasm_f32x4_mul(av, wasm_v128_load(b + inner * n + col));
              wasm_v128_store(c, wasm_f32x4_add(wasm_v128_load(c), product));
            }
            for (; col < col_end; ++col) {
              out[row * n + col] += value * b[inner * n + col];
            }
          }
        }
      }
    }
  }
#else
  for (std::uint32_t row = 0; row < m; ++row) {
    for (std::uint32_t col = 0; col < n; ++col) {
      float sum = 0.0f;
      for (std::uint32_t inner = 0; inner < k; ++inner) {
        sum += a[row * k + inner] * b[inner * n + col];
      }
      out[row * n + col] = sum;
    }
  }
#endif
}
