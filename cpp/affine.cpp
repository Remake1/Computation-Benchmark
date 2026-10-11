#include <cstdint>

extern "C" void affine(const float* x, const float* y, float* out,
                       std::uint32_t n, std::uint32_t k) {
  for (std::uint32_t i = 0; i < n; ++i) {
    float t = 0.5f * x[i] + y[i];
    for (std::uint32_t step = 0; step < k; ++step) {
      t = 0.99609375f * t + 0.001953125f;
    }
    out[i] = t;
  }
}

extern "C" void matmul(const float* a, const float* b, float* out,
                       std::uint32_t m, std::uint32_t n, std::uint32_t k) {
  for (std::uint32_t row = 0; row < m; ++row) {
    for (std::uint32_t col = 0; col < n; ++col) {
      float sum = 0.0f;
      for (std::uint32_t inner = 0; inner < k; ++inner) {
        sum += a[row * k + inner] * b[inner * n + col];
      }
      out[row * n + col] = sum;
    }
  }
}
