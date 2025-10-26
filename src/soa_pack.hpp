#pragma once

#include <cstddef>
#include <cstring>

namespace soa {

constexpr int kLane =
#if defined(__AVX2__) || defined(ADMC_USE_AVX2)
    4
#elif defined(__ARM_NEON) || defined(ADMC_USE_NEON)
    2
#else
    1
#endif
    ;

struct Block {
  int start = 0;
  int count = 0;
};

inline void load4d(const double* src, double* __restrict lanebuf) {
  if constexpr (kLane == 1) {
    lanebuf[0] = src[0];
  } else {
    std::memcpy(lanebuf, src,
                sizeof(double) * static_cast<std::size_t>(kLane));
  }
}

inline void store4d(double* dst, const double* __restrict lanebuf) {
  if constexpr (kLane == 1) {
    dst[0] = lanebuf[0];
  } else {
    std::memcpy(dst, lanebuf,
                sizeof(double) * static_cast<std::size_t>(kLane));
  }
}

inline void load4d_masked(const double* src,
                          double* __restrict lanebuf,
                          int count) {
  if constexpr (kLane == 1) {
    (void)count;
    lanebuf[0] = src[0];
  } else {
    for (int i = 0; i < kLane; ++i) {
      lanebuf[i] = (i < count) ? src[i] : 0.0;
    }
  }
}

inline void store4d_masked(double* dst,
                           const double* __restrict lanebuf,
                           int count) {
  if constexpr (kLane == 1) {
    (void)count;
    dst[0] = lanebuf[0];
  } else {
    for (int i = 0; i < kLane; ++i) {
      if (i < count) {
        dst[i] = lanebuf[i];
      }
    }
  }
}

inline void load_vec3(const double* x,
                      const double* y,
                      const double* z,
                      double* __restrict lx,
                      double* __restrict ly,
                      double* __restrict lz,
                      int count) {
  load4d_masked(x, lx, count);
  load4d_masked(y, ly, count);
  load4d_masked(z, lz, count);
}

inline void store_vec3(double* x,
                       double* y,
                       double* z,
                       const double* __restrict lx,
                       const double* __restrict ly,
                       const double* __restrict lz,
                       int count) {
  store4d_masked(x, lx, count);
  store4d_masked(y, ly, count);
  store4d_masked(z, lz, count);
}

}  // namespace soa
