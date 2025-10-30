#include "solver_vec_lane.hpp"

#if defined(ADMC_ENABLE_AVX2) && !defined(__ARM_NEON)
#include <immintrin.h>

namespace admc {
namespace {
inline __m256 load8(const float* p) { return _mm256_load_ps(p); }
inline void store8(float* p, __m256 v) { _mm256_store_ps(p, v); }
}

void solve_normal_lane_avx2(TileSpan t) {
  const int N = t.count;
  if (N <= 0) {
    return;
  }
  const __m256 zero = _mm256_set1_ps(0.0f);
  for (int i = 0; i < N; i += 8) {
    __m256 nx = load8(t.n_x + i);
    __m256 ny = load8(t.n_y + i);
    __m256 nz = load8(t.n_z + i);

    __m256 vAx = load8(t.vAx + i);
    __m256 vAy = load8(t.vAy + i);
    __m256 vAz = load8(t.vAz + i);
    __m256 vBx = load8(t.vBx + i);
    __m256 vBy = load8(t.vBy + i);
    __m256 vBz = load8(t.vBz + i);

    __m256 rvx = _mm256_sub_ps(vBx, vAx);
    __m256 rvy = _mm256_sub_ps(vBy, vAy);
    __m256 rvz = _mm256_sub_ps(vBz, vAz);
    __m256 vrel_n = _mm256_fmadd_ps(rvx, nx,
                                    _mm256_fmadd_ps(rvy, ny, _mm256_mul_ps(rvz, nz)));

    __m256 k = load8(t.k_n + i);
    __m256 tgt = load8(t.target_n + i);
    __m256 jn0 = load8(t.jn + i);

    __m256 num = _mm256_sub_ps(tgt, vrel_n);
    __m256 dj = _mm256_div_ps(num, k);

    __m256 jn1 = _mm256_add_ps(jn0, dj);
    __m256 jn_new = _mm256_max_ps(jn1, zero);
    dj = _mm256_sub_ps(jn_new, jn0);

    __m256 impx = _mm256_mul_ps(dj, nx);
    __m256 impy = _mm256_mul_ps(dj, ny);
    __m256 impz = _mm256_mul_ps(dj, nz);

    __m256 invMA = load8(t.invMassA + i);
    __m256 invMB = load8(t.invMassB + i);

    __m256 dVAx = _mm256_mul_ps(invMA, impx);
    __m256 dVAy = _mm256_mul_ps(invMA, impy);
    __m256 dVAz = _mm256_mul_ps(invMA, impz);
    __m256 dVBx = _mm256_mul_ps(invMB, impx);
    __m256 dVBy = _mm256_mul_ps(invMB, impy);
    __m256 dVBz = _mm256_mul_ps(invMB, impz);

    vAx = _mm256_sub_ps(vAx, dVAx);
    vAy = _mm256_sub_ps(vAy, dVAy);
    vAz = _mm256_sub_ps(vAz, dVAz);
    vBx = _mm256_add_ps(vBx, dVBx);
    vBy = _mm256_add_ps(vBy, dVBy);
    vBz = _mm256_add_ps(vBz, dVBz);

    store8(t.vAx + i, vAx);
    store8(t.vAy + i, vAy);
    store8(t.vAz + i, vAz);
    store8(t.vBx + i, vBx);
    store8(t.vBy + i, vBy);
    store8(t.vBz + i, vBz);
    store8(t.jn + i, jn_new);
  }
}

}  // namespace admc
#else
namespace admc {
void solve_normal_lane_avx2(TileSpan) {}
}  // namespace admc
#endif

#if defined(ADMC_ENABLE_NEON) && defined(__ARM_NEON)
#include <arm_neon.h>

namespace admc {
namespace {
inline float32x4_t load4(const float* p) { return vld1q_f32(p); }
inline void store4(float* p, float32x4_t v) { vst1q_f32(p, v); }
inline float32x4_t div_ps(float32x4_t num, float32x4_t den) {
  float32x4_t recip = vrecpeq_f32(den);
  recip = vmulq_f32(vrecpsq_f32(den, recip), recip);
  recip = vmulq_f32(vrecpsq_f32(den, recip), recip);
  return vmulq_f32(num, recip);
}

inline void solve_chunk4(const TileSpan& t, int offset) {
  float32x4_t nx = load4(t.n_x + offset);
  float32x4_t ny = load4(t.n_y + offset);
  float32x4_t nz = load4(t.n_z + offset);

  float32x4_t vAx = load4(t.vAx + offset);
  float32x4_t vAy = load4(t.vAy + offset);
  float32x4_t vAz = load4(t.vAz + offset);
  float32x4_t vBx = load4(t.vBx + offset);
  float32x4_t vBy = load4(t.vBy + offset);
  float32x4_t vBz = load4(t.vBz + offset);

  float32x4_t rvx = vsubq_f32(vBx, vAx);
  float32x4_t rvy = vsubq_f32(vBy, vAy);
  float32x4_t rvz = vsubq_f32(vBz, vAz);
  float32x4_t vrel_n = vmulq_f32(rvx, nx);
  vrel_n = vmlaq_f32(vrel_n, rvy, ny);
  vrel_n = vmlaq_f32(vrel_n, rvz, nz);

  float32x4_t k = load4(t.k_n + offset);
  float32x4_t tgt = load4(t.target_n + offset);
  float32x4_t jn0 = load4(t.jn + offset);

  float32x4_t num = vsubq_f32(tgt, vrel_n);
  float32x4_t dj = div_ps(num, k);

  float32x4_t jn1 = vaddq_f32(jn0, dj);
  float32x4_t jn_new = vmaxq_f32(jn1, vdupq_n_f32(0.0f));
  dj = vsubq_f32(jn_new, jn0);

  float32x4_t impx = vmulq_f32(dj, nx);
  float32x4_t impy = vmulq_f32(dj, ny);
  float32x4_t impz = vmulq_f32(dj, nz);

  float32x4_t invMA = load4(t.invMassA + offset);
  float32x4_t invMB = load4(t.invMassB + offset);

  float32x4_t dVAx = vmulq_f32(invMA, impx);
  float32x4_t dVAy = vmulq_f32(invMA, impy);
  float32x4_t dVAz = vmulq_f32(invMA, impz);
  float32x4_t dVBx = vmulq_f32(invMB, impx);
  float32x4_t dVBy = vmulq_f32(invMB, impy);
  float32x4_t dVBz = vmulq_f32(invMB, impz);

  vAx = vsubq_f32(vAx, dVAx);
  vAy = vsubq_f32(vAy, dVAy);
  vAz = vsubq_f32(vAz, dVAz);
  vBx = vaddq_f32(vBx, dVBx);
  vBy = vaddq_f32(vBy, dVBy);
  vBz = vaddq_f32(vBz, dVBz);

  store4(t.vAx + offset, vAx);
  store4(t.vAy + offset, vAy);
  store4(t.vAz + offset, vAz);
  store4(t.vBx + offset, vBx);
  store4(t.vBy + offset, vBy);
  store4(t.vBz + offset, vBz);
  store4(t.jn + offset, jn_new);
}
}  // namespace

void solve_normal_lane_neon(TileSpan t) {
  const int N = t.count;
  if (N <= 0) {
    return;
  }
  for (int i = 0; i < N; i += 8) {
    solve_chunk4(t, i);
    solve_chunk4(t, i + 4);
  }
}

}  // namespace admc
#else
namespace admc {
void solve_normal_lane_neon(TileSpan) {}
}  // namespace admc
#endif

