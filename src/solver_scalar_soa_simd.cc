#include "solver_scalar_soa_simd.hpp"

#include "soa_pack.hpp"
#include "solver_scalar_soa.hpp"
#include "types.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

#if defined(ADMC_USE_AVX2)
#include <immintrin.h>
#elif defined(ADMC_USE_NEON)
#include <arm_neon.h>
#endif

namespace {

using math::Vec3;

using Clock = std::chrono::steady_clock;
using DurationMs = std::chrono::duration<double, std::milli>;

double elapsed_ms(const Clock::time_point& begin, const Clock::time_point& end) {
  return DurationMs(end - begin).count();
}

#if defined(ADMC_USE_AVX2)
struct VecD {
  __m256d v;

  VecD() = default;
  explicit VecD(__m256d value) : v(value) {}

  static VecD load(const double* ptr) { return VecD{_mm256_loadu_pd(ptr)}; }

  static VecD load_masked(const double* ptr, int count) {
    alignas(32) double lane[soa::kLane] = {};
    soa::load4d_masked(ptr, lane, count);
    return load(lane);
  }

  static VecD broadcast(double value) { return VecD{_mm256_set1_pd(value)}; }

  void store(double* ptr) const { _mm256_storeu_pd(ptr, v); }
};

inline VecD add(VecD a, VecD b) { return VecD{_mm256_add_pd(a.v, b.v)}; }

inline VecD sub(VecD a, VecD b) { return VecD{_mm256_sub_pd(a.v, b.v)}; }

inline VecD mul(VecD a, VecD b) { return VecD{_mm256_mul_pd(a.v, b.v)}; }

inline VecD max(VecD a, VecD b) { return VecD{_mm256_max_pd(a.v, b.v)}; }

inline VecD min(VecD a, VecD b) { return VecD{_mm256_min_pd(a.v, b.v)}; }

inline VecD sqrt(VecD a) { return VecD{_mm256_sqrt_pd(a.v)}; }

inline VecD negate(VecD a) { return VecD{_mm256_sub_pd(_mm256_setzero_pd(), a.v)}; }
#elif defined(ADMC_USE_NEON)
struct VecD {
  float64x2_t v;

  VecD() = default;
  explicit VecD(float64x2_t value) : v(value) {}

  static VecD load(const double* ptr) {
    return VecD{vld1q_f64(ptr)};
  }

  static VecD load_masked(const double* ptr, int count) {
    double lane[soa::kLane] = {};
    soa::load4d_masked(ptr, lane, count);
    return load(lane);
  }

  static VecD broadcast(double value) { return VecD{vdupq_n_f64(value)}; }

  void store(double* ptr) const { vst1q_f64(ptr, v); }
};

inline VecD add(VecD a, VecD b) { return VecD{vaddq_f64(a.v, b.v)}; }

inline VecD sub(VecD a, VecD b) { return VecD{vsubq_f64(a.v, b.v)}; }

inline VecD mul(VecD a, VecD b) { return VecD{vmulq_f64(a.v, b.v)}; }

inline VecD max(VecD a, VecD b) { return VecD{vmaxq_f64(a.v, b.v)}; }

inline VecD min(VecD a, VecD b) { return VecD{vminq_f64(a.v, b.v)}; }

inline VecD sqrt(VecD a) { return VecD{vsqrtq_f64(a.v)}; }

inline VecD negate(VecD a) { return VecD{vnegq_f64(a.v)}; }
#else
struct VecD {
  double v = 0.0;

  VecD() = default;
  explicit VecD(double value) : v(value) {}

  static VecD load(const double* ptr) { return VecD{*ptr}; }

  static VecD load_masked(const double* ptr, int /*count*/) {
    return load(ptr);
  }

  static VecD broadcast(double value) { return VecD{value}; }

  void store(double* ptr) const { *ptr = v; }
};

inline VecD add(VecD a, VecD b) { return VecD{a.v + b.v}; }

inline VecD sub(VecD a, VecD b) { return VecD{a.v - b.v}; }

inline VecD mul(VecD a, VecD b) { return VecD{a.v * b.v}; }

inline VecD max(VecD a, VecD b) { return VecD{std::max(a.v, b.v)}; }

inline VecD min(VecD a, VecD b) { return VecD{std::min(a.v, b.v)}; }

inline VecD sqrt(VecD a) { return VecD{std::sqrt(a.v)}; }

inline VecD negate(VecD a) { return VecD{-a.v}; }
#endif

}  // namespace

namespace soa_simd {

void update_normal_batch(const double* target,
                         const double* v_rel,
                         const double* k,
                         double* impulses,
                         int count) {
  int processed = 0;
  while (processed < count) {
    const int lanes = std::min(soa::kLane, count - processed);

    alignas(32) double target_lane[soa::kLane] = {};
    alignas(32) double vrel_lane[soa::kLane] = {};
    alignas(32) double inv_lane[soa::kLane] = {};
    alignas(32) double impulse_lane[soa::kLane] = {};

    soa::load4d_masked(target + processed, target_lane, lanes);
    soa::load4d_masked(v_rel + processed, vrel_lane, lanes);
    soa::load4d_masked(impulses + processed, impulse_lane, lanes);

    for (int i = 0; i < lanes; ++i) {
      inv_lane[i] = (k[processed + i] > 0.0) ? (1.0 / k[processed + i]) : 0.0;
    }

    VecD target_v = VecD::load(target_lane);
    VecD vrel_v = VecD::load(vrel_lane);
    VecD inv_v = VecD::load(inv_lane);
    VecD impulse_v = VecD::load(impulse_lane);

    impulse_v = add(impulse_v, mul(sub(target_v, vrel_v), inv_v));

    impulse_v.store(impulse_lane);
    soa::store4d_masked(impulses + processed, impulse_lane, lanes);

    processed += lanes;
  }
}

void update_tangent_batch(const double* target,
                          const double* v_rel,
                          const double* k,
                          double* impulses,
                          int count) {
  update_normal_batch(target, v_rel, k, impulses, count);
}

void apply_impulses_batch(std::vector<RigidBody>& bodies,
                          const RowSOA& rows,
                          const double* delta_jn,
                          const double* delta_jt1,
                          const double* delta_jt2,
                          int start,
                          int count) {
  for (int lane = 0; lane < count; ++lane) {
    const int idx = start + lane;
    if (idx >= rows.N) {
      break;
    }

    const int ia = rows.a[idx];
    const int ib = rows.b[idx];
    if (ia < 0 || ib < 0 || ia >= static_cast<int>(bodies.size()) ||
        ib >= static_cast<int>(bodies.size())) {
      continue;
    }

    RigidBody& A = bodies[ia];
    RigidBody& B = bodies[ib];
    const double dj_n = delta_jn[lane];
    const double dj_t1 = delta_jt1[lane];
    const double dj_t2 = delta_jt2[lane];

    const double impulse_x = rows.nx[idx] * dj_n + rows.t1x[idx] * dj_t1 +
                             rows.t2x[idx] * dj_t2;
    const double impulse_y = rows.ny[idx] * dj_n + rows.t1y[idx] * dj_t1 +
                             rows.t2y[idx] * dj_t2;
    const double impulse_z = rows.nz[idx] * dj_n + rows.t1z[idx] * dj_t1 +
                             rows.t2z[idx] * dj_t2;

    A.v.x -= impulse_x * A.invMass;
    A.v.y -= impulse_y * A.invMass;
    A.v.z -= impulse_z * A.invMass;
    B.v.x += impulse_x * B.invMass;
    B.v.y += impulse_y * B.invMass;
    B.v.z += impulse_z * B.invMass;

    const double dw_ax = dj_n * rows.TWn_a_x[idx] +
                         dj_t1 * rows.TWt1_a_x[idx] +
                         dj_t2 * rows.TWt2_a_x[idx];
    const double dw_ay = dj_n * rows.TWn_a_y[idx] +
                         dj_t1 * rows.TWt1_a_y[idx] +
                         dj_t2 * rows.TWt2_a_y[idx];
    const double dw_az = dj_n * rows.TWn_a_z[idx] +
                         dj_t1 * rows.TWt1_a_z[idx] +
                         dj_t2 * rows.TWt2_a_z[idx];
    const double dw_bx = dj_n * rows.TWn_b_x[idx] +
                         dj_t1 * rows.TWt1_b_x[idx] +
                         dj_t2 * rows.TWt2_b_x[idx];
    const double dw_by = dj_n * rows.TWn_b_y[idx] +
                         dj_t1 * rows.TWt1_b_y[idx] +
                         dj_t2 * rows.TWt2_b_y[idx];
    const double dw_bz = dj_n * rows.TWn_b_z[idx] +
                         dj_t1 * rows.TWt1_b_z[idx] +
                         dj_t2 * rows.TWt2_b_z[idx];

    A.w.x -= dw_ax;
    A.w.y -= dw_ay;
    A.w.z -= dw_az;
    B.w.x += dw_bx;
    B.w.y += dw_by;
    B.w.z += dw_bz;
  }
}

}  // namespace soa_simd

namespace {

struct ContactBatch {
  alignas(32) double nx[soa::kLane] = {};
  alignas(32) double ny[soa::kLane] = {};
  alignas(32) double nz[soa::kLane] = {};
  alignas(32) double t1x[soa::kLane] = {};
  alignas(32) double t1y[soa::kLane] = {};
  alignas(32) double t1z[soa::kLane] = {};
  alignas(32) double t2x[soa::kLane] = {};
  alignas(32) double t2y[soa::kLane] = {};
  alignas(32) double t2z[soa::kLane] = {};
  alignas(32) double raxn_x[soa::kLane] = {};
  alignas(32) double raxn_y[soa::kLane] = {};
  alignas(32) double raxn_z[soa::kLane] = {};
  alignas(32) double rbxn_x[soa::kLane] = {};
  alignas(32) double rbxn_y[soa::kLane] = {};
  alignas(32) double rbxn_z[soa::kLane] = {};
  alignas(32) double raxt1_x[soa::kLane] = {};
  alignas(32) double raxt1_y[soa::kLane] = {};
  alignas(32) double raxt1_z[soa::kLane] = {};
  alignas(32) double rbxt1_x[soa::kLane] = {};
  alignas(32) double rbxt1_y[soa::kLane] = {};
  alignas(32) double rbxt1_z[soa::kLane] = {};
  alignas(32) double raxt2_x[soa::kLane] = {};
  alignas(32) double raxt2_y[soa::kLane] = {};
  alignas(32) double raxt2_z[soa::kLane] = {};
  alignas(32) double rbxt2_x[soa::kLane] = {};
  alignas(32) double rbxt2_y[soa::kLane] = {};
  alignas(32) double rbxt2_z[soa::kLane] = {};
  alignas(32) double TWn_a_x[soa::kLane] = {};
  alignas(32) double TWn_a_y[soa::kLane] = {};
  alignas(32) double TWn_a_z[soa::kLane] = {};
  alignas(32) double TWn_b_x[soa::kLane] = {};
  alignas(32) double TWn_b_y[soa::kLane] = {};
  alignas(32) double TWn_b_z[soa::kLane] = {};
  alignas(32) double TWt1_a_x[soa::kLane] = {};
  alignas(32) double TWt1_a_y[soa::kLane] = {};
  alignas(32) double TWt1_a_z[soa::kLane] = {};
  alignas(32) double TWt1_b_x[soa::kLane] = {};
  alignas(32) double TWt1_b_y[soa::kLane] = {};
  alignas(32) double TWt1_b_z[soa::kLane] = {};
  alignas(32) double TWt2_a_x[soa::kLane] = {};
  alignas(32) double TWt2_a_y[soa::kLane] = {};
  alignas(32) double TWt2_a_z[soa::kLane] = {};
  alignas(32) double TWt2_b_x[soa::kLane] = {};
  alignas(32) double TWt2_b_y[soa::kLane] = {};
  alignas(32) double TWt2_b_z[soa::kLane] = {};
  alignas(32) double inv_k_n[soa::kLane] = {};
  alignas(32) double inv_k_t1[soa::kLane] = {};
  alignas(32) double inv_k_t2[soa::kLane] = {};
  alignas(32) double bias[soa::kLane] = {};
  alignas(32) double bounce[soa::kLane] = {};
  alignas(32) double mu[soa::kLane] = {};
  alignas(32) double jn[soa::kLane] = {};
  alignas(32) double jt1[soa::kLane] = {};
  alignas(32) double jt2[soa::kLane] = {};
  alignas(32) double jn_pre_clamp[soa::kLane] = {};
  alignas(32) double jn_new[soa::kLane] = {};
  alignas(32) double v_rel_t1[soa::kLane] = {};
  alignas(32) double v_rel_t2[soa::kLane] = {};
  alignas(32) double dvx[soa::kLane] = {};
  alignas(32) double dvy[soa::kLane] = {};
  alignas(32) double dvz[soa::kLane] = {};
  alignas(32) double wAx[soa::kLane] = {};
  alignas(32) double wAy[soa::kLane] = {};
  alignas(32) double wAz[soa::kLane] = {};
  alignas(32) double wBx[soa::kLane] = {};
  alignas(32) double wBy[soa::kLane] = {};
  alignas(32) double wBz[soa::kLane] = {};
};

}  // namespace

void solve_scalar_soa_simd(std::vector<RigidBody>& bodies,
                           std::vector<Contact>& contacts,
                           RowSOA& rows,
                           JointSOA& joints,
                           const SoaParams& params,
                           SolverDebugInfo* debug_info) {
  const auto solver_begin = Clock::now();
  const int iterations = std::max(1, params.iterations);

  if (debug_info) {
    debug_info->reset();
  }

  for (RigidBody& body : bodies) {
    body.syncDerived();
  }

  const auto warmstart_begin = Clock::now();
  if (!params.warm_start) {
    const std::size_t contact_count = rows.size();
    std::fill_n(rows.jn.begin(), contact_count, 0.0);
    std::fill_n(rows.jt1.begin(), contact_count, 0.0);
    std::fill_n(rows.jt2.begin(), contact_count, 0.0);
    std::fill_n(joints.j.begin(), joints.size(), 0.0);
  } else {
    for (std::size_t i = 0; i < rows.size(); ++i) {
      const int ia = rows.a[i];
      const int ib = rows.b[i];
      if (ia < 0 || ib < 0 || ia >= static_cast<int>(bodies.size()) ||
          ib >= static_cast<int>(bodies.size())) {
        if (debug_info) {
          ++debug_info->invalid_contact_indices;
        }
        continue;
      }

      RigidBody& A = bodies[ia];
      RigidBody& B = bodies[ib];

      const double jn = rows.jn[i];
      const double jt1 = rows.jt1[i];
      const double jt2 = rows.jt2[i];

      if (std::fabs(jn) <= math::kEps && std::fabs(jt1) <= math::kEps &&
          std::fabs(jt2) <= math::kEps) {
        continue;
      }

      if (debug_info) {
        ++debug_info->warmstart_contact_impulses;
      }

      const double impulse_x = rows.nx[i] * jn + rows.t1x[i] * jt1 +
                               rows.t2x[i] * jt2;
      const double impulse_y = rows.ny[i] * jn + rows.t1y[i] * jt1 +
                               rows.t2y[i] * jt2;
      const double impulse_z = rows.nz[i] * jn + rows.t1z[i] * jt1 +
                               rows.t2z[i] * jt2;

      A.v.x -= impulse_x * A.invMass;
      A.v.y -= impulse_y * A.invMass;
      A.v.z -= impulse_z * A.invMass;
      B.v.x += impulse_x * B.invMass;
      B.v.y += impulse_y * B.invMass;
      B.v.z += impulse_z * B.invMass;

      const double dw_ax = jn * rows.TWn_a_x[i] + jt1 * rows.TWt1_a_x[i] +
                           jt2 * rows.TWt2_a_x[i];
      const double dw_ay = jn * rows.TWn_a_y[i] + jt1 * rows.TWt1_a_y[i] +
                           jt2 * rows.TWt2_a_y[i];
      const double dw_az = jn * rows.TWn_a_z[i] + jt1 * rows.TWt1_a_z[i] +
                           jt2 * rows.TWt2_a_z[i];
      const double dw_bx = jn * rows.TWn_b_x[i] + jt1 * rows.TWt1_b_x[i] +
                           jt2 * rows.TWt2_b_x[i];
      const double dw_by = jn * rows.TWn_b_y[i] + jt1 * rows.TWt1_b_y[i] +
                           jt2 * rows.TWt2_b_y[i];
      const double dw_bz = jn * rows.TWn_b_z[i] + jt1 * rows.TWt1_b_z[i] +
                           jt2 * rows.TWt2_b_z[i];

      A.w.x -= dw_ax;
      A.w.y -= dw_ay;
      A.w.z -= dw_az;
      B.w.x += dw_bx;
      B.w.y += dw_by;
      B.w.z += dw_bz;
    }

    for (std::size_t i = 0; i < joints.size(); ++i) {
      const int ia = joints.a[i];
      const int ib = joints.b[i];
      if (ia < 0 || ib < 0 || ia >= static_cast<int>(bodies.size()) ||
          ib >= static_cast<int>(bodies.size())) {
        if (debug_info) {
          ++debug_info->invalid_joint_indices;
        }
        continue;
      }

      if (std::fabs(joints.j[i]) <= math::kEps) {
        continue;
      }

      if (debug_info) {
        ++debug_info->warmstart_joint_impulses;
      }

      RigidBody& A = bodies[ia];
      RigidBody& B = bodies[ib];
      const Vec3 impulse = joints.d[i] * joints.j[i];
      A.applyImpulse(-impulse, joints.ra[i]);
      B.applyImpulse(impulse, joints.rb[i]);
    }
  }

  if (debug_info) {
    const auto warmstart_end = Clock::now();
    debug_info->timings.solver_warmstart_ms +=
        elapsed_ms(warmstart_begin, warmstart_end);
  }

  ContactBatch batch;

  const auto iteration_begin = Clock::now();
  for (int it = 0; it < iterations; ++it) {
    for (int start = 0; start < rows.N; start += soa::kLane) {
      const int lanes = std::min(soa::kLane, rows.N - start);

      bool lane_valid[soa::kLane] = {};
      RigidBody* bodyA[soa::kLane] = {};
      RigidBody* bodyB[soa::kLane] = {};
      double invMassA[soa::kLane] = {};
      double invMassB[soa::kLane] = {};

      soa::load4d_masked(rows.nx.data() + start, batch.nx, lanes);
      soa::load4d_masked(rows.ny.data() + start, batch.ny, lanes);
      soa::load4d_masked(rows.nz.data() + start, batch.nz, lanes);
      soa::load4d_masked(rows.t1x.data() + start, batch.t1x, lanes);
      soa::load4d_masked(rows.t1y.data() + start, batch.t1y, lanes);
      soa::load4d_masked(rows.t1z.data() + start, batch.t1z, lanes);
      soa::load4d_masked(rows.t2x.data() + start, batch.t2x, lanes);
      soa::load4d_masked(rows.t2y.data() + start, batch.t2y, lanes);
      soa::load4d_masked(rows.t2z.data() + start, batch.t2z, lanes);
      soa::load4d_masked(rows.raxn_x.data() + start, batch.raxn_x, lanes);
      soa::load4d_masked(rows.raxn_y.data() + start, batch.raxn_y, lanes);
      soa::load4d_masked(rows.raxn_z.data() + start, batch.raxn_z, lanes);
      soa::load4d_masked(rows.rbxn_x.data() + start, batch.rbxn_x, lanes);
      soa::load4d_masked(rows.rbxn_y.data() + start, batch.rbxn_y, lanes);
      soa::load4d_masked(rows.rbxn_z.data() + start, batch.rbxn_z, lanes);
      soa::load4d_masked(rows.raxt1_x.data() + start, batch.raxt1_x, lanes);
      soa::load4d_masked(rows.raxt1_y.data() + start, batch.raxt1_y, lanes);
      soa::load4d_masked(rows.raxt1_z.data() + start, batch.raxt1_z, lanes);
      soa::load4d_masked(rows.rbxt1_x.data() + start, batch.rbxt1_x, lanes);
      soa::load4d_masked(rows.rbxt1_y.data() + start, batch.rbxt1_y, lanes);
      soa::load4d_masked(rows.rbxt1_z.data() + start, batch.rbxt1_z, lanes);
      soa::load4d_masked(rows.raxt2_x.data() + start, batch.raxt2_x, lanes);
      soa::load4d_masked(rows.raxt2_y.data() + start, batch.raxt2_y, lanes);
      soa::load4d_masked(rows.raxt2_z.data() + start, batch.raxt2_z, lanes);
      soa::load4d_masked(rows.rbxt2_x.data() + start, batch.rbxt2_x, lanes);
      soa::load4d_masked(rows.rbxt2_y.data() + start, batch.rbxt2_y, lanes);
      soa::load4d_masked(rows.rbxt2_z.data() + start, batch.rbxt2_z, lanes);
      soa::load4d_masked(rows.TWn_a_x.data() + start, batch.TWn_a_x, lanes);
      soa::load4d_masked(rows.TWn_a_y.data() + start, batch.TWn_a_y, lanes);
      soa::load4d_masked(rows.TWn_a_z.data() + start, batch.TWn_a_z, lanes);
      soa::load4d_masked(rows.TWn_b_x.data() + start, batch.TWn_b_x, lanes);
      soa::load4d_masked(rows.TWn_b_y.data() + start, batch.TWn_b_y, lanes);
      soa::load4d_masked(rows.TWn_b_z.data() + start, batch.TWn_b_z, lanes);
      soa::load4d_masked(rows.TWt1_a_x.data() + start, batch.TWt1_a_x, lanes);
      soa::load4d_masked(rows.TWt1_a_y.data() + start, batch.TWt1_a_y, lanes);
      soa::load4d_masked(rows.TWt1_a_z.data() + start, batch.TWt1_a_z, lanes);
      soa::load4d_masked(rows.TWt1_b_x.data() + start, batch.TWt1_b_x, lanes);
      soa::load4d_masked(rows.TWt1_b_y.data() + start, batch.TWt1_b_y, lanes);
      soa::load4d_masked(rows.TWt1_b_z.data() + start, batch.TWt1_b_z, lanes);
      soa::load4d_masked(rows.TWt2_a_x.data() + start, batch.TWt2_a_x, lanes);
      soa::load4d_masked(rows.TWt2_a_y.data() + start, batch.TWt2_a_y, lanes);
      soa::load4d_masked(rows.TWt2_a_z.data() + start, batch.TWt2_a_z, lanes);
      soa::load4d_masked(rows.TWt2_b_x.data() + start, batch.TWt2_b_x, lanes);
      soa::load4d_masked(rows.TWt2_b_y.data() + start, batch.TWt2_b_y, lanes);
      soa::load4d_masked(rows.TWt2_b_z.data() + start, batch.TWt2_b_z, lanes);
      soa::load4d_masked(rows.inv_k_n.data() + start, batch.inv_k_n, lanes);
      soa::load4d_masked(rows.inv_k_t1.data() + start, batch.inv_k_t1, lanes);
      soa::load4d_masked(rows.inv_k_t2.data() + start, batch.inv_k_t2, lanes);
      soa::load4d_masked(rows.bias.data() + start, batch.bias, lanes);
      soa::load4d_masked(rows.bounce.data() + start, batch.bounce, lanes);
      soa::load4d_masked(rows.mu.data() + start, batch.mu, lanes);
      soa::load4d_masked(rows.jn.data() + start, batch.jn, lanes);
      soa::load4d_masked(rows.jt1.data() + start, batch.jt1, lanes);
      soa::load4d_masked(rows.jt2.data() + start, batch.jt2, lanes);

      for (int lane = 0; lane < lanes; ++lane) {
        const int idx = start + lane;
        const int ia = rows.a[idx];
        const int ib = rows.b[idx];

        if (ia < 0 || ib < 0 || ia >= static_cast<int>(bodies.size()) ||
            ib >= static_cast<int>(bodies.size())) {
          if (debug_info) {
            ++debug_info->invalid_contact_indices;
          }
          batch.mu[lane] = 0.0;
          batch.inv_k_n[lane] = 0.0;
          batch.inv_k_t1[lane] = 0.0;
          batch.inv_k_t2[lane] = 0.0;
          batch.bias[lane] = 0.0;
          batch.bounce[lane] = 0.0;
          lane_valid[lane] = false;
          continue;
        }

        lane_valid[lane] = true;
        bodyA[lane] = &bodies[ia];
        bodyB[lane] = &bodies[ib];
        invMassA[lane] = bodies[ia].invMass;
        invMassB[lane] = bodies[ib].invMass;

        batch.dvx[lane] = bodies[ib].v.x - bodies[ia].v.x;
        batch.dvy[lane] = bodies[ib].v.y - bodies[ia].v.y;
        batch.dvz[lane] = bodies[ib].v.z - bodies[ia].v.z;

        batch.wAx[lane] = bodies[ia].w.x;
        batch.wAy[lane] = bodies[ia].w.y;
        batch.wAz[lane] = bodies[ia].w.z;
        batch.wBx[lane] = bodies[ib].w.x;
        batch.wBy[lane] = bodies[ib].w.y;
        batch.wBz[lane] = bodies[ib].w.z;
      }

      VecD dvx_v = VecD::load(batch.dvx);
      VecD dvy_v = VecD::load(batch.dvy);
      VecD dvz_v = VecD::load(batch.dvz);
      VecD nx_v = VecD::load(batch.nx);
      VecD ny_v = VecD::load(batch.ny);
      VecD nz_v = VecD::load(batch.nz);
      VecD wAx_v = VecD::load(batch.wAx);
      VecD wAy_v = VecD::load(batch.wAy);
      VecD wAz_v = VecD::load(batch.wAz);
      VecD wBx_v = VecD::load(batch.wBx);
      VecD wBy_v = VecD::load(batch.wBy);
      VecD wBz_v = VecD::load(batch.wBz);
      VecD raxn_x_v = VecD::load(batch.raxn_x);
      VecD raxn_y_v = VecD::load(batch.raxn_y);
      VecD raxn_z_v = VecD::load(batch.raxn_z);
      VecD rbxn_x_v = VecD::load(batch.rbxn_x);
      VecD rbxn_y_v = VecD::load(batch.rbxn_y);
      VecD rbxn_z_v = VecD::load(batch.rbxn_z);

      VecD wA_dot_raxn = add(mul(wAx_v, raxn_x_v),
                              add(mul(wAy_v, raxn_y_v), mul(wAz_v, raxn_z_v)));
      VecD wB_dot_rbxn = add(mul(wBx_v, rbxn_x_v),
                              add(mul(wBy_v, rbxn_y_v), mul(wBz_v, rbxn_z_v)));

      VecD v_rel_n = add(add(mul(nx_v, dvx_v), mul(ny_v, dvy_v)),
                         mul(nz_v, dvz_v));
      v_rel_n = add(v_rel_n, sub(wB_dot_rbxn, wA_dot_raxn));

      VecD bias_v = VecD::load(batch.bias);
      VecD bounce_v = VecD::load(batch.bounce);
      VecD inv_kn_v = VecD::load(batch.inv_k_n);
      VecD jn_old_v = VecD::load(batch.jn);

      VecD rhs = add(negate(add(v_rel_n, bias_v)), bounce_v);
      VecD jn_candidate_v = add(jn_old_v, mul(rhs, inv_kn_v));
      VecD zero_v = VecD::broadcast(0.0);
      VecD jn_clamped_v = max(jn_candidate_v, zero_v);

      jn_candidate_v.store(batch.jn_pre_clamp);
      jn_clamped_v.store(batch.jn_new);

      for (int lane = 0; lane < lanes; ++lane) {
        const int idx = start + lane;
        if (!lane_valid[lane]) {
          rows.jn[idx] = 0.0;
          rows.jt1[idx] = 0.0;
          rows.jt2[idx] = 0.0;
          continue;
        }

        const double jn_pre = batch.jn_pre_clamp[lane];
        double jn_new = batch.jn_new[lane];
        if (jn_pre < 0.0) {
          if (debug_info) {
            ++debug_info->normal_impulse_clamps;
          }
          jn_new = 0.0;
        }

        const double applied = jn_new - batch.jn[lane];
        rows.jn[idx] = jn_new;
        batch.jn[lane] = jn_new;

        if (std::fabs(applied) > math::kEps) {
          const double impulse_x = applied * batch.nx[lane];
          const double impulse_y = applied * batch.ny[lane];
          const double impulse_z = applied * batch.nz[lane];
          RigidBody& A = *bodyA[lane];
          RigidBody& B = *bodyB[lane];

          A.v.x -= impulse_x * invMassA[lane];
          A.v.y -= impulse_y * invMassA[lane];
          A.v.z -= impulse_z * invMassA[lane];
          B.v.x += impulse_x * invMassB[lane];
          B.v.y += impulse_y * invMassB[lane];
          B.v.z += impulse_z * invMassB[lane];

          A.w.x -= applied * batch.TWn_a_x[lane];
          A.w.y -= applied * batch.TWn_a_y[lane];
          A.w.z -= applied * batch.TWn_a_z[lane];
          B.w.x += applied * batch.TWn_b_x[lane];
          B.w.y += applied * batch.TWn_b_y[lane];
          B.w.z += applied * batch.TWn_b_z[lane];

          batch.dvx[lane] = B.v.x - A.v.x;
          batch.dvy[lane] = B.v.y - A.v.y;
          batch.dvz[lane] = B.v.z - A.v.z;
          batch.wAx[lane] = A.w.x;
          batch.wAy[lane] = A.w.y;
          batch.wAz[lane] = A.w.z;
          batch.wBx[lane] = B.w.x;
          batch.wBy[lane] = B.w.y;
          batch.wBz[lane] = B.w.z;
        } else {
          batch.dvx[lane] = bodyB[lane]->v.x - bodyA[lane]->v.x;
          batch.dvy[lane] = bodyB[lane]->v.y - bodyA[lane]->v.y;
          batch.dvz[lane] = bodyB[lane]->v.z - bodyA[lane]->v.z;
          batch.wAx[lane] = bodyA[lane]->w.x;
          batch.wAy[lane] = bodyA[lane]->w.y;
          batch.wAz[lane] = bodyA[lane]->w.z;
          batch.wBx[lane] = bodyB[lane]->w.x;
          batch.wBy[lane] = bodyB[lane]->w.y;
          batch.wBz[lane] = bodyB[lane]->w.z;
        }
      }

      dvx_v = VecD::load(batch.dvx);
      dvy_v = VecD::load(batch.dvy);
      dvz_v = VecD::load(batch.dvz);
      wAx_v = VecD::load(batch.wAx);
      wAy_v = VecD::load(batch.wAy);
      wAz_v = VecD::load(batch.wAz);
      wBx_v = VecD::load(batch.wBx);
      wBy_v = VecD::load(batch.wBy);
      wBz_v = VecD::load(batch.wBz);

      VecD raxt1_x_v = VecD::load(batch.raxt1_x);
      VecD raxt1_y_v = VecD::load(batch.raxt1_y);
      VecD raxt1_z_v = VecD::load(batch.raxt1_z);
      VecD rbxt1_x_v = VecD::load(batch.rbxt1_x);
      VecD rbxt1_y_v = VecD::load(batch.rbxt1_y);
      VecD rbxt1_z_v = VecD::load(batch.rbxt1_z);
      VecD raxt2_x_v = VecD::load(batch.raxt2_x);
      VecD raxt2_y_v = VecD::load(batch.raxt2_y);
      VecD raxt2_z_v = VecD::load(batch.raxt2_z);
      VecD rbxt2_x_v = VecD::load(batch.rbxt2_x);
      VecD rbxt2_y_v = VecD::load(batch.rbxt2_y);
      VecD rbxt2_z_v = VecD::load(batch.rbxt2_z);
      VecD t1x_v = VecD::load(batch.t1x);
      VecD t1y_v = VecD::load(batch.t1y);
      VecD t1z_v = VecD::load(batch.t1z);
      VecD t2x_v = VecD::load(batch.t2x);
      VecD t2y_v = VecD::load(batch.t2y);
      VecD t2z_v = VecD::load(batch.t2z);

      VecD wA_dot_raxt1 = add(mul(wAx_v, raxt1_x_v),
                               add(mul(wAy_v, raxt1_y_v),
                                   mul(wAz_v, raxt1_z_v)));
      VecD wB_dot_rbxt1 = add(mul(wBx_v, rbxt1_x_v),
                               add(mul(wBy_v, rbxt1_y_v),
                                   mul(wBz_v, rbxt1_z_v)));
      VecD wA_dot_raxt2 = add(mul(wAx_v, raxt2_x_v),
                               add(mul(wAy_v, raxt2_y_v),
                                   mul(wAz_v, raxt2_z_v)));
      VecD wB_dot_rbxt2 = add(mul(wBx_v, rbxt2_x_v),
                               add(mul(wBy_v, rbxt2_y_v),
                                   mul(wBz_v, rbxt2_z_v)));

      VecD v_rel_t1_v = add(add(mul(t1x_v, dvx_v), mul(t1y_v, dvy_v)),
                            mul(t1z_v, dvz_v));
      v_rel_t1_v = add(v_rel_t1_v, sub(wB_dot_rbxt1, wA_dot_raxt1));

      VecD v_rel_t2_v = add(add(mul(t2x_v, dvx_v), mul(t2y_v, dvy_v)),
                            mul(t2z_v, dvz_v));
      v_rel_t2_v = add(v_rel_t2_v, sub(wB_dot_rbxt2, wA_dot_raxt2));

      v_rel_t1_v.store(batch.v_rel_t1);
      v_rel_t2_v.store(batch.v_rel_t2);

      for (int lane = 0; lane < lanes; ++lane) {
        const int idx = start + lane;
        if (!lane_valid[lane]) {
          rows.jt1[idx] = 0.0;
          rows.jt2[idx] = 0.0;
          batch.jt1[lane] = 0.0;
          batch.jt2[lane] = 0.0;
          continue;
        }

        const double mu = batch.mu[lane];
        if (mu <= math::kEps) {
          rows.jt1[idx] = 0.0;
          rows.jt2[idx] = 0.0;
          batch.jt1[lane] = 0.0;
          batch.jt2[lane] = 0.0;
          continue;
        }

        const double jt1_old = batch.jt1[lane];
        const double jt2_old = batch.jt2[lane];
        double jt1_candidate =
            jt1_old + (-batch.v_rel_t1[lane]) * batch.inv_k_t1[lane];
        double jt2_candidate =
            jt2_old + (-batch.v_rel_t2[lane]) * batch.inv_k_t2[lane];

        const double friction_max =
            mu * std::max(batch.jn_new[lane], 0.0);
        const double jt_mag_sq =
            jt1_candidate * jt1_candidate + jt2_candidate * jt2_candidate;
        const double friction_max_sq = friction_max * friction_max;
        double scale = 1.0;
        if (jt_mag_sq > friction_max_sq &&
            jt_mag_sq > math::kEps * math::kEps) {
          const double jt_mag = std::sqrt(jt_mag_sq);
          scale = (friction_max > 0.0) ? (friction_max / jt_mag) : 0.0;
          if (debug_info) {
            ++debug_info->tangent_projections;
          }
        }

        jt1_candidate *= scale;
        jt2_candidate *= scale;

        const double delta_jt1 = jt1_candidate - jt1_old;
        const double delta_jt2 = jt2_candidate - jt2_old;
        rows.jt1[idx] = jt1_candidate;
        rows.jt2[idx] = jt2_candidate;
        batch.jt1[lane] = jt1_candidate;
        batch.jt2[lane] = jt2_candidate;

        if (std::fabs(delta_jt1) > math::kEps ||
            std::fabs(delta_jt2) > math::kEps) {
          RigidBody& A = *bodyA[lane];
          RigidBody& B = *bodyB[lane];

          const double impulse_x = delta_jt1 * batch.t1x[lane] +
                                   delta_jt2 * batch.t2x[lane];
          const double impulse_y = delta_jt1 * batch.t1y[lane] +
                                   delta_jt2 * batch.t2y[lane];
          const double impulse_z = delta_jt1 * batch.t1z[lane] +
                                   delta_jt2 * batch.t2z[lane];

          A.v.x -= impulse_x * invMassA[lane];
          A.v.y -= impulse_y * invMassA[lane];
          A.v.z -= impulse_z * invMassA[lane];
          B.v.x += impulse_x * invMassB[lane];
          B.v.y += impulse_y * invMassB[lane];
          B.v.z += impulse_z * invMassB[lane];

          A.w.x -= delta_jt1 * batch.TWt1_a_x[lane] +
                   delta_jt2 * batch.TWt2_a_x[lane];
          A.w.y -= delta_jt1 * batch.TWt1_a_y[lane] +
                   delta_jt2 * batch.TWt2_a_y[lane];
          A.w.z -= delta_jt1 * batch.TWt1_a_z[lane] +
                   delta_jt2 * batch.TWt2_a_z[lane];
          B.w.x += delta_jt1 * batch.TWt1_b_x[lane] +
                   delta_jt2 * batch.TWt2_b_x[lane];
          B.w.y += delta_jt1 * batch.TWt1_b_y[lane] +
                   delta_jt2 * batch.TWt2_b_y[lane];
          B.w.z += delta_jt1 * batch.TWt1_b_z[lane] +
                   delta_jt2 * batch.TWt2_b_z[lane];
        }
      }
    }

    for (std::size_t i = 0; i < joints.size(); ++i) {
      const int ia = joints.a[i];
      const int ib = joints.b[i];
      if (ia < 0 || ib < 0 || ia >= static_cast<int>(bodies.size()) ||
          ib >= static_cast<int>(bodies.size())) {
        if (debug_info) {
          ++debug_info->invalid_joint_indices;
        }
        continue;
      }

      const double denom = joints.k[i] + joints.gamma[i];
      if (denom <= math::kEps) {
        if (debug_info) {
          ++debug_info->singular_joint_denominators;
        }
        continue;
      }

      RigidBody& A = bodies[ia];
      RigidBody& B = bodies[ib];
      const Vec3 va = A.v + math::cross(A.w, joints.ra[i]);
      const Vec3 vb = B.v + math::cross(B.w, joints.rb[i]);
      const double v_rel_d = math::dot(joints.d[i], vb - va);

      double j_new = joints.j[i] - (v_rel_d + joints.bias[i]) / denom;
      if (joints.rope[i] && j_new < 0.0) {
        if (debug_info) {
          ++debug_info->rope_clamps;
        }
        j_new = 0.0;
      }

      const double applied = j_new - joints.j[i];
      joints.j[i] = j_new;

      if (std::fabs(applied) > math::kEps) {
        const Vec3 impulse = applied * joints.d[i];
        A.applyImpulse(-impulse, joints.ra[i]);
        B.applyImpulse(impulse, joints.rb[i]);
      }
    }
  }

  if (debug_info) {
    const auto iteration_end = Clock::now();
    debug_info->timings.solver_iterations_ms +=
        elapsed_ms(iteration_begin, iteration_end);
  }

  for (std::size_t i = 0; i < rows.size(); ++i) {
    const int idx = rows.indices[i];
    if (idx < 0 || idx >= static_cast<int>(contacts.size())) {
      if (debug_info) {
        ++debug_info->invalid_contact_indices;
      }
      continue;
    }
    Contact& c = contacts[static_cast<std::size_t>(idx)];
    c.jn = rows.jn[i];
    c.jt1 = rows.jt1[i];
    c.jt2 = rows.jt2[i];
    c.mu = rows.mu[i];
    c.e = rows.e[i];
    c.bias = rows.bias[i];
    c.bounce = rows.bounce[i];
    c.C = rows.C[i];
  }

  const auto integrate_begin = Clock::now();
  for (RigidBody& body : bodies) {
    body.integrate(params.dt);
  }
  if (debug_info) {
    const auto integrate_end = Clock::now();
    debug_info->timings.solver_integrate_ms +=
        elapsed_ms(integrate_begin, integrate_end);
    debug_info->timings.solver_total_ms +=
        elapsed_ms(solver_begin, integrate_end);
  }
}

void solve_scalar_soa_simd(std::vector<RigidBody>& bodies,
                           std::vector<Contact>& contacts,
                           RowSOA& rows,
                           const SoaParams& params,
                           SolverDebugInfo* debug_info) {
  static JointSOA empty_joints;
  empty_joints.clear();
  solve_scalar_soa_simd(bodies, contacts, rows, empty_joints, params,
                        debug_info);
}

