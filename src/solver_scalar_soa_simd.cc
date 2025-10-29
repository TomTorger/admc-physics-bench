#include "solver_scalar_soa_simd.hpp"

#include "soa_pack.hpp"
#include "solver_scalar_soa.hpp"
#include "types.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

#if defined(ADMC_USE_AVX2)
#include <immintrin.h>
#elif defined(ADMC_USE_NEON)
#include <arm_neon.h>
#endif

namespace {

using math::Vec3;

using Clock = std::chrono::steady_clock;
using DurationMs = std::chrono::duration<double, std::milli>;

constexpr double kStaticFrictionSpeedThreshold = 0.1;

double elapsed_ms(const Clock::time_point& begin, const Clock::time_point& end) {
  return DurationMs(end - begin).count();
}

inline void apply_impulse_to_body(const RigidBody& body,
                                  SolverBodySoA& state,
                                  int body_index,
                                  const Vec3& impulse,
                                  const Vec3& r) {
  const int slot = state.slot_for_body(body_index);
  if (slot < 0) {
    return;
  }

  state.vx[static_cast<std::size_t>(slot)] += impulse.x * body.invMass;
  state.vy[static_cast<std::size_t>(slot)] += impulse.y * body.invMass;
  state.vz[static_cast<std::size_t>(slot)] += impulse.z * body.invMass;

  const Vec3 torque = math::cross(r, impulse);
  const Vec3 delta_w = body.invInertiaWorld * torque;
  state.wx[static_cast<std::size_t>(slot)] += delta_w.x;
  state.wy[static_cast<std::size_t>(slot)] += delta_w.y;
  state.wz[static_cast<std::size_t>(slot)] += delta_w.z;
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
  int lanes = 0;
  int start = 0;
  int bodyA_index[soa::kLane] = {};
  int bodyB_index[soa::kLane] = {};
  int bodyA_slot[soa::kLane] = {};
  int bodyB_slot[soa::kLane] = {};
  double invMassA[soa::kLane] = {};
  double invMassB[soa::kLane] = {};
  bool lane_valid[soa::kLane] = {};
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

  SolverBodySoA body_state;
  body_state.initialize(bodies, rows, joints);
  body_state.load_from(bodies);

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

      const RigidBody& A = bodies[ia];
      const RigidBody& B = bodies[ib];

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

      const int slot_a = body_state.slot_for_body(ia);
      const int slot_b = body_state.slot_for_body(ib);
      if (slot_a < 0 || slot_b < 0) {
        continue;
      }

      body_state.vx[static_cast<std::size_t>(slot_a)] -= impulse_x * A.invMass;
      body_state.vy[static_cast<std::size_t>(slot_a)] -= impulse_y * A.invMass;
      body_state.vz[static_cast<std::size_t>(slot_a)] -= impulse_z * A.invMass;
      body_state.vx[static_cast<std::size_t>(slot_b)] += impulse_x * B.invMass;
      body_state.vy[static_cast<std::size_t>(slot_b)] += impulse_y * B.invMass;
      body_state.vz[static_cast<std::size_t>(slot_b)] += impulse_z * B.invMass;

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

      body_state.wx[static_cast<std::size_t>(slot_a)] -= dw_ax;
      body_state.wy[static_cast<std::size_t>(slot_a)] -= dw_ay;
      body_state.wz[static_cast<std::size_t>(slot_a)] -= dw_az;
      body_state.wx[static_cast<std::size_t>(slot_b)] += dw_bx;
      body_state.wy[static_cast<std::size_t>(slot_b)] += dw_by;
      body_state.wz[static_cast<std::size_t>(slot_b)] += dw_bz;
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

      const RigidBody& A = bodies[ia];
      const RigidBody& B = bodies[ib];
      const Vec3 impulse = joints.d[i] * joints.j[i];
      apply_impulse_to_body(A, body_state, ia, -impulse, joints.ra[i]);
      apply_impulse_to_body(B, body_state, ib, impulse, joints.rb[i]);
    }
  }

  if (debug_info) {
    const auto warmstart_end = Clock::now();
    debug_info->timings.solver_warmstart_ms +=
        elapsed_ms(warmstart_begin, warmstart_end);
  }

  std::vector<ContactBatch> contact_batches;
  if (rows.N > 0) {
    const int num_batches = (rows.N + soa::kLane - 1) / soa::kLane;
    contact_batches.reserve(static_cast<std::size_t>(num_batches));
    for (int start = 0; start < rows.N; start += soa::kLane) {
      ContactBatch batch;
      const int lanes = std::min(soa::kLane, rows.N - start);
      batch.lanes = lanes;
      batch.start = start;

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

        batch.bodyA_index[lane] = ia;
        batch.bodyB_index[lane] = ib;
        const int slot_a = body_state.slot_for_body(ia);
        const int slot_b = body_state.slot_for_body(ib);
        batch.bodyA_slot[lane] = slot_a;
        batch.bodyB_slot[lane] = slot_b;
        if (ia < 0 || ib < 0 || ia >= static_cast<int>(bodies.size()) ||
            ib >= static_cast<int>(bodies.size()) || slot_a < 0 ||
            slot_b < 0) {
          if (debug_info) {
            ++debug_info->invalid_contact_indices;
          }
          batch.mu[lane] = 0.0;
          batch.inv_k_n[lane] = 0.0;
          batch.inv_k_t1[lane] = 0.0;
          batch.inv_k_t2[lane] = 0.0;
          batch.bias[lane] = 0.0;
          batch.bounce[lane] = 0.0;
          batch.lane_valid[lane] = false;
          continue;
        }

        batch.lane_valid[lane] = true;
        batch.invMassA[lane] = bodies[ia].invMass;
        batch.invMassB[lane] = bodies[ib].invMass;
      }

      contact_batches.push_back(batch);
    }
  }

  auto solve_joint_iteration_scalar = [&](std::size_t i) {
    const int ia = joints.a[i];
    const int ib = joints.b[i];
    if (ia < 0 || ib < 0 || ia >= static_cast<int>(bodies.size()) ||
        ib >= static_cast<int>(bodies.size())) {
      if (debug_info) {
        ++debug_info->invalid_joint_indices;
      }
      return;
    }

    const bool active = (params.beta > math::kEps) ||
                        (joints.beta[i] > math::kEps) ||
                        (joints.gamma[i] > math::kEps);
    if (!active) {
      return;
    }

    const double denom = joints.k[i] + joints.gamma[i];
    if (denom <= math::kEps) {
      if (debug_info) {
        ++debug_info->singular_joint_denominators;
      }
      return;
    }

    const int slot_a = body_state.slot_for_body(ia);
    const int slot_b = body_state.slot_for_body(ib);
    if (slot_a < 0 || slot_b < 0) {
      if (debug_info) {
        ++debug_info->invalid_joint_indices;
      }
      return;
    }

    const RigidBody& A = bodies[ia];
    const RigidBody& B = bodies[ib];
    const Vec3 va(body_state.vx[static_cast<std::size_t>(slot_a)],
                  body_state.vy[static_cast<std::size_t>(slot_a)],
                  body_state.vz[static_cast<std::size_t>(slot_a)]);
    const Vec3 wa(body_state.wx[static_cast<std::size_t>(slot_a)],
                  body_state.wy[static_cast<std::size_t>(slot_a)],
                  body_state.wz[static_cast<std::size_t>(slot_a)]);
    const Vec3 vb(body_state.vx[static_cast<std::size_t>(slot_b)],
                  body_state.vy[static_cast<std::size_t>(slot_b)],
                  body_state.vz[static_cast<std::size_t>(slot_b)]);
    const Vec3 wb(body_state.wx[static_cast<std::size_t>(slot_b)],
                  body_state.wy[static_cast<std::size_t>(slot_b)],
                  body_state.wz[static_cast<std::size_t>(slot_b)]);
    const Vec3 va_world = va + math::cross(wa, joints.ra[i]);
    const Vec3 vb_world = vb + math::cross(wb, joints.rb[i]);
    const double v_rel_d = math::dot(joints.d[i], vb_world - va_world);

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
      apply_impulse_to_body(A, body_state, ia, -impulse, joints.ra[i]);
      apply_impulse_to_body(B, body_state, ib, impulse, joints.rb[i]);
    }
  };

  const auto iteration_begin = Clock::now();
  for (int it = 0; it < iterations; ++it) {
    for (ContactBatch& batch : contact_batches) {
      const int lanes = batch.lanes;
      if (lanes <= 0) {
        continue;
      }

      for (int lane = 0; lane < lanes; ++lane) {
        if (!batch.lane_valid[lane]) {
          batch.dvx[lane] = 0.0;
          batch.dvy[lane] = 0.0;
          batch.dvz[lane] = 0.0;
          batch.wAx[lane] = 0.0;
          batch.wAy[lane] = 0.0;
          batch.wAz[lane] = 0.0;
          batch.wBx[lane] = 0.0;
          batch.wBy[lane] = 0.0;
          batch.wBz[lane] = 0.0;
          continue;
        }

        const int slot_a = batch.bodyA_slot[lane];
        const int slot_b = batch.bodyB_slot[lane];
        batch.dvx[lane] = body_state.vx[static_cast<std::size_t>(slot_b)] -
                          body_state.vx[static_cast<std::size_t>(slot_a)];
        batch.dvy[lane] = body_state.vy[static_cast<std::size_t>(slot_b)] -
                          body_state.vy[static_cast<std::size_t>(slot_a)];
        batch.dvz[lane] = body_state.vz[static_cast<std::size_t>(slot_b)] -
                          body_state.vz[static_cast<std::size_t>(slot_a)];
        batch.wAx[lane] = body_state.wx[static_cast<std::size_t>(slot_a)];
        batch.wAy[lane] = body_state.wy[static_cast<std::size_t>(slot_a)];
        batch.wAz[lane] = body_state.wz[static_cast<std::size_t>(slot_a)];
        batch.wBx[lane] = body_state.wx[static_cast<std::size_t>(slot_b)];
        batch.wBy[lane] = body_state.wy[static_cast<std::size_t>(slot_b)];
        batch.wBz[lane] = body_state.wz[static_cast<std::size_t>(slot_b)];
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
        const int idx = batch.start + lane;
        if (!batch.lane_valid[lane]) {
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
          const int slot_a = batch.bodyA_slot[lane];
          const int slot_b = batch.bodyB_slot[lane];
          body_state.vx[static_cast<std::size_t>(slot_a)] -=
              impulse_x * batch.invMassA[lane];
          body_state.vy[static_cast<std::size_t>(slot_a)] -=
              impulse_y * batch.invMassA[lane];
          body_state.vz[static_cast<std::size_t>(slot_a)] -=
              impulse_z * batch.invMassA[lane];
          body_state.vx[static_cast<std::size_t>(slot_b)] +=
              impulse_x * batch.invMassB[lane];
          body_state.vy[static_cast<std::size_t>(slot_b)] +=
              impulse_y * batch.invMassB[lane];
          body_state.vz[static_cast<std::size_t>(slot_b)] +=
              impulse_z * batch.invMassB[lane];

          body_state.wx[static_cast<std::size_t>(slot_a)] -=
              applied * batch.TWn_a_x[lane];
          body_state.wy[static_cast<std::size_t>(slot_a)] -=
              applied * batch.TWn_a_y[lane];
          body_state.wz[static_cast<std::size_t>(slot_a)] -=
              applied * batch.TWn_a_z[lane];
          body_state.wx[static_cast<std::size_t>(slot_b)] +=
              applied * batch.TWn_b_x[lane];
          body_state.wy[static_cast<std::size_t>(slot_b)] +=
              applied * batch.TWn_b_y[lane];
          body_state.wz[static_cast<std::size_t>(slot_b)] +=
              applied * batch.TWn_b_z[lane];

          batch.dvx[lane] = body_state.vx[static_cast<std::size_t>(slot_b)] -
                            body_state.vx[static_cast<std::size_t>(slot_a)];
          batch.dvy[lane] = body_state.vy[static_cast<std::size_t>(slot_b)] -
                            body_state.vy[static_cast<std::size_t>(slot_a)];
          batch.dvz[lane] = body_state.vz[static_cast<std::size_t>(slot_b)] -
                            body_state.vz[static_cast<std::size_t>(slot_a)];
          batch.wAx[lane] = body_state.wx[static_cast<std::size_t>(slot_a)];
          batch.wAy[lane] = body_state.wy[static_cast<std::size_t>(slot_a)];
          batch.wAz[lane] = body_state.wz[static_cast<std::size_t>(slot_a)];
          batch.wBx[lane] = body_state.wx[static_cast<std::size_t>(slot_b)];
          batch.wBy[lane] = body_state.wy[static_cast<std::size_t>(slot_b)];
          batch.wBz[lane] = body_state.wz[static_cast<std::size_t>(slot_b)];
        } else {
          const int slot_a = batch.bodyA_slot[lane];
          const int slot_b = batch.bodyB_slot[lane];
          batch.dvx[lane] = body_state.vx[static_cast<std::size_t>(slot_b)] -
                            body_state.vx[static_cast<std::size_t>(slot_a)];
          batch.dvy[lane] = body_state.vy[static_cast<std::size_t>(slot_b)] -
                            body_state.vy[static_cast<std::size_t>(slot_a)];
          batch.dvz[lane] = body_state.vz[static_cast<std::size_t>(slot_b)] -
                            body_state.vz[static_cast<std::size_t>(slot_a)];
          batch.wAx[lane] = body_state.wx[static_cast<std::size_t>(slot_a)];
          batch.wAy[lane] = body_state.wy[static_cast<std::size_t>(slot_a)];
          batch.wAz[lane] = body_state.wz[static_cast<std::size_t>(slot_a)];
          batch.wBx[lane] = body_state.wx[static_cast<std::size_t>(slot_b)];
          batch.wBy[lane] = body_state.wy[static_cast<std::size_t>(slot_b)];
          batch.wBz[lane] = body_state.wz[static_cast<std::size_t>(slot_b)];
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
        const int idx = batch.start + lane;
        if (!batch.lane_valid[lane]) {
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

        const double vt_mag = std::sqrt(batch.v_rel_t1[lane] * batch.v_rel_t1[lane] +
                                        batch.v_rel_t2[lane] * batch.v_rel_t2[lane]);
        const double jt1_old = batch.jt1[lane];
        const double jt2_old = batch.jt2[lane];
        double jt1_candidate =
            jt1_old + (-batch.v_rel_t1[lane]) * batch.inv_k_t1[lane];
        double jt2_candidate =
            jt2_old + (-batch.v_rel_t2[lane]) * batch.inv_k_t2[lane];

        double friction_max =
            mu * std::max(batch.jn_new[lane], 0.0);
        const double jt_mag_sq =
            jt1_candidate * jt1_candidate + jt2_candidate * jt2_candidate;
        double friction_max_sq = friction_max * friction_max;
        double scale = 1.0;
        if (jt_mag_sq > friction_max_sq &&
            jt_mag_sq > math::kEps * math::kEps) {
          const double jt_mag = std::sqrt(jt_mag_sq);
          if (mu > 0.0 && vt_mag < kStaticFrictionSpeedThreshold) {
            const double required_jn = jt_mag / mu;
            const double delta_needed = required_jn - batch.jn_new[lane];
            if (delta_needed > math::kEps) {
              const int slot_a = batch.bodyA_slot[lane];
              const int slot_b = batch.bodyB_slot[lane];
              const double impulse_x = delta_needed * batch.nx[lane];
              const double impulse_y = delta_needed * batch.ny[lane];
              const double impulse_z = delta_needed * batch.nz[lane];

              body_state.vx[static_cast<std::size_t>(slot_a)] -=
                  impulse_x * batch.invMassA[lane];
              body_state.vy[static_cast<std::size_t>(slot_a)] -=
                  impulse_y * batch.invMassA[lane];
              body_state.vz[static_cast<std::size_t>(slot_a)] -=
                  impulse_z * batch.invMassA[lane];
              body_state.vx[static_cast<std::size_t>(slot_b)] +=
                  impulse_x * batch.invMassB[lane];
              body_state.vy[static_cast<std::size_t>(slot_b)] +=
                  impulse_y * batch.invMassB[lane];
              body_state.vz[static_cast<std::size_t>(slot_b)] +=
                  impulse_z * batch.invMassB[lane];

              body_state.wx[static_cast<std::size_t>(slot_a)] -=
                  delta_needed * batch.TWn_a_x[lane];
              body_state.wy[static_cast<std::size_t>(slot_a)] -=
                  delta_needed * batch.TWn_a_y[lane];
              body_state.wz[static_cast<std::size_t>(slot_a)] -=
                  delta_needed * batch.TWn_a_z[lane];
              body_state.wx[static_cast<std::size_t>(slot_b)] +=
                  delta_needed * batch.TWn_b_x[lane];
              body_state.wy[static_cast<std::size_t>(slot_b)] +=
                  delta_needed * batch.TWn_b_y[lane];
              body_state.wz[static_cast<std::size_t>(slot_b)] +=
                  delta_needed * batch.TWn_b_z[lane];

              batch.dvx[lane] = body_state.vx[static_cast<std::size_t>(slot_b)] -
                                body_state.vx[static_cast<std::size_t>(slot_a)];
              batch.dvy[lane] = body_state.vy[static_cast<std::size_t>(slot_b)] -
                                body_state.vy[static_cast<std::size_t>(slot_a)];
              batch.dvz[lane] = body_state.vz[static_cast<std::size_t>(slot_b)] -
                                body_state.vz[static_cast<std::size_t>(slot_a)];
              batch.wAx[lane] = body_state.wx[static_cast<std::size_t>(slot_a)];
              batch.wAy[lane] = body_state.wy[static_cast<std::size_t>(slot_a)];
              batch.wAz[lane] = body_state.wz[static_cast<std::size_t>(slot_a)];
              batch.wBx[lane] = body_state.wx[static_cast<std::size_t>(slot_b)];
              batch.wBy[lane] = body_state.wy[static_cast<std::size_t>(slot_b)];
              batch.wBz[lane] = body_state.wz[static_cast<std::size_t>(slot_b)];

              batch.jn_new[lane] = required_jn;
              batch.jn[lane] = required_jn;
              rows.jn[idx] = required_jn;
              friction_max = mu * std::max(batch.jn_new[lane], 0.0);
              friction_max_sq = friction_max * friction_max;
            }
          }
          if (jt_mag_sq > friction_max_sq &&
              jt_mag_sq > math::kEps * math::kEps) {
            scale = (friction_max > 0.0) ? (friction_max / jt_mag) : 0.0;
            if (debug_info) {
              ++debug_info->tangent_projections;
            }
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
          const int slot_a = batch.bodyA_slot[lane];
          const int slot_b = batch.bodyB_slot[lane];

          const double impulse_x = delta_jt1 * batch.t1x[lane] +
                                   delta_jt2 * batch.t2x[lane];
          const double impulse_y = delta_jt1 * batch.t1y[lane] +
                                   delta_jt2 * batch.t2y[lane];
          const double impulse_z = delta_jt1 * batch.t1z[lane] +
                                   delta_jt2 * batch.t2z[lane];

          body_state.vx[static_cast<std::size_t>(slot_a)] -=
              impulse_x * batch.invMassA[lane];
          body_state.vy[static_cast<std::size_t>(slot_a)] -=
              impulse_y * batch.invMassA[lane];
          body_state.vz[static_cast<std::size_t>(slot_a)] -=
              impulse_z * batch.invMassA[lane];
          body_state.vx[static_cast<std::size_t>(slot_b)] +=
              impulse_x * batch.invMassB[lane];
          body_state.vy[static_cast<std::size_t>(slot_b)] +=
              impulse_y * batch.invMassB[lane];
          body_state.vz[static_cast<std::size_t>(slot_b)] +=
              impulse_z * batch.invMassB[lane];

          body_state.wx[static_cast<std::size_t>(slot_a)] -=
              delta_jt1 * batch.TWt1_a_x[lane] +
              delta_jt2 * batch.TWt2_a_x[lane];
          body_state.wy[static_cast<std::size_t>(slot_a)] -=
              delta_jt1 * batch.TWt1_a_y[lane] +
              delta_jt2 * batch.TWt2_a_y[lane];
          body_state.wz[static_cast<std::size_t>(slot_a)] -=
              delta_jt1 * batch.TWt1_a_z[lane] +
              delta_jt2 * batch.TWt2_a_z[lane];
          body_state.wx[static_cast<std::size_t>(slot_b)] +=
              delta_jt1 * batch.TWt1_b_x[lane] +
              delta_jt2 * batch.TWt2_b_x[lane];
          body_state.wy[static_cast<std::size_t>(slot_b)] +=
              delta_jt1 * batch.TWt1_b_y[lane] +
              delta_jt2 * batch.TWt2_b_y[lane];
          body_state.wz[static_cast<std::size_t>(slot_b)] +=
              delta_jt1 * batch.TWt1_b_z[lane] +
              delta_jt2 * batch.TWt2_b_z[lane];

          batch.dvx[lane] = body_state.vx[static_cast<std::size_t>(slot_b)] -
                            body_state.vx[static_cast<std::size_t>(slot_a)];
          batch.dvy[lane] = body_state.vy[static_cast<std::size_t>(slot_b)] -
                            body_state.vy[static_cast<std::size_t>(slot_a)];
          batch.dvz[lane] = body_state.vz[static_cast<std::size_t>(slot_b)] -
                            body_state.vz[static_cast<std::size_t>(slot_a)];
          batch.wAx[lane] = body_state.wx[static_cast<std::size_t>(slot_a)];
          batch.wAy[lane] = body_state.wy[static_cast<std::size_t>(slot_a)];
          batch.wAz[lane] = body_state.wz[static_cast<std::size_t>(slot_a)];
          batch.wBx[lane] = body_state.wx[static_cast<std::size_t>(slot_b)];
          batch.wBy[lane] = body_state.wy[static_cast<std::size_t>(slot_b)];
          batch.wBz[lane] = body_state.wz[static_cast<std::size_t>(slot_b)];
        }
      }
    }

    for (std::size_t i = 0; i < joints.size(); ++i) {
      solve_joint_iteration_scalar(i);
    }
  }

  for (int pass = 0; pass < 3; ++pass) {
    for (std::size_t i = 0; i < joints.size(); ++i) {
      solve_joint_iteration_scalar(i);
    }
  }

  for (int pass = 0; pass < 2; ++pass) {
    for (std::size_t i = 0; i < joints.size(); ++i) {
      const int ia = joints.a[i];
      const int ib = joints.b[i];
      if (ia < 0 || ib < 0 || ia >= static_cast<int>(bodies.size()) ||
          ib >= static_cast<int>(bodies.size())) {
        continue;
      }

      RigidBody& A = bodies[ia];
      RigidBody& B = bodies[ib];
      const Vec3 ra = joints.ra[i];
      const Vec3 rb = joints.rb[i];
      const Vec3 pa = A.x + ra;
      const Vec3 pb = B.x + rb;
      const Vec3 delta = pb - pa;
      const double dist = math::length(delta);
      if (dist <= math::kEps) {
        continue;
      }
      const double rest = joints.rest[i];
      const double error = dist - rest;
      if (std::fabs(error) <= 1e-6) {
        continue;
      }
      if (joints.rope[i] && error < 0.0) {
        continue;
      }

      const bool projection_enabled = (params.beta > math::kEps) ||
                                      (joints.beta[i] > math::kEps) ||
                                      (joints.gamma[i] > math::kEps);
      if (!projection_enabled) {
        continue;
      }

      const Vec3 dir = math::normalize_safe(delta);
      double k = joints.k[i];
      if (k <= math::kEps) {
        continue;
      }

      const double impulse_mag = -error / k;
      const Vec3 impulse = impulse_mag * dir;

      if (A.invMass > math::kEps) {
        A.x -= impulse * A.invMass;
      }
      if (B.invMass > math::kEps) {
        B.x += impulse * B.invMass;
      }

      auto apply_rotation = [](RigidBody& body, const Vec3& offset,
                                const Vec3& impulse_vec) {
        const Vec3 torque = math::cross(offset, impulse_vec);
        const Vec3 ang = body.invInertiaWorld * torque;
        const double angle = math::length(ang);
        if (angle <= math::kEps) {
          return;
        }
        const Vec3 axis = ang / angle;
        body.q = math::Quat::from_axis_angle(axis, angle) * body.q;
        body.q.normalize();
      };

      if (math::length2(ra) > math::kEps * math::kEps &&
          A.invMass > math::kEps) {
        apply_rotation(A, ra, impulse);
      }
      if (math::length2(rb) > math::kEps * math::kEps &&
          B.invMass > math::kEps) {
        apply_rotation(B, rb, -impulse);
      }

      A.syncDerived();
      B.syncDerived();
    }
  }

  if (debug_info) {
    const auto iteration_end = Clock::now();
    debug_info->timings.solver_iterations_ms +=
        elapsed_ms(iteration_begin, iteration_end);
  }

  body_state.store_to(bodies);

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

