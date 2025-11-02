#include "solver_scalar_soa_native.hpp"

#include "concurrency/task_pool.hpp"

#include "platform.hpp"
#include "solver/thread_scratch.hpp"
#include "soa_pack.hpp"
#include "solver_scalar_soa.hpp"
#include "types.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

namespace {

using math::Vec3;
using admc::ThreadScratch;

using Clock = std::chrono::steady_clock;
using DurationMs = std::chrono::duration<double, std::milli>;

constexpr double kStaticFrictionSpeedThreshold = 0.1;
constexpr double kStaticFrictionSpeedThresholdSq =
    kStaticFrictionSpeedThreshold * kStaticFrictionSpeedThreshold;

double elapsed_ms(const Clock::time_point& begin, const Clock::time_point& end) {
  return DurationMs(end - begin).count();
}

struct ScopedAccumulator {
  Clock::time_point begin;
  double* target = nullptr;

  explicit ScopedAccumulator(double* slot) : begin(Clock::now()), target(slot) {}
  ~ScopedAccumulator() {
    if (target) {
      *target += elapsed_ms(begin, Clock::now());
    }
  }
};

struct BodySoA {
  std::span<double> vx;
  std::span<double> vy;
  std::span<double> vz;
  std::span<double> wx;
  std::span<double> wy;
  std::span<double> wz;

  explicit BodySoA(const ThreadScratch::BodyState& state)
      : vx(state.vx),
        vy(state.vy),
        vz(state.vz),
        wx(state.wx),
        wy(state.wy),
        wz(state.wz) {}

  void load_from(const std::vector<RigidBody>& bodies,
                 const std::vector<int>& active_indices) {
    for (int index : active_indices) {
      const std::size_t i = static_cast<std::size_t>(index);
      vx[i] = bodies[i].v.x;
      vy[i] = bodies[i].v.y;
      vz[i] = bodies[i].v.z;
      wx[i] = bodies[i].w.x;
      wy[i] = bodies[i].w.y;
      wz[i] = bodies[i].w.z;
    }
  }

  void store_to(std::vector<RigidBody>& bodies,
                const std::vector<int>& active_indices) const {
    for (int index : active_indices) {
      const std::size_t i = static_cast<std::size_t>(index);
      bodies[i].v.x = vx[i];
      bodies[i].v.y = vy[i];
      bodies[i].v.z = vz[i];
      bodies[i].w.x = wx[i];
      bodies[i].w.y = wy[i];
      bodies[i].w.z = wz[i];
    }
  }
};

inline void apply_impulse_to_body(const RigidBody& body,
                                  BodySoA& state,
                                  int index,
                                  const Vec3& impulse,
                                  const Vec3& r) {
  state.vx[static_cast<std::size_t>(index)] += impulse.x * body.invMass;
  state.vy[static_cast<std::size_t>(index)] += impulse.y * body.invMass;
  state.vz[static_cast<std::size_t>(index)] += impulse.z * body.invMass;

  const Vec3 torque = math::cross(r, impulse);
  const Vec3 delta_w = body.invInertiaWorld * torque;
  state.wx[static_cast<std::size_t>(index)] += delta_w.x;
  state.wy[static_cast<std::size_t>(index)] += delta_w.y;
  state.wz[static_cast<std::size_t>(index)] += delta_w.z;
}

struct ContactBatch;
struct BatchColoring;

BatchColoring color_contact_batches(const std::vector<ContactBatch>& batches,
                                    int body_count);

double solve_normal_batch(ContactBatch& batch,
                          BodySoA& body_state,
                          SolverDebugInfo* debug_info);

double solve_friction_batch(ContactBatch& batch,
                            BodySoA& body_state,
                            SolverDebugInfo* debug_info);

inline double clamp_tangent(double jt1,
                            double jt2,
                            double limit,
                            double limit_sq,
                            bool* clamped);

admc::TaskPool& native_solver_pool(unsigned requested_threads) {
  struct Holder {
    std::mutex mutex;
    std::unique_ptr<admc::TaskPool> pool;
    unsigned threads = 0;
  };
  static Holder holder;

  const unsigned desired = std::max(1u, requested_threads);
  std::lock_guard<std::mutex> lock(holder.mutex);
  if (!holder.pool || holder.threads != desired) {
    holder.pool.reset();
    holder.pool = std::make_unique<admc::TaskPool>(desired);
    holder.threads = desired;
  }
  return *holder.pool;
}

#if ADMC_HAS_AVX2
#include <immintrin.h>
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
#elif ADMC_HAS_NEON
#include <arm_neon.h>
struct VecD {
  float64x2_t v;

  VecD() = default;
  explicit VecD(float64x2_t value) : v(value) {}

  static VecD load(const double* ptr) { return VecD{vld1q_f64(ptr)}; }

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

  static VecD load_masked(const double* ptr, int /*count*/) { return load(ptr); }

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

struct ContactBatch {
  int lanes = 0;
  int start = 0;
  const double* nx = nullptr;
  const double* ny = nullptr;
  const double* nz = nullptr;
  const double* t1x = nullptr;
  const double* t1y = nullptr;
  const double* t1z = nullptr;
  const double* t2x = nullptr;
  const double* t2y = nullptr;
  const double* t2z = nullptr;
  const double* raxn_x = nullptr;
  const double* raxn_y = nullptr;
  const double* raxn_z = nullptr;
  const double* rbxn_x = nullptr;
  const double* rbxn_y = nullptr;
  const double* rbxn_z = nullptr;
  const double* raxt1_x = nullptr;
  const double* raxt1_y = nullptr;
  const double* raxt1_z = nullptr;
  const double* rbxt1_x = nullptr;
  const double* rbxt1_y = nullptr;
  const double* rbxt1_z = nullptr;
  const double* raxt2_x = nullptr;
  const double* raxt2_y = nullptr;
  const double* raxt2_z = nullptr;
  const double* rbxt2_x = nullptr;
  const double* rbxt2_y = nullptr;
  const double* rbxt2_z = nullptr;
  const double* TWn_a_x = nullptr;
  const double* TWn_a_y = nullptr;
  const double* TWn_a_z = nullptr;
  const double* TWn_b_x = nullptr;
  const double* TWn_b_y = nullptr;
  const double* TWn_b_z = nullptr;
  const double* TWt1_a_x = nullptr;
  const double* TWt1_a_y = nullptr;
  const double* TWt1_a_z = nullptr;
  const double* TWt1_b_x = nullptr;
  const double* TWt1_b_y = nullptr;
  const double* TWt1_b_z = nullptr;
  const double* TWt2_a_x = nullptr;
  const double* TWt2_a_y = nullptr;
  const double* TWt2_a_z = nullptr;
  const double* TWt2_b_x = nullptr;
  const double* TWt2_b_y = nullptr;
  const double* TWt2_b_z = nullptr;
  const double* inv_k_n = nullptr;
  const double* inv_k_t1 = nullptr;
  const double* inv_k_t2 = nullptr;
  const double* bias = nullptr;
  const double* bounce = nullptr;
  const double* mu = nullptr;
  double* jn = nullptr;
  double* jt1 = nullptr;
  double* jt2 = nullptr;
  int bodyA_index[soa::kLane] = {};
  int bodyB_index[soa::kLane] = {};
  double invMassA[soa::kLane] = {};
  double invMassB[soa::kLane] = {};
  bool lane_valid[soa::kLane] = {};
  bool has_friction = false;
  alignas(32) double jn_pre_clamp[soa::kLane] = {};
  alignas(32) double jn_new[soa::kLane] = {};
  alignas(32) double rel_t1[soa::kLane] = {};
  alignas(32) double rel_t2[soa::kLane] = {};
  alignas(32) double jt1_candidate[soa::kLane] = {};
  alignas(32) double jt2_candidate[soa::kLane] = {};
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

inline void apply_body_delta(ContactBatch& batch,
                             int body_index,
                             double dvx,
                             double dvy,
                             double dvz,
                             double dwx,
                             double dwy,
                             double dwz) {
  if (body_index < 0) {
    return;
  }
  for (int lane = 0; lane < batch.lanes; ++lane) {
    if (!batch.lane_valid[lane]) {
      continue;
    }
    if (batch.bodyA_index[lane] == body_index) {
      batch.dvx[lane] -= dvx;
      batch.dvy[lane] -= dvy;
      batch.dvz[lane] -= dvz;
      batch.wAx[lane] += dwx;
      batch.wAy[lane] += dwy;
      batch.wAz[lane] += dwz;
    }
    if (batch.bodyB_index[lane] == body_index) {
      batch.dvx[lane] += dvx;
      batch.dvy[lane] += dvy;
      batch.dvz[lane] += dvz;
      batch.wBx[lane] += dwx;
      batch.wBy[lane] += dwy;
      batch.wBz[lane] += dwz;
    }
  }
}

inline void compute_relative_velocities(ContactBatch& batch,
                                        BodySoA& body_state) {
  for (int lane = 0; lane < batch.lanes; ++lane) {
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
    const int ia = batch.bodyA_index[lane];
    const int ib = batch.bodyB_index[lane];
    batch.dvx[lane] = body_state.vx[ib] - body_state.vx[ia];
    batch.dvy[lane] = body_state.vy[ib] - body_state.vy[ia];
    batch.dvz[lane] = body_state.vz[ib] - body_state.vz[ia];
    batch.wAx[lane] = body_state.wx[ia];
    batch.wAy[lane] = body_state.wy[ia];
    batch.wAz[lane] = body_state.wz[ia];
    batch.wBx[lane] = body_state.wx[ib];
    batch.wBy[lane] = body_state.wy[ib];
    batch.wBz[lane] = body_state.wz[ib];
  }
}

struct BatchColoring {
  std::vector<int> batch_color;
  std::vector<std::vector<int>> batches_per_color;
  std::vector<char> color_has_friction;
};

BatchColoring color_contact_batches(const std::vector<ContactBatch>& batches,
                                    int body_count) {
  BatchColoring result;
  const std::size_t batch_count = batches.size();
  result.batch_color.assign(batch_count, -1);
  if (batch_count == 0 || body_count <= 0) {
    return result;
  }

  std::vector<std::vector<int>> body_colors(static_cast<std::size_t>(body_count));
  std::vector<int> color_stamp;
  std::vector<int> used_colors;
  int stamp = 0;
  int max_color = -1;

  auto mark_color = [&](int color) {
    if (color < 0) {
      return;
    }
    if (static_cast<std::size_t>(color) >= color_stamp.size()) {
      color_stamp.resize(static_cast<std::size_t>(color) + 1, -1);
    }
    if (color_stamp[static_cast<std::size_t>(color)] != stamp) {
      color_stamp[static_cast<std::size_t>(color)] = stamp;
      used_colors.push_back(color);
    }
  };

  for (std::size_t i = 0; i < batch_count; ++i) {
    const ContactBatch& batch = batches[i];
    ++stamp;
    used_colors.clear();

    for (int lane = 0; lane < batch.lanes; ++lane) {
      const int a = batch.bodyA_index[lane];
      const int b = batch.bodyB_index[lane];
      if (a >= 0 && a < body_count) {
        for (int color : body_colors[static_cast<std::size_t>(a)]) {
          mark_color(color);
        }
      }
      if (b >= 0 && b < body_count) {
        for (int color : body_colors[static_cast<std::size_t>(b)]) {
          mark_color(color);
        }
      }
    }

    int color = 0;
    while (true) {
      if (static_cast<std::size_t>(color) >= color_stamp.size()) {
        color_stamp.resize(static_cast<std::size_t>(color) + 1, -1);
        break;
      }
      if (color_stamp[static_cast<std::size_t>(color)] != stamp) {
        break;
      }
      ++color;
    }

    result.batch_color[i] = color;
    max_color = std::max(max_color, color);

    for (int lane = 0; lane < batch.lanes; ++lane) {
      const int a = batch.bodyA_index[lane];
      const int b = batch.bodyB_index[lane];
      if (a >= 0 && a < body_count) {
        body_colors[static_cast<std::size_t>(a)].push_back(color);
      }
      if (b >= 0 && b < body_count) {
        body_colors[static_cast<std::size_t>(b)].push_back(color);
      }
    }
  }

  if (max_color < 0) {
    return result;
  }

  result.batches_per_color.resize(static_cast<std::size_t>(max_color) + 1);
  result.color_has_friction.resize(static_cast<std::size_t>(max_color) + 1, 0);
  for (std::size_t i = 0; i < batch_count; ++i) {
    const int color = result.batch_color[i];
    if (color < 0) {
      continue;
    }
    result.batches_per_color[static_cast<std::size_t>(color)].push_back(static_cast<int>(i));
    if (batches[i].has_friction) {
      result.color_has_friction[static_cast<std::size_t>(color)] = 1;
    }
  }

  return result;
}

double solve_normal_batch(ContactBatch& batch,
                          BodySoA& body_state,
                          SolverDebugInfo* debug_info) {
  VecD dvx_v = VecD::load(batch.dvx);
  VecD dvy_v = VecD::load(batch.dvy);
  VecD dvz_v = VecD::load(batch.dvz);
  VecD nx_v = VecD::load_masked(batch.nx, batch.lanes);
  VecD ny_v = VecD::load_masked(batch.ny, batch.lanes);
  VecD nz_v = VecD::load_masked(batch.nz, batch.lanes);
  VecD wAx_v = VecD::load(batch.wAx);
  VecD wAy_v = VecD::load(batch.wAy);
  VecD wAz_v = VecD::load(batch.wAz);
  VecD wBx_v = VecD::load(batch.wBx);
  VecD wBy_v = VecD::load(batch.wBy);
  VecD wBz_v = VecD::load(batch.wBz);
  VecD raxn_x_v = VecD::load_masked(batch.raxn_x, batch.lanes);
  VecD raxn_y_v = VecD::load_masked(batch.raxn_y, batch.lanes);
  VecD raxn_z_v = VecD::load_masked(batch.raxn_z, batch.lanes);
  VecD rbxn_x_v = VecD::load_masked(batch.rbxn_x, batch.lanes);
  VecD rbxn_y_v = VecD::load_masked(batch.rbxn_y, batch.lanes);
  VecD rbxn_z_v = VecD::load_masked(batch.rbxn_z, batch.lanes);

  VecD wA_dot_raxn = add(mul(wAx_v, raxn_x_v),
                         add(mul(wAy_v, raxn_y_v), mul(wAz_v, raxn_z_v)));
  VecD wB_dot_rbxn = add(mul(wBx_v, rbxn_x_v),
                         add(mul(wBy_v, rbxn_y_v), mul(wBz_v, rbxn_z_v)));

  VecD v_rel_n = add(add(mul(nx_v, dvx_v), mul(ny_v, dvy_v)),
                     mul(nz_v, dvz_v));
  v_rel_n = add(v_rel_n, sub(wB_dot_rbxn, wA_dot_raxn));

  VecD bias_v = VecD::load_masked(batch.bias, batch.lanes);
  VecD bounce_v = VecD::load_masked(batch.bounce, batch.lanes);
  VecD inv_kn_v = VecD::load_masked(batch.inv_k_n, batch.lanes);
  VecD jn_old_v = VecD::load_masked(batch.jn, batch.lanes);

  VecD rhs = add(negate(add(v_rel_n, bias_v)), bounce_v);
  VecD jn_candidate_v = add(jn_old_v, mul(rhs, inv_kn_v));
  VecD zero_v = VecD::broadcast(0.0);
  VecD jn_clamped_v = max(jn_candidate_v, zero_v);

  jn_candidate_v.store(batch.jn_pre_clamp);
  jn_clamped_v.store(batch.jn_new);

  double max_delta = 0.0;
  for (int lane = 0; lane < batch.lanes; ++lane) {
    if (!batch.lane_valid[lane]) {
      batch.jn[lane] = 0.0;
      batch.jt1[lane] = 0.0;
      batch.jt2[lane] = 0.0;
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
    max_delta = std::max(max_delta, std::fabs(applied));
    batch.jn[lane] = jn_new;

    if (std::fabs(applied) > math::kEps) {
      const double impulse_x = applied * batch.nx[lane];
      const double impulse_y = applied * batch.ny[lane];
      const double impulse_z = applied * batch.nz[lane];
      const int ia = batch.bodyA_index[lane];
      const int ib = batch.bodyB_index[lane];
      const double inv_mass_a = batch.invMassA[lane];
      const double inv_mass_b = batch.invMassB[lane];
      const double delta_vax = -impulse_x * inv_mass_a;
      const double delta_vay = -impulse_y * inv_mass_a;
      const double delta_vaz = -impulse_z * inv_mass_a;
      const double delta_vbx = impulse_x * inv_mass_b;
      const double delta_vby = impulse_y * inv_mass_b;
      const double delta_vbz = impulse_z * inv_mass_b;

      const double delta_wax = -applied * batch.TWn_a_x[lane];
      const double delta_way = -applied * batch.TWn_a_y[lane];
      const double delta_waz = -applied * batch.TWn_a_z[lane];
      const double delta_wbx = applied * batch.TWn_b_x[lane];
      const double delta_wby = applied * batch.TWn_b_y[lane];
      const double delta_wbz = applied * batch.TWn_b_z[lane];

      body_state.vx[ia] += delta_vax;
      body_state.vy[ia] += delta_vay;
      body_state.vz[ia] += delta_vaz;
      body_state.vx[ib] += delta_vbx;
      body_state.vy[ib] += delta_vby;
      body_state.vz[ib] += delta_vbz;

      body_state.wx[ia] += delta_wax;
      body_state.wy[ia] += delta_way;
      body_state.wz[ia] += delta_waz;
      body_state.wx[ib] += delta_wbx;
      body_state.wy[ib] += delta_wby;
      body_state.wz[ib] += delta_wbz;

      apply_body_delta(batch, ia, delta_vax, delta_vay, delta_vaz, delta_wax,
                       delta_way, delta_waz);
      apply_body_delta(batch, ib, delta_vbx, delta_vby, delta_vbz, delta_wbx,
                       delta_wby, delta_wbz);
    }
  }

  return max_delta;
}

double solve_friction_batch(ContactBatch& batch,
                            BodySoA& body_state,
                            SolverDebugInfo* debug_info) {
  if (!batch.has_friction) {
    return 0.0;
  }

  VecD dvx_v = VecD::load(batch.dvx);
  VecD dvy_v = VecD::load(batch.dvy);
  VecD dvz_v = VecD::load(batch.dvz);
  VecD wAx_v = VecD::load(batch.wAx);
  VecD wAy_v = VecD::load(batch.wAy);
  VecD wAz_v = VecD::load(batch.wAz);
  VecD wBx_v = VecD::load(batch.wBx);
  VecD wBy_v = VecD::load(batch.wBy);
  VecD wBz_v = VecD::load(batch.wBz);

  VecD t1x_v = VecD::load_masked(batch.t1x, batch.lanes);
  VecD t1y_v = VecD::load_masked(batch.t1y, batch.lanes);
  VecD t1z_v = VecD::load_masked(batch.t1z, batch.lanes);
  VecD t2x_v = VecD::load_masked(batch.t2x, batch.lanes);
  VecD t2y_v = VecD::load_masked(batch.t2y, batch.lanes);
  VecD t2z_v = VecD::load_masked(batch.t2z, batch.lanes);

  VecD raxt1_x_v = VecD::load_masked(batch.raxt1_x, batch.lanes);
  VecD raxt1_y_v = VecD::load_masked(batch.raxt1_y, batch.lanes);
  VecD raxt1_z_v = VecD::load_masked(batch.raxt1_z, batch.lanes);
  VecD rbxt1_x_v = VecD::load_masked(batch.rbxt1_x, batch.lanes);
  VecD rbxt1_y_v = VecD::load_masked(batch.rbxt1_y, batch.lanes);
  VecD rbxt1_z_v = VecD::load_masked(batch.rbxt1_z, batch.lanes);
  VecD raxt2_x_v = VecD::load_masked(batch.raxt2_x, batch.lanes);
  VecD raxt2_y_v = VecD::load_masked(batch.raxt2_y, batch.lanes);
  VecD raxt2_z_v = VecD::load_masked(batch.raxt2_z, batch.lanes);
  VecD rbxt2_x_v = VecD::load_masked(batch.rbxt2_x, batch.lanes);
  VecD rbxt2_y_v = VecD::load_masked(batch.rbxt2_y, batch.lanes);
  VecD rbxt2_z_v = VecD::load_masked(batch.rbxt2_z, batch.lanes);

  VecD wA_dot_raxt1 = add(mul(wAx_v, raxt1_x_v),
                          add(mul(wAy_v, raxt1_y_v), mul(wAz_v, raxt1_z_v)));
  VecD wB_dot_rbxt1 = add(mul(wBx_v, rbxt1_x_v),
                          add(mul(wBy_v, rbxt1_y_v), mul(wBz_v, rbxt1_z_v)));
  VecD wA_dot_raxt2 = add(mul(wAx_v, raxt2_x_v),
                          add(mul(wAy_v, raxt2_y_v), mul(wAz_v, raxt2_z_v)));
  VecD wB_dot_rbxt2 = add(mul(wBx_v, rbxt2_x_v),
                          add(mul(wBy_v, rbxt2_y_v), mul(wBz_v, rbxt2_z_v)));

  VecD t1_rel = add(add(mul(t1x_v, dvx_v), mul(t1y_v, dvy_v)),
                    mul(t1z_v, dvz_v));
  VecD t2_rel = add(add(mul(t2x_v, dvx_v), mul(t2y_v, dvy_v)),
                    mul(t2z_v, dvz_v));
  t1_rel = add(t1_rel, sub(wB_dot_rbxt1, wA_dot_raxt1));
  t2_rel = add(t2_rel, sub(wB_dot_rbxt2, wA_dot_raxt2));
  t1_rel.store(batch.rel_t1);
  t2_rel.store(batch.rel_t2);

  VecD inv_kt1_v = VecD::load_masked(batch.inv_k_t1, batch.lanes);
  VecD inv_kt2_v = VecD::load_masked(batch.inv_k_t2, batch.lanes);
  VecD jt1_old_v = VecD::load_masked(batch.jt1, batch.lanes);
  VecD jt2_old_v = VecD::load_masked(batch.jt2, batch.lanes);

  VecD jt1_candidate_v = sub(jt1_old_v, mul(t1_rel, inv_kt1_v));
  VecD jt2_candidate_v = sub(jt2_old_v, mul(t2_rel, inv_kt2_v));
  jt1_candidate_v.store(batch.jt1_candidate);
  jt2_candidate_v.store(batch.jt2_candidate);

  double max_delta = 0.0;
  for (int lane = 0; lane < batch.lanes; ++lane) {
    if (!batch.lane_valid[lane]) {
      continue;
    }
    double jt1_new = batch.jt1_candidate[lane];
    double jt2_new = batch.jt2_candidate[lane];
    bool clamped = false;
    const double limit = batch.mu[lane] * std::max(batch.jn[lane], 0.0);
    const double vt1 = batch.rel_t1[lane];
    const double vt2 = batch.rel_t2[lane];
    const double vt_sq = vt1 * vt1 + vt2 * vt2;
    if (limit <= 0.0) {
      jt1_new = 0.0;
      jt2_new = 0.0;
    } else if (vt_sq < kStaticFrictionSpeedThresholdSq) {
      jt1_new = 0.0;
      jt2_new = 0.0;
    }

    const double limit_sq = limit * limit;
    const double scale =
        clamp_tangent(jt1_new, jt2_new, limit, limit_sq, &clamped);
    if (clamped && debug_info) {
      ++debug_info->tangent_projections;
    }
    jt1_new *= scale;
    jt2_new *= scale;

    const double dj1 = jt1_new - batch.jt1[lane];
    const double dj2 = jt2_new - batch.jt2[lane];
    const double delta_t_mag = std::sqrt(dj1 * dj1 + dj2 * dj2);
    max_delta = std::max(max_delta, delta_t_mag);
    batch.jt1[lane] = jt1_new;
    batch.jt2[lane] = jt2_new;
    if (std::fabs(dj1) > math::kEps || std::fabs(dj2) > math::kEps) {
      const int ia = batch.bodyA_index[lane];
      const int ib = batch.bodyB_index[lane];
      const double impulse_x = dj1 * batch.t1x[lane] + dj2 * batch.t2x[lane];
      const double impulse_y = dj1 * batch.t1y[lane] + dj2 * batch.t2y[lane];
      const double impulse_z = dj1 * batch.t1z[lane] + dj2 * batch.t2z[lane];

      const double inv_mass_a = batch.invMassA[lane];
      const double inv_mass_b = batch.invMassB[lane];
      const double delta_vax = -impulse_x * inv_mass_a;
      const double delta_vay = -impulse_y * inv_mass_a;
      const double delta_vaz = -impulse_z * inv_mass_a;
      const double delta_vbx = impulse_x * inv_mass_b;
      const double delta_vby = impulse_y * inv_mass_b;
      const double delta_vbz = impulse_z * inv_mass_b;

      const double delta_wax =
          -(dj1 * batch.TWt1_a_x[lane] + dj2 * batch.TWt2_a_x[lane]);
      const double delta_way =
          -(dj1 * batch.TWt1_a_y[lane] + dj2 * batch.TWt2_a_y[lane]);
      const double delta_waz =
          -(dj1 * batch.TWt1_a_z[lane] + dj2 * batch.TWt2_a_z[lane]);
      const double delta_wbx =
          dj1 * batch.TWt1_b_x[lane] + dj2 * batch.TWt2_b_x[lane];
      const double delta_wby =
          dj1 * batch.TWt1_b_y[lane] + dj2 * batch.TWt2_b_y[lane];
      const double delta_wbz =
          dj1 * batch.TWt1_b_z[lane] + dj2 * batch.TWt2_b_z[lane];

      body_state.vx[ia] += delta_vax;
      body_state.vy[ia] += delta_vay;
      body_state.vz[ia] += delta_vaz;
      body_state.vx[ib] += delta_vbx;
      body_state.vy[ib] += delta_vby;
      body_state.vz[ib] += delta_vbz;

      body_state.wx[ia] += delta_wax;
      body_state.wy[ia] += delta_way;
      body_state.wz[ia] += delta_waz;
      body_state.wx[ib] += delta_wbx;
      body_state.wy[ib] += delta_wby;
      body_state.wz[ib] += delta_wbz;

      apply_body_delta(batch, ia, delta_vax, delta_vay, delta_vaz, delta_wax,
                       delta_way, delta_waz);
      apply_body_delta(batch, ib, delta_vbx, delta_vby, delta_vbz, delta_wbx,
                       delta_wby, delta_wbz);
    }
  }

  return max_delta;
}

inline double clamp_tangent(double jt1,
                            double jt2,
                            double limit,
                            double limit_sq,
                            bool* clamped) {
  const double mag_sq = jt1 * jt1 + jt2 * jt2;
  if (mag_sq <= limit_sq) {
    return 1.0;
  }
  if (mag_sq <= math::kEps * math::kEps) {
    return 1.0;
  }
  *clamped = true;
  const double inv_mag = 1.0 / std::sqrt(mag_sq);
  return limit * inv_mag;
}

std::vector<int> collect_active_body_indices(const RowSOA& rows,
                                             const JointSOA& joints,
                                             std::size_t body_count) {
  std::vector<int> active;
  if (body_count == 0) {
    return active;
  }
  std::vector<char> seen(body_count, 0);
  auto mark = [&](int idx) {
    if (idx < 0 || idx >= static_cast<int>(body_count)) {
      return;
    }
    if (seen[static_cast<std::size_t>(idx)]) {
      return;
    }
    seen[static_cast<std::size_t>(idx)] = 1;
    active.push_back(idx);
  };
  const std::size_t contact_count = rows.size();
  for (std::size_t i = 0; i < contact_count; ++i) {
    mark(rows.a[i]);
    mark(rows.b[i]);
  }
  const std::size_t joint_count = joints.size();
  for (std::size_t i = 0; i < joint_count; ++i) {
    mark(joints.a[i]);
    mark(joints.b[i]);
  }
  return active;
}

inline void sanitize_solver_timings(SolverDebugInfo* info,
                                    const SoaNativeStats& stats,
                                    double measured_total_ms) {
  if (!info) {
    return;
  }
  info->timings = stats.to_breakdown();
  if (info->timings.solver_total_ms <= 0.0) {
    info->timings.solver_total_ms = measured_total_ms;
  }
  if (info->timings.total_step_ms <= 0.0) {
    info->timings.total_step_ms =
        std::max(info->timings.solver_total_ms, measured_total_ms);
  }
}

}  // namespace

void solve_scalar_soa_native(std::vector<RigidBody>& bodies,
                             std::vector<Contact>& contacts,
                             RowSOA& rows,
                             JointSOA& joints,
                             const SoaParams& params,
                             SolverDebugInfo* debug_info) {
  const auto solver_begin = Clock::now();
  const int iterations = std::max(1, params.iterations);
  SoaNativeStats stats;

  if (debug_info) {
    debug_info->reset();
  }

  for (RigidBody& body : bodies) {
    body.syncDerived();
  }

  ThreadScratch& scratch = admc::tls_scratch();
  BodySoA body_state(scratch.acquire_body_state(bodies.size()));
  std::fill(body_state.vx.begin(), body_state.vx.end(), 0.0);
  std::fill(body_state.vy.begin(), body_state.vy.end(), 0.0);
  std::fill(body_state.vz.begin(), body_state.vz.end(), 0.0);
  std::fill(body_state.wx.begin(), body_state.wx.end(), 0.0);
  std::fill(body_state.wy.begin(), body_state.wy.end(), 0.0);
  std::fill(body_state.wz.begin(), body_state.wz.end(), 0.0);
  const std::vector<int> active_bodies =
      collect_active_body_indices(rows, joints, bodies.size());
  {
    ScopedAccumulator accum(&stats.staging_ms);
    body_state.load_from(bodies, active_bodies);
  }

  if (!params.warm_start) {
    const std::size_t contact_count = rows.size();
    std::fill_n(rows.jn.begin(), contact_count, 0.0);
    std::fill_n(rows.jt1.begin(), contact_count, 0.0);
    std::fill_n(rows.jt2.begin(), contact_count, 0.0);
    std::fill_n(joints.j.begin(), joints.size(), 0.0);
  } else {
    ScopedAccumulator accum(&stats.warmstart_ms);
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

      body_state.vx[ia] -= impulse_x * A.invMass;
      body_state.vy[ia] -= impulse_y * A.invMass;
      body_state.vz[ia] -= impulse_z * A.invMass;
      body_state.vx[ib] += impulse_x * B.invMass;
      body_state.vy[ib] += impulse_y * B.invMass;
      body_state.vz[ib] += impulse_z * B.invMass;

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

      body_state.wx[ia] -= dw_ax;
      body_state.wy[ia] -= dw_ay;
      body_state.wz[ia] -= dw_az;
      body_state.wx[ib] += dw_bx;
      body_state.wy[ib] += dw_by;
      body_state.wz[ib] += dw_bz;
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

  std::vector<ContactBatch> contact_batches;
  if (rows.N > 0) {
    const int num_batches = (rows.N + soa::kLane - 1) / soa::kLane;
    contact_batches.reserve(static_cast<std::size_t>(num_batches));
    for (int start = 0; start < rows.N; start += soa::kLane) {
      ContactBatch batch;
      batch.start = start;
      batch.lanes = std::min(soa::kLane, rows.N - start);
      batch.nx = rows.nx.data() + start;
      batch.ny = rows.ny.data() + start;
      batch.nz = rows.nz.data() + start;
      batch.t1x = rows.t1x.data() + start;
      batch.t1y = rows.t1y.data() + start;
      batch.t1z = rows.t1z.data() + start;
      batch.t2x = rows.t2x.data() + start;
      batch.t2y = rows.t2y.data() + start;
      batch.t2z = rows.t2z.data() + start;
      batch.raxn_x = rows.raxn_x.data() + start;
      batch.raxn_y = rows.raxn_y.data() + start;
      batch.raxn_z = rows.raxn_z.data() + start;
      batch.rbxn_x = rows.rbxn_x.data() + start;
      batch.rbxn_y = rows.rbxn_y.data() + start;
      batch.rbxn_z = rows.rbxn_z.data() + start;
      batch.raxt1_x = rows.raxt1_x.data() + start;
      batch.raxt1_y = rows.raxt1_y.data() + start;
      batch.raxt1_z = rows.raxt1_z.data() + start;
      batch.rbxt1_x = rows.rbxt1_x.data() + start;
      batch.rbxt1_y = rows.rbxt1_y.data() + start;
      batch.rbxt1_z = rows.rbxt1_z.data() + start;
      batch.raxt2_x = rows.raxt2_x.data() + start;
      batch.raxt2_y = rows.raxt2_y.data() + start;
      batch.raxt2_z = rows.raxt2_z.data() + start;
      batch.rbxt2_x = rows.rbxt2_x.data() + start;
      batch.rbxt2_y = rows.rbxt2_y.data() + start;
      batch.rbxt2_z = rows.rbxt2_z.data() + start;
      batch.TWn_a_x = rows.TWn_a_x.data() + start;
      batch.TWn_a_y = rows.TWn_a_y.data() + start;
      batch.TWn_a_z = rows.TWn_a_z.data() + start;
      batch.TWn_b_x = rows.TWn_b_x.data() + start;
      batch.TWn_b_y = rows.TWn_b_y.data() + start;
      batch.TWn_b_z = rows.TWn_b_z.data() + start;
      batch.TWt1_a_x = rows.TWt1_a_x.data() + start;
      batch.TWt1_a_y = rows.TWt1_a_y.data() + start;
      batch.TWt1_a_z = rows.TWt1_a_z.data() + start;
      batch.TWt1_b_x = rows.TWt1_b_x.data() + start;
      batch.TWt1_b_y = rows.TWt1_b_y.data() + start;
      batch.TWt1_b_z = rows.TWt1_b_z.data() + start;
      batch.TWt2_a_x = rows.TWt2_a_x.data() + start;
      batch.TWt2_a_y = rows.TWt2_a_y.data() + start;
      batch.TWt2_a_z = rows.TWt2_a_z.data() + start;
      batch.TWt2_b_x = rows.TWt2_b_x.data() + start;
      batch.TWt2_b_y = rows.TWt2_b_y.data() + start;
      batch.TWt2_b_z = rows.TWt2_b_z.data() + start;
      batch.inv_k_n = rows.inv_k_n.data() + start;
      batch.inv_k_t1 = rows.inv_k_t1.data() + start;
      batch.inv_k_t2 = rows.inv_k_t2.data() + start;
      batch.bias = rows.bias.data() + start;
      batch.bounce = rows.bounce.data() + start;
      batch.mu = rows.mu.data() + start;
      batch.jn = rows.jn.data() + start;
      batch.jt1 = rows.jt1.data() + start;
      batch.jt2 = rows.jt2.data() + start;

      for (int lane = 0; lane < batch.lanes; ++lane) {
        const int idx = start + lane;
        const int ia = rows.a[idx];
        const int ib = rows.b[idx];
        const bool valid = ia >= 0 && ib >= 0 &&
                           ia < static_cast<int>(bodies.size()) &&
                           ib < static_cast<int>(bodies.size());
        batch.lane_valid[lane] = valid;
        if (!valid) {
          if (debug_info) {
            ++debug_info->invalid_contact_indices;
          }
          batch.bodyA_index[lane] = 0;
          batch.bodyB_index[lane] = 0;
          batch.invMassA[lane] = 0.0;
          batch.invMassB[lane] = 0.0;
          continue;
        }
        batch.bodyA_index[lane] = ia;
        batch.bodyB_index[lane] = ib;
        batch.invMassA[lane] = bodies[ia].invMass;
        batch.invMassB[lane] = bodies[ib].invMass;
        if (batch.mu[lane] > 0.0) {
          batch.has_friction = true;
        }
      }

      compute_relative_velocities(batch, body_state);
      contact_batches.push_back(batch);
    }
  }


#if defined(ADMC_ENABLE_PARALLEL) && !defined(ADMC_DETERMINISTIC)
  const bool allow_parallel = params.use_threads && params.thread_count > 1;
#else
  const bool allow_parallel = false;
#endif
  BatchColoring color_data;
  admc::TaskPool* pool = nullptr;
  unsigned worker_count = 1;
  std::vector<SolverDebugInfo> worker_debug;
  if (allow_parallel) {
    color_data =
        color_contact_batches(contact_batches,
                              static_cast<int>(body_state.vx.size()));
    if (!color_data.batches_per_color.empty()) {
      pool = &native_solver_pool(static_cast<unsigned>(params.thread_count));
      worker_count = std::max(1u, pool->worker_count());
      if (debug_info) {
        worker_debug.resize(worker_count);
      }
    }
  }

  const double convergence_threshold = params.convergence_threshold;
  const bool use_convergence = convergence_threshold > 0.0;
  for (int iter = 0; iter < iterations; ++iter) {
    double normal_max_delta = 0.0;
    double friction_max_delta = 0.0;
    const bool parallel_ready =
        allow_parallel && pool && !color_data.batches_per_color.empty();

    if (!parallel_ready) {
      ScopedAccumulator normal_timer(&stats.normal_ms);
      for (ContactBatch& batch : contact_batches) {
        normal_max_delta = std::max(
            normal_max_delta,
            solve_normal_batch(batch, body_state, debug_info));
      }
    } else {
      ScopedAccumulator normal_timer(&stats.normal_ms);
      for (std::size_t color = 0;
           color < color_data.batches_per_color.size(); ++color) {
        const auto& bucket = color_data.batches_per_color[color];
        if (bucket.empty()) {
          continue;
        }
        std::atomic<std::size_t> cursor{0};
        std::vector<double> local_max(worker_count, 0.0);
        if (debug_info) {
          for (auto& dbg : worker_debug) {
            dbg.reset();
          }
        }
        for (unsigned w = 0; w < worker_count; ++w) {
          pool->enqueue([&, w]() {
            double max_delta = 0.0;
            SolverDebugInfo* dbg =
                debug_info ? &worker_debug[w] : nullptr;
            while (true) {
              const std::size_t idx =
                  cursor.fetch_add(1, std::memory_order_relaxed);
              if (idx >= bucket.size()) {
                break;
              }
              ContactBatch& batch =
                  contact_batches[static_cast<std::size_t>(bucket[idx])];
              max_delta = std::max(
                  max_delta, solve_normal_batch(batch, body_state, dbg));
            }
            local_max[w] = max_delta;
          });
        }
        pool->wait_idle();
        if (debug_info) {
          for (const auto& dbg : worker_debug) {
            debug_info->accumulate(dbg);
          }
        }
        double color_max = 0.0;
        for (double value : local_max) {
          color_max = std::max(color_max, value);
        }
        normal_max_delta = std::max(normal_max_delta, color_max);
      }
    }

    if (!parallel_ready) {
      ScopedAccumulator friction_timer(&stats.friction_ms);
      for (ContactBatch& batch : contact_batches) {
        friction_max_delta = std::max(
            friction_max_delta,
            solve_friction_batch(batch, body_state, debug_info));
      }
    } else {
      ScopedAccumulator friction_timer(&stats.friction_ms);
      for (std::size_t color = 0;
           color < color_data.batches_per_color.size(); ++color) {
        if (!color_data.color_has_friction.empty() &&
            !color_data.color_has_friction[color]) {
          continue;
        }
        const auto& bucket = color_data.batches_per_color[color];
        if (bucket.empty()) {
          continue;
        }
        std::atomic<std::size_t> cursor{0};
        std::vector<double> local_max(worker_count, 0.0);
        if (debug_info) {
          for (auto& dbg : worker_debug) {
            dbg.reset();
          }
        }
        for (unsigned w = 0; w < worker_count; ++w) {
          pool->enqueue([&, w]() {
            double max_delta = 0.0;
            SolverDebugInfo* dbg =
                debug_info ? &worker_debug[w] : nullptr;
            while (true) {
              const std::size_t idx =
                  cursor.fetch_add(1, std::memory_order_relaxed);
              if (idx >= bucket.size()) {
                break;
              }
              ContactBatch& batch =
                  contact_batches[static_cast<std::size_t>(bucket[idx])];
              max_delta = std::max(
                  max_delta, solve_friction_batch(batch, body_state, dbg));
            }
            local_max[w] = max_delta;
          });
        }
        pool->wait_idle();
        if (debug_info) {
          for (const auto& dbg : worker_debug) {
            debug_info->accumulate(dbg);
          }
        }
        double color_max = 0.0;
        for (double value : local_max) {
          color_max = std::max(color_max, value);
        }
        friction_max_delta = std::max(friction_max_delta, color_max);
      }
    }

    const double iteration_max_delta =
        std::max(normal_max_delta, friction_max_delta);
    if (use_convergence && iteration_max_delta < convergence_threshold) {
      break;
    }
  }

  {
    ScopedAccumulator accum(&stats.writeback_ms);
    body_state.store_to(bodies, active_bodies);
  }

  const auto solver_end = Clock::now();
  const double measured_total_ms = elapsed_ms(solver_begin, solver_end);
  sanitize_solver_timings(debug_info, stats, measured_total_ms);
}

void solve_scalar_soa_native(std::vector<RigidBody>& bodies,
                             std::vector<Contact>& contacts,
                             RowSOA& rows,
                             const SoaParams& params,
                             SolverDebugInfo* debug_info) {
  static JointSOA empty_joints;
  empty_joints.clear();
  solve_scalar_soa_native(bodies, contacts, rows, empty_joints, params,
                          debug_info);
}

void solve_scalar_soa_native(std::vector<RigidBody>& bodies,
                             std::vector<Contact>& contacts,
                             RowSOA& rows,
                             JointSOA& joints,
                             const SolverParams& params,
                             SolverDebugInfo* debug_info) {
  SoaParams derived;
  static_cast<SolverParams&>(derived) = params;
  solve_scalar_soa_native(bodies, contacts, rows, joints, derived, debug_info);
}

void solve_scalar_soa_native(std::vector<RigidBody>& bodies,
                             std::vector<Contact>& contacts,
                             RowSOA& rows,
                             const SolverParams& params,
                             SolverDebugInfo* debug_info) {
  SoaParams derived;
  static_cast<SolverParams&>(derived) = params;
  solve_scalar_soa_native(bodies, contacts, rows, derived, debug_info);
}
