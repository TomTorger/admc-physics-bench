#include "solver_scalar_soa_spheres.hpp"

#include "solver_vec_lane.hpp"
#include "types.hpp"

#if defined(_MSC_VER)
#define ADMC_RESTRICT __restrict
#else
#define ADMC_RESTRICT __restrict__
#endif

namespace {

#if defined(ADMC_ENABLE_AVX2) && !defined(__ARM_NEON)
constexpr bool kHasAvx2 = true;
#else
constexpr bool kHasAvx2 = false;
#endif

#if defined(ADMC_ENABLE_NEON) && defined(__ARM_NEON)
constexpr bool kHasNeon = true;
#else
constexpr bool kHasNeon = false;
#endif

inline void apply_impulse_to_body(int index,
                                  float impulse_x,
                                  float impulse_y,
                                  float impulse_z,
                                  float* ADMC_RESTRICT vx,
                                  float* ADMC_RESTRICT vy,
                                  float* ADMC_RESTRICT vz,
                                  const float* ADMC_RESTRICT inv_mass) {
  if (index < 0) {
    return;
  }
  const float inv_m = inv_mass[index];
  if (inv_m <= 0.0f) {
    return;
  }
  vx[index] -= inv_m * impulse_x;
  vy[index] -= inv_m * impulse_y;
  vz[index] -= inv_m * impulse_z;
}

inline void apply_impulse_to_body_b(int index,
                                    float impulse_x,
                                    float impulse_y,
                                    float impulse_z,
                                    float* ADMC_RESTRICT vx,
                                    float* ADMC_RESTRICT vy,
                                    float* ADMC_RESTRICT vz,
                                    const float* ADMC_RESTRICT inv_mass) {
  if (index < 0) {
    return;
  }
  const float inv_m = inv_mass[index];
  if (inv_m <= 0.0f) {
    return;
  }
  vx[index] += inv_m * impulse_x;
  vy[index] += inv_m * impulse_y;
  vz[index] += inv_m * impulse_z;
}

}  // namespace

namespace soa {

void solve_tile_normals_spheres(Tile& tile, int iterations) {
  const int rows = tile.size();
  if (rows <= 0 || iterations <= 0) {
    return;
  }

  int* ADMC_RESTRICT bodyA = tile.bodyA.data();
  int* ADMC_RESTRICT bodyB = tile.bodyB.data();
  float* ADMC_RESTRICT nx = tile.n_x.data();
  float* ADMC_RESTRICT ny = tile.n_y.data();
  float* ADMC_RESTRICT nz = tile.n_z.data();
  float* ADMC_RESTRICT target = tile.target_n.data();
  float* ADMC_RESTRICT inv_k_n = tile.inv_k_n.data();
  float* ADMC_RESTRICT j_n = tile.j_n.data();

  float* ADMC_RESTRICT vx = tile.bodies.vx.data();
  float* ADMC_RESTRICT vy = tile.bodies.vy.data();
  float* ADMC_RESTRICT vz = tile.bodies.vz.data();
  float* ADMC_RESTRICT inv_mass = tile.bodies.invMass.data();

  const bool lane_available = (kHasAvx2 || kHasNeon);
  const int lane_width = lane_available ? 8 : 0;
  const int bulk = (lane_width > 0) ? (rows / lane_width) * lane_width : 0;

  SoaAlignedVector<float> lane_vAx;
  SoaAlignedVector<float> lane_vAy;
  SoaAlignedVector<float> lane_vAz;
  SoaAlignedVector<float> lane_vBx;
  SoaAlignedVector<float> lane_vBy;
  SoaAlignedVector<float> lane_vBz;
  SoaAlignedVector<float> lane_invMassA;
  SoaAlignedVector<float> lane_invMassB;
  SoaAlignedVector<float> lane_jn_prev;

  if (bulk > 0) {
    lane_vAx.resize(static_cast<std::size_t>(bulk));
    lane_vAy.resize(static_cast<std::size_t>(bulk));
    lane_vAz.resize(static_cast<std::size_t>(bulk));
    lane_vBx.resize(static_cast<std::size_t>(bulk));
    lane_vBy.resize(static_cast<std::size_t>(bulk));
    lane_vBz.resize(static_cast<std::size_t>(bulk));
    lane_invMassA.resize(static_cast<std::size_t>(bulk));
    lane_invMassB.resize(static_cast<std::size_t>(bulk));
    lane_jn_prev.resize(static_cast<std::size_t>(bulk));

    for (int i = 0; i < bulk; ++i) {
      const int a = bodyA[i];
      const int b = bodyB[i];
      lane_invMassA[static_cast<std::size_t>(i)] =
          (a >= 0) ? inv_mass[a] : 0.0f;
      lane_invMassB[static_cast<std::size_t>(i)] =
          (b >= 0) ? inv_mass[b] : 0.0f;
    }
  }

  admc::TileSpan span{};
  if (bulk > 0) {
    span.n_x = tile.n_x.data();
    span.n_y = tile.n_y.data();
    span.n_z = tile.n_z.data();
    span.rAx = tile.rAx.data();
    span.rAy = tile.rAy.data();
    span.rAz = tile.rAz.data();
    span.rBx = tile.rBx.data();
    span.rBy = tile.rBy.data();
    span.rBz = tile.rBz.data();
    span.bodyA = reinterpret_cast<const int32_t*>(tile.bodyA.data());
    span.bodyB = reinterpret_cast<const int32_t*>(tile.bodyB.data());
    span.k_n = tile.inv_k_n.data();
    span.target_n = target;
    span.jn = j_n;
    span.vAx = lane_vAx.data();
    span.vAy = lane_vAy.data();
    span.vAz = lane_vAz.data();
    span.vBx = lane_vBx.data();
    span.vBy = lane_vBy.data();
    span.vBz = lane_vBz.data();
    span.invMassA = lane_invMassA.data();
    span.invMassB = lane_invMassB.data();
    span.count = bulk;
    span.lanes = lane_width;
  }

  for (int iter = 0; iter < iterations; ++iter) {
    if (bulk > 0) {
      for (int i = 0; i < bulk; ++i) {
        const int a = bodyA[i];
        const int b = bodyB[i];
        if (a >= 0) {
          lane_vAx[static_cast<std::size_t>(i)] = vx[a];
          lane_vAy[static_cast<std::size_t>(i)] = vy[a];
          lane_vAz[static_cast<std::size_t>(i)] = vz[a];
        } else {
          lane_vAx[static_cast<std::size_t>(i)] = 0.0f;
          lane_vAy[static_cast<std::size_t>(i)] = 0.0f;
          lane_vAz[static_cast<std::size_t>(i)] = 0.0f;
        }
        if (b >= 0) {
          lane_vBx[static_cast<std::size_t>(i)] = vx[b];
          lane_vBy[static_cast<std::size_t>(i)] = vy[b];
          lane_vBz[static_cast<std::size_t>(i)] = vz[b];
        } else {
          lane_vBx[static_cast<std::size_t>(i)] = 0.0f;
          lane_vBy[static_cast<std::size_t>(i)] = 0.0f;
          lane_vBz[static_cast<std::size_t>(i)] = 0.0f;
        }
        lane_jn_prev[static_cast<std::size_t>(i)] = j_n[i];
      }

      if (kHasAvx2) {
        solve_normal_lane_avx2(span);
      } else if (kHasNeon) {
        solve_normal_lane_neon(span);
      }

      for (int i = 0; i < bulk; ++i) {
        const float applied = j_n[i] - lane_jn_prev[static_cast<std::size_t>(i)];
        if (applied == 0.0f) {
          continue;
        }
        const float jx = applied * nx[i];
        const float jy = applied * ny[i];
        const float jz = applied * nz[i];
        const int a = bodyA[i];
        const int b = bodyB[i];
        if (a >= 0) {
          apply_impulse_to_body(a, jx, jy, jz, vx, vy, vz, inv_mass);
        }
        if (b >= 0) {
          apply_impulse_to_body_b(b, jx, jy, jz, vx, vy, vz, inv_mass);
        }
      }
    }

#pragma omp simd
    for (int i = bulk; i < rows; ++i) {
      const int a = bodyA[i];
      const int b = bodyB[i];

      float vax = 0.0f;
      float vay = 0.0f;
      float vaz = 0.0f;
      if (a >= 0) {
        vax = vx[a];
        vay = vy[a];
        vaz = vz[a];
      }
      float vbx = 0.0f;
      float vby = 0.0f;
      float vbz = 0.0f;
      if (b >= 0) {
        vbx = vx[b];
        vby = vy[b];
        vbz = vz[b];
      }

      const float nx_i = nx[i];
      const float ny_i = ny[i];
      const float nz_i = nz[i];

      const float dvx = vbx - vax;
      const float dvy = vby - vay;
      const float dvz = vbz - vaz;
      const float vrel_n = dvx * nx_i + dvy * ny_i + dvz * nz_i;

      const float target_i = target[i];
      const float inv_kn_i = inv_k_n[i];

      float jn_old = j_n[i];
      float jn_new = jn_old + (target_i - vrel_n) * inv_kn_i;
      if (jn_new < 0.0f) {
        jn_new = 0.0f;
      }
      const float applied = jn_new - jn_old;
      j_n[i] = jn_new;

      if (applied == 0.0f) {
        continue;
      }

      const float jx = applied * nx_i;
      const float jy = applied * ny_i;
      const float jz = applied * nz_i;

      if (a >= 0) {
        apply_impulse_to_body(a, jx, jy, jz, vx, vy, vz, inv_mass);
      }
      if (b >= 0) {
        apply_impulse_to_body_b(b, jx, jy, jz, vx, vy, vz, inv_mass);
      }
    }
  }
}

}  // namespace soa

