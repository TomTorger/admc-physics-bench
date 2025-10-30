#include "solver_scalar_soa_spheres.hpp"

#if defined(_MSC_VER)
#define ADMC_RESTRICT __restrict
#else
#define ADMC_RESTRICT __restrict__
#endif

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

  for (int iter = 0; iter < iterations; ++iter) {
#pragma omp simd
    for (int i = 0; i < rows; ++i) {
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

      const float ima = (a >= 0) ? inv_mass[a] : 0.0f;
      if (a >= 0 && ima > 0.0f) {
        vax -= ima * jx;
        vay -= ima * jy;
        vaz -= ima * jz;
        vx[a] = vax;
        vy[a] = vay;
        vz[a] = vaz;
      }

      const float imb = (b >= 0) ? inv_mass[b] : 0.0f;
      if (b >= 0 && imb > 0.0f) {
        vbx += imb * jx;
        vby += imb * jy;
        vbz += imb * jz;
        vx[b] = vbx;
        vy[b] = vby;
        vz[b] = vbz;
      }
    }
  }
}

}  // namespace soa

