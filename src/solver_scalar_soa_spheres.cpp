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
  float* ADMC_RESTRICT k_n = tile.k_n.data();
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

      double vax = 0.0;
      double vay = 0.0;
      double vaz = 0.0;
      if (a >= 0) {
        vax = static_cast<double>(vx[a]);
        vay = static_cast<double>(vy[a]);
        vaz = static_cast<double>(vz[a]);
      }
      double vbx = 0.0;
      double vby = 0.0;
      double vbz = 0.0;
      if (b >= 0) {
        vbx = static_cast<double>(vx[b]);
        vby = static_cast<double>(vy[b]);
        vbz = static_cast<double>(vz[b]);
      }

      const double nx_i = static_cast<double>(nx[i]);
      const double ny_i = static_cast<double>(ny[i]);
      const double nz_i = static_cast<double>(nz[i]);

      const double dvx = vbx - vax;
      const double dvy = vby - vay;
      const double dvz = vbz - vaz;
      const double vrel_n = dvx * nx_i + dvy * ny_i + dvz * nz_i;

      const double target_i = static_cast<double>(target[i]);
      const double kn_i = static_cast<double>(k_n[i]);

      double jn_new = static_cast<double>(j_n[i]) + (target_i - vrel_n) / kn_i;
      if (jn_new < 0.0) {
        jn_new = 0.0;
      }
      const double applied = jn_new - static_cast<double>(j_n[i]);
      j_n[i] = static_cast<float>(jn_new);

      const double jx = applied * nx_i;
      const double jy = applied * ny_i;
      const double jz = applied * nz_i;

      const double ima = (a >= 0) ? static_cast<double>(inv_mass[a]) : 0.0;
      const double imb = (b >= 0) ? static_cast<double>(inv_mass[b]) : 0.0;

      if (a >= 0 && ima > 0.0) {
        vax -= ima * jx;
        vay -= ima * jy;
        vaz -= ima * jz;
        vx[a] = static_cast<float>(vax);
        vy[a] = static_cast<float>(vay);
        vz[a] = static_cast<float>(vaz);
      }

      if (b >= 0 && imb > 0.0) {
        vbx += imb * jx;
        vby += imb * jy;
        vbz += imb * jz;
        vx[b] = static_cast<float>(vbx);
        vy[b] = static_cast<float>(vby);
        vz[b] = static_cast<float>(vbz);
      }
    }
  }
}

}  // namespace soa

