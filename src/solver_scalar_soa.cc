#include "solver_scalar_soa.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <numeric>
#include <sstream>

namespace {
using math::Vec3;

using Clock = std::chrono::steady_clock;
using DurationMs = std::chrono::duration<double, std::milli>;

constexpr double kPi = 3.14159265358979323846264338327950288;
constexpr double kWarmstartRotationCosThreshold =
    std::cos(15.0 * kPi / 180.0);

#if defined(_MSC_VER)
#define ADMC_RESTRICT __restrict
#else
#define ADMC_RESTRICT __restrict__
#endif

enum RowFlags : std::uint8_t {
  kRowHasFriction = 1u << 0,
  kRowIsParticle = 1u << 1,
};

struct TileContactRef {
  int row = -1;
  int local_a = -1;
  int local_b = -1;
};

struct LocalBodyState {
  int body = -1;
  Vec3 v;
  Vec3 w;
  double inv_mass = 0.0;
};

struct Tile {
  std::vector<int> rows;
  std::vector<int> body_ids;
  std::vector<TileContactRef> contacts;
  std::vector<int> normals_only;
  std::vector<int> particles;
  std::vector<int> frictional;
};

struct TileSolveScratch {
  std::vector<LocalBodyState> bodies;

  void resize_bodies(std::size_t count) { bodies.resize(count); }
};

int find_or_add_local_body(int body, std::vector<int>& local_ids) {
  for (std::size_t i = 0; i < local_ids.size(); ++i) {
    if (local_ids[i] == body) {
      return static_cast<int>(i);
    }
  }
  local_ids.push_back(body);
  return static_cast<int>(local_ids.size() - 1);
}

std::vector<Tile> build_tiles(const RowSOA& rows,
                              std::size_t body_count,
                              int tile_size) {
  std::vector<Tile> tiles;
  if (rows.N <= 0 || body_count == 0 || tile_size <= 0) {
    return tiles;
  }

  std::vector<int> parent(body_count);
  std::iota(parent.begin(), parent.end(), 0);
  std::vector<int> rank(body_count, 0);

  auto find_set = [&](int x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  };

  auto union_sets = [&](int a, int b) {
    a = find_set(a);
    b = find_set(b);
    if (a == b) {
      return;
    }
    if (rank[a] < rank[b]) {
      parent[a] = b;
    } else if (rank[a] > rank[b]) {
      parent[b] = a;
    } else {
      parent[b] = a;
      ++rank[a];
    }
  };

  for (int i = 0; i < rows.N; ++i) {
    const int a = rows.a[i];
    const int b = rows.b[i];
    if (a >= 0 && b >= 0 && a < static_cast<int>(body_count) &&
        b < static_cast<int>(body_count)) {
      union_sets(a, b);
    }
  }

  std::map<int, std::vector<int>> islands;
  for (int i = 0; i < rows.N; ++i) {
    const int a = rows.a[i];
    const int b = rows.b[i];
    int root = -1;
    if (a >= 0 && a < static_cast<int>(body_count)) {
      root = find_set(a);
    } else if (b >= 0 && b < static_cast<int>(body_count)) {
      root = find_set(b);
    }
    if (root < 0) {
      continue;
    }
    islands[root].push_back(i);
  }

  for (auto& entry : islands) {
    auto& island_rows = entry.second;
    std::stable_sort(island_rows.begin(), island_rows.end(),
                     [&](int lhs, int rhs) {
                       if (rows.a[lhs] != rows.a[rhs]) {
                         return rows.a[lhs] < rows.a[rhs];
                       }
                       if (rows.b[lhs] != rows.b[rhs]) {
                         return rows.b[lhs] < rows.b[rhs];
                       }
                       return lhs < rhs;
                     });
    for (std::size_t offset = 0; offset < island_rows.size();
         offset += static_cast<std::size_t>(tile_size)) {
      const std::size_t end =
          std::min(island_rows.size(), offset + static_cast<std::size_t>(tile_size));
      Tile tile;
      tile.rows.assign(island_rows.begin() + static_cast<std::ptrdiff_t>(offset),
                       island_rows.begin() + static_cast<std::ptrdiff_t>(end));
      tile.body_ids.reserve(tile.rows.size() * 2);
      tile.contacts.reserve(tile.rows.size());
      tile.normals_only.reserve(tile.rows.size());
      tile.particles.reserve(tile.rows.size());
      tile.frictional.reserve(tile.rows.size());

      for (int row : tile.rows) {
        TileContactRef ref;
        ref.row = row;

        const int a = rows.a[row];
        const int b = rows.b[row];
        if (a >= 0 && a < static_cast<int>(body_count)) {
          ref.local_a = find_or_add_local_body(a, tile.body_ids);
        }
        if (b >= 0 && b < static_cast<int>(body_count)) {
          ref.local_b = find_or_add_local_body(b, tile.body_ids);
        }

        tile.contacts.push_back(ref);
        const int contact_index = static_cast<int>(tile.contacts.size() - 1);
        const std::uint8_t flags = rows.flags[row];
        if ((flags & kRowHasFriction) == 0) {
          tile.normals_only.push_back(contact_index);
        } else if (flags & kRowIsParticle) {
          tile.particles.push_back(contact_index);
        } else {
          tile.frictional.push_back(contact_index);
        }
      }

      tiles.push_back(std::move(tile));
    }
  }

  return tiles;
}

void scatter_tile(const TileSolveScratch& scratch,
                  std::vector<RigidBody>& bodies) {
  for (const LocalBodyState& state : scratch.bodies) {
    if (state.body < 0 || state.body >= static_cast<int>(bodies.size())) {
      continue;
    }
    RigidBody& body = bodies[static_cast<std::size_t>(state.body)];
    body.v = state.v;
    body.w = state.w;
  }
}

struct TileWorkView {
  TileSolveScratch* scratch = nullptr;
  std::vector<RigidBody>* bodies = nullptr;
  RowSOA* rows = nullptr;
  SolverDebugInfo* debug = nullptr;
};

void stage_tile(TileWorkView& view, const Tile& tile) {
  TileSolveScratch& scratch = *view.scratch;
  scratch.resize_bodies(tile.body_ids.size());

  for (std::size_t i = 0; i < tile.body_ids.size(); ++i) {
    LocalBodyState state;
    state.body = tile.body_ids[i];
    if (state.body >= 0 &&
        state.body < static_cast<int>(view.bodies->size())) {
      const RigidBody& rb =
          (*view.bodies)[static_cast<std::size_t>(state.body)];
      state.v = rb.v;
      state.w = rb.w;
      state.inv_mass = rb.invMass;
    } else {
      state.v = Vec3();
      state.w = Vec3();
      state.inv_mass = 0.0;
    }
    scratch.bodies[i] = state;
  }
}

void solve_rows_normals_only(TileWorkView& view, const Tile& tile) {
  TileSolveScratch& scratch = *view.scratch;
  RowSOA& rows = *view.rows;
  const auto& contacts = tile.contacts;
  auto& bodies = scratch.bodies;
  double* ADMC_RESTRICT jn = rows.jn.data();
  double* ADMC_RESTRICT jt1 = rows.jt1.data();
  double* ADMC_RESTRICT jt2 = rows.jt2.data();
  const double* ADMC_RESTRICT nx = rows.nx.data();
  const double* ADMC_RESTRICT ny = rows.ny.data();
  const double* ADMC_RESTRICT nz = rows.nz.data();
  const double* ADMC_RESTRICT bias = rows.bias.data();
  const double* ADMC_RESTRICT bounce = rows.bounce.data();
  const double* ADMC_RESTRICT inv_k_n = rows.inv_k_n.data();
  const double* ADMC_RESTRICT raxn_x = rows.raxn_x.data();
  const double* ADMC_RESTRICT raxn_y = rows.raxn_y.data();
  const double* ADMC_RESTRICT raxn_z = rows.raxn_z.data();
  const double* ADMC_RESTRICT rbxn_x = rows.rbxn_x.data();
  const double* ADMC_RESTRICT rbxn_y = rows.rbxn_y.data();
  const double* ADMC_RESTRICT rbxn_z = rows.rbxn_z.data();
  const double* ADMC_RESTRICT TWn_a_x = rows.TWn_a_x.data();
  const double* ADMC_RESTRICT TWn_a_y = rows.TWn_a_y.data();
  const double* ADMC_RESTRICT TWn_a_z = rows.TWn_a_z.data();
  const double* ADMC_RESTRICT TWn_b_x = rows.TWn_b_x.data();
  const double* ADMC_RESTRICT TWn_b_y = rows.TWn_b_y.data();
  const double* ADMC_RESTRICT TWn_b_z = rows.TWn_b_z.data();

#if defined(__clang__)
#pragma clang loop vectorize(enable)
#endif
  for (std::size_t idx = 0; idx < tile.normals_only.size(); ++idx) {
    const int contact_index = tile.normals_only[idx];
    if (contact_index < 0 || contact_index >= static_cast<int>(contacts.size())) {
      continue;
    }
    const TileContactRef& ref = contacts[static_cast<std::size_t>(contact_index)];
    if (ref.local_a < 0 || ref.local_b < 0 ||
        ref.local_a >= static_cast<int>(bodies.size()) ||
        ref.local_b >= static_cast<int>(bodies.size())) {
      if (view.debug) {
        ++view.debug->invalid_contact_indices;
      }
      continue;
    }

    LocalBodyState& A = bodies[static_cast<std::size_t>(ref.local_a)];
    LocalBodyState& B = bodies[static_cast<std::size_t>(ref.local_b)];
    const int row = ref.row;

    double dvx = B.v.x - A.v.x;
    double dvy = B.v.y - A.v.y;
    double dvz = B.v.z - A.v.z;

    double wAx = A.w.x;
    double wAy = A.w.y;
    double wAz = A.w.z;
    double wBx = B.w.x;
    double wBy = B.w.y;
    double wBz = B.w.z;

    const double wA_dot_raxn =
        wAx * raxn_x[row] + wAy * raxn_y[row] + wAz * raxn_z[row];
    const double wB_dot_rbxn =
        wBx * rbxn_x[row] + wBy * rbxn_y[row] + wBz * rbxn_z[row];

    const double v_rel_n =
        nx[row] * dvx + ny[row] * dvy + nz[row] * dvz + wB_dot_rbxn - wA_dot_raxn;

    const double rhs = -(v_rel_n + bias[row] - bounce[row]);
    const double delta_jn = rhs * inv_k_n[row];
    const double jn_old = jn[row];
    double jn_candidate = jn_old + delta_jn;
    if (jn_candidate < 0.0) {
      if (view.debug) {
        ++view.debug->normal_impulse_clamps;
      }
      jn_candidate = 0.0;
    }
    jn[row] = jn_candidate;
    jt1[row] = 0.0;
    jt2[row] = 0.0;

    const double applied_n = jn_candidate - jn_old;
    if (std::fabs(applied_n) <= math::kEps) {
      continue;
    }

    const double impulse_x = nx[row] * applied_n;
    const double impulse_y = ny[row] * applied_n;
    const double impulse_z = nz[row] * applied_n;
    const double inv_mass_sum = A.inv_mass + B.inv_mass;

    A.v.x -= impulse_x * A.inv_mass;
    A.v.y -= impulse_y * A.inv_mass;
    A.v.z -= impulse_z * A.inv_mass;
    B.v.x += impulse_x * B.inv_mass;
    B.v.y += impulse_y * B.inv_mass;
    B.v.z += impulse_z * B.inv_mass;

    dvx += impulse_x * inv_mass_sum;
    dvy += impulse_y * inv_mass_sum;
    dvz += impulse_z * inv_mass_sum;

    const double TWn_ax = TWn_a_x[row];
    const double TWn_ay = TWn_a_y[row];
    const double TWn_az = TWn_a_z[row];
    const double TWn_bx = TWn_b_x[row];
    const double TWn_by = TWn_b_y[row];
    const double TWn_bz = TWn_b_z[row];

    A.w.x -= applied_n * TWn_ax;
    A.w.y -= applied_n * TWn_ay;
    A.w.z -= applied_n * TWn_az;
    B.w.x += applied_n * TWn_bx;
    B.w.y += applied_n * TWn_by;
    B.w.z += applied_n * TWn_bz;

    (void)dvx;
    (void)dvy;
    (void)dvz;
  }
}

void solve_rows_particles(TileWorkView& view, const Tile& tile) {
  TileSolveScratch& scratch = *view.scratch;
  RowSOA& rows = *view.rows;
  const auto& contacts = tile.contacts;
  auto& bodies = scratch.bodies;
  double* ADMC_RESTRICT jn = rows.jn.data();
  double* ADMC_RESTRICT jt1 = rows.jt1.data();
  double* ADMC_RESTRICT jt2 = rows.jt2.data();
  const double* ADMC_RESTRICT nx = rows.nx.data();
  const double* ADMC_RESTRICT ny = rows.ny.data();
  const double* ADMC_RESTRICT nz = rows.nz.data();
  const double* ADMC_RESTRICT t1x = rows.t1x.data();
  const double* ADMC_RESTRICT t1y = rows.t1y.data();
  const double* ADMC_RESTRICT t1z = rows.t1z.data();
  const double* ADMC_RESTRICT t2x = rows.t2x.data();
  const double* ADMC_RESTRICT t2y = rows.t2y.data();
  const double* ADMC_RESTRICT t2z = rows.t2z.data();
  const double* ADMC_RESTRICT bias = rows.bias.data();
  const double* ADMC_RESTRICT bounce = rows.bounce.data();
  const double* ADMC_RESTRICT inv_k_n = rows.inv_k_n.data();
  const double* ADMC_RESTRICT inv_k_t1 = rows.inv_k_t1.data();
  const double* ADMC_RESTRICT inv_k_t2 = rows.inv_k_t2.data();
  const double* ADMC_RESTRICT mu = rows.mu.data();

#if defined(__clang__)
#pragma clang loop vectorize(enable)
#endif
  for (std::size_t idx = 0; idx < tile.particles.size(); ++idx) {
    const int contact_index = tile.particles[idx];
    if (contact_index < 0 || contact_index >= static_cast<int>(contacts.size())) {
      continue;
    }
    const TileContactRef& ref = contacts[static_cast<std::size_t>(contact_index)];
    if (ref.local_a < 0 || ref.local_b < 0 ||
        ref.local_a >= static_cast<int>(bodies.size()) ||
        ref.local_b >= static_cast<int>(bodies.size())) {
      if (view.debug) {
        ++view.debug->invalid_contact_indices;
      }
      continue;
    }

    LocalBodyState& A = bodies[static_cast<std::size_t>(ref.local_a)];
    LocalBodyState& B = bodies[static_cast<std::size_t>(ref.local_b)];
    const int row = ref.row;

    const double dvx = B.v.x - A.v.x;
    const double dvy = B.v.y - A.v.y;
    const double dvz = B.v.z - A.v.z;

    const double v_rel_n = nx[row] * dvx + ny[row] * dvy + nz[row] * dvz;
    const double rhs = -(v_rel_n + bias[row] - bounce[row]);
    const double delta_jn = rhs * inv_k_n[row];
    const double jn_old = jn[row];
    double jn_candidate = jn_old + delta_jn;
    if (jn_candidate < 0.0) {
      if (view.debug) {
        ++view.debug->normal_impulse_clamps;
      }
      jn_candidate = 0.0;
    }
    jn[row] = jn_candidate;

    const double applied_n = jn_candidate - jn_old;
    if (std::fabs(applied_n) > math::kEps) {
      const double impulse_x = nx[row] * applied_n;
      const double impulse_y = ny[row] * applied_n;
      const double impulse_z = nz[row] * applied_n;
      A.v.x -= impulse_x * A.inv_mass;
      A.v.y -= impulse_y * A.inv_mass;
      A.v.z -= impulse_z * A.inv_mass;
      B.v.x += impulse_x * B.inv_mass;
      B.v.y += impulse_y * B.inv_mass;
      B.v.z += impulse_z * B.inv_mass;
    }

    if (mu[row] <= math::kEps) {
      jt1[row] = 0.0;
      jt2[row] = 0.0;
      continue;
    }

    const double v_rel_t1 = t1x[row] * dvx + t1y[row] * dvy + t1z[row] * dvz;
    const double v_rel_t2 = t2x[row] * dvx + t2y[row] * dvy + t2z[row] * dvz;
    double jt1_candidate = jt1[row] + (-v_rel_t1) * inv_k_t1[row];
    double jt2_candidate = jt2[row] + (-v_rel_t2) * inv_k_t2[row];

    const double friction_max = mu[row] * std::max(jn[row], 0.0);
    const double friction_max_sq = friction_max * friction_max;
    const double jt_mag_sq =
        jt1_candidate * jt1_candidate + jt2_candidate * jt2_candidate;
    double scale = 1.0;
    if (jt_mag_sq > friction_max_sq && jt_mag_sq > math::kEps * math::kEps) {
      const double jt_mag = std::sqrt(jt_mag_sq);
      scale = (friction_max > 0.0) ? (friction_max / jt_mag) : 0.0;
      if (view.debug) {
        ++view.debug->tangent_projections;
      }
    }

    jt1_candidate *= scale;
    jt2_candidate *= scale;
    const double delta_jt1 = jt1_candidate - jt1[row];
    const double delta_jt2 = jt2_candidate - jt2[row];
    jt1[row] = jt1_candidate;
    jt2[row] = jt2_candidate;

    if (std::fabs(delta_jt1) <= math::kEps &&
        std::fabs(delta_jt2) <= math::kEps) {
      continue;
    }

    const double impulse_x = delta_jt1 * t1x[row] + delta_jt2 * t2x[row];
    const double impulse_y = delta_jt1 * t1y[row] + delta_jt2 * t2y[row];
    const double impulse_z = delta_jt1 * t1z[row] + delta_jt2 * t2z[row];

    A.v.x -= impulse_x * A.inv_mass;
    A.v.y -= impulse_y * A.inv_mass;
    A.v.z -= impulse_z * A.inv_mass;
    B.v.x += impulse_x * B.inv_mass;
    B.v.y += impulse_y * B.inv_mass;
    B.v.z += impulse_z * B.inv_mass;
  }
}

void solve_rows_frictional(TileWorkView& view, const Tile& tile) {
  TileSolveScratch& scratch = *view.scratch;
  RowSOA& rows = *view.rows;
  const auto& contacts = tile.contacts;
  auto& bodies = scratch.bodies;
  double* ADMC_RESTRICT jn = rows.jn.data();
  double* ADMC_RESTRICT jt1 = rows.jt1.data();
  double* ADMC_RESTRICT jt2 = rows.jt2.data();
  const double* ADMC_RESTRICT nx = rows.nx.data();
  const double* ADMC_RESTRICT ny = rows.ny.data();
  const double* ADMC_RESTRICT nz = rows.nz.data();
  const double* ADMC_RESTRICT t1x = rows.t1x.data();
  const double* ADMC_RESTRICT t1y = rows.t1y.data();
  const double* ADMC_RESTRICT t1z = rows.t1z.data();
  const double* ADMC_RESTRICT t2x = rows.t2x.data();
  const double* ADMC_RESTRICT t2y = rows.t2y.data();
  const double* ADMC_RESTRICT t2z = rows.t2z.data();
  const double* ADMC_RESTRICT bias = rows.bias.data();
  const double* ADMC_RESTRICT bounce = rows.bounce.data();
  const double* ADMC_RESTRICT inv_k_n = rows.inv_k_n.data();
  const double* ADMC_RESTRICT inv_k_t1 = rows.inv_k_t1.data();
  const double* ADMC_RESTRICT inv_k_t2 = rows.inv_k_t2.data();
  const double* ADMC_RESTRICT mu = rows.mu.data();
  const double* ADMC_RESTRICT raxn_x = rows.raxn_x.data();
  const double* ADMC_RESTRICT raxn_y = rows.raxn_y.data();
  const double* ADMC_RESTRICT raxn_z = rows.raxn_z.data();
  const double* ADMC_RESTRICT rbxn_x = rows.rbxn_x.data();
  const double* ADMC_RESTRICT rbxn_y = rows.rbxn_y.data();
  const double* ADMC_RESTRICT rbxn_z = rows.rbxn_z.data();
  const double* ADMC_RESTRICT raxt1_x = rows.raxt1_x.data();
  const double* ADMC_RESTRICT raxt1_y = rows.raxt1_y.data();
  const double* ADMC_RESTRICT raxt1_z = rows.raxt1_z.data();
  const double* ADMC_RESTRICT rbxt1_x = rows.rbxt1_x.data();
  const double* ADMC_RESTRICT rbxt1_y = rows.rbxt1_y.data();
  const double* ADMC_RESTRICT rbxt1_z = rows.rbxt1_z.data();
  const double* ADMC_RESTRICT raxt2_x = rows.raxt2_x.data();
  const double* ADMC_RESTRICT raxt2_y = rows.raxt2_y.data();
  const double* ADMC_RESTRICT raxt2_z = rows.raxt2_z.data();
  const double* ADMC_RESTRICT rbxt2_x = rows.rbxt2_x.data();
  const double* ADMC_RESTRICT rbxt2_y = rows.rbxt2_y.data();
  const double* ADMC_RESTRICT rbxt2_z = rows.rbxt2_z.data();
  const double* ADMC_RESTRICT TWn_a_x = rows.TWn_a_x.data();
  const double* ADMC_RESTRICT TWn_a_y = rows.TWn_a_y.data();
  const double* ADMC_RESTRICT TWn_a_z = rows.TWn_a_z.data();
  const double* ADMC_RESTRICT TWn_b_x = rows.TWn_b_x.data();
  const double* ADMC_RESTRICT TWn_b_y = rows.TWn_b_y.data();
  const double* ADMC_RESTRICT TWn_b_z = rows.TWn_b_z.data();
  const double* ADMC_RESTRICT TWt1_a_x = rows.TWt1_a_x.data();
  const double* ADMC_RESTRICT TWt1_a_y = rows.TWt1_a_y.data();
  const double* ADMC_RESTRICT TWt1_a_z = rows.TWt1_a_z.data();
  const double* ADMC_RESTRICT TWt1_b_x = rows.TWt1_b_x.data();
  const double* ADMC_RESTRICT TWt1_b_y = rows.TWt1_b_y.data();
  const double* ADMC_RESTRICT TWt1_b_z = rows.TWt1_b_z.data();
  const double* ADMC_RESTRICT TWt2_a_x = rows.TWt2_a_x.data();
  const double* ADMC_RESTRICT TWt2_a_y = rows.TWt2_a_y.data();
  const double* ADMC_RESTRICT TWt2_a_z = rows.TWt2_a_z.data();
  const double* ADMC_RESTRICT TWt2_b_x = rows.TWt2_b_x.data();
  const double* ADMC_RESTRICT TWt2_b_y = rows.TWt2_b_y.data();
  const double* ADMC_RESTRICT TWt2_b_z = rows.TWt2_b_z.data();

#if defined(__clang__)
#pragma clang loop vectorize(enable)
#endif
  for (std::size_t idx = 0; idx < tile.frictional.size(); ++idx) {
    const int contact_index = tile.frictional[idx];
    if (contact_index < 0 || contact_index >= static_cast<int>(contacts.size())) {
      continue;
    }
    const TileContactRef& ref = contacts[static_cast<std::size_t>(contact_index)];
    if (ref.local_a < 0 || ref.local_b < 0 ||
        ref.local_a >= static_cast<int>(bodies.size()) ||
        ref.local_b >= static_cast<int>(bodies.size())) {
      if (view.debug) {
        ++view.debug->invalid_contact_indices;
      }
      continue;
    }

    LocalBodyState& A = bodies[static_cast<std::size_t>(ref.local_a)];
    LocalBodyState& B = bodies[static_cast<std::size_t>(ref.local_b)];
    const int row = ref.row;

    double dvx = B.v.x - A.v.x;
    double dvy = B.v.y - A.v.y;
    double dvz = B.v.z - A.v.z;

    double wAx = A.w.x;
    double wAy = A.w.y;
    double wAz = A.w.z;
    double wBx = B.w.x;
    double wBy = B.w.y;
    double wBz = B.w.z;

    const double wA_dot_raxn =
        wAx * raxn_x[row] + wAy * raxn_y[row] + wAz * raxn_z[row];
    const double wB_dot_rbxn =
        wBx * rbxn_x[row] + wBy * rbxn_y[row] + wBz * rbxn_z[row];

    const double v_rel_n =
        nx[row] * dvx + ny[row] * dvy + nz[row] * dvz + wB_dot_rbxn - wA_dot_raxn;

    const double rhs = -(v_rel_n + bias[row] - bounce[row]);
    const double delta_jn = rhs * inv_k_n[row];
    const double jn_old = jn[row];
    double jn_candidate = jn_old + delta_jn;
    if (jn_candidate < 0.0) {
      if (view.debug) {
        ++view.debug->normal_impulse_clamps;
      }
      jn_candidate = 0.0;
    }
    jn[row] = jn_candidate;

    const double applied_n = jn_candidate - jn_old;
    if (std::fabs(applied_n) > math::kEps) {
      const double impulse_x = nx[row] * applied_n;
      const double impulse_y = ny[row] * applied_n;
      const double impulse_z = nz[row] * applied_n;
      const double inv_mass_sum = A.inv_mass + B.inv_mass;

      A.v.x -= impulse_x * A.inv_mass;
      A.v.y -= impulse_y * A.inv_mass;
      A.v.z -= impulse_z * A.inv_mass;
      B.v.x += impulse_x * B.inv_mass;
      B.v.y += impulse_y * B.inv_mass;
      B.v.z += impulse_z * B.inv_mass;

      dvx += impulse_x * inv_mass_sum;
      dvy += impulse_y * inv_mass_sum;
      dvz += impulse_z * inv_mass_sum;

      const double TWn_ax = TWn_a_x[row];
      const double TWn_ay = TWn_a_y[row];
      const double TWn_az = TWn_a_z[row];
      const double TWn_bx = TWn_b_x[row];
      const double TWn_by = TWn_b_y[row];
      const double TWn_bz = TWn_b_z[row];

      A.w.x -= applied_n * TWn_ax;
      A.w.y -= applied_n * TWn_ay;
      A.w.z -= applied_n * TWn_az;
      B.w.x += applied_n * TWn_bx;
      B.w.y += applied_n * TWn_by;
      B.w.z += applied_n * TWn_bz;

      wAx -= applied_n * TWn_ax;
      wAy -= applied_n * TWn_ay;
      wAz -= applied_n * TWn_az;
      wBx += applied_n * TWn_bx;
      wBy += applied_n * TWn_by;
      wBz += applied_n * TWn_bz;
    }

    if (mu[row] <= math::kEps) {
      jt1[row] = 0.0;
      jt2[row] = 0.0;
      continue;
    }

    const double wA_dot_raxt1 =
        wAx * raxt1_x[row] + wAy * raxt1_y[row] + wAz * raxt1_z[row];
    const double wB_dot_rbxt1 =
        wBx * rbxt1_x[row] + wBy * rbxt1_y[row] + wBz * rbxt1_z[row];
    const double wA_dot_raxt2 =
        wAx * raxt2_x[row] + wAy * raxt2_y[row] + wAz * raxt2_z[row];
    const double wB_dot_rbxt2 =
        wBx * rbxt2_x[row] + wBy * rbxt2_y[row] + wBz * rbxt2_z[row];

    const double v_rel_t1 = t1x[row] * dvx + t1y[row] * dvy + t1z[row] * dvz +
                            wB_dot_rbxt1 - wA_dot_raxt1;
    const double v_rel_t2 = t2x[row] * dvx + t2y[row] * dvy + t2z[row] * dvz +
                            wB_dot_rbxt2 - wA_dot_raxt2;

    double jt1_candidate = jt1[row] + (-v_rel_t1) * inv_k_t1[row];
    double jt2_candidate = jt2[row] + (-v_rel_t2) * inv_k_t2[row];

    const double friction_max = mu[row] * std::max(jn[row], 0.0);
    const double friction_max_sq = friction_max * friction_max;
    const double jt_mag_sq =
        jt1_candidate * jt1_candidate + jt2_candidate * jt2_candidate;
    double scale = 1.0;
    if (jt_mag_sq > friction_max_sq && jt_mag_sq > math::kEps * math::kEps) {
      const double jt_mag = std::sqrt(jt_mag_sq);
      scale = (friction_max > 0.0) ? (friction_max / jt_mag) : 0.0;
      if (view.debug) {
        ++view.debug->tangent_projections;
      }
    }

    jt1_candidate *= scale;
    jt2_candidate *= scale;
    const double delta_jt1 = jt1_candidate - jt1[row];
    const double delta_jt2 = jt2_candidate - jt2[row];
    jt1[row] = jt1_candidate;
    jt2[row] = jt2_candidate;

    if (std::fabs(delta_jt1) <= math::kEps &&
        std::fabs(delta_jt2) <= math::kEps) {
      continue;
    }

    const double impulse_x = delta_jt1 * t1x[row] + delta_jt2 * t2x[row];
    const double impulse_y = delta_jt1 * t1y[row] + delta_jt2 * t2y[row];
    const double impulse_z = delta_jt1 * t1z[row] + delta_jt2 * t2z[row];

    A.v.x -= impulse_x * A.inv_mass;
    A.v.y -= impulse_y * A.inv_mass;
    A.v.z -= impulse_z * A.inv_mass;
    B.v.x += impulse_x * B.inv_mass;
    B.v.y += impulse_y * B.inv_mass;
    B.v.z += impulse_z * B.inv_mass;

    A.w.x -= delta_jt1 * TWt1_a_x[row] + delta_jt2 * TWt2_a_x[row];
    A.w.y -= delta_jt1 * TWt1_a_y[row] + delta_jt2 * TWt2_a_y[row];
    A.w.z -= delta_jt1 * TWt1_a_z[row] + delta_jt2 * TWt2_a_z[row];
    B.w.x += delta_jt1 * TWt1_b_x[row] + delta_jt2 * TWt2_b_x[row];
    B.w.y += delta_jt1 * TWt1_b_y[row] + delta_jt2 * TWt2_b_y[row];
    B.w.z += delta_jt1 * TWt1_b_z[row] + delta_jt2 * TWt2_b_z[row];
  }
}

void solve_tile(TileWorkView& view, const Tile& tile) {
  if (!tile.normals_only.empty()) {
    solve_rows_normals_only(view, tile);
  }
  if (!tile.particles.empty()) {
    solve_rows_particles(view, tile);
  }
  if (!tile.frictional.empty()) {
    solve_rows_frictional(view, tile);
  }
}

double elapsed_ms(const Clock::time_point& begin, const Clock::time_point& end) {
  return DurationMs(end - begin).count();
}

Vec3 make_tangent(const Vec3& n) {
  if (std::fabs(n.x) < 0.57735026919) {
    return math::normalize_safe(math::cross(Vec3(1.0, 0.0, 0.0), n));
  }
  return math::normalize_safe(math::cross(Vec3(0.0, 1.0, 0.0), n));
}

Vec3 orthonormalize(const Vec3& n, const Vec3& t) {
  Vec3 tangent = t - math::dot(t, n) * n;
  tangent = math::normalize_safe(tangent);
  if (math::length2(tangent) <= math::kEps * math::kEps) {
    tangent = make_tangent(n);
  }
  return tangent;
}

SoaParams make_soa_params(const SolverParams& params) {
  SoaParams derived;
  static_cast<SolverParams&>(derived) = params;
  if (derived.thread_count <= 0) {
    derived.thread_count = 1;
  }
  if (derived.block_size <= 0) {
    derived.block_size = 1;
  }
  if (derived.tile_size <= 0) {
    derived.tile_size = 128;
  }
  if (derived.max_contacts_per_tile <= 0) {
    derived.max_contacts_per_tile = derived.tile_size;
  } else {
    derived.tile_size = derived.max_contacts_per_tile;
  }
  derived.max_contacts_per_tile = derived.tile_size;
  return derived;
}

}  // namespace

void solve_scalar_soa_simd(std::vector<RigidBody>& bodies,
                           std::vector<Contact>& contacts,
                           RowSOA& rows,
                           JointSOA& joints,
                           const SoaParams& params,
                           SolverDebugInfo* debug_info);

void solve_scalar_soa_simd(std::vector<RigidBody>& bodies,
                           std::vector<Contact>& contacts,
                           RowSOA& rows,
                           const SoaParams& params,
                           SolverDebugInfo* debug_info);

void solve_scalar_soa_mt(std::vector<RigidBody>& bodies,
                         std::vector<Contact>& contacts,
                         RowSOA& rows,
                         JointSOA& joints,
                         const SoaParams& params,
                         SolverDebugInfo* debug_info);

void solve_scalar_soa_mt(std::vector<RigidBody>& bodies,
                         std::vector<Contact>& contacts,
                         RowSOA& rows,
                         const SoaParams& params,
                         SolverDebugInfo* debug_info);

std::string solver_debug_summary(const SolverDebugInfo& info) {
  std::ostringstream oss;
  oss << "invalid_contacts=" << info.invalid_contact_indices
      << ", invalid_joints=" << info.invalid_joint_indices
      << ", warmstart_contacts=" << info.warmstart_contact_impulses
      << ", warmstart_joints=" << info.warmstart_joint_impulses
      << ", normal_clamps=" << info.normal_impulse_clamps
      << ", tangent_projections=" << info.tangent_projections
      << ", rope_clamps=" << info.rope_clamps
      << ", singular_joint_denoms=" << info.singular_joint_denominators
      << ", solver_ms=" << info.timings.solver_total_ms
      << ", warmstart_ms=" << info.timings.solver_warmstart_ms
      << ", iteration_ms=" << info.timings.solver_iterations_ms
      << ", integrate_ms=" << info.timings.solver_integrate_ms;
  return oss.str();
}

void build_soa(const std::vector<RigidBody>& bodies,
               const std::vector<Contact>& contacts,
               const SoaParams& params,
               RowSOA& rows) {
  const std::size_t capacity = contacts.size();
  if (rows.indices.size() < capacity) {
    auto resize_all = [&](std::size_t size) {
      rows.a.resize(size);
      rows.b.resize(size);
      rows.nx.resize(size);
      rows.ny.resize(size);
      rows.nz.resize(size);
      rows.t1x.resize(size);
      rows.t1y.resize(size);
      rows.t1z.resize(size);
      rows.t2x.resize(size);
      rows.t2y.resize(size);
      rows.t2z.resize(size);
      rows.rax.resize(size);
      rows.ray.resize(size);
      rows.raz.resize(size);
      rows.rbx.resize(size);
      rows.rby.resize(size);
      rows.rbz.resize(size);
      rows.raxn_x.resize(size);
      rows.raxn_y.resize(size);
      rows.raxn_z.resize(size);
      rows.rbxn_x.resize(size);
      rows.rbxn_y.resize(size);
      rows.rbxn_z.resize(size);
      rows.raxt1_x.resize(size);
      rows.raxt1_y.resize(size);
      rows.raxt1_z.resize(size);
      rows.rbxt1_x.resize(size);
      rows.rbxt1_y.resize(size);
      rows.rbxt1_z.resize(size);
      rows.raxt2_x.resize(size);
      rows.raxt2_y.resize(size);
      rows.raxt2_z.resize(size);
      rows.rbxt2_x.resize(size);
      rows.rbxt2_y.resize(size);
      rows.rbxt2_z.resize(size);
      rows.TWn_a_x.resize(size);
      rows.TWn_a_y.resize(size);
      rows.TWn_a_z.resize(size);
      rows.TWn_b_x.resize(size);
      rows.TWn_b_y.resize(size);
      rows.TWn_b_z.resize(size);
      rows.TWt1_a_x.resize(size);
      rows.TWt1_a_y.resize(size);
      rows.TWt1_a_z.resize(size);
      rows.TWt1_b_x.resize(size);
      rows.TWt1_b_y.resize(size);
      rows.TWt1_b_z.resize(size);
      rows.TWt2_a_x.resize(size);
      rows.TWt2_a_y.resize(size);
      rows.TWt2_a_z.resize(size);
      rows.TWt2_b_x.resize(size);
      rows.TWt2_b_y.resize(size);
      rows.TWt2_b_z.resize(size);
      rows.k_n.resize(size);
      rows.k_t1.resize(size);
      rows.k_t2.resize(size);
      rows.inv_k_n.resize(size);
      rows.inv_k_t1.resize(size);
      rows.inv_k_t2.resize(size);
      rows.mu.resize(size);
      rows.e.resize(size);
      rows.bias.resize(size);
      rows.bounce.resize(size);
      rows.C.resize(size);
      rows.jn.resize(size);
      rows.jt1.resize(size);
      rows.jt2.resize(size);
      rows.flags.resize(size);
      rows.types.resize(size);
      rows.indices.resize(size);
    };
    resize_all(capacity);
  }

  std::size_t write_index = 0;

  for (std::size_t i = 0; i < contacts.size(); ++i) {
    const Contact& c = contacts[i];
    if (c.a < 0 || c.b < 0 || c.a >= static_cast<int>(bodies.size()) ||
        c.b >= static_cast<int>(bodies.size())) {
      continue;
    }

    const RigidBody& A = bodies[c.a];
    const RigidBody& B = bodies[c.b];

    Vec3 n = c.n;
    Vec3 t1 = c.t1;
    Vec3 t2 = c.t2;
    if (math::length2(n) <= math::kEps * math::kEps) {
      n = Vec3(1.0, 0.0, 0.0);
    }
    if (math::length2(t1) <= math::kEps * math::kEps) {
      t1 = make_tangent(n);
    }
    if (math::length2(t2) <= math::kEps * math::kEps) {
      t2 = math::normalize_safe(math::cross(n, t1));
    }

    Vec3 ra = c.ra;
    Vec3 rb = c.rb;
    if (math::length2(ra) <= math::kEps * math::kEps) {
      ra = c.p - A.x;
    }
    if (math::length2(rb) <= math::kEps * math::kEps) {
      rb = c.p - B.x;
    }

    Vec3 ra_cross_n = c.ra_cross_n;
    if (math::length2(ra_cross_n) <= math::kEps * math::kEps) {
      ra_cross_n = math::cross(ra, n);
    }
    Vec3 rb_cross_n = c.rb_cross_n;
    if (math::length2(rb_cross_n) <= math::kEps * math::kEps) {
      rb_cross_n = math::cross(rb, n);
    }
    Vec3 ra_cross_t1 = c.ra_cross_t1;
    if (math::length2(ra_cross_t1) <= math::kEps * math::kEps) {
      ra_cross_t1 = math::cross(ra, t1);
    }
    Vec3 rb_cross_t1 = c.rb_cross_t1;
    if (math::length2(rb_cross_t1) <= math::kEps * math::kEps) {
      rb_cross_t1 = math::cross(rb, t1);
    }
    Vec3 ra_cross_t2 = c.ra_cross_t2;
    if (math::length2(ra_cross_t2) <= math::kEps * math::kEps) {
      ra_cross_t2 = math::cross(ra, t2);
    }
    Vec3 rb_cross_t2 = c.rb_cross_t2;
    if (math::length2(rb_cross_t2) <= math::kEps * math::kEps) {
      rb_cross_t2 = math::cross(rb, t2);
    }

    Vec3 TWn_a = c.TWn_a;
    if (math::length2(TWn_a) <= math::kEps * math::kEps) {
      TWn_a = A.invInertiaWorld * ra_cross_n;
    }
    Vec3 TWn_b = c.TWn_b;
    if (math::length2(TWn_b) <= math::kEps * math::kEps) {
      TWn_b = B.invInertiaWorld * rb_cross_n;
    }
    Vec3 TWt1_a = c.TWt1_a;
    if (math::length2(TWt1_a) <= math::kEps * math::kEps) {
      TWt1_a = A.invInertiaWorld * ra_cross_t1;
    }
    Vec3 TWt1_b = c.TWt1_b;
    if (math::length2(TWt1_b) <= math::kEps * math::kEps) {
      TWt1_b = B.invInertiaWorld * rb_cross_t1;
    }
    Vec3 TWt2_a = c.TWt2_a;
    if (math::length2(TWt2_a) <= math::kEps * math::kEps) {
      TWt2_a = A.invInertiaWorld * ra_cross_t2;
    }
    Vec3 TWt2_b = c.TWt2_b;
    if (math::length2(TWt2_b) <= math::kEps * math::kEps) {
      TWt2_b = B.invInertiaWorld * rb_cross_t2;
    }

    double k_n = c.k_n;
    if (k_n <= math::kEps) {
      k_n = A.invMass + B.invMass;
      k_n += math::dot(ra_cross_n, TWn_a) + math::dot(rb_cross_n, TWn_b);
      if (k_n <= math::kEps) {
        k_n = 1.0;
      }
    }
    double k_t1 = c.k_t1;
    if (k_t1 <= math::kEps) {
      k_t1 = A.invMass + B.invMass;
      k_t1 += math::dot(ra_cross_t1, TWt1_a) + math::dot(rb_cross_t1, TWt1_b);
      if (k_t1 <= math::kEps) {
        k_t1 = 1.0;
      }
    }
    double k_t2 = c.k_t2;
    if (k_t2 <= math::kEps) {
      k_t2 = A.invMass + B.invMass;
      k_t2 += math::dot(ra_cross_t2, TWt2_a) + math::dot(rb_cross_t2, TWt2_b);
      if (k_t2 <= math::kEps) {
        k_t2 = 1.0;
      }
    }

    const Vec3 va = A.v + math::cross(A.w, ra);
    const Vec3 vb = B.v + math::cross(B.w, rb);
    const double v_rel_n = math::dot(n, vb - va);
    const double restitution = std::max(c.e, params.restitution);
    const double bounce = (v_rel_n < 0.0) ? (-restitution * v_rel_n) : 0.0;
    const double bias = c.bias;
    const double mu = std::max(c.mu, params.mu);
    const double violation = (std::fabs(c.C) <= math::kEps) ? 0.0 : c.C;

    rows.indices[write_index] = static_cast<int>(i);
    rows.a[write_index] = c.a;
    rows.b[write_index] = c.b;
    rows.nx[write_index] = n.x;
    rows.ny[write_index] = n.y;
    rows.nz[write_index] = n.z;
    rows.t1x[write_index] = t1.x;
    rows.t1y[write_index] = t1.y;
    rows.t1z[write_index] = t1.z;
    rows.t2x[write_index] = t2.x;
    rows.t2y[write_index] = t2.y;
    rows.t2z[write_index] = t2.z;
    rows.rax[write_index] = ra.x;
    rows.ray[write_index] = ra.y;
    rows.raz[write_index] = ra.z;
    rows.rbx[write_index] = rb.x;
    rows.rby[write_index] = rb.y;
    rows.rbz[write_index] = rb.z;
    rows.raxn_x[write_index] = ra_cross_n.x;
    rows.raxn_y[write_index] = ra_cross_n.y;
    rows.raxn_z[write_index] = ra_cross_n.z;
    rows.rbxn_x[write_index] = rb_cross_n.x;
    rows.rbxn_y[write_index] = rb_cross_n.y;
    rows.rbxn_z[write_index] = rb_cross_n.z;
    rows.raxt1_x[write_index] = ra_cross_t1.x;
    rows.raxt1_y[write_index] = ra_cross_t1.y;
    rows.raxt1_z[write_index] = ra_cross_t1.z;
    rows.rbxt1_x[write_index] = rb_cross_t1.x;
    rows.rbxt1_y[write_index] = rb_cross_t1.y;
    rows.rbxt1_z[write_index] = rb_cross_t1.z;
    rows.raxt2_x[write_index] = ra_cross_t2.x;
    rows.raxt2_y[write_index] = ra_cross_t2.y;
    rows.raxt2_z[write_index] = ra_cross_t2.z;
    rows.rbxt2_x[write_index] = rb_cross_t2.x;
    rows.rbxt2_y[write_index] = rb_cross_t2.y;
    rows.rbxt2_z[write_index] = rb_cross_t2.z;
    rows.TWn_a_x[write_index] = TWn_a.x;
    rows.TWn_a_y[write_index] = TWn_a.y;
    rows.TWn_a_z[write_index] = TWn_a.z;
    rows.TWn_b_x[write_index] = TWn_b.x;
    rows.TWn_b_y[write_index] = TWn_b.y;
    rows.TWn_b_z[write_index] = TWn_b.z;
    rows.TWt1_a_x[write_index] = TWt1_a.x;
    rows.TWt1_a_y[write_index] = TWt1_a.y;
    rows.TWt1_a_z[write_index] = TWt1_a.z;
    rows.TWt1_b_x[write_index] = TWt1_b.x;
    rows.TWt1_b_y[write_index] = TWt1_b.y;
    rows.TWt1_b_z[write_index] = TWt1_b.z;
    rows.TWt2_a_x[write_index] = TWt2_a.x;
    rows.TWt2_a_y[write_index] = TWt2_a.y;
    rows.TWt2_a_z[write_index] = TWt2_a.z;
    rows.TWt2_b_x[write_index] = TWt2_b.x;
    rows.TWt2_b_y[write_index] = TWt2_b.y;
    rows.TWt2_b_z[write_index] = TWt2_b.z;
    rows.k_n[write_index] = k_n;
    rows.k_t1[write_index] = k_t1;
    rows.k_t2[write_index] = k_t2;
    rows.inv_k_n[write_index] = 1.0 / k_n;
    rows.inv_k_t1[write_index] = 1.0 / k_t1;
    rows.inv_k_t2[write_index] = 1.0 / k_t2;
    const bool allow_warm_start = params.warm_start && (violation < -params.slop);
    const bool allow_friction_warmstart = allow_warm_start && (mu > math::kEps);
    rows.jn[write_index] = allow_warm_start ? c.jn : 0.0;
    rows.jt1[write_index] = allow_friction_warmstart ? c.jt1 : 0.0;
    rows.jt2[write_index] = allow_friction_warmstart ? c.jt2 : 0.0;
    rows.mu[write_index] = mu;
    rows.e[write_index] = restitution;
    rows.bias[write_index] = bias;
    rows.bounce[write_index] = bounce;
    rows.C[write_index] = violation;
    std::uint8_t flags = 0;
    if (mu > math::kEps) {
      flags |= 0x1;
    }
    if (c.type == Contact::Type::kSphereSphere) {
      flags |= 0x2;
    }
    rows.flags[write_index] = flags;
    rows.types[write_index] = static_cast<std::uint8_t>(c.type);

    ++write_index;
  }

  rows.N = static_cast<int>(write_index);
}

RowSOA build_soa(const std::vector<RigidBody>& bodies,
                 const std::vector<Contact>& contacts,
                 const SolverParams& params) {
  return build_soa(bodies, contacts, make_soa_params(params));
}

RowSOA build_soa(const std::vector<RigidBody>& bodies,
                 const std::vector<Contact>& contacts,
                 const SoaParams& params) {
  RowSOA rows;
  build_soa(bodies, contacts, params, rows);
  return rows;
}

void build_soa(const std::vector<RigidBody>& bodies,
               const std::vector<Contact>& contacts,
               const SolverParams& params,
               RowSOA& rows) {
  build_soa(bodies, contacts, make_soa_params(params), rows);
}

void solve_scalar_soa_scalar(std::vector<RigidBody>& bodies,
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

  if (params.warm_start) {
    for (std::size_t i = 0; i < rows.size(); ++i) {
      const int idx = rows.indices[i];
      if (idx < 0 || idx >= static_cast<int>(contacts.size())) {
        continue;
      }
      Contact& c = contacts[static_cast<std::size_t>(idx)];
      const Vec3 curr_t1(rows.t1x[i], rows.t1y[i], rows.t1z[i]);
      const Vec3 curr_t2(rows.t2x[i], rows.t2y[i], rows.t2z[i]);
      const double prev_len_sq = math::length2(c.prev_t1);
      const double curr_len_sq = math::length2(curr_t1);
      if (prev_len_sq > math::kEps && curr_len_sq > math::kEps) {
        const double inv_len = 1.0 / std::sqrt(prev_len_sq * curr_len_sq);
        const double cos_angle = math::dot(c.prev_t1, curr_t1) * inv_len;
        if (cos_angle < kWarmstartRotationCosThreshold) {
          rows.jt1[i] = 0.0;
          rows.jt2[i] = 0.0;
          c.jt1 = 0.0;
          c.jt2 = 0.0;
        }
      }
      c.prev_t1 = curr_t1;
      c.prev_t2 = curr_t2;
    }
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
      const double impulse_x = rows.nx[i] * jn + rows.t1x[i] * jt1 +
                               rows.t2x[i] * jt2;
      const double impulse_y = rows.ny[i] * jn + rows.t1y[i] * jt1 +
                               rows.t2y[i] * jt2;
      const double impulse_z = rows.nz[i] * jn + rows.t1z[i] * jt1 +
                               rows.t2z[i] * jt2;
      if (debug_info) {
        ++debug_info->warmstart_contact_impulses;
      }

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

      RigidBody& A = bodies[ia];
      RigidBody& B = bodies[ib];
      const Vec3 impulse = joints.d[i] * joints.j[i];
      if (debug_info) {
        ++debug_info->warmstart_joint_impulses;
      }
      A.applyImpulse(-impulse, joints.ra[i]);
      B.applyImpulse(impulse, joints.rb[i]);
    }
  }

  if (debug_info) {
    const auto warmstart_end = Clock::now();
    debug_info->timings.solver_warmstart_ms +=
        elapsed_ms(warmstart_begin, warmstart_end);
  }

  auto solve_joint_iteration = [&](std::size_t i) {
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
  };

  const int tile_size =
      std::max(1, (params.tile_size > 0) ? params.tile_size : params.max_contacts_per_tile);
  std::vector<Tile> tiles = build_tiles(rows, bodies.size(), tile_size);
  TileSolveScratch scratch;
  TileWorkView view{&scratch, &bodies, &rows, debug_info};

  const auto iteration_begin = Clock::now();
  for (int it = 0; it < iterations; ++it) {
    for (const Tile& tile : tiles) {
      if (tile.contacts.empty()) {
        continue;
      }
      stage_tile(view, tile);
      solve_tile(view, tile);
      scatter_tile(scratch, bodies);
    }

    for (std::size_t i = 0; i < joints.size(); ++i) {
      solve_joint_iteration(i);
    }
  }

  for (int pass = 0; pass < 3; ++pass) {
    for (std::size_t i = 0; i < joints.size(); ++i) {
      solve_joint_iteration(i);
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
    c.prev_t1 = Vec3(rows.t1x[i], rows.t1y[i], rows.t1z[i]);
    c.prev_t2 = Vec3(rows.t2x[i], rows.t2y[i], rows.t2z[i]);
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

void solve_scalar_soa_scalar(std::vector<RigidBody>& bodies,
                             std::vector<Contact>& contacts,
                             RowSOA& rows,
                             const SoaParams& params,
                             SolverDebugInfo* debug_info) {
  static JointSOA empty_joints;
  empty_joints.clear();
  solve_scalar_soa_scalar(bodies, contacts, rows, empty_joints, params,
                          debug_info);
}

void solve_scalar_soa(std::vector<RigidBody>& bodies,
                      std::vector<Contact>& contacts,
                      RowSOA& rows,
                      JointSOA& joints,
                      const SoaParams& params,
                      SolverDebugInfo* debug_info) {
  SoaParams effective = params;
  if (effective.thread_count <= 0) {
    effective.thread_count = 1;
  }
  if (effective.block_size <= 0) {
    effective.block_size = 1;
  }

#if defined(ADMC_USE_THREADS)
  if (effective.use_threads && effective.thread_count > 1) {
    solve_scalar_soa_mt(bodies, contacts, rows, joints, effective, debug_info);
    return;
  }
#endif

#if defined(ADMC_USE_AVX2) || defined(ADMC_USE_NEON)
  if (effective.use_simd) {
    solve_scalar_soa_simd(bodies, contacts, rows, joints, effective, debug_info);
    return;
  }
#endif

  solve_scalar_soa_scalar(bodies, contacts, rows, joints, effective, debug_info);
}

void solve_scalar_soa(std::vector<RigidBody>& bodies,
                      std::vector<Contact>& contacts,
                      RowSOA& rows,
                      JointSOA& joints,
                      const SolverParams& params,
                      SolverDebugInfo* debug_info) {
  solve_scalar_soa(bodies, contacts, rows, joints, make_soa_params(params),
                   debug_info);
}

void solve_scalar_soa(std::vector<RigidBody>& bodies,
                      std::vector<Contact>& contacts,
                      RowSOA& rows,
                      const SoaParams& params,
                      SolverDebugInfo* debug_info) {
  SoaParams effective = params;
  if (effective.thread_count <= 0) {
    effective.thread_count = 1;
  }
  if (effective.block_size <= 0) {
    effective.block_size = 1;
  }

#if defined(ADMC_USE_THREADS)
  if (effective.use_threads && effective.thread_count > 1) {
    solve_scalar_soa_mt(bodies, contacts, rows, effective, debug_info);
    return;
  }
#endif

#if defined(ADMC_USE_AVX2) || defined(ADMC_USE_NEON)
  if (effective.use_simd) {
    solve_scalar_soa_simd(bodies, contacts, rows, effective, debug_info);
    return;
  }
#endif

  solve_scalar_soa_scalar(bodies, contacts, rows, effective, debug_info);
}

void solve_scalar_soa(std::vector<RigidBody>& bodies,
                      std::vector<Contact>& contacts,
                      RowSOA& rows,
                      const SolverParams& params,
                      SolverDebugInfo* debug_info) {
  solve_scalar_soa(bodies, contacts, rows, make_soa_params(params), debug_info);
}
