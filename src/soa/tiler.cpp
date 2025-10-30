#include "soa/tiler.hpp"

#include <algorithm>
#include <numeric>
#include <unordered_map>

namespace soa {
namespace {

int find_or_add_local(int global,
                      std::vector<int>& map,
                      std::vector<int>& touched,
                      Tile::LocalBodies& bodies,
                      const World& world) {
  if (global < 0 || global >= static_cast<int>(map.size())) {
    return -1;
  }
  int& entry = map[static_cast<std::size_t>(global)];
  if (entry >= 0) {
    return entry;
  }
  entry = bodies.localBodyCount();
  touched.push_back(global);
  bodies.globalId.push_back(global);
  const RigidBody& body = world.body(global);
  bodies.invMass.push_back(static_cast<float>(body.invMass));
  bodies.vx.push_back(static_cast<float>(body.v.x));
  bodies.vy.push_back(static_cast<float>(body.v.y));
  bodies.vz.push_back(static_cast<float>(body.v.z));
  return entry;
}

}  // namespace

std::vector<Tile> build_tiles_from_contacts(const World& world,
                                            const ContactManifold& cm,
                                            const BuildTilesParams& params) {
  std::vector<Tile> result;
  if (!world.valid() || !cm.valid()) {
    return result;
  }

  const int body_count = world.bodyCount();
  const int contact_count = cm.size();
  if (contact_count <= 0 || params.maxTileRows <= 0) {
    return result;
  }

  std::vector<int> parent(static_cast<std::size_t>(body_count));
  std::vector<int> rank(static_cast<std::size_t>(body_count), 0);
  std::iota(parent.begin(), parent.end(), 0);

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

  const auto& contacts = cm.contacts();
  for (int i = 0; i < contact_count; ++i) {
    const Contact& c = contacts[static_cast<std::size_t>(i)];
    if (c.a >= 0 && c.b >= 0 && c.a < body_count && c.b < body_count) {
      union_sets(c.a, c.b);
    }
  }

  std::unordered_map<int, std::vector<int>> islands;
  islands.reserve(static_cast<std::size_t>(contact_count));
  for (int i = 0; i < contact_count; ++i) {
    const Contact& c = contacts[static_cast<std::size_t>(i)];
    int root = -1;
    if (c.a >= 0 && c.a < body_count) {
      root = find_set(c.a);
    } else if (c.b >= 0 && c.b < body_count) {
      root = find_set(c.b);
    }
    if (root >= 0) {
      islands[root].push_back(i);
    }
  }

  std::vector<int> local_map(static_cast<std::size_t>(body_count), -1);
  std::vector<int> touched_globals;

  for (auto& entry : islands) {
    std::vector<int>& rows = entry.second;
    for (std::size_t start = 0; start < rows.size();
         start += static_cast<std::size_t>(params.maxTileRows)) {
      const std::size_t end =
          std::min(rows.size(), start + static_cast<std::size_t>(params.maxTileRows));
      const int row_count = static_cast<int>(end - start);
      Tile tile;
      tile.resize(row_count);
      tile.bodies.clear();
      touched_globals.clear();

      for (int local = 0; local < row_count; ++local) {
        const int contact_index = rows[static_cast<std::size_t>(start) +
                                      static_cast<std::size_t>(local)];
        tile.contactIndices[static_cast<std::size_t>(local)] = contact_index;
        const Contact& contact = contacts[static_cast<std::size_t>(contact_index)];

        const int local_a = find_or_add_local(contact.a, local_map, touched_globals,
                                             tile.bodies, world);
        const int local_b = find_or_add_local(contact.b, local_map, touched_globals,
                                             tile.bodies, world);
        tile.bodyA[static_cast<std::size_t>(local)] = local_a;
        tile.bodyB[static_cast<std::size_t>(local)] = local_b;

        const float nx_val = static_cast<float>(contact.n.x);
        const float ny_val = static_cast<float>(contact.n.y);
        const float nz_val = static_cast<float>(contact.n.z);
        tile.n_x[static_cast<std::size_t>(local)] = nx_val;
        tile.n_y[static_cast<std::size_t>(local)] = ny_val;
        tile.n_z[static_cast<std::size_t>(local)] = nz_val;

        tile.rAx[static_cast<std::size_t>(local)] =
            static_cast<float>(contact.ra.x);
        tile.rAy[static_cast<std::size_t>(local)] =
            static_cast<float>(contact.ra.y);
        tile.rAz[static_cast<std::size_t>(local)] =
            static_cast<float>(contact.ra.z);

        tile.rBx[static_cast<std::size_t>(local)] =
            static_cast<float>(contact.rb.x);
        tile.rBy[static_cast<std::size_t>(local)] =
            static_cast<float>(contact.rb.y);
        tile.rBz[static_cast<std::size_t>(local)] =
            static_cast<float>(contact.rb.z);

        float inv_mass_a = 0.0f;
        float inv_mass_b = 0.0f;
        if (local_a >= 0) {
          inv_mass_a = tile.bodies.invMass[static_cast<std::size_t>(local_a)];
        }
        if (local_b >= 0) {
          inv_mass_b = tile.bodies.invMass[static_cast<std::size_t>(local_b)];
        }
        float kn = inv_mass_a + inv_mass_b;
        if (kn <= 0.0f) {
          kn = 1.0f;
        }
        tile.k_n[static_cast<std::size_t>(local)] = kn;

        double vax0 = 0.0;
        double vay0 = 0.0;
        double vaz0 = 0.0;
        if (local_a >= 0) {
          vax0 = static_cast<double>(tile.bodies.vx[static_cast<std::size_t>(local_a)]);
          vay0 = static_cast<double>(tile.bodies.vy[static_cast<std::size_t>(local_a)]);
          vaz0 = static_cast<double>(tile.bodies.vz[static_cast<std::size_t>(local_a)]);
        }
        double vbx0 = 0.0;
        double vby0 = 0.0;
        double vbz0 = 0.0;
        if (local_b >= 0) {
          vbx0 = static_cast<double>(tile.bodies.vx[static_cast<std::size_t>(local_b)]);
          vby0 = static_cast<double>(tile.bodies.vy[static_cast<std::size_t>(local_b)]);
          vbz0 = static_cast<double>(tile.bodies.vz[static_cast<std::size_t>(local_b)]);
        }

        const double v_rel_n_initial =
            (vbx0 - vax0) * static_cast<double>(nx_val) +
            (vby0 - vay0) * static_cast<double>(ny_val) +
            (vbz0 - vaz0) * static_cast<double>(nz_val);
        const double restitution = std::max(contact.e, params.restitution);
        const double bounce =
            (v_rel_n_initial < 0.0) ? (-restitution * v_rel_n_initial) : 0.0;
        tile.target_n[static_cast<std::size_t>(local)] =
            static_cast<float>(-(contact.bias - bounce));

        const bool allow_warm_start =
            params.warmStart && (contact.C < -params.slop);
        const float warm_start_impulse =
            allow_warm_start ? static_cast<float>(contact.jn) : 0.0f;
        tile.j_n[static_cast<std::size_t>(local)] = warm_start_impulse;

        if (allow_warm_start && warm_start_impulse != 0.0f) {
          const float jx = warm_start_impulse * tile.n_x[static_cast<std::size_t>(local)];
          const float jy = warm_start_impulse * tile.n_y[static_cast<std::size_t>(local)];
          const float jz = warm_start_impulse * tile.n_z[static_cast<std::size_t>(local)];
          if (local_a >= 0 && inv_mass_a > 0.0f) {
            tile.bodies.vx[static_cast<std::size_t>(local_a)] -= inv_mass_a * jx;
            tile.bodies.vy[static_cast<std::size_t>(local_a)] -= inv_mass_a * jy;
            tile.bodies.vz[static_cast<std::size_t>(local_a)] -= inv_mass_a * jz;
          }
          if (local_b >= 0 && inv_mass_b > 0.0f) {
            tile.bodies.vx[static_cast<std::size_t>(local_b)] += inv_mass_b * jx;
            tile.bodies.vy[static_cast<std::size_t>(local_b)] += inv_mass_b * jy;
            tile.bodies.vz[static_cast<std::size_t>(local_b)] += inv_mass_b * jz;
          }
        }
      }

      for (int g : touched_globals) {
        local_map[static_cast<std::size_t>(g)] = -1;
      }
      touched_globals.clear();

      result.push_back(std::move(tile));
    }
  }

  return result;
}

void flush_tile(const Tile& tile, World& world, ContactManifold& cm) {
  if (!world.valid() || !cm.valid()) {
    return;
  }

  auto& bodies = world.bodies();
  for (int i = 0; i < tile.localBodyCount(); ++i) {
    const int global = tile.bodies.globalId[static_cast<std::size_t>(i)];
    if (global < 0 || global >= static_cast<int>(bodies.size())) {
      continue;
    }
    RigidBody& body = bodies[static_cast<std::size_t>(global)];
    body.v.x = static_cast<double>(tile.bodies.vx[static_cast<std::size_t>(i)]);
    body.v.y = static_cast<double>(tile.bodies.vy[static_cast<std::size_t>(i)]);
    body.v.z = static_cast<double>(tile.bodies.vz[static_cast<std::size_t>(i)]);
  }

  auto& contacts = cm.contacts();
  for (int row = 0; row < tile.size(); ++row) {
    const int index = tile.contactIndices[static_cast<std::size_t>(row)];
    if (index < 0 || index >= static_cast<int>(contacts.size())) {
      continue;
    }
    contacts[static_cast<std::size_t>(index)].jn =
        static_cast<double>(tile.j_n[static_cast<std::size_t>(row)]);
  }
}

}  // namespace soa

