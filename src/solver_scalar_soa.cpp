#include "solver_scalar_soa.hpp"

#include "soa/tiler.hpp"
#include "solver_scalar_soa_spheres.hpp"

#include <vector>

namespace soa {

void solve_soa(World& world, ContactManifold& cm, const SolverParams& params) {
  if (!world.valid() || !cm.valid()) {
    return;
  }

  BuildTilesParams build_params;
  build_params.maxTileRows = (params.tile_rows > 0) ? params.tile_rows : params.tile_size;
  if (build_params.maxTileRows <= 0) {
    build_params.maxTileRows = 128;
  }
  build_params.spheresOnly = params.spheres_only;
  build_params.frictionless = params.frictionless;
  build_params.warmStart = params.warm_start;
  build_params.restitution = params.restitution;
  build_params.slop = params.slop;

  if (!(build_params.spheresOnly && build_params.frictionless)) {
    RowSOA rows;
    build_soa(world.bodies(), cm.contacts(), params, rows);
    ::solve_scalar_soa(world.bodies(), cm.contacts(), rows, params);
    return;
  }

  std::vector<Tile> tiles = build_tiles_from_contacts(world, cm, build_params);
  if (tiles.empty()) {
    return;
  }

  for (auto& tile : tiles) {
    solve_tile_normals_spheres(tile, params.iterations);
    flush_tile(tile, world, cm);
  }
}

}  // namespace soa

