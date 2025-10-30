#pragma once

#include "soa/world.hpp"

#include <vector>

namespace soa {

template <typename T>
using AlignedVec = SoaAlignedVector<T>;

struct Tile {
  AlignedVec<int> bodyA;
  AlignedVec<int> bodyB;
  AlignedVec<float> n_x, n_y, n_z;
  AlignedVec<float> rAx, rAy, rAz;
  AlignedVec<float> rBx, rBy, rBz;
  AlignedVec<float> k_n;
  AlignedVec<float> inv_k_n;
  AlignedVec<float> target_n;
  AlignedVec<float> j_n;

  struct LocalBodies {
    std::vector<int> globalId;
    AlignedVec<float> invMass;
    AlignedVec<float> vx;
    AlignedVec<float> vy;
    AlignedVec<float> vz;

    int localBodyCount() const {
      return static_cast<int>(globalId.size());
    }

    void resize(int count) {
      globalId.resize(static_cast<std::size_t>(count));
      invMass.resize(static_cast<std::size_t>(count));
      vx.resize(static_cast<std::size_t>(count));
      vy.resize(static_cast<std::size_t>(count));
      vz.resize(static_cast<std::size_t>(count));
    }

    void clear() {
      globalId.clear();
      invMass.clear();
      vx.clear();
      vy.clear();
      vz.clear();
    }
  } bodies;

  int size() const { return static_cast<int>(bodyA.size()); }
  int localBodyCount() const {
    return static_cast<int>(bodies.globalId.size());
  }

  std::vector<int> contactIndices;

  void resize(int rows) {
    bodyA.resize(static_cast<std::size_t>(rows));
    bodyB.resize(static_cast<std::size_t>(rows));
    n_x.resize(static_cast<std::size_t>(rows));
    n_y.resize(static_cast<std::size_t>(rows));
    n_z.resize(static_cast<std::size_t>(rows));
    rAx.resize(static_cast<std::size_t>(rows));
    rAy.resize(static_cast<std::size_t>(rows));
    rAz.resize(static_cast<std::size_t>(rows));
    rBx.resize(static_cast<std::size_t>(rows));
    rBy.resize(static_cast<std::size_t>(rows));
    rBz.resize(static_cast<std::size_t>(rows));
    k_n.resize(static_cast<std::size_t>(rows));
    inv_k_n.resize(static_cast<std::size_t>(rows));
    target_n.resize(static_cast<std::size_t>(rows));
    j_n.resize(static_cast<std::size_t>(rows));
    contactIndices.resize(static_cast<std::size_t>(rows));
  }
};

struct BuildTilesParams {
  int maxTileRows = 128;
  bool spheresOnly = false;
  bool frictionless = false;
  bool warmStart = true;
  double restitution = 0.0;
  double slop = 0.0;
};

std::vector<Tile> build_tiles_from_contacts(const World& world,
                                            const ContactManifold& cm,
                                            const BuildTilesParams& params);

void flush_tile(const Tile& tile, World& world, ContactManifold& cm);

}  // namespace soa

