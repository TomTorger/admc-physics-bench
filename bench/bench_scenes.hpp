// bench/bench_scenes.hpp
#pragma once
#include <string>
#include <utility>
#include <vector>
#include <algorithm>

#include "bench_cli.hpp"

// Project scene factories & types (this is the correct header in your tree)
#include "scenes.hpp"

namespace bench {

// ---------------------------------------------------------------------------
// Scene construction helpers (mirrors the original monolith behavior)
// ---------------------------------------------------------------------------

inline bool make_scene_by_name(const std::string& name, Scene* scene) {
  if (!scene) return false;

  if (name == "two_spheres") {
    *scene = make_two_spheres_head_on();
  } else if (name == "spheres_cloud_1024") {
    *scene = make_spheres_box_cloud(1024);
  } else if (name == "spheres_cloud_4096") {
    *scene = make_spheres_box_cloud(4096);
  } else if (name == "spheres_cloud_8192") {
    *scene = make_spheres_box_cloud(8192);
  } else if (name == "spheres_cloud_10k") {
    *scene = make_spheres_box_cloud(10000);
  } else if (name == "spheres_cloud_50k") {
    *scene = make_spheres_box_cloud(50000);
  } else if (name == "spheres_cloud_10k_fric") {
    // Scene geometry equals 10k spheres; friction is enabled via solver params
    *scene = make_spheres_box_cloud(10000);
  } else if (name == "box_stack_4") {
    *scene = make_box_stack(4);
  } else if (name == "box_stack") {
    *scene = make_box_stack(8);
  } else if (name == "pendulum") {
    *scene = make_pendulum(1);
  } else if (name == "chain_64") {
    *scene = make_chain_64();
  } else if (name == "rope_256") {
    *scene = make_rope_256();
  } else {
    return false;
  }
  return true;
}

// Support parametric scene names like `--scene=spheres_cloud --sizes=1024,8192`
inline bool make_scene_with_size(const std::string& name,
                                 int size,
                                 Scene* scene,
                                 std::string* resolved_name) {
  if (!scene) return false;

  if (name == "spheres_cloud") {
    const int count = (size > 0) ? size : 1024;
    *scene = make_spheres_box_cloud(count);
    if (resolved_name) *resolved_name = "spheres_cloud_" + std::to_string(count);
    return true;
  }

  if (size > 0) {
    // Try a sized alias like "box_stack_8" or "spheres_cloud_4096"
    const std::string sized = name + "_" + std::to_string(size);
    if (make_scene_by_name(sized, scene)) {
      if (resolved_name) *resolved_name = sized;
      return true;
    }
  }

  if (make_scene_by_name(name, scene)) {
    if (resolved_name) *resolved_name = name;
    return true;
  }

  return false;
}

// ---------------------------------------------------------------------------
// Build the scene list from CLI config (used by bench_main.cc)
// ---------------------------------------------------------------------------
inline std::vector<std::pair<std::string, Scene>>
resolve_scenes_from_config(const BenchConfig& cfg) {
  struct Req { std::string name; int size; };

  // If user passed sizes, expand parametric scene (only first scene is used as the base)
  std::vector<Req> reqs;
  if (!cfg.sizes.empty() && !cfg.scenes.empty()) {
    for (int sz : cfg.sizes) {
      reqs.push_back({cfg.scenes.front(), sz});
    }
  } else {
    for (const auto& s : cfg.scenes) {
      reqs.push_back({s, -1});
    }
  }

  std::vector<std::pair<std::string, Scene>> out;
  out.reserve(reqs.size());

  for (const auto& r : reqs) {
    Scene scene;
    std::string label;
    if (!make_scene_with_size(r.name, r.size, &scene, &label)) {
      // best-effort: skip invalid names but continue others
      continue;
    }
    out.emplace_back(std::move(label), std::move(scene));
  }

  return out;
}

} // namespace bench
