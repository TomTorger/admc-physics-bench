#include "metrics.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace {
std::array<math::Vec3, 10> make_directions() {
  const std::array<math::Vec3, 10> raw = {
      math::Vec3(1.0, 0.0, 0.0), math::Vec3(-1.0, 0.0, 0.0),
      math::Vec3(0.0, 1.0, 0.0), math::Vec3(0.0, -1.0, 0.0),
      math::Vec3(0.0, 0.0, 1.0), math::Vec3(0.0, 0.0, -1.0),
      math::Vec3(0.531176, -0.847845, 0.013421),
      math::Vec3(-0.271321, 0.349121, 0.897932),
      math::Vec3(0.713512, 0.204913, -0.669324),
      math::Vec3(-0.402156, -0.551249, -0.730441)};

  std::array<math::Vec3, 10> dirs = raw;
  for (math::Vec3& d : dirs) {
    d = math::normalize_safe(d);
  }
  return dirs;
}

const std::array<math::Vec3, 10> kDirections = make_directions();

inline double directional_sum(const std::vector<RigidBody>& bodies,
                              const math::Vec3& dir) {
  double sum = 0.0;
  for (const RigidBody& b : bodies) {
    if (b.invMass <= math::kEps) {
      continue;
    }
    const double mass = 1.0 / b.invMass;
    const math::Vec3 momentum = b.v * mass;
    sum += math::dot(momentum, dir);
  }
  return sum;
}
}  // namespace

Drift directional_momentum_drift(const std::vector<RigidBody>& pre,
                                 const std::vector<RigidBody>& post) {
  Drift drift;
  for (const math::Vec3& dir : kDirections) {
    const double before = directional_sum(pre, dir);
    const double after = directional_sum(post, dir);
    drift.max_abs = std::max(drift.max_abs, std::fabs(after - before));
  }
  return drift;
}

double constraint_penetration_Linf(const std::vector<Contact>& contacts) {
  double max_pen = 0.0;
  for (const Contact& c : contacts) {
    max_pen = std::max(max_pen, std::fabs(c.penetration));
  }
  return max_pen;
}

