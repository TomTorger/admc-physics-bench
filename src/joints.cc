#include "joints.hpp"

#include <algorithm>

namespace {
constexpr double kDirEps = 1e-9;
constexpr double kJointBiasBoost = 1.05;
}

void build_distance_joint_rows(const std::vector<RigidBody>& bodies,
                               std::vector<DistanceJoint>& joints,
                               double /*dt*/) {
  for (DistanceJoint& joint : joints) {
    if (joint.a < 0 || joint.b < 0 || joint.a >= static_cast<int>(bodies.size()) ||
        joint.b >= static_cast<int>(bodies.size())) {
      continue;
    }

    const RigidBody& A = bodies[static_cast<std::size_t>(joint.a)];
    const RigidBody& B = bodies[static_cast<std::size_t>(joint.b)];

    const Vec3 world_la = A.q.rotate(joint.la);
    const Vec3 world_lb = B.q.rotate(joint.lb);

    joint.pa = A.x + world_la;
    joint.pb = B.x + world_lb;
    joint.ra = joint.pa - A.x;
    joint.rb = joint.pb - B.x;

    const Vec3 delta = joint.pb - joint.pa;
    const double dist = math::length(delta);
    Vec3 dir = joint.d_hat;
    if (dist > kDirEps) {
      dir = delta / std::max(dist, kDirEps);
    } else if (math::length2(dir) <= math::kEps * math::kEps) {
      dir = Vec3(1.0, 0.0, 0.0);
    }
    joint.d_hat = math::normalize_safe(dir);
    joint.C = dist - joint.rest;
  }
}

void build_joint_soa(const std::vector<RigidBody>& bodies,
                     const std::vector<DistanceJoint>& joints,
                     double dt,
                     JointSOA& rows) {
  rows.clear();
  rows.reserve(joints.size());

  const double dt_sq = (dt > math::kEps) ? (dt * dt) : 0.0;
  const double inv_dt = (dt > math::kEps) ? (1.0 / dt) : 0.0;

  for (std::size_t i = 0; i < joints.size(); ++i) {
    const DistanceJoint& joint = joints[i];
    if (joint.a < 0 || joint.b < 0 || joint.a >= static_cast<int>(bodies.size()) ||
        joint.b >= static_cast<int>(bodies.size())) {
      continue;
    }

    const RigidBody& A = bodies[static_cast<std::size_t>(joint.a)];
    const RigidBody& B = bodies[static_cast<std::size_t>(joint.b)];

    Vec3 dir = math::normalize_safe(joint.d_hat);
    if (math::length2(dir) <= math::kEps * math::kEps) {
      dir = Vec3(1.0, 0.0, 0.0);
    }

    const Vec3 ra = joint.ra;
    const Vec3 rb = joint.rb;
    const Vec3 ra_cross_d = math::cross(ra, dir);
    const Vec3 rb_cross_d = math::cross(rb, dir);

    double k = A.invMass + B.invMass;
    k += math::dot(ra_cross_d, A.invInertiaWorld * ra_cross_d);
    k += math::dot(rb_cross_d, B.invInertiaWorld * rb_cross_d);
    if (k <= math::kEps) {
      k = 1.0;
    }

    double gamma = 0.0;
    if (dt_sq > 0.0) {
      gamma = joint.compliance / dt_sq;
    }

    double bias = 0.0;
    if (dt > math::kEps) {
      bias = (joint.beta * inv_dt) * joint.C * kJointBiasBoost;
    }

    rows.indices.push_back(static_cast<int>(i));
    rows.a.push_back(joint.a);
    rows.b.push_back(joint.b);
    rows.d.push_back(dir);
    rows.ra.push_back(ra);
    rows.rb.push_back(rb);
    rows.k.push_back(k);
    rows.gamma.push_back(gamma);
    rows.bias.push_back(bias);
    rows.j.push_back(joint.jd);
    rows.rope.push_back(static_cast<uint8_t>(joint.rope ? 1 : 0));
    rows.C.push_back(joint.C);
    rows.rest.push_back(joint.rest);
    rows.beta.push_back(joint.beta);
  }
}

JointSOA build_joint_soa(const std::vector<RigidBody>& bodies,
                         const std::vector<DistanceJoint>& joints,
                         double dt) {
  JointSOA rows;
  build_joint_soa(bodies, joints, dt, rows);
  return rows;
}

void scatter_joint_impulses(const JointSOA& joint_rows,
                            std::vector<DistanceJoint>& joints) {
  for (std::size_t i = 0; i < joint_rows.size(); ++i) {
    const int idx = joint_rows.indices[i];
    if (idx < 0 || idx >= static_cast<int>(joints.size())) {
      continue;
    }
    joints[static_cast<std::size_t>(idx)].jd = joint_rows.j[i];
  }
}

