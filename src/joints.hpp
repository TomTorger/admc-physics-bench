#pragma once

#include "types.hpp"

#include <cstdint>
#include <vector>

struct DistanceJoint {
  int a = -1;
  int b = -1;
  Vec3 la;
  Vec3 lb;
  double rest = 1.0;
  double compliance = 0.0;
  double beta = 0.2;
  Vec3 pa;
  Vec3 pb;
  Vec3 d_hat;
  Vec3 ra;
  Vec3 rb;
  double C = 0.0;
  double jd = 0.0;
  bool rope = false;
};

void build_distance_joint_rows(const std::vector<RigidBody>& bodies,
                               std::vector<DistanceJoint>& joints,
                               double dt);

struct JointSOA {
  std::vector<int> a;
  std::vector<int> b;
  std::vector<Vec3> d;
  std::vector<Vec3> ra;
  std::vector<Vec3> rb;
  std::vector<double> k;
  std::vector<double> gamma;
  std::vector<double> bias;
  std::vector<double> j;
  std::vector<uint8_t> rope;
  std::vector<double> C;
  std::vector<int> indices;

  std::size_t size() const { return a.size(); }

  void clear() {
    a.clear();
    b.clear();
    d.clear();
    ra.clear();
    rb.clear();
    k.clear();
    gamma.clear();
    bias.clear();
    j.clear();
    rope.clear();
    C.clear();
    indices.clear();
  }

  void reserve(std::size_t capacity) {
    a.reserve(capacity);
    b.reserve(capacity);
    d.reserve(capacity);
    ra.reserve(capacity);
    rb.reserve(capacity);
    k.reserve(capacity);
    gamma.reserve(capacity);
    bias.reserve(capacity);
    j.reserve(capacity);
    rope.reserve(capacity);
    C.reserve(capacity);
    indices.reserve(capacity);
  }
};

JointSOA build_joint_soa(const std::vector<RigidBody>& bodies,
                         const std::vector<DistanceJoint>& joints,
                         double dt);

void build_joint_soa(const std::vector<RigidBody>& bodies,
                     const std::vector<DistanceJoint>& joints,
                     double dt,
                     JointSOA& rows);

void scatter_joint_impulses(const JointSOA& joint_rows,
                            std::vector<DistanceJoint>& joints);

