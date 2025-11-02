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
  std::vector<double> rest;
  std::vector<double> beta;
  std::vector<int> indices;

  std::size_t size() const { return a.size(); }

  void ensure_capacity(std::size_t capacity) {
    if (a.capacity() < capacity) {
      reserve(capacity);
    }
  }

  void clear_but_keep_capacity() {
    a.resize(0);
    b.resize(0);
    d.resize(0);
    ra.resize(0);
    rb.resize(0);
    k.resize(0);
    gamma.resize(0);
    bias.resize(0);
    j.resize(0);
    rope.resize(0);
    C.resize(0);
    rest.resize(0);
    beta.resize(0);
    indices.resize(0);
  }

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
    rest.clear();
    beta.clear();
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
    rest.reserve(capacity);
    beta.reserve(capacity);
    indices.reserve(capacity);
  }

  void resize(std::size_t size) {
    a.resize(size);
    b.resize(size);
    d.resize(size);
    ra.resize(size);
    rb.resize(size);
    k.resize(size);
    gamma.resize(size);
    bias.resize(size);
    j.resize(size);
    rope.resize(size);
    C.resize(size);
    rest.resize(size);
    beta.resize(size);
    indices.resize(size);
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

