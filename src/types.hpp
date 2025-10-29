#pragma once

#include "math.hpp"

#include <vector>
#include <cstdint>
#include <algorithm>

using math::Mat3;
using math::Quat;
using math::Vec3;

//! Rigid body state used by scalar solvers.
struct RigidBody {
  Vec3 x;                 //!< World-space position.
  Quat q;                 //!< Unit orientation (Hamilton).
  Vec3 v;                 //!< Linear velocity.
  Vec3 w;                 //!< Angular velocity.
  double invMass = 0.0;   //!< Inverse mass; zero indicates static.
  Mat3 invInertiaLocal;   //!< Inverse inertia tensor in local frame.
  Mat3 invInertiaWorld;   //!< Inverse inertia tensor in world frame.

  //! Constructs a static body at the origin.
  RigidBody();

  //! Synchronizes derived quantities using the current orientation.
  void syncDerived();

  //! Applies an impulse at offset r from the center of mass.
  void applyImpulse(const Vec3& P, const Vec3& r);

  //! Integrates velocities and orientation forward by dt.
  void integrate(double dt);
};

//! Contact point description between two bodies.
struct Contact {
  enum class Type : std::uint8_t {
    kUnknown = 0,
    kSphereSphere,
    kSpherePlane,
  };

  int a = -1;
  int b = -1;
  Vec3 p;      //!< World contact point.
  Vec3 n;      //!< Contact normal from body a to body b.
  Vec3 ra;     //!< Offset from body a COM to contact.
  Vec3 rb;     //!< Offset from body b COM to contact.
  Vec3 t1;     //!< First orthonormal tangent.
  Vec3 t2;     //!< Second orthonormal tangent.
  Vec3 ra_cross_n;
  Vec3 rb_cross_n;
  Vec3 ra_cross_t1;
  Vec3 rb_cross_t1;
  Vec3 ra_cross_t2;
  Vec3 rb_cross_t2;
  Vec3 TWn_a;
  Vec3 TWn_b;
  Vec3 TWt1_a;
  Vec3 TWt1_b;
  Vec3 TWt2_a;
  Vec3 TWt2_b;
  double e = 0.0;
  double mu = 0.0;
  double bias = 0.0;
  double bounce = 0.0; //!< Stored restitution target for scalar solvers.
  double C = 0.0; //!< Constraint violation (negative when penetrating).
  double jn = 0.0; //!< Warm-start accumulator for normal impulse.
  double jt1 = 0.0; //!< Warm-start accumulator for first friction tangent.
  double jt2 = 0.0; //!< Warm-start accumulator for second friction tangent.
  double k_n = 0.0;
  double k_t1 = 0.0;
  double k_t2 = 0.0;
  double radius_a = 0.0; //!< Optional cached radius for body a (spheres).
  double radius_b = 0.0; //!< Optional cached radius for body b (spheres).
  double plane_offset = 0.0; //!< Optional plane offset when involving a plane.
  Type type = Type::kUnknown;
};

struct RowSOA;
struct JointSOA;

//! Dual structure-of-arrays body storage used by SIMD solvers.
struct SolverBodySoA {
  std::vector<int> body_of_slot;   //!< Mapping from SoA slot to body index.
  std::vector<int> slot_of_body;   //!< Mapping from body index to SoA slot.

  std::vector<double> vx;
  std::vector<double> vy;
  std::vector<double> vz;
  std::vector<double> wx;
  std::vector<double> wy;
  std::vector<double> wz;

  //! Resets the storage to cover the bodies referenced by contacts / joints.
  void initialize(const std::vector<RigidBody>& bodies,
                  const RowSOA& rows,
                  const JointSOA& joints);

  //! Copies AoS velocities into the SoA buffers for active bodies.
  void load_from(const std::vector<RigidBody>& bodies);

  //! Flushes SoA velocities back to the underlying AoS body storage.
  void store_to(std::vector<RigidBody>& bodies) const;

  //! Returns the SoA slot for a body index or -1 if the body is inactive.
  int slot_for_body(int body_index) const {
    if (body_index < 0 || static_cast<std::size_t>(body_index) >= slot_of_body.size()) {
      return -1;
    }
    return slot_of_body[static_cast<std::size_t>(body_index)];
  }

  std::size_t size() const { return body_of_slot.size(); }
};


//! Structure-of-arrays representation of contacts for batched solves.
struct RowSOA {
  int N = 0;

  std::vector<int> a;
  std::vector<int> b;

  // Contact frames (unit vectors)
  std::vector<double> nx, ny, nz;
  std::vector<double> t1x, t1y, t1z;
  std::vector<double> t2x, t2y, t2z;

  // Offsets from center of mass to contact point
  std::vector<double> rax, ray, raz;
  std::vector<double> rbx, rby, rbz;

  // Cross terms r x d
  std::vector<double> raxn_x, raxn_y, raxn_z;
  std::vector<double> rbxn_x, rbxn_y, rbxn_z;
  std::vector<double> raxt1_x, raxt1_y, raxt1_z;
  std::vector<double> rbxt1_x, rbxt1_y, rbxt1_z;
  std::vector<double> raxt2_x, raxt2_y, raxt2_z;
  std::vector<double> rbxt2_x, rbxt2_y, rbxt2_z;

  // Inertia-multiplied cross terms TW = I^-1 * (r x d)
  std::vector<double> TWn_a_x, TWn_a_y, TWn_a_z;
  std::vector<double> TWn_b_x, TWn_b_y, TWn_b_z;
  std::vector<double> TWt1_a_x, TWt1_a_y, TWt1_a_z;
  std::vector<double> TWt1_b_x, TWt1_b_y, TWt1_b_z;
  std::vector<double> TWt2_a_x, TWt2_a_y, TWt2_a_z;
  std::vector<double> TWt2_b_x, TWt2_b_y, TWt2_b_z;

  // Effective masses per direction
  std::vector<double> k_n, k_t1, k_t2;
  // Cached reciprocals avoid divisions inside the solver hot loop.
  std::vector<double> inv_k_n, inv_k_t1, inv_k_t2;

  // Material / bias terms per contact
  std::vector<double> mu, e, bias, bounce, C;

  // Accumulated impulses (warm-start)
  std::vector<double> jn, jt1, jt2;

  std::vector<int> indices; //!< Mapping back to original contact indices.

  std::size_t size() const { return static_cast<std::size_t>(N); }

  void clear() {
    N = 0;
    a.clear();
    b.clear();
    nx.clear();
    ny.clear();
    nz.clear();
    t1x.clear();
    t1y.clear();
    t1z.clear();
    t2x.clear();
    t2y.clear();
    t2z.clear();
    rax.clear();
    ray.clear();
    raz.clear();
    rbx.clear();
    rby.clear();
    rbz.clear();
    raxn_x.clear();
    raxn_y.clear();
    raxn_z.clear();
    rbxn_x.clear();
    rbxn_y.clear();
    rbxn_z.clear();
    raxt1_x.clear();
    raxt1_y.clear();
    raxt1_z.clear();
    rbxt1_x.clear();
    rbxt1_y.clear();
    rbxt1_z.clear();
    raxt2_x.clear();
    raxt2_y.clear();
    raxt2_z.clear();
    rbxt2_x.clear();
    rbxt2_y.clear();
    rbxt2_z.clear();
    TWn_a_x.clear();
    TWn_a_y.clear();
    TWn_a_z.clear();
    TWn_b_x.clear();
    TWn_b_y.clear();
    TWn_b_z.clear();
    TWt1_a_x.clear();
    TWt1_a_y.clear();
    TWt1_a_z.clear();
    TWt1_b_x.clear();
    TWt1_b_y.clear();
    TWt1_b_z.clear();
    TWt2_a_x.clear();
    TWt2_a_y.clear();
    TWt2_a_z.clear();
    TWt2_b_x.clear();
    TWt2_b_y.clear();
    TWt2_b_z.clear();
    k_n.clear();
    k_t1.clear();
    k_t2.clear();
    inv_k_n.clear();
    inv_k_t1.clear();
    inv_k_t2.clear();
    mu.clear();
    e.clear();
    bias.clear();
    bounce.clear();
    C.clear();
    jn.clear();
    jt1.clear();
    jt2.clear();
    indices.clear();
  }

  void reserve(std::size_t capacity) {
    a.reserve(capacity);
    b.reserve(capacity);
    nx.reserve(capacity);
    ny.reserve(capacity);
    nz.reserve(capacity);
    t1x.reserve(capacity);
    t1y.reserve(capacity);
    t1z.reserve(capacity);
    t2x.reserve(capacity);
    t2y.reserve(capacity);
    t2z.reserve(capacity);
    rax.reserve(capacity);
    ray.reserve(capacity);
    raz.reserve(capacity);
    rbx.reserve(capacity);
    rby.reserve(capacity);
    rbz.reserve(capacity);
    raxn_x.reserve(capacity);
    raxn_y.reserve(capacity);
    raxn_z.reserve(capacity);
    rbxn_x.reserve(capacity);
    rbxn_y.reserve(capacity);
    rbxn_z.reserve(capacity);
    raxt1_x.reserve(capacity);
    raxt1_y.reserve(capacity);
    raxt1_z.reserve(capacity);
    rbxt1_x.reserve(capacity);
    rbxt1_y.reserve(capacity);
    rbxt1_z.reserve(capacity);
    raxt2_x.reserve(capacity);
    raxt2_y.reserve(capacity);
    raxt2_z.reserve(capacity);
    rbxt2_x.reserve(capacity);
    rbxt2_y.reserve(capacity);
    rbxt2_z.reserve(capacity);
    TWn_a_x.reserve(capacity);
    TWn_a_y.reserve(capacity);
    TWn_a_z.reserve(capacity);
    TWn_b_x.reserve(capacity);
    TWn_b_y.reserve(capacity);
    TWn_b_z.reserve(capacity);
    TWt1_a_x.reserve(capacity);
    TWt1_a_y.reserve(capacity);
    TWt1_a_z.reserve(capacity);
    TWt1_b_x.reserve(capacity);
    TWt1_b_y.reserve(capacity);
    TWt1_b_z.reserve(capacity);
    TWt2_a_x.reserve(capacity);
    TWt2_a_y.reserve(capacity);
    TWt2_a_z.reserve(capacity);
    TWt2_b_x.reserve(capacity);
    TWt2_b_y.reserve(capacity);
    TWt2_b_z.reserve(capacity);
    k_n.reserve(capacity);
    k_t1.reserve(capacity);
    k_t2.reserve(capacity);
    inv_k_n.reserve(capacity);
    inv_k_t1.reserve(capacity);
    inv_k_t2.reserve(capacity);
    mu.reserve(capacity);
    e.reserve(capacity);
    bias.reserve(capacity);
    bounce.reserve(capacity);
    C.reserve(capacity);
    jn.reserve(capacity);
    jt1.reserve(capacity);
    jt2.reserve(capacity);
    indices.reserve(capacity);
  }
};

// ----------------------- Inline implementation -----------------------

inline RigidBody::RigidBody()
    : x(0.0, 0.0, 0.0),
      q(),
      v(0.0, 0.0, 0.0),
      w(0.0, 0.0, 0.0),
      invMass(0.0),
      invInertiaLocal(),
      invInertiaWorld() {}

inline void RigidBody::syncDerived() {
  const Mat3 R = math::from_quat(q);
  invInertiaWorld = R * invInertiaLocal * R.transposed();
}

inline void RigidBody::applyImpulse(const Vec3& P, const Vec3& r) {
  v += P * invMass;
  const Vec3 torque = math::cross(r, P);
  w += invInertiaWorld * torque;
}

inline void RigidBody::integrate(double dt) {
  x += v * dt;

  const math::Vec3 omega = w;
  const math::Quat omega_q(0.0, omega.x, omega.y, omega.z);
  const math::Quat dq = omega_q * q;
  q.w += 0.5 * dq.w * dt;
  q.x += 0.5 * dq.x * dt;
  q.y += 0.5 * dq.y * dt;
  q.z += 0.5 * dq.z * dt;
  q.normalize();

  syncDerived();
}

