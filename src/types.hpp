#pragma once

#include "math.hpp"
#include "platform.hpp"

#include <cstdint>
#include <memory>
#include <new>
#include <vector>

template <typename T, std::size_t Alignment>
struct AlignedAllocator {
  using value_type = T;

  AlignedAllocator() noexcept = default;

  template <class U>
  constexpr AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

  [[nodiscard]] T* allocate(std::size_t n) {
    if (n == 0) {
      return nullptr;
    }
    void* ptr = admc_aligned_alloc(n * sizeof(T), Alignment);
    return static_cast<T*>(ptr);
  }

  void deallocate(T* p, std::size_t) noexcept {
    if (!p) {
      return;
    }
    admc_aligned_free(p);
  }

  template <typename U>
  struct rebind {
    using other = AlignedAllocator<U, Alignment>;
  };
};

template <typename T, typename U, std::size_t Alignment>
bool operator==(const AlignedAllocator<T, Alignment>&,
                const AlignedAllocator<U, Alignment>&) noexcept {
  return true;
}

template <typename T, typename U, std::size_t Alignment>
bool operator!=(const AlignedAllocator<T, Alignment>&,
                const AlignedAllocator<U, Alignment>&) noexcept {
  return false;
}

template <typename T>
using SoaAlignedVector = std::vector<T, AlignedAllocator<T, 64>>;

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
  Vec3 prev_t1;
  Vec3 prev_t2;
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

//! Structure-of-arrays representation of contacts for batched solves.
struct RowSOA {
  int N = 0;

  std::vector<int> a;
  std::vector<int> b;

  // Contact frames (unit vectors)
  SoaAlignedVector<double> nx, ny, nz;
  SoaAlignedVector<double> t1x, t1y, t1z;
  SoaAlignedVector<double> t2x, t2y, t2z;

  // Offsets from center of mass to contact point
  SoaAlignedVector<double> rax, ray, raz;
  SoaAlignedVector<double> rbx, rby, rbz;

  // Cross terms r x d
  SoaAlignedVector<double> raxn_x, raxn_y, raxn_z;
  SoaAlignedVector<double> rbxn_x, rbxn_y, rbxn_z;
  SoaAlignedVector<double> raxt1_x, raxt1_y, raxt1_z;
  SoaAlignedVector<double> rbxt1_x, rbxt1_y, rbxt1_z;
  SoaAlignedVector<double> raxt2_x, raxt2_y, raxt2_z;
  SoaAlignedVector<double> rbxt2_x, rbxt2_y, rbxt2_z;

  // Inertia-multiplied cross terms TW = I^-1 * (r x d)
  SoaAlignedVector<double> TWn_a_x, TWn_a_y, TWn_a_z;
  SoaAlignedVector<double> TWn_b_x, TWn_b_y, TWn_b_z;
  SoaAlignedVector<double> TWt1_a_x, TWt1_a_y, TWt1_a_z;
  SoaAlignedVector<double> TWt1_b_x, TWt1_b_y, TWt1_b_z;
  SoaAlignedVector<double> TWt2_a_x, TWt2_a_y, TWt2_a_z;
  SoaAlignedVector<double> TWt2_b_x, TWt2_b_y, TWt2_b_z;

  // Effective masses per direction
  SoaAlignedVector<double> k_n, k_t1, k_t2;
  // Cached reciprocals avoid divisions inside the solver hot loop.
  SoaAlignedVector<double> inv_k_n, inv_k_t1, inv_k_t2;

  // Material / bias terms per contact
  SoaAlignedVector<double> mu, e, bias, bounce, C;

  // Accumulated impulses (warm-start)
  SoaAlignedVector<double> jn, jt1, jt2;

  std::vector<std::uint8_t> flags;
  std::vector<std::uint8_t> types;
  std::vector<int> indices; //!< Mapping back to original contact indices.

  std::size_t size() const { return static_cast<std::size_t>(N); }

  void ensure_capacity(std::size_t capacity) {
    if (a.capacity() < capacity) {
      reserve(capacity);
    }
  }

  void clear_but_keep_capacity() {
    N = 0;
    a.resize(0);
    b.resize(0);
    nx.resize(0);
    ny.resize(0);
    nz.resize(0);
    t1x.resize(0);
    t1y.resize(0);
    t1z.resize(0);
    t2x.resize(0);
    t2y.resize(0);
    t2z.resize(0);
    rax.resize(0);
    ray.resize(0);
    raz.resize(0);
    rbx.resize(0);
    rby.resize(0);
    rbz.resize(0);
    raxn_x.resize(0);
    raxn_y.resize(0);
    raxn_z.resize(0);
    rbxn_x.resize(0);
    rbxn_y.resize(0);
    rbxn_z.resize(0);
    raxt1_x.resize(0);
    raxt1_y.resize(0);
    raxt1_z.resize(0);
    rbxt1_x.resize(0);
    rbxt1_y.resize(0);
    rbxt1_z.resize(0);
    raxt2_x.resize(0);
    raxt2_y.resize(0);
    raxt2_z.resize(0);
    rbxt2_x.resize(0);
    rbxt2_y.resize(0);
    rbxt2_z.resize(0);
    TWn_a_x.resize(0);
    TWn_a_y.resize(0);
    TWn_a_z.resize(0);
    TWn_b_x.resize(0);
    TWn_b_y.resize(0);
    TWn_b_z.resize(0);
    TWt1_a_x.resize(0);
    TWt1_a_y.resize(0);
    TWt1_a_z.resize(0);
    TWt1_b_x.resize(0);
    TWt1_b_y.resize(0);
    TWt1_b_z.resize(0);
    TWt2_a_x.resize(0);
    TWt2_a_y.resize(0);
    TWt2_a_z.resize(0);
    TWt2_b_x.resize(0);
    TWt2_b_y.resize(0);
    TWt2_b_z.resize(0);
    k_n.resize(0);
    k_t1.resize(0);
    k_t2.resize(0);
    inv_k_n.resize(0);
    inv_k_t1.resize(0);
    inv_k_t2.resize(0);
    mu.resize(0);
    e.resize(0);
    bias.resize(0);
    bounce.resize(0);
    C.resize(0);
    jn.resize(0);
    jt1.resize(0);
    jt2.resize(0);
    flags.resize(0);
    types.resize(0);
    indices.resize(0);
  }

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
    flags.clear();
    types.clear();
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
    flags.reserve(capacity);
    types.reserve(capacity);
    indices.reserve(capacity);
  }

  void resize(std::size_t size) {
    a.resize(size);
    b.resize(size);
    nx.resize(size);
    ny.resize(size);
    nz.resize(size);
    t1x.resize(size);
    t1y.resize(size);
    t1z.resize(size);
    t2x.resize(size);
    t2y.resize(size);
    t2z.resize(size);
    rax.resize(size);
    ray.resize(size);
    raz.resize(size);
    rbx.resize(size);
    rby.resize(size);
    rbz.resize(size);
    raxn_x.resize(size);
    raxn_y.resize(size);
    raxn_z.resize(size);
    rbxn_x.resize(size);
    rbxn_y.resize(size);
    rbxn_z.resize(size);
    raxt1_x.resize(size);
    raxt1_y.resize(size);
    raxt1_z.resize(size);
    rbxt1_x.resize(size);
    rbxt1_y.resize(size);
    rbxt1_z.resize(size);
    raxt2_x.resize(size);
    raxt2_y.resize(size);
    raxt2_z.resize(size);
    rbxt2_x.resize(size);
    rbxt2_y.resize(size);
    rbxt2_z.resize(size);
    TWn_a_x.resize(size);
    TWn_a_y.resize(size);
    TWn_a_z.resize(size);
    TWn_b_x.resize(size);
    TWn_b_y.resize(size);
    TWn_b_z.resize(size);
    TWt1_a_x.resize(size);
    TWt1_a_y.resize(size);
    TWt1_a_z.resize(size);
    TWt1_b_x.resize(size);
    TWt1_b_y.resize(size);
    TWt1_b_z.resize(size);
    TWt2_a_x.resize(size);
    TWt2_a_y.resize(size);
    TWt2_a_z.resize(size);
    TWt2_b_x.resize(size);
    TWt2_b_y.resize(size);
    TWt2_b_z.resize(size);
    k_n.resize(size);
    k_t1.resize(size);
    k_t2.resize(size);
    inv_k_n.resize(size);
    inv_k_t1.resize(size);
    inv_k_t2.resize(size);
    mu.resize(size);
    e.resize(size);
    bias.resize(size);
    bounce.resize(size);
    C.resize(size);
    jn.resize(size);
    jt1.resize(size);
    jt2.resize(size);
    flags.resize(size);
    types.resize(size);
    indices.resize(size);
    N = static_cast<int>(size);
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

