#pragma once

#include <array>
#include <cmath>

namespace math {

//! Machine epsilon threshold for vector/quaternion safety checks.
constexpr double kEps = 1e-12;

//! Returns true when |x| <= eps.
inline bool nearly_zero(double x, double eps = kEps) {
  return std::fabs(x) <= eps;
}

// ----------------------- Vec3 -----------------------
//! Simple 3D vector of doubles.
struct Vec3 {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;

  constexpr Vec3() = default;
  constexpr Vec3(double xx, double yy, double zz) : x(xx), y(yy), z(zz) {}

  //! Access component by index (0=x,1=y,2=z).
  double& operator[](int i) {
    return (&x)[i];
  }

  //! Const access component by index (0=x,1=y,2=z).
  const double& operator[](int i) const {
    return (&x)[i];
  }

  //! Unary minus.
  constexpr Vec3 operator-() const { return Vec3(-x, -y, -z); }

  //! Vector addition.
  constexpr Vec3 operator+(const Vec3& v) const {
    return Vec3(x + v.x, y + v.y, z + v.z);
  }

  //! Vector subtraction.
  constexpr Vec3 operator-(const Vec3& v) const {
    return Vec3(x - v.x, y - v.y, z - v.z);
  }

  //! Scalar multiplication.
  constexpr Vec3 operator*(double s) const {
    return Vec3(x * s, y * s, z * s);
  }

  //! Scalar division.
  Vec3 operator/(double s) const {
    const double inv = 1.0 / s;
    return (*this) * inv;
  }

  //! Compound addition.
  Vec3& operator+=(const Vec3& v) {
    x += v.x; y += v.y; z += v.z;
    return *this;
  }

  //! Compound subtraction.
  Vec3& operator-=(const Vec3& v) {
    x -= v.x; y -= v.y; z -= v.z;
    return *this;
  }

  //! Compound scalar multiplication.
  Vec3& operator*=(double s) {
    x *= s; y *= s; z *= s;
    return *this;
  }

  //! Compound scalar division.
  Vec3& operator/=(double s) {
    const double inv = 1.0 / s;
    return (*this) *= inv;
  }
};

//! Scalar * vector multiplication.
inline Vec3 operator*(double s, const Vec3& v) {
  return v * s;
}

//! Dot product.
inline double dot(const Vec3& a, const Vec3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

//! Right-handed cross product.
inline Vec3 cross(const Vec3& a, const Vec3& b) {
  return Vec3(a.y * b.z - a.z * b.y,
              a.z * b.x - a.x * b.z,
              a.x * b.y - a.y * b.x);
}

//! Squared length.
inline double length2(const Vec3& v) {
  return dot(v, v);
}

//! Euclidean length.
inline double length(const Vec3& v) {
  return std::sqrt(length2(v));
}

//! Safe normalization (returns zero vector when the input is tiny).
inline Vec3 normalize_safe(const Vec3& v, double eps = kEps) {
  const double len_sq = length2(v);
  if (len_sq <= eps * eps) {
    return Vec3();
  }
  const double inv_len = 1.0 / std::sqrt(len_sq);
  return v * inv_len;
}

// ----------------------- Mat3 (row-major) -----------------------
//! Row-major 3x3 matrix.
struct Mat3 {
  std::array<double, 9> m{ {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0} };

  constexpr Mat3() = default;
  explicit constexpr Mat3(const std::array<double, 9>& values) : m(values) {}

  //! Identity matrix.
  static Mat3 identity() {
    return Mat3({{1.0, 0.0, 0.0,
                  0.0, 1.0, 0.0,
                  0.0, 0.0, 1.0}});
  }

  //! Diagonal matrix from vector.
  static Mat3 diagonal(const Vec3& d) {
    return Mat3({{d.x, 0.0, 0.0,
                  0.0, d.y, 0.0,
                  0.0, 0.0, d.z}});
  }

  //! Outer product a * b^T.
  static Mat3 outer(const Vec3& a, const Vec3& b) {
    return Mat3({{a.x * b.x, a.x * b.y, a.x * b.z,
                  a.y * b.x, a.y * b.y, a.y * b.z,
                  a.z * b.x, a.z * b.y, a.z * b.z}});
  }

  //! Transposed matrix.
  Mat3 transposed() const {
    return Mat3({{m[0], m[3], m[6],
                  m[1], m[4], m[7],
                  m[2], m[5], m[8]}});
  }

  //! Row vector (0-based index).
  Vec3 row(int i) const {
    const int base = i * 3;
    return Vec3(m[base], m[base + 1], m[base + 2]);
  }

  //! Column vector (0-based index).
  Vec3 col(int j) const {
    return Vec3(m[j], m[j + 3], m[j + 6]);
  }

  //! Matrix-vector multiplication.
  Vec3 operator*(const Vec3& v) const {
    return Vec3(dot(row(0), v), dot(row(1), v), dot(row(2), v));
  }

  //! Matrix-matrix multiplication.
  Mat3 operator*(const Mat3& B) const {
    Mat3 result;
    for (int r = 0; r < 3; ++r) {
      const Vec3 row_r = row(r);
      for (int c = 0; c < 3; ++c) {
        const Vec3 col_c = B.col(c);
        result.m[r * 3 + c] = dot(row_r, col_c);
      }
    }
    return result;
  }
};

// ----------------------- Quat (Hamilton) -----------------------
//! Hamilton quaternion (w + xi + yj + zk).
struct Quat {
  double w = 1.0;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;

  constexpr Quat() = default;
  constexpr Quat(double ww, double xx, double yy, double zz)
      : w(ww), x(xx), y(yy), z(zz) {}

  //! Normalizes the quaternion in-place, guarding tiny norms.
  void normalize(double eps = kEps) {
    const double n = std::sqrt(w * w + x * x + y * y + z * z);
    if (n <= eps) {
      w = 1.0; x = y = z = 0.0;
      return;
    }
    const double inv = 1.0 / n;
    w *= inv; x *= inv; y *= inv; z *= inv;
  }

  //! Returns a normalized copy.
  Quat normalized(double eps = kEps) const {
    Quat q = *this;
    q.normalize(eps);
    return q;
  }

  //! Builds quaternion from axis-angle (axis need not be unit length).
  static Quat from_axis_angle(const Vec3& axis, double radians) {
    const Vec3 naxis = normalize_safe(axis);
    const double half = 0.5 * radians;
    const double s = std::sin(half);
    const double c = std::cos(half);
    return Quat(c, naxis.x * s, naxis.y * s, naxis.z * s);
  }

  //! Quaternion multiplication (composition).
  Quat operator*(const Quat& q) const {
    return Quat(w * q.w - x * q.x - y * q.y - z * q.z,
                w * q.x + x * q.w + y * q.z - z * q.y,
                w * q.y - x * q.z + y * q.w + z * q.x,
                w * q.z + x * q.y - y * q.x + z * q.w);
  }

  //! Rotates a vector by the quaternion.
  Vec3 rotate(const Vec3& v) const {
    const Vec3 qv(x, y, z);
    const Vec3 t = 2.0 * cross(qv, v);
    return v + w * t + cross(qv, t);
  }
};

//! Rotation matrix from quaternion (active rotation).
inline Mat3 from_quat(const Quat& q_in) {
  const Quat q = q_in.normalized();
  const double ww = q.w * q.w;
  const double xx = q.x * q.x;
  const double yy = q.y * q.y;
  const double zz = q.z * q.z;
  const double xy = q.x * q.y;
  const double xz = q.x * q.z;
  const double yz = q.y * q.z;
  const double wx = q.w * q.x;
  const double wy = q.w * q.y;
  const double wz = q.w * q.z;

  return Mat3({{1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy),
                2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx),
                2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)}});
}

}  // namespace math

