#pragma once

#include <cstddef>
#include <cstdint>

namespace admc {

struct TileSpan {
  const float* __restrict n_x;
  const float* __restrict n_y;
  const float* __restrict n_z;

  const float* __restrict rAx;
  const float* __restrict rAy;
  const float* __restrict rAz;
  const float* __restrict rBx;
  const float* __restrict rBy;
  const float* __restrict rBz;

  const int32_t* __restrict bodyA;
  const int32_t* __restrict bodyB;

  const float* __restrict k_n;
  const float* __restrict target_n;
  float* __restrict jn;

  float* __restrict vAx;
  float* __restrict vAy;
  float* __restrict vAz;
  float* __restrict vBx;
  float* __restrict vBy;
  float* __restrict vBz;

  const float* __restrict invMassA;
  const float* __restrict invMassB;

  int count;
  int lanes;
};

void solve_normal_lane_avx2(TileSpan tile);
void solve_normal_lane_neon(TileSpan tile);

}  // namespace admc

