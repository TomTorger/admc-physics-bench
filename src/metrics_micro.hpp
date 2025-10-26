#pragma once

#include <cstddef>

struct MicroMetrics {
  double flops_per_row = 0.0;
  double bytes_per_row = 0.0;
  std::size_t rows_processed = 0;

  void reset() {
    flops_per_row = 0.0;
    bytes_per_row = 0.0;
    rows_processed = 0;
  }

  void accumulate(const MicroMetrics& other) {
    flops_per_row += other.flops_per_row;
    bytes_per_row += other.bytes_per_row;
    rows_processed += other.rows_processed;
  }
};

MicroMetrics estimate_contact_metrics(std::size_t rows, bool include_friction);
