#include "metrics_micro.hpp"

MicroMetrics estimate_contact_metrics(std::size_t rows, bool include_friction) {
  MicroMetrics metrics;
  metrics.rows_processed = rows;
  const double base_flops = include_friction ? 120.0 : 80.0;
  const double base_bytes = include_friction ? 256.0 : 192.0;
  metrics.flops_per_row = base_flops;
  metrics.bytes_per_row = base_bytes;
  return metrics;
}
