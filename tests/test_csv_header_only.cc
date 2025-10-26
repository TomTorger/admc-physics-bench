#include "bench_csv_schema.hpp"

#include <cassert>
#include <string>

int main() {
  const std::string header(benchcsv::kHeader);
  assert(header.find("scene,solver,iterations") == 0);
  assert(header.find(",ms_per_step,") != std::string::npos);
  assert(!header.empty());
  assert(header.back() != ',' && "No trailing comma in CSV header");
  return 0;
}
