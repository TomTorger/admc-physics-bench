#include "bench_cli.hpp"
#include "bench_runner.hpp"
#include "bench_output.hpp"
#include "bench_scenes.hpp"
#include "bench_utils.hpp"

#include <iostream>
#include <map>

int main(int argc, char** argv) {
  using namespace bench;

  std::vector<char*> passthrough;
  BenchConfig cfg = parse_cli(argc, argv, passthrough);

  print_header("ADMC Physics Bench", cfg.threads);

  // 1️⃣ Resolve scenes
  auto scenes = resolve_scenes_from_config(cfg);
  if (scenes.empty()) {
    std::cerr << "No valid scenes found from CLI.\n";
    return 1;
  }

  // 2️⃣ Run all solver/scene combinations
  std::vector<BenchResult> all_results;
  std::map<std::pair<std::string, std::string>, double> single_thread_baseline;

  for (auto& [scene_name, scene] : scenes) {
    std::cout << "\n--- Scene: " << scene_name << " ---\n";
    auto results = run_suite_for_scene(scene_name, scene, cfg);

    for (auto& r : results) {
      std::optional<double> baseline;
      if (cfg.threads == 1)
        single_thread_baseline[{r.scene, r.solver}] = r.ms_per_step;
      else if (auto it = single_thread_baseline.find({r.scene, r.solver});
               it != single_thread_baseline.end())
        baseline = it->second;

      print_result_line(r, baseline);
      all_results.push_back(std::move(r));
    }
  }

  // 3️⃣ Summary output
  print_summary_table(all_results);

  // 4️⃣ Write CSV results
  if (cfg.csv_mode) {
    write_results_csv(all_results, cfg.csv_path);
    std::cout << "\nCSV results written to " << (cfg.csv_path.empty() ? default_results_csv_path() : cfg.csv_path)
              << "\n";
  }

  std::cout << "\nAll benchmarks complete.\n";
  return 0;
}
