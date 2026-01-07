#include "benchmarks.h"
#include "datastructures.h"

#include <algorithm>
#include <experimental/meta>
#include <iostream>
#include <sstream>
#include <string>

#include <chrono>
#include <fstream>
#include <likwid.h>
#include <numeric>
#include <random>

using Clock = std::chrono::high_resolution_clock;
using unit = std::milli;

size_t max_results_size = 65536; // 2^16

struct FileOpts {
  std::string input = "";  // Option "--input <string>"
  std::string input2 = ""; // Option "--input2 <string>"
  std::string output = ""; // Option "--output <string>"
  int repetitions = 5;     // Option "--repetitions <int>"
};

/* */
struct ValidationFiles;
consteval {
  std::vector<std::meta::info> specs;
  auto benchmarks = members_of(^^kernels, std::meta::access_context::current());
  for (auto b : benchmarks) {
    specs.push_back(data_member_spec(^^std::string, {.name = identifier_of(b)}));
  }
  define_aggregate(^^ValidationFiles, specs );
}

FileOpts opts;
ValidationFiles validation; // Files for validation
std::ostream *output;

/* Get the data member of validation that corresponds to the given benchmark name */
consteval std::meta::info get_validation_file_by_name(std::string_view name) {
  for (const auto v : nonstatic_data_members_of(^^ValidationFiles,
                                            std::meta::access_context::current())) {
    if (identifier_of(v) == name) {
      return v;
    }
  }
}

/* Parse command-line options.
   Taken from https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p2996r13.html#parsing-command-line-options */
void parse_options(std::span<std::string_view const> args) {
  // Parse benchmark configuration options
  constexpr auto ctx = std::meta::access_context::current();
  template for (constexpr auto dm :
                define_static_array(nonstatic_data_members_of(^^FileOpts, ctx))) {
    auto it = std::find_if(args.begin(), args.end(), [](std::string_view arg) {
      return arg.starts_with("--") && arg.substr(2) == identifier_of(dm);
    });

    if (it == args.end()) {
      // no option provided, use default
      continue;
    } else if (it + 1 == args.end()) {
      std::cerr << "Option " << *it << " is missing a value\n";
      std::exit(EXIT_FAILURE);
    }

    using T = typename[:type_of(dm):];
    auto iss = std::stringstream(it[1]);
    if (iss >> opts.[:dm:]; !iss) {
      std::cerr << "Failed to parse option " << *it << " into a "
                << display_string_of(^^T) << '\n';
      std::exit(EXIT_FAILURE);
    }
  }

  // Use first input file as second input file if not provided
  if (opts.input2.empty()) {
    opts.input2 = opts.input;
  }

  // Parse validation files if provided
  template for (constexpr auto dm :
                define_static_array(nonstatic_data_members_of(^^ValidationFiles, ctx))) {
    auto it = std::find_if(args.begin(), args.end(), [](std::string_view arg) {
      return arg.starts_with("--validation_") && arg.substr(13) == identifier_of(dm);
    });

    if (it == args.end()) {
      // no option provided, use default
      continue;
    } else if (it + 1 == args.end()) {
      std::cerr << "Option " << *it << " is missing a value\n";
      std::exit(EXIT_FAILURE);
    }

    using T = typename[:type_of(dm):];
    auto iss = std::stringstream(it[1]);
    if (iss >> validation.[:dm:]; !iss) {
      std::cerr << "Failed to parse option " << *it << " into a "
                << display_string_of(^^T) << '\n';
      std::exit(EXIT_FAILURE);
    }
  }
}

/* Read Lorentzvector (pt, eta, phi, e) data from the given CSV file into the
 * container. */
template <typename Container>
void ReadData(Container &v, std::string filename) {
  std::ifstream is(filename);
  if (is.is_open()) {
    std::string line;
    for (size_t i = 0; i < v.size(); ++i) {
      if (!getline(is, line)) {
        throw std::runtime_error("Not enough data in file " + filename +
                                 " for requested size " +
                                 std::to_string(v.size()));
        break;
      }

      std::stringstream ss(line);
      std::string token;
      std::vector<std::string> temp;

      getline(ss, token, ',');
      v[i].pt = std::stod(token);
      getline(ss, token, ',');
      v[i].eta = std::stod(token);
      getline(ss, token, ',');
      v[i].phi = std::stod(token);
      getline(ss, token, ',');
      v[i].e = std::stod(token);
    }
    is.close();
  } else {
    throw std::runtime_error("Failed to open file " + filename +
                             " for reading");
  }
}

template <typename Container, std::meta::info BenchmarkFunc,
          typename... ExtraArgs>
void RunBenchmark(size_t in_size, size_t out_size, ExtraArgs... extra_args) {
  std::vector<double> measured_times;

  for (int _ = 0; _ < opts.repetitions; ++_) {
    Container v1(in_size), v2(in_size);
    ReadData(v1, opts.input);
    ReadData(v2, opts.input2);

    // Cap the results size to avoid excessive memory usage
    std::vector<double> results(std::min(out_size, max_results_size));

    // Measure time taken by the benchmark function
    auto start = Clock::now();
    [:substitute(BenchmarkFunc, {^^Container}):](v1, v2, results,
                                                 extra_args...);
    auto end = Clock::now();

    // Validate results if a validation file is provided
    auto validation_file = validation.[: get_validation_file_by_name(identifier_of(BenchmarkFunc)) :];
    if (!validation_file.empty()) {
      std::ifstream val_file(validation_file);
      if (val_file.is_open()) {
        std::string line;
        size_t idx = 0;

        while (getline(val_file, line)) {
            std::cerr << std::setprecision(20);
            if (std::fabs(std::stod(line) - results[idx]) > 1e-15) {
            std::cerr << "Validation failed at index " << idx << ": expected "
                      << line << ", got " << results[idx] << std::endl;
            break;
          }
          idx++;
        }
      }
    }

    std::chrono::duration<double, unit> elapsed = end - start;
    measured_times.push_back(elapsed.count());
  }

  // Print configuration and timing information in csv format
  double min = *std::ranges::min_element(measured_times);
  double max = *std::ranges::max_element(measured_times);
  double avg = std::reduce(measured_times.begin(), measured_times.end(), 0.0) /
               measured_times.size();
  double stddev =
      std::sqrt(std::reduce(measured_times.begin(), measured_times.end(), 0.0,
                            [avg](double acc, double t) {
                              return acc + (t - avg) * (t - avg);
                            }) /
                measured_times.size());
  *output << identifier_of(BenchmarkFunc) << ","
          << Container::get_partitions_string() << "," << in_size << ","
          << sizeof(Container) << "," << display_string_of(dealias(^^unit))
          << "," << min << "," << max << "," << avg << "," << stddev
          << std::endl;
}

/* Run all benchmarks defined in benchmarks.h. */
template <typename Container> void RunAllBenchmarks(size_t n) {
  RunBenchmark<Container, ^^kernels::InvariantMassSequential>(n, n);

  std::vector<size_t> indices(n);
  std::iota(begin(indices), end(indices), 0);
  std::mt19937 rng(123);
  std::shuffle(begin(indices), end(indices), rng);
  RunBenchmark<Container, ^^kernels::InvariantMassRandom>(n, n, indices);

  RunBenchmark<Container, ^^kernels::DeltaR2Pairwise>(n, n * n);
}

template <typename T> std::vector<size_t> GetProblemSizes() {
  std::vector<size_t> sizes;
  auto err = topology_init();
  if (err < 0) {
    fprintf(stderr, "Failed to initialize LIKWID's topology module\n");
    return sizes;
  }

  CpuTopology_t topo = get_cpuTopology();

  // Fits in L1 Cache
  sizes.push_back(topo->cacheLevels[0].size / sizeof(T) / 3);
  // Does not fit in any cache
  sizes.push_back(topo->cacheLevels[topo->numCacheLevels - 1].size / sizeof(T));

//   sizes.push_back(10);

  return sizes;
}

int main(int argc, char *argv[]) {
  parse_options(std::vector<std::string_view>(argv + 1, argv + argc));

  std::ofstream output_file(opts.output);
  if (output_file.is_open()) {
    output = &output_file;
  } else {
    output = &std::cout;
  }
  *output << "benchmark,partitions,problem_size,container_byte_size,time_ratio,"
             "min,max,"
             "avg,stddev\n";

  for (size_t n : GetProblemSizes<Particle>()) {
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({0, 1, 2, 3}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({0, 1, 3, 2}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({0, 2, 1, 3}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({0, 2, 3, 1}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({0, 3, 1, 2}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({0, 3, 2, 1}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({1, 0, 2, 3}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({1, 0, 3, 2}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({1, 2, 0, 3}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({1, 2, 3, 0}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({1, 3, 0, 2}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({1, 3, 2, 0}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({2, 0, 1, 3}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({2, 0, 3, 1}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({2, 1, 0, 3}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({2, 1, 3, 0}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({2, 3, 0, 1}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({2, 3, 1, 0}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({3, 0, 1, 2}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({3, 0, 2, 1}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({3, 1, 0, 2}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({3, 1, 2, 0}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({3, 2, 0, 1}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({3, 2, 1, 0}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({0, 1, 2}).data()>, SubParticle<SplitOp({3}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({0, 2, 1}).data()>, SubParticle<SplitOp({3}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({1, 0, 2}).data()>, SubParticle<SplitOp({3}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({1, 2, 0}).data()>, SubParticle<SplitOp({3}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({2, 0, 1}).data()>, SubParticle<SplitOp({3}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({2, 1, 0}).data()>, SubParticle<SplitOp({3}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({0, 1, 3}).data()>, SubParticle<SplitOp({2}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({0, 3, 1}).data()>, SubParticle<SplitOp({2}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({1, 0, 3}).data()>, SubParticle<SplitOp({2}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({1, 3, 0}).data()>, SubParticle<SplitOp({2}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({3, 0, 1}).data()>, SubParticle<SplitOp({2}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({3, 1, 0}).data()>, SubParticle<SplitOp({2}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({0, 1}).data()>, SubParticle<SplitOp({2, 3}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({1, 0}).data()>, SubParticle<SplitOp({2, 3}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({1, 0}).data()>, SubParticle<SplitOp({2, 3}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({1, 0}).data()>, SubParticle<SplitOp({3, 2}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({0, 1}).data()>, SubParticle<SplitOp({2}).data()>, SubParticle<SplitOp({3}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({1, 0}).data()>, SubParticle<SplitOp({2}).data()>, SubParticle<SplitOp({3}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({0, 2, 3}).data()>, SubParticle<SplitOp({1}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({0, 3, 2}).data()>, SubParticle<SplitOp({1}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({2, 0, 3}).data()>, SubParticle<SplitOp({1}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({2, 3, 0}).data()>, SubParticle<SplitOp({1}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({3, 0, 2}).data()>, SubParticle<SplitOp({1}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({3, 2, 0}).data()>, SubParticle<SplitOp({1}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({0, 2}).data()>, SubParticle<SplitOp({1, 3}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({2, 0}).data()>, SubParticle<SplitOp({1, 3}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({2, 0}).data()>, SubParticle<SplitOp({1, 3}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({2, 0}).data()>, SubParticle<SplitOp({3, 1}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({0, 2}).data()>, SubParticle<SplitOp({1}).data()>, SubParticle<SplitOp({3}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({2, 0}).data()>, SubParticle<SplitOp({1}).data()>, SubParticle<SplitOp({3}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({0, 3}).data()>, SubParticle<SplitOp({1, 2}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({3, 0}).data()>, SubParticle<SplitOp({1, 2}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({3, 0}).data()>, SubParticle<SplitOp({1, 2}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({3, 0}).data()>, SubParticle<SplitOp({2, 1}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({0}).data()>, SubParticle<SplitOp({1, 2, 3}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({0}).data()>, SubParticle<SplitOp({1, 3, 2}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({0}).data()>, SubParticle<SplitOp({2, 1, 3}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({0}).data()>, SubParticle<SplitOp({2, 3, 1}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({0}).data()>, SubParticle<SplitOp({3, 1, 2}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({0}).data()>, SubParticle<SplitOp({3, 2, 1}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({0}).data()>, SubParticle<SplitOp({1, 2}).data()>, SubParticle<SplitOp({3}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({0}).data()>, SubParticle<SplitOp({2, 1}).data()>, SubParticle<SplitOp({3}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({0, 3}).data()>, SubParticle<SplitOp({1}).data()>, SubParticle<SplitOp({2}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({3, 0}).data()>, SubParticle<SplitOp({1}).data()>, SubParticle<SplitOp({2}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({0}).data()>, SubParticle<SplitOp({1, 3}).data()>, SubParticle<SplitOp({2}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({0}).data()>, SubParticle<SplitOp({3, 1}).data()>, SubParticle<SplitOp({2}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({0}).data()>, SubParticle<SplitOp({1}).data()>, SubParticle<SplitOp({2, 3}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({0}).data()>, SubParticle<SplitOp({1}).data()>, SubParticle<SplitOp({3, 2}).data()>>>(n);
		RunAllBenchmarks<PartitionedContainer<ParticleRef, SubParticle<SplitOp({0}).data()>, SubParticle<SplitOp({1}).data()>, SubParticle<SplitOp({2}).data()>, SubParticle<SplitOp({3}).data()>>>(n);
	}
	return 0;
}
