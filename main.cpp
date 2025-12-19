#include "benchmarks.h"
#include "datastructures.h"

#include <string>
#include <experimental/meta>
#include <iostream>
#include <algorithm>
#include <sstream>

#include <chrono>
#include <fstream>
#include <likwid.h>
#include <numeric>
#include <random>

using Clock = std::chrono::high_resolution_clock;

size_t max_results_size = 1048576; // 2^20

struct FileOpts {
  std::string input = "";            // Option "--input <string>"
  std::string input2 = "";           // Option "--input2 <string>"
  std::string output = "";           // Option "--output <string>"
  std::string validation = "";       // Option "--validation <string>"
};

FileOpts opts;
std::ostream *output;

/* Parse command-line options. */
template<typename Opts>
auto parse_options(std::span<std::string_view const> args) -> Opts {
  Opts opts;

  constexpr auto ctx = std::meta::access_context::current();
  template for (constexpr auto dm : define_static_array(nonstatic_data_members_of(^^Opts, ctx))) {
    auto it = std::find_if(args.begin(), args.end(),
      [](std::string_view arg){
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

  if (opts.input2.empty()) {
    opts.input2 = opts.input;
  }

  return opts;
}

/* Read Lorentzvector (pt, eta, phi, e) data from the given CSV file into the container. */
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
      v[i].pt = std::stof(token);
      getline(ss, token, ',');
      v[i].eta = std::stof(token);
      getline(ss, token, ',');
      v[i].phi = std::stof(token);
      getline(ss, token, ',');
      v[i].e = std::stof(token);
    }
    is.close();
  } else {
    throw std::runtime_error("Failed to open file " + filename + " for reading");
  }
}

template <typename Container, std::meta::info BenchmarkFunc,
          typename... ExtraArgs>
void RunBenchmark(size_t in_size, size_t out_size, ExtraArgs... extra_args) {
  Container v1(in_size), v2(in_size);
  ReadData(v1, opts.input);
  ReadData(v2, opts.input2);

  // Cap the results size to avoid excessive memory usage
  std::vector<double> results(std::min(out_size, max_results_size));

  auto start = Clock::now();
  [:substitute(BenchmarkFunc, {^^Container}):](v1, v2, results, extra_args...);
  auto end = Clock::now();

  // Print configuration and timing information in csv format
  std::chrono::duration<double, std::milli> elapsed = end - start;
  *output << identifier_of(BenchmarkFunc) << ","
          << Container::get_partitions_string() << "," << in_size << ","
          << sizeof(Container) << "," << elapsed.count() << std::endl;
}

/* Run all benchmarks defined in benchmarks.h. */
template <typename Container> void RunAllBenchmarks(size_t n) {
  std::vector<size_t> indices(n);
  std::iota(begin(indices), end(indices), n);
  std::mt19937 rng(std::random_device{}());
  std::shuffle(begin(indices), end(indices), rng);
  RunBenchmark<Container, ^^kernels::InvariantMassRandom>(n, n, indices);

  RunBenchmark<Container, ^^kernels::InvariantMassSequential>(n, n);
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
  sizes.push_back(topo->cacheLevels[topo->numCacheLevels - 1].size * 2 /
                  sizeof(T));

  return sizes;
}

int main(int argc, char *argv[]) {
  opts = parse_options<FileOpts>(std::vector<std::string_view>(argv+1, argv+argc));
  std::ofstream output_file(opts.output);
  if (output_file.is_open()) {
    output = &output_file;
  } else {
    output = &std::cout;
  }
  *output << "benchmark,partitions,problem_size,container_byte_size,time_ms\n";

  for (size_t n : GetProblemSizes<Particle>()) {
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({0, 1, 2, 3}).data()>>>(n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({0, 1, 3, 2}).data()>>>(n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({0, 2, 1, 3}).data()>>>(n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({0, 2, 3, 1}).data()>>>(n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({0, 3, 1, 2}).data()>>>(n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({0, 3, 2, 1}).data()>>>(n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({1, 0, 2, 3}).data()>>>(n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({1, 0, 3, 2}).data()>>>(n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({1, 2, 0, 3}).data()>>>(n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({1, 2, 3, 0}).data()>>>(n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({1, 3, 0, 2}).data()>>>(n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({1, 3, 2, 0}).data()>>>(n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({2, 0, 1, 3}).data()>>>(n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({2, 0, 3, 1}).data()>>>(n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({2, 1, 0, 3}).data()>>>(n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({2, 1, 3, 0}).data()>>>(n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({2, 3, 0, 1}).data()>>>(n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({2, 3, 1, 0}).data()>>>(n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({3, 0, 1, 2}).data()>>>(n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({3, 0, 2, 1}).data()>>>(n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({3, 1, 0, 2}).data()>>>(n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({3, 1, 2, 0}).data()>>>(n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({3, 2, 0, 1}).data()>>>(n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({3, 2, 1, 0}).data()>>>(n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({0, 1, 2}).data()>,
        SubParticle<SplitOp({3}).data()>>>(n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({0, 2, 1}).data()>,
        SubParticle<SplitOp({3}).data()>>>(n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({1, 0, 2}).data()>,
        SubParticle<SplitOp({3}).data()>>>(n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({1, 2, 0}).data()>,
        SubParticle<SplitOp({3}).data()>>>(n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({2, 0, 1}).data()>,
        SubParticle<SplitOp({3}).data()>>>(n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({2, 1, 0}).data()>,
        SubParticle<SplitOp({3}).data()>>>(n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({0, 1, 3}).data()>,
        SubParticle<SplitOp({2}).data()>>>(n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({0, 3, 1}).data()>,
        SubParticle<SplitOp({2}).data()>>>(n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({1, 0, 3}).data()>,
        SubParticle<SplitOp({2}).data()>>>(n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({1, 3, 0}).data()>,
        SubParticle<SplitOp({2}).data()>>>(n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({3, 0, 1}).data()>,
        SubParticle<SplitOp({2}).data()>>>(n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({3, 1, 0}).data()>,
        SubParticle<SplitOp({2}).data()>>>(n);
    RunAllBenchmarks<
        PartitionedContainer<ParticleRef, SubParticle<SplitOp({0, 1}).data()>,
                             SubParticle<SplitOp({2, 3}).data()>>>(n);
    RunAllBenchmarks<
        PartitionedContainer<ParticleRef, SubParticle<SplitOp({1, 0}).data()>,
                             SubParticle<SplitOp({2, 3}).data()>>>(n);
    RunAllBenchmarks<
        PartitionedContainer<ParticleRef, SubParticle<SplitOp({1, 0}).data()>,
                             SubParticle<SplitOp({2, 3}).data()>>>(n);
    RunAllBenchmarks<
        PartitionedContainer<ParticleRef, SubParticle<SplitOp({1, 0}).data()>,
                             SubParticle<SplitOp({3, 2}).data()>>>(n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({0, 1}).data()>,
        SubParticle<SplitOp({2}).data()>, SubParticle<SplitOp({3}).data()>>>(n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({1, 0}).data()>,
        SubParticle<SplitOp({2}).data()>, SubParticle<SplitOp({3}).data()>>>(n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({0, 2, 3}).data()>,
        SubParticle<SplitOp({1}).data()>>>(n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({0, 3, 2}).data()>,
        SubParticle<SplitOp({1}).data()>>>(n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({2, 0, 3}).data()>,
        SubParticle<SplitOp({1}).data()>>>(n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({2, 3, 0}).data()>,
        SubParticle<SplitOp({1}).data()>>>(n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({3, 0, 2}).data()>,
        SubParticle<SplitOp({1}).data()>>>(n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({3, 2, 0}).data()>,
        SubParticle<SplitOp({1}).data()>>>(n);
    RunAllBenchmarks<
        PartitionedContainer<ParticleRef, SubParticle<SplitOp({0, 2}).data()>,
                             SubParticle<SplitOp({1, 3}).data()>>>(n);
    RunAllBenchmarks<
        PartitionedContainer<ParticleRef, SubParticle<SplitOp({2, 0}).data()>,
                             SubParticle<SplitOp({1, 3}).data()>>>(n);
    RunAllBenchmarks<
        PartitionedContainer<ParticleRef, SubParticle<SplitOp({2, 0}).data()>,
                             SubParticle<SplitOp({1, 3}).data()>>>(n);
    RunAllBenchmarks<
        PartitionedContainer<ParticleRef, SubParticle<SplitOp({2, 0}).data()>,
                             SubParticle<SplitOp({3, 1}).data()>>>(n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({0, 2}).data()>,
        SubParticle<SplitOp({1}).data()>, SubParticle<SplitOp({3}).data()>>>(n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({2, 0}).data()>,
        SubParticle<SplitOp({1}).data()>, SubParticle<SplitOp({3}).data()>>>(n);
    RunAllBenchmarks<
        PartitionedContainer<ParticleRef, SubParticle<SplitOp({0, 3}).data()>,
                             SubParticle<SplitOp({1, 2}).data()>>>(n);
    RunAllBenchmarks<
        PartitionedContainer<ParticleRef, SubParticle<SplitOp({3, 0}).data()>,
                             SubParticle<SplitOp({1, 2}).data()>>>(n);
    RunAllBenchmarks<
        PartitionedContainer<ParticleRef, SubParticle<SplitOp({3, 0}).data()>,
                             SubParticle<SplitOp({1, 2}).data()>>>(n);
    RunAllBenchmarks<
        PartitionedContainer<ParticleRef, SubParticle<SplitOp({3, 0}).data()>,
                             SubParticle<SplitOp({2, 1}).data()>>>(n);
    RunAllBenchmarks<
        PartitionedContainer<ParticleRef, SubParticle<SplitOp({0}).data()>,
                             SubParticle<SplitOp({1, 2, 3}).data()>>>(n);
    RunAllBenchmarks<
        PartitionedContainer<ParticleRef, SubParticle<SplitOp({0}).data()>,
                             SubParticle<SplitOp({1, 3, 2}).data()>>>(n);
    RunAllBenchmarks<
        PartitionedContainer<ParticleRef, SubParticle<SplitOp({0}).data()>,
                             SubParticle<SplitOp({2, 1, 3}).data()>>>(n);
    RunAllBenchmarks<
        PartitionedContainer<ParticleRef, SubParticle<SplitOp({0}).data()>,
                             SubParticle<SplitOp({2, 3, 1}).data()>>>(n);
    RunAllBenchmarks<
        PartitionedContainer<ParticleRef, SubParticle<SplitOp({0}).data()>,
                             SubParticle<SplitOp({3, 1, 2}).data()>>>(n);
    RunAllBenchmarks<
        PartitionedContainer<ParticleRef, SubParticle<SplitOp({0}).data()>,
                             SubParticle<SplitOp({3, 2, 1}).data()>>>(n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({0}).data()>,
        SubParticle<SplitOp({1, 2}).data()>, SubParticle<SplitOp({3}).data()>>>(
        n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({0}).data()>,
        SubParticle<SplitOp({2, 1}).data()>, SubParticle<SplitOp({3}).data()>>>(
        n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({0, 3}).data()>,
        SubParticle<SplitOp({1}).data()>, SubParticle<SplitOp({2}).data()>>>(n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({3, 0}).data()>,
        SubParticle<SplitOp({1}).data()>, SubParticle<SplitOp({2}).data()>>>(n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({0}).data()>,
        SubParticle<SplitOp({1, 3}).data()>, SubParticle<SplitOp({2}).data()>>>(
        n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({0}).data()>,
        SubParticle<SplitOp({3, 1}).data()>, SubParticle<SplitOp({2}).data()>>>(
        n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({0}).data()>,
        SubParticle<SplitOp({1}).data()>, SubParticle<SplitOp({2, 3}).data()>>>(
        n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({0}).data()>,
        SubParticle<SplitOp({1}).data()>, SubParticle<SplitOp({3, 2}).data()>>>(
        n);
    RunAllBenchmarks<PartitionedContainer<
        ParticleRef, SubParticle<SplitOp({0}).data()>,
        SubParticle<SplitOp({1}).data()>, SubParticle<SplitOp({2}).data()>,
        SubParticle<SplitOp({3}).data()>>>(n);
  }
  return 0;
}
