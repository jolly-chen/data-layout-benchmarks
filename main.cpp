#include "benchmarks.h"
#include "datastructures.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <likwid.h>
#include <numeric>
#include <random>

using Clock = std::chrono::high_resolution_clock;

size_t max_results_size = 1048576; // 2^20

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
  }
}

template <typename Container, std::meta::info BenchmarkFunc,
          typename... ExtraArgs>
void RunBenchmark(size_t in_size, size_t out_size, ExtraArgs... extra_args) {
  Container v1(in_size), v2(in_size);
  ReadData(v1, "/data/data-layout-benchmarks/data/500k.csv");
  ReadData(v2, "/data/data-layout-benchmarks/data/500k.csv");

  // Cap the results size to avoid excessive memory usage
  std::vector<double> results(std::min(out_size, max_results_size));

  auto start = Clock::now();
  [:substitute(BenchmarkFunc, {^^Container}):](v1, v2, results, extra_args...);
  auto end = Clock::now();

  // Print configuration and timing information in csv format
  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << identifier_of(BenchmarkFunc) << ","
            << Container::get_partitions_string() << "," << in_size << ","
            << sizeof(Container) << "," << elapsed.count() << std::endl;
}

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
  sizes.push_back(topo->cacheLevels[topo->numCacheLevels - 1].size * 2 / sizeof(T));

  return sizes;
}

int main() {
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
