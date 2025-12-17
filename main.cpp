#include "benchmarks.h"
#include "datastructures.h"
#include <chrono>
#include <iostream>
#include <fstream>

using Clock = std::chrono::high_resolution_clock;

template <typename Container>
void read_data(Container &v, std::string filename) {
  std::ifstream is(filename);
  if (is.is_open()) {
    std::string line;
    for (size_t i = 0; i < v.size(); ++i) {
      getline(is, line);
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

template <typename Container, typename BenchmarkFunc>
void RunBenchmark(size_t in_size, size_t out_size, BenchmarkFunc func) {
  Container v1(in_size), v2(in_size);
  read_data(v1, "/data/data-layout-benchmarks/data/100k.csv");
  read_data(v2, "/data/data-layout-benchmarks/data/100k.csv");

  std::vector<double> results(out_size);

  auto start = Clock::now();
  func(v1, v2, results);
  auto end = Clock::now();

  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";
}

template <typename Container> void RunAllBenchmarks(size_t n) {
  std::cout << "Running InvariantMassRandom\n";
  RunBenchmark<Container>(n, n, kernels::InvariantMassRandom<Container>);

  std::cout << "Running InvariantMassSequential\n";
  RunBenchmark<Container>(n, n, kernels::InvariantMassSequential<Container>);

  std::cout << "Running DeltaR2Pairwise\n";
  RunBenchmark<Container>(n, n*n, kernels::DeltaR2Pairwise<Container>);
}

int main() {
  RunAllBenchmarks<PartitionedContainer<
      Particle, SubParticle<SplitOp({0, 1, 2, 3}).data()>>>(10);
  return 0;
}
