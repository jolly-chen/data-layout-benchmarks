#include "benchmarks.h"
#include "datastructures.h"
#include <chrono>
#include <fstream>
#include <iostream>

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


template <typename Container, std::meta::info BenchmarkFunc>
void RunBenchmark(size_t in_size, size_t out_size) {
  Container v1(in_size), v2(in_size);
  read_data(v1, "/data/data-layout-benchmarks/data/100k.csv");
  read_data(v2, "/data/data-layout-benchmarks/data/100k.csv");

  std::vector<double> results(out_size);

  auto start = Clock::now();
  [: substitute(BenchmarkFunc, { ^^Container }) :](v1, v2, results);
  auto end = Clock::now();

  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << identifier_of(BenchmarkFunc)
            << "," << Container::get_partitions_string()
            << "," << in_size
            << "," << sizeof(Container)
            << "," << elapsed.count() << std::endl;
}

template <typename Container> void RunAllBenchmarks(size_t n) {
  RunBenchmark<Container, ^^kernels::InvariantMassRandom>(n, n);
  RunBenchmark<Container, ^^kernels::InvariantMassSequential>(n, n);
  RunBenchmark<Container, ^^kernels::DeltaR2Pairwise>(n, n*n);
}

int main() {
  for (size_t input_size : {1000, 10000, 100000}) {
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({0, 1, 2, 3}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({0, 1, 3, 2}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({0, 2, 1, 3}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({0, 2, 3, 1}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({0, 3, 1, 2}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({0, 3, 2, 1}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({1, 0, 2, 3}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({1, 0, 3, 2}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({1, 2, 0, 3}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({1, 2, 3, 0}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({1, 3, 0, 2}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({1, 3, 2, 0}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({2, 0, 1, 3}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({2, 0, 3, 1}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({2, 1, 0, 3}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({2, 1, 3, 0}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({2, 3, 0, 1}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({2, 3, 1, 0}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({3, 0, 1, 2}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({3, 0, 2, 1}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({3, 1, 0, 2}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({3, 1, 2, 0}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({3, 2, 0, 1}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({3, 2, 1, 0}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({0, 1, 2}).data()>, SubParticle<SplitOp({3}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({0, 2, 1}).data()>, SubParticle<SplitOp({3}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({1, 0, 2}).data()>, SubParticle<SplitOp({3}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({1, 2, 0}).data()>, SubParticle<SplitOp({3}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({2, 0, 1}).data()>, SubParticle<SplitOp({3}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({2, 1, 0}).data()>, SubParticle<SplitOp({3}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({0, 1, 3}).data()>, SubParticle<SplitOp({2}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({0, 3, 1}).data()>, SubParticle<SplitOp({2}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({1, 0, 3}).data()>, SubParticle<SplitOp({2}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({1, 3, 0}).data()>, SubParticle<SplitOp({2}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({3, 0, 1}).data()>, SubParticle<SplitOp({2}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({3, 1, 0}).data()>, SubParticle<SplitOp({2}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({0, 1}).data()>, SubParticle<SplitOp({2, 3}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({1, 0}).data()>, SubParticle<SplitOp({2, 3}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({1, 0}).data()>, SubParticle<SplitOp({2, 3}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({1, 0}).data()>, SubParticle<SplitOp({3, 2}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({0, 1}).data()>, SubParticle<SplitOp({2}).data()>, SubParticle<SplitOp({3}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({1, 0}).data()>, SubParticle<SplitOp({2}).data()>, SubParticle<SplitOp({3}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({0, 2, 3}).data()>, SubParticle<SplitOp({1}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({0, 3, 2}).data()>, SubParticle<SplitOp({1}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({2, 0, 3}).data()>, SubParticle<SplitOp({1}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({2, 3, 0}).data()>, SubParticle<SplitOp({1}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({3, 0, 2}).data()>, SubParticle<SplitOp({1}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({3, 2, 0}).data()>, SubParticle<SplitOp({1}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({0, 2}).data()>, SubParticle<SplitOp({1, 3}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({2, 0}).data()>, SubParticle<SplitOp({1, 3}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({2, 0}).data()>, SubParticle<SplitOp({1, 3}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({2, 0}).data()>, SubParticle<SplitOp({3, 1}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({0, 2}).data()>, SubParticle<SplitOp({1}).data()>, SubParticle<SplitOp({3}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({2, 0}).data()>, SubParticle<SplitOp({1}).data()>, SubParticle<SplitOp({3}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({0, 3}).data()>, SubParticle<SplitOp({1, 2}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({3, 0}).data()>, SubParticle<SplitOp({1, 2}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({3, 0}).data()>, SubParticle<SplitOp({1, 2}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({3, 0}).data()>, SubParticle<SplitOp({2, 1}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({0}).data()>, SubParticle<SplitOp({1, 2, 3}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({0}).data()>, SubParticle<SplitOp({1, 3, 2}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({0}).data()>, SubParticle<SplitOp({2, 1, 3}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({0}).data()>, SubParticle<SplitOp({2, 3, 1}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({0}).data()>, SubParticle<SplitOp({3, 1, 2}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({0}).data()>, SubParticle<SplitOp({3, 2, 1}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({0}).data()>, SubParticle<SplitOp({1, 2}).data()>, SubParticle<SplitOp({3}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({0}).data()>, SubParticle<SplitOp({2, 1}).data()>, SubParticle<SplitOp({3}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({0, 3}).data()>, SubParticle<SplitOp({1}).data()>, SubParticle<SplitOp({2}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({3, 0}).data()>, SubParticle<SplitOp({1}).data()>, SubParticle<SplitOp({2}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({0}).data()>, SubParticle<SplitOp({1, 3}).data()>, SubParticle<SplitOp({2}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({0}).data()>, SubParticle<SplitOp({3, 1}).data()>, SubParticle<SplitOp({2}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({0}).data()>, SubParticle<SplitOp({1}).data()>, SubParticle<SplitOp({2, 3}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({0}).data()>, SubParticle<SplitOp({1}).data()>, SubParticle<SplitOp({3, 2}).data()>>>(input_size);
		RunAllBenchmarks<PartitionedContainer<Particle, SubParticle<SplitOp({0}).data()>, SubParticle<SplitOp({1}).data()>, SubParticle<SplitOp({2}).data()>, SubParticle<SplitOp({3}).data()>>>(input_size);
	}	
	return 0;
}
