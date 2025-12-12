#include "benchmarks.h"
#include "datastructures.h"
#include <chrono>
#include <iostream>

using Clock = std::chrono::high_resolution_clock;

template <typename T, typename... Partitions>
void RunInvariantMassRandom(size_t n) {
  using PC = PartitionedContainer<T, Partitions...>;
  PC v1(n), v2(n);

  std::vector<double> results(n);

  auto start = Clock::now();
  InvariantMassRandom(v1, v2, results);
  auto end = Clock::now();

  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";
}

int main() {
  RunInvariantMassRandom<Particle, SubParticle<SplitOp({0, 1, 2}).data()>, SubParticle<SplitOp({3}).data()>>(10000000);
  return 0;
}

// https://godbolt.org/z/xhTMKPsTK