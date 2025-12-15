#include "benchmarks.h"
#include "datastructures.h"
#include <chrono>
#include <iostream>

using Clock = std::chrono::high_resolution_clock;

// template <typename T> void RunInvariantMassRandom() {
//   std::vector<T> v1(1000);
//   std::vector<T> v2(1000);
//   std::vector<double> results(1000);

//   auto start = Clock::now();
//   InvariantMassRandom(v1, v2, results);
//   auto end = Clock::now();

//   std::chrono::duration<double> elapsed = end - start;
//   std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";
// }


template <typename T>
void RunInvariantMassRandomPartitioned() {
  using PC = PartitionedContainer<T, SubParticle<SplitOp({0, 1, 2}).data()>, SubParticle<SplitOp({3}).data()>>;
  PC v1(10), v2(10);

  v1[0].pt = 0;

  // std::vector<double> results(1000);

  // auto start = Clock::now();
  // InvariantMassRandom(v1, v2, results);
  // auto end = Clock::now();

  // std::chrono::duration<double> elapsed = end - start;
  // std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";
}

int main() {
  //   RunInvariantMassRandom<Particle>();
  RunInvariantMassRandomPartitioned<Particle>();
  return 0;
}

// https://godbolt.org/z/xhTMKPsTK