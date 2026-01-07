#include "datastructures.h"
#include <algorithm>    
#include <random>
#include <iostream>

template <typename T> void VectorSum(const T &v1, const T &v2) {
  std::cout << "Running VectorSum test... ";

  assert(v1.size() == v2.size());
  const size_t n = v1.size();
  std::vector<float> results(n);

  // Initialize vectors
  for (size_t i = 0; i < n; i++) {
    v1[i].pt = i;
    v1[i].phi = i;
    v2[i].pt = i;
    v2[i].phi = i;
  }

  // Sum pt and phi members
  for (size_t i = 0; i < n; i++) {
    results[i] = v1[i].pt + v1[i].phi + v2[i].pt + v2[i].phi;
  }

  // Verify results
  for (size_t i = 0; i < n; i++) {
    if (results[i] != i * 4) {
      std::cout << "FAILED\n";
      std::cerr << "Error at index " << i << ": expected " << i * 4 << ", got "
                << results[i] << "\n";
      return;
    }
  }

  std::cout << "PASSED\n";
}

template <typename T> void VerifyContiguousAllocation01_23(const T &p) {
  std::cout << "Running VerifyContiguousAllocation01_23... ";

  // Test that pt-eta and phi-e elements are allocated contiguously
  size_t n = p.size();
  size_t pt0_eta0_diff = reinterpret_cast<char *>(&(p[0].eta)) -
                         reinterpret_cast<char *>(&(p[0].pt));
  size_t eta0_pt1_diff = reinterpret_cast<char *>(&(p[1].pt)) -
                         reinterpret_cast<char *>(&(p[0].eta));
  size_t phi0_e0_diff = reinterpret_cast<char *>(&(p[0].e)) -
                        reinterpret_cast<char *>(&(p[0].phi));
  size_t e0_phi1_diff = reinterpret_cast<char *>(&(p[1].phi)) -
                        reinterpret_cast<char *>(&(p[0].e));

  if (pt0_eta0_diff != sizeof(float)) {
    std::cout << "FAILED\n";
    std::cerr
        << "Non-contiguous allocation detected between pt[0] and eta[0]\n";
    return;
  }

  if (eta0_pt1_diff != sizeof(float)) {
    std::cout << "FAILED\n";
    std::cerr
        << "Non-contiguous allocation detected between eta[0] and pt[1]\n";
    return;
  }

  if (phi0_e0_diff != sizeof(float)) {
    std::cout << "FAILED\n";
    std::cerr << "Non-contiguous allocation detected between phi[0] and e[0]\n";
    return;
  }

  if (e0_phi1_diff != sizeof(float)) {
    std::cout << "FAILED\n";
    std::cerr << "Non-contiguous allocation detected between e[0] and phi[1]\n";
    return;
  }

  // Test that the different members are allocated contiguously
  auto pt_start = reinterpret_cast<char *>(&p[0].pt);
  auto phi_start = reinterpret_cast<char *>(&p[0].phi);
  if (phi_start != pt_start + 2 * n * sizeof(float)) {
    std::cout << "FAILED\n";
    std::cerr << "Members are not allocated contiguously\n";
    return;
  }

  std::cout << "PASSED\n";
}

template <typename T> void VerifyContiguousAllocation0_1_2_3(const T &p) {
  std::cout << "Running VerifyContiguousAllocation0_1_2_3... ";

  // Test that elements for each member are allocated contiguously
  size_t n = p.size();
  size_t pt_diff = reinterpret_cast<char *>(&(p[1].pt)) -
                   reinterpret_cast<char *>(&(p[0].pt));
  size_t eta_diff = reinterpret_cast<char *>(&(p[1].eta)) -
                    reinterpret_cast<char *>(&(p[0].eta));
  size_t phi_diff = reinterpret_cast<char *>(&(p[1].phi)) -
                    reinterpret_cast<char *>(&(p[0].phi));
  size_t e_diff =
      reinterpret_cast<char *>(&(p[1].e)) - reinterpret_cast<char *>(&(p[0].e));

  if (pt_diff != sizeof(float)) {
    std::cout << "FAILED\n";
    std::cerr << "Elements of pt are not contiguous\n";
    return;
  }

  if (eta_diff != sizeof(float)) {
    std::cout << "FAILED\n";
    std::cerr << "Elements of eta are not contiguous\n";
    return;
  }

  if (phi_diff != sizeof(float)) {
    std::cout << "FAILED\n";
    std::cerr << "Elements of phi are not contiguous\n";
    return;
  }

  if (e_diff != sizeof(float)) {
    std::cout << "FAILED\n";
    std::cerr << "Elements of e are not contiguous\n";
    return;
  }

  // Test that the different members are allocated contiguously
  auto pt_start = reinterpret_cast<char *>(&p[0].pt);
  auto eta_start = reinterpret_cast<char *>(&p[0].eta);
  auto phi_start = reinterpret_cast<char *>(&p[0].phi);
  auto e_start = reinterpret_cast<char *>(&p[0].e);
  if (eta_start != pt_start + n * sizeof(float) ||
      phi_start != eta_start + n * sizeof(float) ||
      e_start != phi_start + n * sizeof(float)) {
    std::cout << "FAILED\n";
    std::cerr << "Members are not allocated contiguously\n";
    return;
  }

  std::cout << "PASSED\n";
}

int main() {
  // using Container01_23 =
  //     PartitionedContainer<Particle, SubParticle<SplitOp({0, 1}).data()>,
  //                          SubParticle<SplitOp({2, 3}).data()>>;
  // Container01_23 v1(1000), v2(1000);

  // VectorSum(v1, v2);
  // VerifyContiguousAllocation01_23(v1);

  // using Container0_1_2_3 =
  //     PartitionedContainer<Particle, SubParticle<SplitOp({0}).data()>,
  //                          SubParticle<SplitOp({1}).data()>,
  //                          SubParticle<SplitOp({2}).data()>,
  //                          SubParticle<SplitOp({3}).data()>>;
  // Container0_1_2_3 v3(10);
  // VerifyContiguousAllocation0_1_2_3(v3);


    // create and initialize an array                                                                                                   
    std::vector<size_t> indices(10);
    std::iota(begin(indices), end(indices), 0);
    std::mt19937_64 rng(123);
    std::shuffle(begin(indices), end(indices), rng);

    // copy the contents of the array to output                                                                            
    for (size_t i = 0; i < indices.size(); ++i) {
      std::cout << indices[i] << " ";
    }
    std::cout << std::endl;
}
