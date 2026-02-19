#include "test.h"
#include "struct_transformer.h"
#include <algorithm>
#include <iostream>
#include <random>

template <typename T> void VectorSum(const T &v1, const T &v2, const std::string &tname) {
  std::cout << "Running VectorSum test with " << tname << "..."<< std::endl;

  assert(v1.size() == v2.size());
  const size_t n = v1.size();
  std::vector<float> results(n);

  // Initialize vectors
  for (size_t i = 0; i < n; i++) {
    v1[i].x = i;
    v1[i].y = i;
    v2[i].x = i;
    v2[i].y = i;
  }

  // Sum x and y members
  for (size_t i = 0; i < n; i++) {
    results[i] = v1[i].x + v1[i].y + v2[i].x + v2[i].y;
  }

  // Verify results
  for (size_t i = 0; i < n; i++) {
    if (results[i] != i * 4) {
      std::cout << "\033[1;31mFAILED\033[0m\n";
      std::cerr << "\tError at index " << i << ": expected " << i * 4
                << ", got " << results[i] << "\n";
      return;
    }
  }

  std::cout << "\033[1;32mPASSED\033[0m\n";
}

template <typename T> void VerifyAllocation01_23(const T &p, const std::string &tname) {
  std::cout << "Running VerifyAllocation01_23 with " << tname << "..."<< std::endl;

  // Test that x-y and z-w elements are allocated ly (i.e., xyx and
  // zwz)
  size_t n = p.size();
  size_t x0_y0_diff =
      reinterpret_cast<char *>(&(p[0].y)) - reinterpret_cast<char *>(&(p[0].x));
  size_t y0_x1_diff =
      reinterpret_cast<char *>(&(p[1].x)) - reinterpret_cast<char *>(&(p[0].y));
  size_t z0_w0_diff =
      reinterpret_cast<char *>(&(p[0].w)) - reinterpret_cast<char *>(&(p[0].z));
  size_t w0_z1_diff =
      reinterpret_cast<char *>(&(p[1].z)) - reinterpret_cast<char *>(&(p[0].w));

  if (x0_y0_diff != sizeof(double)) {
    std::cout << "\033[1;31mFAILED\033[0m\n";
    std::cerr
        << "\tNon-contiguous allocation detected between x[0] and y[0]: gap of "
        << x0_y0_diff << "\n";
    return;
  }

  if (y0_x1_diff != sizeof(double)) {
    std::cout << "\033[1;31mFAILED\033[0m\n";
    std::cerr
        << "\tNon-contiguous allocation detected between y[0] and x[1]: gap of "
        << y0_x1_diff << "\n";
    return;
  }

  if (z0_w0_diff != sizeof(float)) {
    std::cout << "\033[1;31mFAILED\033[0m\n";
    std::cerr
        << "\tNon-contiguous allocation detected between z[0] and w[0]: gap of "
        << z0_w0_diff << "\n";
    return;
  }

  if (w0_z1_diff != sizeof(float)) {
    std::cout << "\033[1;31mFAILED\033[0m\n";
    std::cerr
        << "\tNon-contiguous allocation detected between w[0] and z[1]: gap of "
        << w0_z1_diff << "\n";
    return;
  }

  // Test that the different members are allocated ly (i.e.,
  // xyxyxy...zwzwzw...)
  auto last_y = reinterpret_cast<char *>(&p[n - 1].y) + sizeof(p[n - 1].y);
  auto first_z = reinterpret_cast<char *>(&p[0].z);
  if (last_y != first_z) {
    std::cout << "\033[1;31mFAILED\033[0m\n";
    std::cerr << "\tMembers are not allocated contiguously: end of first "
                 "partition is "
              << last_y << " and the start of second partition is " << first_z
              << "\n";
    return;
  }

  std::cout << "\033[1;32mPASSED\033[0m\n";
}

template <typename T> void VerifyAllocation0_1_2_3(const T &p, const std::string &tname) {
  std::cout << "Running VerifyAllocation0_1_2_3 with " << tname << "..." << std::endl;
  bool failed = false;

  // Test that elements for er1_h member are allocated ly (i.e., xx and
  // yy and zz and ww)
  size_t x_diff =
      reinterpret_cast<char *>(&(p[1].x)) - reinterpret_cast<char *>(&(p[0].x));
  size_t y_diff =
      reinterpret_cast<char *>(&(p[1].y)) - reinterpret_cast<char *>(&(p[0].y));
  size_t z_diff =
      reinterpret_cast<char *>(&(p[1].z)) - reinterpret_cast<char *>(&(p[0].z));
  size_t w_diff =
      reinterpret_cast<char *>(&(p[1].w)) - reinterpret_cast<char *>(&(p[0].w));

  if (x_diff != sizeof(int)) {
    std::cout << "\033[1;31mFAILED\033[0m\n";
    std::cerr << "\tElements of x are not : gap of " << x_diff
              << "\n";
    failed = true;
  }

  if (y_diff != sizeof(double)) {
    std::cout << "\033[1;31mFAILED\033[0m\n";
    std::cerr << "\tElements of y are not : gap of " << y_diff
              << "\n";
    failed = true;
  }

  if (z_diff != sizeof(float)) {
    std::cout << "\033[1;31mFAILED\033[0m\n";
    std::cerr << "\tElements of z are not : gap of " << z_diff
              << "\n";
    failed = true;
  }

  if (w_diff != sizeof(char)) {
    std::cout << "\033[1;31mFAILED\033[0m\n";
    std::cerr << "\tElements of w are not : gap of " << w_diff
              << "\n";
    failed = true;
  }

  // Test that the different members are allocated ly (i.e.,
  // xxxx...yyyy...zzzz...wwww...)
  size_t n = p.size();
  auto x_end =
      reinterpret_cast<char *>(&p[0].x) + AlignSize(n * sizeof(int), 64);
  auto y_start = reinterpret_cast<char *>(&p[0].y);
  auto y_end = y_start + AlignSize(n * sizeof(double), 64);
  auto z_start = reinterpret_cast<char *>(&p[0].z);
  auto z_end = z_start + AlignSize(n * sizeof(float), 64);
  auto w_start = reinterpret_cast<char *>(&p[0].w);
  if (y_start != x_end || z_start != y_end || w_start != z_end) {
    std::cout << "\033[1;31mFAILED\033[0m\n";
    std::cerr << "\tMembers are not allocated ly. Partition "
                 "boundaries are:\n"
              << "\t\tx: [" << (long long)reinterpret_cast<char *>(&p[0].x)
              << ", " << (long long)x_end << "]\n"
              << "\t\ty: [" << (long long)y_start << ", " << (long long)y_end
              << "]\n"
              << "\t\tz: [" << (long long)z_start << ", " << (long long)z_end
              << "]\n"
              << "\t\tw: [" << (long long)w_start << ", "
              << (long long)(w_start + AlignSize(n * sizeof(char), 64))
              << "]\n";
    failed = true;
  }

  if (!failed) {
    std::cout << "\033[1;32mPASSED\033[0m\n";
  }
}

template <typename T> void VerifyAlignment0_1_23(const T &p, const std::string &tname) {
  bool failed = false;
  std::string err;

  std::cout << "Running VerifyAlignment0_1_23 with " << tname << "..." << std::endl;

  auto x_start = reinterpret_cast<char *>(&p[0].x);
  if ((long long) x_start % 64 != 0) {
    err +=  "\t\tx start: " + std::to_string((long long)x_start) + "\n";
    failed = true;
  }

  auto y_start = reinterpret_cast<char *>(&p[0].y);
  if ((long long) y_start % 64 != 0) {
    err +=  "\t\ty start: " + std::to_string((long long)y_start) + "\n";
    failed = true;
  }

  auto z_start = reinterpret_cast<char *>(&p[0].z);
  if ((long long) z_start % 64 != 0) {
    err +=  "\t\tz start: " + std::to_string((long long)z_start) + "\n";
    failed = true;
  }

  if (!failed) {
    std::cout << "\033[1;32mPASSED\033[0m\n";
  } else {
    std::cout << "\033[1;31mFAILED\033[0m\n";
    std::cerr << "\tPartitions are not aligned to 64-byte boundaries:\n" << err;
  }
}

int main() {
  ////// Python generated structures
  using PyContainerContiguous01_23 = PartitionedContainerContiguous01_23;
  PyContainerContiguous01_23 pc1_01_23(1000, alignment), pc2_01_23(1000, alignment);
  VectorSum(pc1_01_23, pc2_01_23, "PyContainerContiguous01_23");
  VerifyAllocation01_23(pc1_01_23, "PyContainerContiguous01_23");

  using PyContainerContiguous0_1_2_3 = PartitionedContainerContiguous0_1_2_3;
  PyContainerContiguous0_1_2_3 pc1_0_1_2_3(1000, alignment);
  VerifyAllocation0_1_2_3(pc1_0_1_2_3, "PyContainerContiguous0_1_2_3");

  using PyContainerContiguous0_1_23 = PartitionedContainerContiguous0_1_23;
  PyContainerContiguous0_1_23 pc1_0_1_23(1000, alignment);
  VerifyAlignment0_1_23(pc1_0_1_23, "PyContainerContiguous0_1_23");

  using PyContainer0_1_23 = PartitionedContainer0_1_23;
  PyContainer0_1_23 p1_0_1_23(1000, alignment);
  VerifyAlignment0_1_23(p1_0_1_23, "PyContainer0_1_23");
}
