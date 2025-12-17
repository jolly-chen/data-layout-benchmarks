#ifndef BENCHMARKS_H
#define BENCHMARKS_H

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

template <typename T>
inline double ComputeInvariantMass(const T &pt1, const T &eta1, const T &phi1,
                                   const T &e1, const T &pt2, const T &eta2,
                                   const T &phi2, const T &e2) {
  // Conversion from (pt, eta, phi, e) to (x, y, z, mass) coordinate system
  const auto x1 = pt1 * std::cos(phi1);
  const auto y1 = pt1 * std::sin(phi1);
  const auto z1 = pt1 * std::sinh(eta1);
  const auto m1_sq = e1 * e1 - (pt1 * pt1 * std::cosh(eta1) * std::cosh(eta1));

  const auto x2 = pt2 * std::cos(phi2);
  const auto y2 = pt2 * std::sin(phi2);
  const auto z2 = pt2 * std::sinh(eta2);
  const auto m2_sq = e2 * e2 - (pt2 * pt2 * std::cosh(eta2) * std::cosh(eta2));

  // Numerically stable computation of Invariant Masses
  const auto p1_sq = x1 * x1 + y1 * y1 + z1 * z1;
  const auto p2_sq = x2 * x2 + y2 * y2 + z2 * z2;

  const auto r1 = m1_sq / p1_sq;
  const auto r2 = m2_sq / p2_sq;
  const auto x = r1 + r2 + r1 * r2;

  const auto cx = y1 * z2 - y2 * z1;
  const auto cy = x1 * z2 - x2 * z1;
  const auto cz = x1 * y2 - x2 * y1;

  // norm of cross product
  const auto c = std::sqrt(cx * cx + cy * cy + cz * cz);

  // dot product
  const auto d = x1 * x2 + y1 * y2 + z1 * z2;

  const auto a = std::atan2(c, d);

  const auto cos_a = std::cos(a);
  auto y = x;
  if (cos_a >= 0) {
    y = (x + std::sin(a) * std::sin(a)) / (std::sqrt(x + 1) + cos_a);
  } else {
    y = std::sqrt(x + 1) - cos_a;
  }

  const auto z = 2 * std::sqrt(p1_sq * p2_sq);

  return std::sqrt(m1_sq + m2_sq + y * z);
}

template <typename T>
inline double DeltaR2(const T &eta1, const T &phi1, const T &eta2,
                      const T &phi2) {

  const auto deta = eta1 - eta2;
  const auto dphi = phi1 - phi2;
  const auto r = std::fmod(dphi + M_PI, 2 * M_PI);
  if (r < -M_PI) {
    return deta * deta + (r + 2 * M_PI) * (r + 2 * M_PI);
  } else if (r > M_PI) {
    return deta * deta + (r - 2 * M_PI) * (r - 2 * M_PI);
  } else {
    return deta * deta + r * r;
  }
}

namespace kernels {
template <typename T>
inline void InvariantMassSequential(const T &v1, const T &v2,
                                    std::vector<double> &results) {
  assert(v1.size() == v2.size() && v1.size() == results.size());
  const size_t n = v1.size();

  for (size_t i = 0; i < 1; i++) {
    results[i] = ComputeInvariantMass(v1[i].pt, v1[i].eta, v1[i].phi, v1[i].e,
                                      v2[i].pt, v2[i].eta, v2[i].phi, v2[i].e);
  }
}

template <typename T>
inline void InvariantMassRandom(const T &v1, const T &v2,
                                std::vector<double> &results) {
  assert(v1.size() == v2.size());
  const size_t n = v1.size();

  for (size_t i = 0; i < n; i++) {
    size_t idx = rand() % n;
    results[i] =
        ComputeInvariantMass(v1[idx].pt, v1[idx].eta, v1[idx].phi, v1[idx].e,
                             v2[idx].pt, v2[idx].eta, v2[idx].phi, v2[idx].e);
  }
}

template <typename T>
inline void DeltaR2Pairwise(const T &v1, const T &v2,
                            std::vector<double> &results) {
  assert(v1.size() == v2.size());
  const size_t n = v1.size();

  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      results[i * n + j] = DeltaR2(v1[i].eta, v1[i].phi, v2[j].eta, v2[j].phi);
    }
  }
}
} // namespace kernels

#endif