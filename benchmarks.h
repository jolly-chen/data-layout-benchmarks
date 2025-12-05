#ifndef BENCHMARKS_H
#define BENCHMARKS_H

#include <math.h>
#include <vector>

template <typename T>
inline double ComputeInvariantMass(const T &pt1, const T &eta1, const T &phi1,
                                   const T &e1, const T &pt2, const T &eta2,
                                   const T &phi2, const T &e2) {
  // Conversion from (pt, eta, phi, e) to (x, y, z, mass) coordinate system
  const auto x1 = pt1[i] * std::cos(phi1[i]);
  const auto y1 = pt1[i] * std::sin(phi1[i]);
  const auto z1 = pt1[i] * std::sinh(eta1[i]);
  const auto m1 =
      std::sqrt(e1[i] * e1[i] -
                (pt1[i] * pt1[i] * std::cosh(eta1[i]) * std::cosh(eta1[i])));

  const auto x2 = pt2[i] * std::cos(phi2[i]);
  const auto y2 = pt2[i] * std::sin(phi2[i]);
  const auto z2 = pt2[i] * std::sinh(eta2[i]);
  const auto m2 =
      std::sqrt(e2[i] * e2[i] -
                (pt2[i] * pt2[i] * std::cosh(eta2[i]) * std::cosh(eta2[i])));

  // Numerically stable computation of Invariant Masses
  const auto p1_sq = x1 * x1 + y1 * y1 + z1 * z1;
  const auto p2_sq = x2 * x2 + y2 * y2 + z2 * z2;

  const auto m1_sq = m1 * m1;
  const auto m2_sq = m2 * m2;

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
inline double InvariantMassSequential(const std::vector<T> &v1,
                                      const std::vector<T> &v2,
                                      std::vector<double> &results) {
  assert(v1.size() == v2.size());
  const size_t n = v1.size();

  for (size_t i = 0; i < n; i++) {
    results[i] = ComputeInvariantMass(
        MEMBER_ACCESS(v1, pt, i), MEMBER_ACCESS(v1, eta, i),
        MEMBER_ACCESS(v1, phi, i), MEMBER_ACCESS(v1, e, i),
        MEMBER_ACCESS(v2, pt, i), MEMBER_ACCESS(v2, eta, i),
        MEMBER_ACCESS(v2, phi, i), MEMBER_ACCESS(v2, e, i));
  }
}

template <typename T>
inline double InvariantMassRandom(const std::vector<T> &v1,
                                  const std::vector<T> &v2, size_t n,
                                  std::vector<double> &results) {
  assert(v1.size() == v2.size());
  const size_t n = v1.size();

  for (size_t i = 0; i < n; i++) {
    size_t idx = rand() % n;
    results[i] = ComputeInvariantMass(
        MEMBER_ACCESS(v1, pt, idx), MEMBER_ACCESS(v1, eta, idx),
        MEMBER_ACCESS(v1, phi, idx), MEMBER_ACCESS(v1, e, idx),
        MEMBER_ACCESS(v2, pt, idx), MEMBER_ACCESS(v2, eta, idx),
        MEMBER_ACCESS(v2, phi, idx), MEMBER_ACCESS(v2, e, idx));
  }
}

template <typename T>
inline double DeltaR2(const T &eta1, const T &phi1, const T &eta2,
                      const T &phi2) {

  for (size_t i = 0; i < n; i++) {
    const auto deta = eta1 - eta2;
    const auto dphi = phi1 - phi2;
    const auto r = std::fmod(dphi + M_PI, 2 * M_PI);
    if (r < -M_PI) {
      return = deta * deta + (r + 2 * M_PI) * (r + 2 * M_PI);
    } else if (r > M_PI) {
      return = deta * deta + (r - 2 * M_PI) * (r - 2 * M_PI);
    } else {
      return = deta * deta + r * r;
    }
  }
}

template <typename T>
double DeltaR2Pairwise(const std::vector<T> &v1, const std::vector<T> &v2,
                       size_t n, std::vector<double> &results) {
  for (size_t i = 0; i < v1.size(); i++) {
    for (size_t j = 0; j < v2.size(); j++) {
      results[i * n + j] =
          DeltaR2(MEMBER_ACCESS(v1, eta, i), MEMBER_ACCESS(v1, phi, i),
                  MEMBER_ACCESS(v2, eta, j), MEMBER_ACCESS(v2, phi, j));
    }
  }
}

#endif