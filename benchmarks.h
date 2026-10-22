#ifndef BENCHMARKS_H
#define BENCHMARKS_H

#include "datasets.h"
#include "layout_helpers.h"
#include "layouts_index.h"

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <span>
#include <string_view>
#include <tuple>
#include <vector>


template <typename T>
inline double ComputeInvariantMass(const T &pt1, const T &eta1, const T &phi1,
                                   const T &e1, const T &pt2, const T &eta2,
                                   const T &phi2, const T &e2) {
  // Conversion from (pt, eta, phi, e) to (x, y, z, mass) coordinate system
  const auto x1 = pt1 * std::cos(phi1);
  const auto y1 = pt1 * std::sin(phi1);
  const auto z1 = pt1 * std::sinh(eta1);
  const auto p1 = pt1 * std::cosh(eta1);
  const auto m1_sq = e1 * e1 - p1 * p1;

  const auto x2 = pt2 * std::cos(phi2);
  const auto y2 = pt2 * std::sin(phi2);
  const auto z2 = pt2 * std::sinh(eta2);
  const auto p2 = pt2 * std::cosh(eta2);
  const auto m2_sq = e2 * e2 - p2 * p2;

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

  //  const auto dphi = DeltaPhi(phi1, phi2, c);
  //  return (eta1 - eta2) * (eta1 - eta2) + dphi * dphi;
template <typename T>
inline double DeltaR2(const T &eta1, const T &phi1, const T &eta2,
                      const T &phi2) {
  const auto deta = eta1 - eta2;
  auto r = std::fmod(phi2 - phi1, 2.0 * M_PI);
  if (r < -M_PI) {
    r += 2.0 * M_PI;
  } else if (r > M_PI) {
    r -= 2.0 * M_PI;
  }

  return deta * deta + r * r;
}

namespace kernels {
template <typename T>
inline void InvariantMassSequential(const T &v1, const T &v2,
                                    std::span<double> results) {
  assert(v1.size() == v2.size());
  const size_t n = v1.size();

  for (size_t i = 0; i < n; i++) {
    results[i] = ComputeInvariantMass(v1[i].pt, v1[i].eta, v1[i].phi, v1[i].e,
                                      v2[i].pt, v2[i].eta, v2[i].phi, v2[i].e);
  }
}

template <typename T1, typename T2>
inline void InvariantMassRandom(const T1 &v1, const T2 &v2,
                                std::span<double> results,
                                std::span<size_t> indices) {
  assert(v1.size() == v2.size());
  const size_t n = v1.size();

  for (size_t i = 0; i < n; i++) {
    size_t idx = indices[i];
    results[i] =
        ComputeInvariantMass(v1[idx].pt, v1[idx].eta, v1[idx].phi, v1[idx].e,
                             v2[idx].pt, v2[idx].eta, v2[idx].phi, v2[idx].e);
  }
}

constexpr size_t deltar2_max_outer_size = 128;

template <typename T>
inline void DeltaR2Pairwise(const T &v, std::span<double> results) {
  const size_t n = v.size();
  size_t idx = 0;

  for (size_t i = 0; i < std::min(deltar2_max_outer_size, n); i++) {
    for (size_t j = i + 1; j < n; j++) {
      results[idx++] = DeltaR2(v[i].eta, v[i].phi, v[j].eta, v[j].phi);
    }
  }
}

} // namespace kernels

/* Benchmark descriptors: everything the driver in main.cpp needs to know to
 * run a kernel on any layout of its element type. To add a kernel, write a
 * struct that satisfies BenchmarkDescriptor (and BenchmarkFor for the layouts
 * of its value_type), and add it to AllKernels. */
namespace bench {

/* The members of a benchmark descriptor K that do not depend on the layout. */
template <class K>
concept BenchmarkDescriptor = requires {
  // Element type; the kernel runs on every container C with LayoutOf<C, value_type>
  typename K::value_type;
  // Benchmark name, also the key for validation data
  { K::name } -> std::convertible_to<std::string_view>;
  // Name of the validation data to compare the results against
  { K::validation_name } -> std::convertible_to<std::string_view>;
  // True if result i only depends on element i, so validation data for a
  // larger input size also validates a smaller one
  { K::validate_prefix } -> std::convertible_to<bool>;
  // Number of datasets of value_type it reads
  { K::num_inputs } -> std::convertible_to<size_t>;
};

/* The members of a benchmark descriptor K for a layout C of its value_type. */
template <class K, class C>
concept BenchmarkFor =
    BenchmarkDescriptor<K> && LayoutOf<C, typename K::value_type> &&
    // Containers and buffers for layout C, set up outside the timed loop from
    // (n, alignment, datasets)
    std::constructible_from<typename K::template Setup<C>, size_t, size_t, const Datasets &> &&
    requires(typename K::template Setup<C> &s, size_t n) {
      // Bytes of all input and output containers for n elements; problem
      // sizes are chosen from this
      { K::template bytes_for_all<C>(n, n) } -> std::convertible_to<size_t>;
      // The timed part
      K::run(s);
      // Results to validate, in input order
      { K::output(s) } -> std::convertible_to<std::span<const double>>;
    };

/* Largest number of elements for which the containers of K with layout C take
 * up at most target_bytes. */
template <class K, class ...C>
  // requires BenchmarkFor<K, C>
size_t ProblemSize(size_t target_bytes, size_t alignment) {
  size_t lo = 0, hi = target_bytes; // every element takes at least one byte
  while (lo < hi) {
    size_t mid = lo + (hi - lo + 1) / 2;
    if (K::template bytes_for_all<C...>(mid, alignment) <= target_bytes) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  return lo;
}

/* Copy the first c.size() elements of src into c. */
template <class C, class T>
void Fill(C &c, const std::vector<T> &src) {
  for (size_t i = 0; i < c.size(); ++i) {
    c[i] = src[i];
  }
}

struct InvariantMassSequential {
  using value_type = Particle;
  static constexpr std::string_view name = "InvariantMassSequential";
  static constexpr std::string_view validation_name = name;
  static constexpr bool validate_prefix = true;
  static constexpr size_t num_inputs = 2;

  template <class C>
  static size_t bytes_for_all(size_t n, size_t alignment) {
    return 2 * C::bytes_for_all(n, alignment) + n * sizeof(double);
  }

  template <LayoutOf<Particle> C>
  struct Setup {
    C v1, v2;
    std::vector<double> results;

    Setup(size_t n, size_t alignment, const Datasets &d) : v1(n, alignment), v2(n, alignment), results(n) {
      Fill(v1, d.Get<Particle>()[0]);
      Fill(v2, d.Get<Particle>()[1]);
    }
  };

  template <class C>
  static void run(Setup<C> &s) { kernels::InvariantMassSequential(s.v1, s.v2, s.results); }

  template <class C>
  static std::span<const double> output(Setup<C> &s) { return s.results; }
};

struct InvariantMassRandom {
  using value_type = Particle;
  static constexpr std::string_view name = "InvariantMassRandom";
  // output() undoes the shuffle, so the results match the sequential ones
  static constexpr std::string_view validation_name = InvariantMassSequential::name;
  static constexpr bool validate_prefix = true;
  static constexpr size_t num_inputs = 2;
  // Fixed, so every layout and every run visits the elements in the same order
  static constexpr std::mt19937::result_type seed = 42;

  template <class C1, class C2>
  static size_t bytes_for_all(size_t n, size_t alignment) {
    return C1::bytes_for_all(n, alignment) + C2::bytes_for_all(n, alignment) + n * sizeof(double) + n * sizeof(size_t);
  }

  template <LayoutOf<Particle> C1, LayoutOf<Particle> C2>
  struct Setup {
    C1 v1;
    C2 v2;
    std::vector<double> results;
    std::vector<size_t> indices;
    std::vector<double> ordered_results;

    Setup(size_t n, size_t alignment, const Datasets &d)
        : v1(n, alignment), v2(n, alignment), results(n), indices(n) {
      Fill(v1, d.Get<Particle>()[0]);
      Fill(v2, d.Get<Particle>()[1]);

      std::iota(begin(indices), end(indices), 0);
      std::mt19937 rng(seed);
      std::shuffle(begin(indices), end(indices), rng);
    }
  };

  template <class C1, class C2>
  static void run(Setup<C1, C2> &s) { kernels::InvariantMassRandom(s.v1, s.v2, s.results, s.indices); }

  template <class C1, class C2>
  static std::span<const double> output(Setup<C1, C2> &s) {
    s.ordered_results.resize(s.results.size());
    for (size_t i = 0; i < s.results.size(); ++i) {
      s.ordered_results[s.indices[i]] = s.results[i];
    }
    return s.ordered_results;
  }
};


struct InvariantMassRandomOut {
  using value_type = ParticleOut;
  using fixed_layout = layouts::ParticleOut::PartitionedContainer01234;
  static constexpr std::string_view name = "InvariantMassRandomOut";
  // output() undoes the shuffle, so the results match the sequential ones
  static constexpr std::string_view validation_name = InvariantMassRandomOut::name;
  static constexpr bool validate_prefix = true;
  static constexpr size_t num_inputs = 2;
  // Fixed, so every layout and every run visits the elements in the same order
  static constexpr std::mt19937::result_type seed = 42;

  template <class C1, class C2>
  static size_t bytes_for_all(size_t n, size_t alignment) {
    return C1::bytes_for_all(n, alignment) + C2::bytes_for_all(n, alignment)
      + 2 *  layouts::Particle::PartitionedContainer01234::bytes_for_all(n, alignment)
      + n * sizeof(double) + n * sizeof(size_t);
  }

  template <LayoutOf<ParticleOut> C1, LayoutOf<ParticleOut> C2>
  struct Setup {
    layouts::Particle::PartitionedContainer01234 v1;
    layouts::Particle::PartitionedContainer01234 v2;
    C1 results1;
    C2 results2;
    std::vector<double> im_results;
    std::vector<size_t> indices;
    std::vector<double> ordered_results;

    Setup(size_t n, size_t alignment, const Datasets &d)
        : v1(n, alignment), v2(n, alignment), results1(n, alignment), results2(n, alignment), im_results(n), indices(n) {
      Fill(v1, d.Get<Particle>()[0]);
      Fill(v2, d.Get<Particle>()[1]);

      std::iota(begin(indices), end(indices), 0);
      std::mt19937 rng(seed);
      std::shuffle(begin(indices), end(indices), rng);
    }
  };

  template <class C1, class C2>
  static void run(Setup<C1, C2> &s) {
    assert(s.v1.size() == s.v2.size());
    const size_t n = s.v1.size();

    for (size_t i = 0; i < n; i++) {
      size_t idx = s.indices[i];
      const auto x1 = s.v1[idx].pt * std::cos(s.v1[idx].phi);
      const auto y1 = s.v1[idx].pt * std::sin(s.v1[idx].phi);
      const auto z1 = s.v1[idx].pt * std::sinh(s.v1[idx].eta);
      const auto p1 = s.v1[idx].pt * std::cosh(s.v1[idx].eta);
      const auto m1_sq = s.v1[idx].e * s.v1[idx].e - p1 * p1;

      const auto x2 = s.v2[idx].pt * std::cos(s.v2[idx].phi);
      const auto y2 = s.v2[idx].pt * std::sin(s.v2[idx].phi);
      const auto z2 = s.v2[idx].pt * std::sinh(s.v2[idx].eta);
      const auto p2 = s.v2[idx].pt * std::cosh(s.v2[idx].eta);
      const auto m2_sq = s.v2[idx].e * s.v2[idx].e - p2 * p2;

      const auto p1_sq = x1 * x1 + y1 * y1 + z1 * z1;
      const auto p2_sq = x2 * x2 + y2 * y2 + z2 * z2;

      const auto r1 = m1_sq / p1_sq;
      const auto r2 = m2_sq / p2_sq;
      const auto x = r1 + r2 + r1 * r2;

      const auto cx = y1 * z2 - y2 * z1;
      const auto cy = x1 * z2 - x2 * z1;
      const auto cz = x1 * y2 - x2 * y1;
      const auto c = std::sqrt(cx * cx + cy * cy + cz * cz);
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
      const auto mass = std::sqrt(m1_sq + m2_sq + y * z);

      s.results1[i].x = x1;
      s.results1[i].y = y1;
      s.results1[i].z = z1;
      s.results1[i].p = p1;
      s.results2[i].x = x2;
      s.results2[i].y = y2;
      s.results2[i].z = z2;
      s.results2[i].p = p2;
      s.im_results[i] = mass;
    }
  }

  template <class C1, class C2>
  static std::span<const double> output(Setup<C1, C2> &s) {
    s.ordered_results.resize(s.im_results.size());
    for (size_t i = 0; i < s.im_results.size(); ++i) {
      s.ordered_results[s.indices[i]] = s.im_results[i];
    }
    return s.ordered_results;
  }
};

struct FitTracks {
  using value_type = Particle;
  static constexpr std::string_view name = "FitTracks";
  static constexpr std::string_view validation_name = name;
  static constexpr bool validate_prefix = false;
  static constexpr size_t num_inputs = 2;
  static constexpr int max_hits_per_track = 10;

  template <class H, class T>
  static size_t bytes_for_all(size_t n, size_t alignment) {
    return H::bytes_for_all(n * 10, alignment) + T::bytes_for_all(n, alignment);
  }

  template <LayoutOf<Hit> H, LayoutOf<Track> T>
  struct Setup {
    H hits;
    T tracks;

    Setup(size_t n, size_t alignment, const Datasets &d)
        : hits(n * 10, alignment), tracks(n, alignment) {
      Fill(hits, d.Get<Hit>()[0]);

      // hits array size fixed to 10 time the tracks array. Track array initialized cannot be
      // pre-generated because the valid hit_indices depends on the number of hits / problem size.
      srand(123);
      for (size_t i = 0; i < tracks.size(); ++i) {
        tracks[i].n_hits = 4 + rand() % (max_hits_per_track - 3);
        for (int h = 0; h < tracks[i].n_hits; ++h)
            tracks[i].hit_idx[h] = rand() % hits.size();
        tracks[i].chi2 = 0.0f;
      }
    }
  };

  template <class H, class T>
  static void run(Setup<H, T> &s) {
    auto n_tracks = s.tracks.size();
    assert(s.hits.size() == s.tracks.size() * 10);

    for (size_t t = 0; t < n_tracks; ++t) {           // AoS #1: sequential walk
        auto trk = s.tracks[t];

        float cx = 0.f, cy = 0.f, cz = 0.f;
        for (int h = 0; h < trk.n_hits; ++h) {
          size_t idx = trk.hit_idx[h];              // AoS #2: gather via index
          assert(idx < s.hits.size());
          cx += s.hits[idx].x;
          cy += s.hits[idx].y;
          cz += s.hits[idx].z;
        }
        cx /= trk.n_hits; cy /= trk.n_hits; cz /= trk.n_hits;

        float chi2 = 0.f;
        for (int h = 0; h < trk.n_hits; ++h) {
          size_t idx = trk.hit_idx[h];              // AoS #2 again: second gather pass
          assert(idx < s.hits.size());
          float dx = s.hits[idx].x - cx;
          float dy = s.hits[idx].y - cy;
          float dz = s.hits[idx].z - cz;
          float w  = 1.0f / (s.hits[idx].sigma * s.hits[idx].sigma);
          chi2 += w * (dx*dx + dy*dy + dz*dz);
        }
        trk.chi2 = chi2;
    }
  }

  template <class H, class T>
  static std::span<const double> output(Setup<H, T> &s) {
    std::vector<double> chi2s;
    chi2s.reserve(s.tracks.size());
    for (size_t t = 0; t < s.tracks.size(); ++t) {
      chi2s.push_back(s.tracks[t].chi2);
    }
    return chi2s;
  };
};


struct DeltaR2Pairwise {
  using value_type = Particle;
  static constexpr std::string_view name = "DeltaR2Pairwise";
  static constexpr std::string_view validation_name = name;
  static constexpr bool validate_prefix = false;
  static constexpr size_t num_inputs = 1;

  /* Number of pairs the kernel computes for n elements. */
  static constexpr size_t num_results(size_t n) {
    const size_t m = std::min(kernels::deltar2_max_outer_size, n);
    return m * (n - 1) - m * (m - 1) / 2;
  }

  template <class C>
  static size_t bytes_for_all(size_t n, size_t alignment) {
    return C::bytes_for_all(n, alignment) + num_results(n) * sizeof(double);
  }

  template <LayoutOf<Particle> C>
  struct Setup {
    C v;
    std::vector<double> results;

    Setup(size_t n, size_t alignment, const Datasets &d) : v(n, alignment), results(num_results(n)) {
      Fill(v, d.Get<Particle>()[0]);
    }
  };

  template <class C>
  static void run(Setup<C> &s) { kernels::DeltaR2Pairwise(s.v, s.results); }

  template <class C>
  static std::span<const double> output(Setup<C> &s) { return s.results; }
};

/* The benchmarks main.cpp registers. DeltaR2Pairwise is not registered, but
 * can be added here. */
using AllKernels = std::tuple<InvariantMassSequential, InvariantMassRandom, InvariantMassRandomOut>;

} // namespace bench

#endif // BENCHMARKS_H
