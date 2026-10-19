#ifndef CELL_KERNELS_H
#define CELL_KERNELS_H

// A kernel on a struct other than Particle, plugged into the framework without
// changing the driver, the layout generator or the input parser: Cell is
// generated from cell.spec, and main.cpp is compiled with
// -DKERNELS_HEADER="cell_kernels.h" -DKERNELS=plugin::Kernels.

#include "benchmarks.h"
#include "layouts_Cell.h"

#include <span>
#include <string_view>
#include <tuple>
#include <vector>

namespace kernels {
/* Average density of each cell and its two neighbors. */
template <typename T>
inline void Stencil(const T &v, std::span<double> results) {
  for (size_t i = 1; i + 1 < v.size(); i++) {
    results[i] = (v[i - 1].density + v[i].density + v[i + 1].density) / 3.0;
  }
}
} // namespace kernels

namespace plugin {
struct StencilOnCell {
  using value_type = Cell;
  static constexpr std::string_view name = "Stencil";
  static constexpr std::string_view validation_name = name;
  static constexpr bool validate_prefix = false;
  static constexpr size_t num_inputs = 1;

  template <class C>
  static size_t bytes_for_all(size_t n, size_t alignment) {
    return C::bytes_for_all(n, alignment) + n * sizeof(double);
  }

  template <LayoutOf<Cell> C>
  struct Setup {
    C v;
    std::vector<double> results;

    Setup(size_t n, size_t alignment, const Datasets &d) : v(n, alignment), results(n) {
      bench::Fill(v, d.Get<Cell>()[0]);
    }
  };

  template <class C>
  static void run(Setup<C> &s) { kernels::Stencil(s.v, s.results); }

  template <class C>
  static std::span<const double> output(Setup<C> &s) { return s.results; }
};

/* The kernels of the plugin build of main.cpp: Particle and Cell kernels side by side. */
using Kernels = std::tuple<bench::InvariantMassSequential, bench::InvariantMassRandom, StencilOnCell>;
} // namespace plugin

#endif // CELL_KERNELS_H
