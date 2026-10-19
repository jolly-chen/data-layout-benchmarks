// Tests of the Cell plugin: the input parser reads the Cell dataset in
// tests/data, and the stencil kernel gives the same results on every generated
// Cell layout as on a plain std::vector<Cell>.

#include "cell_kernels.h"
#include "layout_helpers.h"

#include <algorithm>
#include <cstdio>
#include <format>
#include <string>
#include <vector>

int failures = 0;

#define CHECK(cond, ...)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      ++failures;                                                              \
      std::printf("\033[1;31mFAILED\033[0m %s:%d: %s: %s\n", __FILE__,        \
                  __LINE__, #cond, std::format(__VA_ARGS__).c_str());          \
    }                                                                          \
  } while (0)

#ifndef TEST_DATA_DIR
#error "TEST_DATA_DIR must be defined"
#endif

constexpr size_t alignment = 64;
constexpr size_t n = 1000;

int main() {
  // cells.csv has only some members, in another order than in cell.spec
  Datasets d;
  d.SetListFile("Cell", TEST_DATA_DIR "/cell_inputs");
  d.Reserve<Cell>(n, 1);
  d.Load();
  const auto &cells = d.Get<Cell>()[0];
  CHECK(cells[3].material == 3 && cells[3].density == 2.5f &&
            cells[3].x == 1.5 && cells[3].y == 0 && cells[3].z == 0,
        "cell 3 read as material={} density={} x={} y={} z={}", cells[3].material, cells[3].density,
        cells[3].x, cells[3].y, cells[3].z);

  // Reference: the kernel on a plain vector, which has the same interface
  std::vector<double> reference(n);
  kernels::Stencil(cells, reference);

  size_t layouts = 0;
  ForEachLayout<Cell>([&]<class C>() {
    ++layouts;
    plugin::StencilOnCell::Setup<C> s(n, alignment, d);
    plugin::StencilOnCell::run(s);
    auto out = plugin::StencilOnCell::output(s);
    CHECK(std::ranges::equal(out, reference), "{}: results differ from the reference", C::to_string());
  });
  CHECK(layouts > 2, "found {} Cell layouts", layouts);

  // The Particle layouts generated next to the Cell layouts are not Cell layouts
  size_t particle_layouts = 0;
  ForEachLayout<Particle>([&]<class C>() { ++particle_layouts; });
  CHECK(particle_layouts == 2, "found {} Particle layouts, expected AoS and SoA", particle_layouts);

  std::printf("Checked the stencil on %zu Cell layouts\n", layouts);
  if (failures == 0) {
    std::printf("\033[1;32mPASSED\033[0m\n");
  } else {
    std::printf("\033[1;31m%d check(s) FAILED\033[0m\n", failures);
  }
  return failures == 0 ? 0 : 1;
}
