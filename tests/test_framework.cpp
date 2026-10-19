// Tests of the framework on the Particle layouts generated for the tests:
// - the driver finds every generated layout, and only layouts of the right struct,
// - problem sizes are the largest that fit the byte targets,
// - the input parser maps CSV columns to members and rejects malformed datasets,
// - every kernel gives the reference results (generated with ROOT) on every layout.
// Reads its datasets from tests/data (TEST_DATA_DIR).

#include "benchmarks.h"
#include "datasets.h"
#include "layout_helpers.h"

#include <cmath>
#include <cstdio>
#include <format>
#include <fstream>
#include <functional>
#include <set>
#include <string>
#include <vector>

#ifndef TEST_DATA_DIR
#error "TEST_DATA_DIR must be defined"
#endif

int failures = 0;

#define CHECK(cond, ...)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      ++failures;                                                              \
      std::printf("\033[1;31mFAILED\033[0m %s:%d: %s: %s\n", __FILE__,        \
                  __LINE__, #cond, std::format(__VA_ARGS__).c_str());          \
    }                                                                          \
  } while (0)

const std::string data_dir = TEST_DATA_DIR;
constexpr size_t alignment = 64;

std::vector<double> ReadReference(const std::string &path) {
  std::vector<double> values;
  std::ifstream is(path);
  for (std::string line; std::getline(is, line);) {
    values.push_back(std::stod(line));
  }
  return values;
}

/* Path of a dataset in tests/data/parser, made to test the input parser. */
std::string ParserData(const std::string &name) { return data_dir + "/parser/" + name; }

/* Whether f throws a std::runtime_error whose message contains text. */
bool Throws(const std::function<void()> &f, const std::string &text) {
  try {
    f();
  } catch (const std::runtime_error &e) {
    if (std::string(e.what()).find(text) != std::string::npos) { return true; }
    std::printf("  unexpected message: %s\n", e.what());
    return false;
  }
  return false;
}

void TestLayoutEnumeration() {
  std::printf("Running TestLayoutEnumeration...\n");
  std::set<std::string> names;
  size_t count = 0;
  ForEachLayout<Particle>([&]<class C>() {
    ++count;
    names.insert(C::to_string());
  });
  // tests/CMakeLists.txt generates every 1000th layout plus AoS and SoA. The
  // names come from generated containers, so the right number of distinct
  // names means all of them were found.
  constexpr size_t expected_layouts = 39;
  CHECK(count == expected_layouts, "found {} layouts, expected {}", count, expected_layouts);
  CHECK(names.size() == count, "{} layouts but {} distinct names", count, names.size());
  CHECK(names.contains("PartitionedContainer0123456"), "AoS layout missing");
  CHECK(names.contains("PartitionedContainer0_1_2_3_4_5_6"), "SoA layout missing");

  size_t others = 0;
  ForEachLayout<int>([&]<class C>() { ++others; });
  CHECK(others == 0, "found {} layouts for a struct without layouts", others);
}

template <class K>
void TestProblemSize() {
  std::printf("Running TestProblemSize<%s>...\n", K::name.data());
  for (size_t target : {size_t{32768}, size_t{33554432}}) {
    ForEachLayout<Particle>([&]<class C>() {
      size_t n = bench::ProblemSize<K, C>(target, alignment);
      size_t bytes = K::template bytes_for_all<C>(n, alignment);
      size_t next = K::template bytes_for_all<C>(n + 1, alignment);
      CHECK(n > 0 && bytes <= target && next > target,
            "{} target {}: n={} uses {} bytes, n+1 uses {}", C::to_string(), target, n, bytes, next);
    });
  }
}

void TestReadCsv() {
  std::printf("Running TestReadCsv...\n");

  // Columns in any order, members without a column are value-initialized
  auto path = ParserData("reordered.csv");
  auto data = ReadCsv<Particle>(path, 2);
  CHECK(data.size() == 2, "read {} rows", data.size());
  CHECK(data[1].pt == 5 && data[1].eta == 6 && data[1].phi == 7 && data[1].e == 8,
        "row 2 read as pt={} eta={} phi={} e={}", data[1].pt, data[1].eta, data[1].phi, data[1].e);
  CHECK(data[0].id == 0 && data[0].charge == 0 && data[0].posCovMatrix[2][2] == 0,
        "members without a column are not zero");

  // Arithmetic members of other types, and whitespace around values
  path = ParserData("types.csv");
  data = ReadCsv<Particle>(path, 1);
  CHECK(data[0].id == -3 && data[0].charge == 1 && data[0].pt == 2.5,
        "read id={} charge={} pt={}", data[0].id, int(data[0].charge), data[0].pt);

  // Malformed datasets
  auto no_header = ParserData("no_header.csv");
  CHECK(Throws([&] { ReadCsv<Particle>(no_header, 1); }, "header row"), "missing header row not reported");
  auto unknown = ParserData("unknown.csv");
  CHECK(Throws([&] { ReadCsv<Particle>(unknown, 1); }, "'mass'"), "unknown column not reported");
  auto array = ParserData("array.csv");
  CHECK(Throws([&] { ReadCsv<Particle>(array, 1); }, "non-arithmetic"), "non-arithmetic member not reported");
  auto twice = ParserData("twice.csv");
  CHECK(Throws([&] { ReadCsv<Particle>(twice, 1); }, "twice"), "duplicate column not reported");
  auto short_file = ParserData("short.csv");
  CHECK(Throws([&] { ReadCsv<Particle>(short_file, 2); }, "Not enough data"), "too few rows not reported");
  auto bad_value = ParserData("bad_value.csv");
  CHECK(Throws([&] { ReadCsv<Particle>(bad_value, 1); }, "not a valid value"), "invalid value not reported");
  auto bad_row = ParserData("bad_row.csv");
  CHECK(Throws([&] { ReadCsv<Particle>(bad_row, 1); }, "has 1 values"), "row with too few values not reported");
}

void TestDatasets() {
  std::printf("Running TestDatasets...\n");
  Datasets d;
  d.SetDefaultListFile(data_dir + "/particle_inputs");
  d.Reserve<Particle>(10, 2);
  d.Reserve<Particle>(1000, 1); // the largest number of rows and files is kept
  d.Load();
  const auto &p = d.Get<Particle>();
  CHECK(p.size() == 2 && p[0].size() == 1000 && p[1].size() == 1000,
        "loaded {} datasets", p.size());

  Datasets too_few;
  too_few.SetListFile("Particle", data_dir + "/one_particle_input");
  too_few.Reserve<Particle>(10, 2);
  CHECK(Throws([&] { too_few.Load(); }, "need 2"), "too few datasets not reported");

  Datasets none;
  none.Reserve<Particle>(10, 1);
  CHECK(Throws([&] { none.Load(); }, "No input specified for Particle"), "missing input not reported");
}

/* Run K on every layout with n elements and compare its output with reference. */
template <class K>
void TestKernel(const Datasets &d, const std::vector<double> &reference, size_t n) {
  std::printf("Running TestKernel<%s> with n=%zu...\n", K::name.data(), n);
  ForEachLayout<Particle>([&]<class C>() {
    typename K::template Setup<C> s(n, alignment, d);
    K::run(s);
    auto out = K::output(s);

    if constexpr (K::validate_prefix) {
      CHECK(out.size() == n && n <= reference.size(), "{}: {} results for n={}, reference has {}", C::to_string(),
            out.size(), n, reference.size());
    } else {
      CHECK(out.size() == reference.size(), "{}: {} results, reference has {}", C::to_string(), out.size(), reference.size());
    }
    size_t mismatches = 0;
    for (size_t i = 0; i < out.size() && i < reference.size(); ++i) {
      mismatches += !(std::abs(out[i] - reference[i]) <= 1e-6);
    }
    CHECK(mismatches == 0, "{}: {} of {} results differ from the reference", C::to_string(), mismatches, out.size());
  });
}

/* Every member read back through a container equals the source element. */
void TestFill(const Datasets &d) {
  std::printf("Running TestFill...\n");
  std::vector<Particle> src = d.Get<Particle>()[0];
  for (size_t i = 0; i < src.size(); ++i) {
    src[i].id = static_cast<int>(i);
    src[i].charge = static_cast<char>(i % 3);
    src[i].posCovMatrix[i % 3][(i + 1) % 3] = static_cast<double>(i);
  }
  ForEachLayout<Particle>([&]<class C>() {
    C c(src.size(), alignment);
    bench::Fill(c, src);
    size_t mismatches = 0;
    for (size_t i = 0; i < src.size(); ++i) {
      auto r = c[i];
      mismatches += r.id != src[i].id || r.pt != src[i].pt || r.eta != src[i].eta || r.phi != src[i].phi ||
                    r.e != src[i].e || r.charge != src[i].charge || r.posCovMatrix != src[i].posCovMatrix;
    }
    CHECK(mismatches == 0, "{}: {} elements differ after Fill", C::to_string(), mismatches);
  });
}

int main() {
  TestLayoutEnumeration();
  TestProblemSize<bench::InvariantMassSequential>();
  TestProblemSize<bench::InvariantMassRandom>();
  TestReadCsv();
  TestDatasets();

  Datasets d;
  d.SetDefaultListFile(data_dir + "/particle_inputs");
  d.Reserve<Particle>(1000, 2);
  d.Load();
  TestFill(d);

  // The reference for 1000 elements also validates fewer elements
  auto invariant_mass = ReadReference(data_dir + "/particles1_particles2_1000_InvariantMassSequential.validation");
  for (size_t n : {size_t{1000}, size_t{37}}) {
    TestKernel<bench::InvariantMassSequential>(d, invariant_mass, n);
    TestKernel<bench::InvariantMassRandom>(d, invariant_mass, n);
  }
  // Not registered by default, but kept working
  TestKernel<bench::DeltaR2Pairwise>(d, ReadReference(data_dir + "/particles1_200_DeltaR2Pairwise.validation"), 200);

  if (failures == 0) {
    std::printf("\033[1;32mPASSED\033[0m\n");
  } else {
    std::printf("\033[1;31m%d check(s) FAILED\033[0m\n", failures);
  }
  return failures == 0 ? 0 : 1;
}
