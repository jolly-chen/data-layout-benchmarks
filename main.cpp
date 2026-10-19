#include <benchmark/benchmark.h> // https://github.com/google/benchmark

#include "benchmarks.h"
#include "datasets.h"
#include "layout_helpers.h"
#include "layouts_index.h"
#include "utils.h"

#include <algorithm>
#include <map>
#include <meta>

#include <likwid.h>

#include <filesystem>
#include <fstream>
#include <ios>
#include <print>

// Input data of the benchmarks, per element type
Datasets datasets;

// // Global options parsed from the command line.
FileOpts opts;

// Information for validation of benchmark results
std::vector<ValidationInfo> validation_data;

size_t Alignment = 64; // Default alignment, will be set based on CPU topology

// Problem sizes are chosen so that the input and output containers of a
// benchmark take up this many bytes, as a multiple of the cache size:
constexpr double SmallTargetL1Factor = 1.0;  // fits in the L1 cache
constexpr double LargeTargetLLCFactor = 2.0; // does not fit in any cache

/* Read validation data from the given file, caching it for later calls. */
const std::vector<double> &ReadValidationData(const std::string &filename) {
  static std::map<std::string, std::vector<double>> cache;
  if (auto it = cache.find(filename); it != cache.end()) { return it->second; }

  std::ifstream is(filename);
  if (!is.is_open()) {
    throw std::runtime_error("Failed to open file " + filename + " for reading");
  }
  std::vector<double> data;
  size_t line_num = 0;
  for (std::string line; getline(is, line);) {
    ++line_num;
    auto token = datasets_detail::Trim(line);
    if (token.empty()) { continue; }
    double value;
    auto [end, ec] = std::from_chars(token.data(), token.data() + token.size(), value);
    if (ec != std::errc{} || end != token.data() + token.size()) {
      throw std::runtime_error(std::format("Line {} of {}: '{}' is not a valid value", line_num, filename, token));
    }
    data.push_back(value);
  }

  return cache.emplace(filename, std::move(data)).first->second;
}

/* Parse validation info from the given CSV file, with one
 * "<benchmark>,<input size>,<validation file>" line per benchmark, and read
 * the validation data, so a missing file is reported before any benchmark
 * runs. Relative paths are relative to the directory of the CSV file. */
void ParseValidationInfo(const std::string &filename) {
  std::ifstream is(filename);
  if (!is.is_open()) {
    throw std::runtime_error("Failed to open validation file " + filename + " for reading");
  }

  const auto dir = std::filesystem::path(filename).parent_path();
  size_t line_num = 0;
  for (std::string line; getline(is, line);) {
    ++line_num;
    if (datasets_detail::Trim(line).empty()) { continue; }

    auto tokens = datasets_detail::SplitCsvLine(line);
    size_t input_size = 0;
    bool valid = tokens.size() == 3 && !tokens[0].empty() && !tokens[2].empty();
    if (valid) {
      auto [end, ec] = std::from_chars(tokens[1].data(), tokens[1].data() + tokens[1].size(), input_size);
      valid = ec == std::errc{} && end == tokens[1].data() + tokens[1].size();
    }
    if (!valid) {
      throw std::runtime_error(std::format(
          "Line {} of {} is not of the form <benchmark>,<input size>,<validation file>: '{}'",
          line_num, filename, datasets_detail::Trim(line)));
    }

    std::filesystem::path path(tokens[2]);
    ValidationInfo info{std::string(tokens[0]), input_size,
                        (path.is_relative() ? dir / path : path).string()};
    ReadValidationData(info.validation_file);
    validation_data.push_back(std::move(info));
  }
}

/* Validate results if a validation file is provided. A file for exactly
 * in_size is preferred; if allow_prefix is set, the file for the smallest
 * larger input size is used otherwise, comparing its first results.size()
 * values. Returns 1 if the results match, -1 if they do not, and 0 if there is
 * no validation file for this benchmark and size. */
int ValidateResults(const std::string_view benchmark_name,
                    std::span<const double> results, const size_t in_size,
                    bool allow_prefix) {
  const ValidationInfo *match = nullptr;
  for (const auto &info : validation_data) {
    if (info.benchmark_name != benchmark_name) { continue; }
    if (info.input_size == in_size) {
      match = &info;
      break;
    }
    if (allow_prefix && info.input_size > in_size &&
        (!match || info.input_size < match->input_size)) {
      match = &info;
    }
  }
  if (!match) { return 0; }

  const auto &expected_results = ReadValidationData(match->validation_file);
  if (expected_results.size() < results.size()) {
    std::fprintf(stderr,
                 "\033[1;31mVALIDATION FAILED\033[0m for benchmark %s with input "
                 "size %zu: %s has %zu values, expected at least %zu\n",
                 benchmark_name.data(), in_size, match->validation_file.c_str(),
                 expected_results.size(), results.size());
    return -1;
  }
  for (size_t j = 0; j < results.size(); ++j) {
    // Written so that NaN results fail too
    if (!(std::abs(results[j] - expected_results[j]) <= 1e-6)) {
      std::fprintf(
          stderr,
          "\033[1;31mVALIDATION FAILED\033[0m for benchmark %s with input "
          "size %zu at index %zu: expected %f, got %f\n",
          benchmark_name.data(), in_size, j, expected_results[j],
          results[j]);
      return -1;
    }
  }
  return 1;
}

// /* Parse command-line options.
//    Taken from
//    https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p2996r13.html#parsing-command-line-options
//  */
void ParseOptions(int &argc, char **argv) {
  CmdLineParser cmdLineParser(argc, argv);
  if (cmdLineParser.CmdOptionExists("--help") ||
      cmdLineParser.CmdOptionExists("-h")) {
    // clang-format off
    std::printf(
        "Usage: ./main [-h] --input [STRUCT=]INPUT_LIST_FILE [--input ...] "
        "[--validation VALIDATION_CONFIG_FILE] [google benchmark options]\n"
        "options:\n"
        "  --help                               Show this help message and exit\n"
        "  --input [STRUCT=]INPUT_LIST_FILE     File listing the input datasets, one path per line, relative to the\n"
        "                                       directory of the list file unless absolute. Each dataset is\n"
        "                                       a CSV file whose first row is a header with names of data members of\n"
        "                                       the struct (e.g. 'pt,eta,phi,e' for Particle), followed by one row per\n"
        "                                       element. Members without a column are zero-initialized. Each dataset\n"
        "                                       needs at least as many rows as the largest problem size. The order\n"
        "                                       of the lines matters: a benchmark with two inputs reads the first\n"
        "                                       dataset into its first container and the second into its second.\n"
        "                                       'STRUCT=' sets the list file for one struct only; without it, the\n"
        "                                       file is used for every struct that has no list file of its own.\n"
        "                                       May be repeated.\n"
        "  --validation VALIDATION_CONFIG_FILE  File containing the benchmark name, input size, and name of the file\n"
        "                                       with data to use for validation, separated by commas and one\n"
        "                                       benchmark per line\n"
      );
       std::exit(EXIT_SUCCESS);
  }

  opts.inputs = cmdLineParser.GetCmdOptions("--input");

  auto validation = cmdLineParser.GetCmdOption("--validation");
  if (!validation.empty()) { opts.validation = validation; }
  // clang-format on

  // Strip our own options so they are not seen by google benchmark's parser.
  cmdLineParser.RemoveParsedOptions();

  std::string inputs;
  for (const auto &input : opts.inputs) {
    if (!inputs.empty()) { inputs += " "; }
    inputs += input;
  }
  benchmark::AddCustomContext("input", inputs);
  benchmark::AddCustomContext("validation", opts.validation);
}

/* Run benchmark K on a container with layout C and n elements. */
template <class K, class C>
  requires bench::BenchmarkFor<K, C>
void BM_Generic(benchmark::State &state, size_t n, size_t target_bytes) {
  typename K::template Setup<C> s(n, Alignment, datasets);

  for (auto _ : state) {
    K::run(s);
    benchmark::DoNotOptimize(s);
    benchmark::ClobberMemory();
  }

  if (!opts.validation.empty()) {
    state.counters["validated"] = ValidateResults(K::validation_name, K::output(s), n, K::validate_prefix);
  }

  state.counters["problem_size"] = n;
  state.counters["bytes_for_all"] = K::template bytes_for_all<C>(n, Alignment);
  state.counters["target_bytes"] = target_bytes;
}

/* Register benchmark K for every generated layout of K::value_type and every
 * target size, and reserve the input data it needs. */
template <bench::BenchmarkDescriptor K>
void Register(const std::vector<size_t> &target_bytes) {
  ForEachLayout<typename K::value_type>([&]<class C>() {
    static_assert(bench::BenchmarkFor<K, C>, "benchmark descriptor does not support this layout");
    for (size_t target : target_bytes) {
      const size_t n = bench::ProblemSize<K, C>(target, Alignment);
      datasets.Reserve<typename K::value_type>(n, K::num_inputs);
      benchmark::RegisterBenchmark(std::string(K::name) + "_" + C::to_string(), BM_Generic<K, C>, n, target)
          ->Unit(benchmark::kMillisecond);
    }
  });
}

template <bench::BenchmarkDescriptor... K>
void RegisterAll(std::tuple<K...> *, const std::vector<size_t> &target_bytes) {
  (Register<K>(target_bytes), ...);
}

int main(int argc, char **argv) {
  benchmark::MaybeReenterWithoutASLR(argc, argv);
  ParseOptions(argc, argv);

  // Get problem sizes and alignment
  auto err = topology_init();
  if (err < 0) {
    fprintf(stderr, "Failed to initialize LIKWID's topology module\n");
    return EXIT_FAILURE;
  }
  CpuTopology_t topo = get_cpuTopology();

  const std::vector<size_t> target_bytes = {
      static_cast<size_t>(SmallTargetL1Factor * topo->cacheLevels[0].size),
      static_cast<size_t>(LargeTargetLLCFactor * topo->cacheLevels[topo->numCacheLevels - 1].size)};
  Alignment = topo->cacheLevels[0].lineSize;

  // Get input data
  if (opts.inputs.empty()) {
    std::cerr << "No input data specified. Exiting." << std::endl;
    return EXIT_FAILURE;
  }
  for (const auto &input : opts.inputs) {
    if (auto eq = input.find('='); eq != std::string::npos) {
      datasets.SetListFile(input.substr(0, eq), input.substr(eq + 1));
    } else {
      datasets.SetDefaultListFile(input);
    }
  }

  // Read validation data if provided
  if (!opts.validation.empty()) {
    try {
      ParseValidationInfo(opts.validation);
    } catch (const std::exception &e) {
      std::cerr << e.what() << std::endl;
      return EXIT_FAILURE;
    }
  }

  benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  //////////////////////////////////////////////////////////////////////////

  RegisterAll(static_cast<bench::AllKernels *>(nullptr), target_bytes);

  try {
    datasets.Load();
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
