#include <benchmark/benchmark.h> // https://github.com/google/benchmark

#include "benchmarks.h"
#include "datasets.h"
#include "layout_helpers.h"
#include "layouts_index.h"
#include "utils.h"

#include <algorithm>
#include <random>

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

/* Parse validation info from the given CSV file. */
void ParseValidationInfo(const std::string filename) {
  std::ifstream is(filename);
  if (is.is_open()) {
    std::string line;
    while (getline(is, line)) {
      std::stringstream ss(line);
      std::string token;
      ValidationInfo info;

      getline(ss, token, ',');
      info.benchmark_name = token;
      getline(ss, token, ',');
      info.input_size = std::stoul(token);
      getline(ss, token, ',');
      info.validation_file = token;

      validation_data.push_back(info);
    }
    is.close();
  } else {
    throw std::runtime_error("Failed to open validation file " + filename +
                             " for reading");
  }
}

/* Read validation data from the given file into the container. */
std::vector<double> ReadValidationData(const std::string filename) {
  std::vector<double> data;

  std::ifstream is(filename);
  if (is.is_open()) {
    std::string line;
    while (getline(is, line)) {
      data.push_back(std::stod(line));
    }
  } else {
    throw std::runtime_error("Failed to open file " + filename +
                             " for reading");
  }

  return data;
}

/* Validate results if a validation file is provided. */
void ValidateResults(const std::string_view benchmark_name,
                     const std::vector<double> &results, const size_t in_size) {
  if (!validation_data.empty()) {
    auto validation_file =
        std::ranges::find_if(validation_data, [=](const ValidationInfo &info) {
          return info.benchmark_name == benchmark_name &&
                 info.input_size == in_size;
        });

    if (validation_file != validation_data.end()) {
      auto expected_results =
          ReadValidationData(validation_file->validation_file);
      if (expected_results.size() < results.size()) {
        std::fprintf(stderr,
                     "\033[1;31mVALIDATION FAILED\033[0m for benchmark %s with "
                     "input size %zu: %s has %zu values, expected at least %zu\n",
                     benchmark_name.data(), in_size,
                     validation_file->validation_file.c_str(),
                     expected_results.size(), results.size());
        return;
      }
      for (size_t j = 0; j < results.size(); ++j) {
        if (std::abs(results[j] - expected_results[j]) > 1e-6) {
          std::fprintf(
              stderr,
              "\033[1;31mVALIDATION FAILED\033[0m for benchmark %s with input "
              "size %zu at index %zu: expected %f, got %f\n",
              benchmark_name.data(), in_size, j, expected_results[j],
              results[j]);
          break;
        }
      }
    }
  }
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

template <class Container>
void BM_InvariantMassSequential(benchmark::State &state, size_t n) {
  const auto &input_data = datasets.Get<Particle>();
  Container v1(n, Alignment), v2(n, Alignment);
  for (size_t i = 0; i < n; ++i) {
    v1[i].pt = input_data[0][i].pt;
    v1[i].eta = input_data[0][i].eta;
    v1[i].phi = input_data[0][i].phi;
    v1[i].e = input_data[0][i].e;
    v2[i].pt = input_data[1][i].pt;
    v2[i].eta = input_data[1][i].eta;
    v2[i].phi = input_data[1][i].phi;
    v2[i].e = input_data[1][i].e;
  }
  std::vector<double> results(n);

  for (auto _ : state) {
    kernels::InvariantMassSequential(v1, v2, results);
    benchmark::DoNotOptimize(results);
    benchmark::ClobberMemory();
  }

  if (!opts.validation.empty()) {
    ValidateResults("InvariantMassSequential", results, n);
    benchmark::DoNotOptimize(results);
    benchmark::ClobberMemory();
  }

  state.counters["problem_size"] = n;
  state.counters["reference_size"] = sizeof(v1[0]);
}

template <class Container>
void BM_InvariantMassRandom(benchmark::State &state, size_t n) {
  const auto &input_data = datasets.Get<Particle>();
  Container v1(n, Alignment), v2(n, Alignment);
  for (size_t i = 0; i < n; ++i) {
    v1[i].pt = input_data[0][i].pt;
    v1[i].eta = input_data[0][i].eta;
    v1[i].phi = input_data[0][i].phi;
    v1[i].e = input_data[0][i].e;
    v2[i].pt = input_data[1][i].pt;
    v2[i].eta = input_data[1][i].eta;
    v2[i].phi = input_data[1][i].phi;
    v2[i].e = input_data[1][i].e;
  }
  std::vector<double> results(n);

  std::vector<size_t> indices(n);
  std::iota(begin(indices), end(indices), 0);
  std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count());
  std::shuffle(begin(indices), end(indices), rng);

  for (auto _ : state) {
    kernels::InvariantMassRandom(v1, v2, results, indices);
    benchmark::DoNotOptimize(results);
    benchmark::ClobberMemory();
  }

  if (!opts.validation.empty()) {
    ValidateResults("InvariantMassRandom", results, n);
    benchmark::DoNotOptimize(results);
    benchmark::ClobberMemory();
  }

  state.counters["problem_size"] = n;
  state.counters["reference_size"] = sizeof(v1[0]);
}

/* Run all benchmarks defined in benchmarks.h. */
template <typename Container>
void RunAllBenchmarks(size_t n) {
  // Register benchmarks for a specific problem size
  benchmark::RegisterBenchmark("BM_InvariantMassSequential", BM_InvariantMassSequential<Container>, n)
    ->Unit(benchmark::kMillisecond)
    ->Name(std::string("InvariantMassSequential_") + std::string(Container::to_string()));

  benchmark::RegisterBenchmark("BM_InvariantMassRandom", BM_InvariantMassRandom<Container>, n)
    ->Unit(benchmark::kMillisecond)
    ->Name(std::string("InvariantMassRandom_") + std::string(Container::to_string()));
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

  std::vector<size_t> problem_sizes;
  problem_sizes.push_back(topo->cacheLevels[0].size / sizeof(Particle) /
                          3); // Fits in L1 Cache
  problem_sizes.push_back(topo->cacheLevels[topo->numCacheLevels - 1].size /
                          sizeof(Particle) / 2); // Does not fit in any cache
  Alignment = topo->cacheLevels[0].lineSize;

  // Get input data: the benchmarks read the first two Particle datasets
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
  datasets.Reserve<Particle>(*std::ranges::max_element(problem_sizes), 2);
  try {
    datasets.Load();
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  // Read validation data if provided
  if (!opts.validation.empty()) {
    ParseValidationInfo(opts.validation);
  }
  
  benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  //////////////////////////////////////////////////////////////////////////

  // Register the benchmarks for every generated layout of Particle
  for (auto &size : problem_sizes) {
    ForEachLayout<Particle>([&]<class C>() { RunAllBenchmarks<C>(size); });
  }

  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
