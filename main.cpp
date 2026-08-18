#include <benchmark/benchmark.h> // https://github.com/google/benchmark

#include "benchmarks.h"
#include "datastructures.h"

#include <algorithm>
#include <random>

#include <likwid.h>

#include <filesystem>
#include <fstream>
#include <print>

#include <meta>

// TODO: make Particle generic
std::vector<std::vector<Particle>> input_data; // Cache for input data

// // Global options parsed from the command line.
FileOpts opts;

// Information for validation of benchmark results
std::vector<ValidationInfo> validation_data;

size_t Alignment = 64; // Default alignment, will be set based on CPU topology

/* Read Lorentzvector (pt, eta, phi, e) data from the given CSV file into
 * the container. */
void ReadData(std::string filename, size_t n,
              std::vector<std::vector<Particle>> &input_data) {
  std::ifstream i_stream(filename);
  if (i_stream.is_open()) {
    size_t file_num = 0;
    for (std::string input_file; getline(i_stream, input_file);) {
      input_data.push_back(std::vector<Particle>(n));

      std::ifstream ii_stream(input_file);
      if (ii_stream.is_open()) {
        std::string line;
        for (size_t i = 0; i < n; ++i) {
          if (!getline(ii_stream, line)) {
            throw std::runtime_error("Not enough data in file " + filename +
                                     " for requested size " +
                                     std::to_string(n));
            break;
          }

          std::stringstream ss(line);
          std::string token;

          getline(ss, token, ',');
          input_data[file_num][i].pt = std::stod(token);
          getline(ss, token, ',');
          input_data[file_num][i].eta = std::stod(token);
          getline(ss, token, ',');
          input_data[file_num][i].phi = std::stod(token);
          getline(ss, token, ',');
          input_data[file_num][i].e = std::stod(token);
        }
        ii_stream.close();
      } else {
        throw std::runtime_error("Failed to open file " + filename +
                                 " for reading");
      }

      file_num++;
    }
  } else {
    throw std::runtime_error("Failed to open file " + filename +
                             " for reading");
  }

  input_data.resize(n);
}

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
      for (size_t j = 0; j < results.size(); ++j) {
        if (std::abs(results[j] - expected_results[j]) > 1e-6) {
          std::fprintf(
              stderr,
              "\033[1;31mVALIDATION FAILED\033[0m for benchmark %s with input1 "
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
        "Usage: ./main [-h] [--input INPUT_CONFIG_FILE] "
        "[--validation VALIDATION_CONFIG_FILE] [--repetitions REPS]\n"
        "options:\n"
        "  --help                               Show this help message and exit\n"
        "  --input INPUT_CONFIG_FILE            File specifying files with input data\n"
        "  --validation VALIDATION_CONFIG_FILE  File containing the benchmark name, input size, and name of the file\n"
        "                                       with data to use for validation, separated by commas and one\n"
        "                                       benchmark per line\n"
      );
       std::exit(EXIT_SUCCESS);
  }

  auto input = cmdLineParser.GetCmdOption("--input");
  if (!input.empty()) { opts.input = input; }

  auto validation = cmdLineParser.GetCmdOption("--validation");
  if (!validation.empty()) { opts.validation = validation; }
  // clang-format on

  benchmark::AddCustomContext("input", opts.input);
  benchmark::AddCustomContext("validation", opts.validation);
}

template <class Container>
void BM_InvariantMassSequential(benchmark::State &state, size_t n) {
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

  // Get input data
  if (!opts.input.empty()) {
    ReadData(opts.input, *std::ranges::max_element(problem_sizes), input_data);
  } else {
    std::cerr << "No input data specified. Exiting." << std::endl;
    return EXIT_FAILURE;
  }

  // Read validation data if provided
  if (!opts.validation.empty()) {
    ParseValidationInfo(opts.validation);
  }

  benchmark::Initialize(&argc, argv);
  // if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  //////////////////////////////////////////////////////////////////////////

  for (auto &size : problem_sizes) {
    // Register benchmarks for each problem size
    template for (constexpr auto &c : std::define_static_array(members_of(
                      ^^containers, std::meta::access_context::current()))) {
      benchmark::RegisterBenchmark("BM_InvariantMassSequential",
                                   BM_InvariantMassSequential<typename[: c
                                   :]>, size)
          ->Unit(benchmark::kMillisecond)
          ->Name(std::string("InvariantMassSequential_") +
          std::string(identifier_of(c)));

      benchmark::RegisterBenchmark("BM_InvariantMassRandom",
                                  BM_InvariantMassRandom<typename[:c:]>, size)
          ->Unit(benchmark::kMillisecond)
          ->Name(std::string("InvariantMassRandom_") +
                std::string(identifier_of(c)));
    }
  }

  //////////////////////////////////////////////////////////////////////////

  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
}