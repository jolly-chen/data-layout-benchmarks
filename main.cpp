#include "benchmarks.h"
#include "datastructures.h"

#include <algorithm>
#include <random>

#include <string>

#include <chrono>
#include <likwid.h>

#include <filesystem>
#include <fstream>
#include <ios>
#include <iostream>
#include <sstream>

#include <papi.h>

#define CHECK_PAPI_RETURN(retval)                                              \
  if (retval != PAPI_OK) {                                                     \
    std::cerr << "PAPI error at " << __FILE__ << ":" << __LINE__ << ": "       \
              << PAPI_strerror(retval) << std::endl;                           \
    exit(1);                                                                   \
  }

int papi_eventset = PAPI_NULL;
size_t papi_nevents = 0;

using Clock = std::chrono::high_resolution_clock;
using unit = std::milli;

/* 2^16, maximum number of results to store to cap memory usage. */
size_t max_results_size = 65536;

std::vector<Particle> input1_data; // Cache for input1 data
std::vector<Particle> input2_data; // Cache for input2 data

// Global options parsed from the command line.
struct FileOpts {
  std::string input1 = "";                  // Option "--input1 <string>"
  std::string input2 = "";                  // Option "--input2 <string>"
  std::string output = "";                  // Option "--output <string>"
  std::string validation = "";              // Option "--validation <string>"
  bool aggregate = false;                   // Option "--aggregate <bool>"
  size_t repetitions = 5;                   // Option "--repetitions <int>"
  size_t warmup = 1;                        // Option "--warmup <int>"
  std::string papi_events = ""; // Option "--papi_events <string>"
};
extern FileOpts opts;
FileOpts opts;

// Information for validation of benchmark results
struct ValidationInfo {
  std::string benchmark_name;
  size_t input_size;
  size_t max_results_size;
  std::string validation_file;
};
std::vector<ValidationInfo> validation_data;

// Output stream for benchmark results. Can be std::cout or a file.
std::ostream *output;

/* Read Lorentzvector (pt, eta, phi, e) data from the given CSV file into
 * the container. */
void ReadData(std::string filename, size_t n,
              std::vector<Particle> &input_data) {
  input_data.resize(n);

  std::ifstream is(filename);
  if (is.is_open()) {
    std::string line;
    for (size_t i = 0; i < n; ++i) {
      if (!getline(is, line)) {
        throw std::runtime_error("Not enough data in file " + filename +
                                 " for requested size " + std::to_string(n));
        break;
      }

      std::stringstream ss(line);
      std::string token;
      std::vector<std::string> temp;

      getline(ss, token, ',');
      input_data[i].pt = std::stod(token);
      getline(ss, token, ',');
      input_data[i].eta = std::stod(token);
      getline(ss, token, ',');
      input_data[i].phi = std::stod(token);
      getline(ss, token, ',');
      input_data[i].e = std::stod(token);
    }
    is.close();
  } else {
    throw std::runtime_error("Failed to open file " + filename +
                             " for reading");
  }
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
      info.max_results_size = std::stoul(token);
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
void ValidateResults(const std::string benchmark_name,
                     const std::vector<double> &results, const size_t in_size) {
  if (!validation_data.empty()) {
    auto validation_file =
        std::ranges::find_if(validation_data, [=](const ValidationInfo &info) {
          return info.benchmark_name == benchmark_name &&
                 info.input_size == in_size &&
                 info.max_results_size == max_results_size;
        });

    if (validation_file != validation_data.end()) {
      auto expected_results =
          ReadValidationData(validation_file->validation_file);
      for (size_t j = 0; j < results.size(); ++j) {
        if (std::abs(results[j] - expected_results[j]) > 1e-6) {
          std::cerr << "\033[1;31mVALIDATION FAILED\033[0m for benchmark "
                    << benchmark_name << " with input1 size " << in_size
                    << " at index " << j << ": expected " << expected_results[j]
                    << ", got " << results[j] << std::endl;
          break;
        }
      }
    }
  }
}

/* Print configuration and timing information in csv format. */
template <typename Container, typename Unit>
void PrintTiming(const std::string &benchmark_name,
                 const std::vector<double> &measured_times,
                 const std::vector<std::vector<long long>> &event_states,
                 const size_t in_size) {
  if (opts.aggregate) {
    auto output_aggregate = [&](auto &arr) {
      double min = *std::ranges::min_element(arr);
      double max = *std::ranges::max_element(arr);
      double avg = std::reduce(arr.begin(), arr.end(), 0.0) / arr.size();
      double stddev =
          std::sqrt(std::reduce(arr.begin(), arr.end(), 0.0,
                                [avg](double acc, double t) {
                                  return acc + (t - avg) * (t - avg);
                                }) /
                    arr.size());
      *output << "," << min << "," << max << "," << avg << "," << stddev;
    };

    *output << benchmark_name << "," << Container::to_string() << "," << in_size
            << "," << sizeof(Container) << "," << unit_to_string<Unit>();
    output_aggregate(measured_times);
    for (size_t e = 0; e < event_states.size(); ++e) {
      output_aggregate(event_states[e]);
    }
    *output << std::endl;
  } else {
    for (size_t r = 0; r < measured_times.size(); ++r) {
      *output << benchmark_name << "," << Container::to_string() << ","
              << in_size << "," << sizeof(Container) << ","
              << unit_to_string<Unit>() << "," << measured_times[r];

      for (size_t e = 0; e < event_states.size(); ++e) {
        *output << "," << event_states[e][r];
      }

      *output << std::endl;
    }
  }
}

/* Run a single benchmark function that takes ONE containers of the given
 * container type.
 */
template <typename Container, typename BenchmarkFunc, typename... ExtraArgs>
void RunBenchmark1(BenchmarkFunc benchmarkfunc, std::string benchmark_name,
                   size_t in_size, size_t alignment, size_t out_size,
                   ExtraArgs... extra_args) {
  std::vector<double> measured_times;

  // Event count per repetition for each event.
  std::vector<std::vector<long long>> event_states(
      papi_nevents, std::vector<long long>(opts.repetitions));
  std::vector<long long> count(papi_nevents);

  for (size_t r = 0; r < opts.repetitions + opts.warmup; ++r) {
    // Initialize input container.
    Container v1(in_size, alignment);
    for (size_t i = 0; i < in_size; ++i) {
      v1[i].pt = input1_data[i].pt;
      v1[i].eta = input1_data[i].eta;
      v1[i].phi = input1_data[i].phi;
      v1[i].e = input1_data[i].e;
    }

    // Cap the results size to avoid excessive memory usage.
    std::vector<double> results(std::min(out_size, max_results_size));

    // Measure time taken by the benchmark function
    if (!opts.papi_events.empty()) {
      PAPI_reset(papi_eventset);
      CHECK_PAPI_RETURN(PAPI_start(papi_eventset));
    }
    auto start = Clock::now();
    benchmarkfunc(v1, results, extra_args...);
    auto end = Clock::now();
    if (!opts.papi_events.empty()) {
      CHECK_PAPI_RETURN(PAPI_stop(papi_eventset, count.data()));
    }

    // Skip warmup iterations
    if (r < opts.warmup) {
      continue;
    }

    // Gather performance counter data
    for (size_t e = 0; e < event_states.size(); ++e) {
      event_states[e][r - opts.warmup] = count[e];
    }

    ValidateResults(benchmark_name, results, in_size);

    std::chrono::duration<double, unit> elapsed = end - start;
    measured_times.push_back(elapsed.count());
  }

  PrintTiming<Container, unit>(benchmark_name, measured_times, event_states,
                               in_size);
}

/* Run a single benchmark function that takes TWO containers of the given
 * container type.
 */
template <typename Container, typename BenchmarkFunc, typename... ExtraArgs>
void RunBenchmark2(BenchmarkFunc benchmarkfunc, std::string benchmark_name,
                   size_t in_size, size_t alignment, size_t out_size,
                   ExtraArgs... extra_args) {
  std::vector<double> measured_times;

  // Event count per repetition for eachkernels::InvariantMassSequential event.
  std::vector<std::vector<long long>> event_states(
      papi_nevents, std::vector<long long>(opts.repetitions));
  std::vector<long long> count(papi_nevents);

  for (size_t r = 0; r < opts.repetitions + opts.warmup; ++r) {
    // Initialize input containers.
    Container v1(in_size, alignment), v2(in_size, alignment);
    for (size_t i = 0; i < in_size; ++i) {
      v1[i].pt = input1_data[i].pt;
      v1[i].eta = input1_data[i].eta;
      v1[i].phi = input1_data[i].phi;
      v1[i].e = input1_data[i].e;
      v2[i].pt = input2_data[i].pt;
      v2[i].eta = input2_data[i].eta;
      v2[i].phi = input2_data[i].phi;
      v2[i].e = input2_data[i].e;
    }

    // Cap the results size to avoid excessive memory usage.
    std::vector<double> results(std::min(out_size, max_results_size));

    // Measure time taken by the benchmark function
    if (!opts.papi_events.empty()) {
      PAPI_reset(papi_eventset);
      CHECK_PAPI_RETURN(PAPI_start(papi_eventset));
    }
    auto start = Clock::now();
    benchmarkfunc(v1, v2, results, extra_args...);
    auto end = Clock::now();
    if (!opts.papi_events.empty()) {
      CHECK_PAPI_RETURN(PAPI_stop(papi_eventset, count.data()));
    }

    // Skip warmup iterations
    if (r < opts.warmup) {
      continue;
    }

    // Gather performance counter data
    for (size_t e = 0; e < event_states.size(); ++e) {
      event_states[e][r - opts.warmup] = count[e];
    }

    ValidateResults(benchmark_name, results, in_size);

    std::chrono::duration<double, unit> elapsed = end - start;
    measured_times.push_back(elapsed.count());
  }

  PrintTiming<Container, unit>(benchmark_name, measured_times, event_states,
                               in_size);
}

/* Run all benchmarks defined in benchmarks.h. */
template <typename Container>
void RunAllBenchmarks(size_t n, size_t alignment) {
  RunBenchmark2<Container>(kernels::InvariantMassSequential<Container>,
                           "InvariantMassSequential", n, alignment, n);

  std::vector<size_t> indices(n);
  std::iota(begin(indices), end(indices), 0);
  std::mt19937 rng(123);
  std::shuffle(begin(indices), end(indices), rng);
  RunBenchmark2<Container>(kernels::InvariantMassRandom<Container>,
                           "InvariantMassRandom", n, alignment, n, indices);

  // RunBenchmark1<Container>(kernels::DeltaR2Pairwise<Container>,
  //                          "DeltaR2Pairwise", n, alignment,
  //                          round(n * (n - 1) / 2));
}

/* Parse command-line options.
   Taken from
   https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p2996r13.html#parsing-command-line-options
 */
void ParseOptions(int &argc, char **argv) {
  CmdLineParser cmdLineParser(argc, argv);
  if (cmdLineParser.CmdOptionExists("--help") ||
      cmdLineParser.CmdOptionExists("-h")) {
    // clang-format off
    std::cout
        << "Usage: ./main [-h] [--input1 INPUT_FILE] [--input2 INPUT_FILE] "
           "[--output OUTPUT_FILE]\n"
        << "              [--validation VALIDATION_FILE] [--repetitions REPS]\n"
        << "options:\n"
        << "  --help                          Show this help message and exit\n"
        << "  --input1 INPUT_FILE             File containing input1 data\n"
        << "  --input2 INPUT_FILE             File containing input2 data\n"
        << "  --output OUTPUT_FILE            File to write results to\n"
        << "  --validation VALIDATION_FILE    File containing the benchmark name, input1 size, max results size, and\n"
        << "                                  name of the file with data to use for validation, separated by commas\n"
        << "                                  and one benchmark per line\n"
        << "  --aggregate {1|0}               Print aggregate results or each repetition (default: 0)\n"
        << "  --repetitions REPS              Number of times to repeat each benchmark (default: 5)\n"
        << "  --warmup WARMUP                 Number of warmup iterations before timing (default: 1)\n"
        << "  --papi_events GROUP             PAPI events to use for counting hardware counters. \n"
        << "                                  Check available events using papi_avail (default: PAPI_TOT_CYC)\n";
       std::exit(EXIT_SUCCESS);
  }

  auto input1 = cmdLineParser.GetCmdOption("--input1");
  if (!input1.empty()) { opts.input1 = input1; }

  auto input2 = cmdLineParser.GetCmdOption("--input2");
  if (!input2.empty()) { opts.input2 = input2; }

  auto output = cmdLineParser.GetCmdOption("--output");
  if (!output.empty()) { opts.output = output; }

  auto validation = cmdLineParser.GetCmdOption("--validation");
  if (!validation.empty()) { opts.validation = validation; }

  auto aggregate = cmdLineParser.GetCmdOption("--aggregate");
  if (!aggregate.empty()) { opts.aggregate = (aggregate == "1"); }

  auto repetitions = cmdLineParser.GetCmdOption("--repetitions");
  if (!repetitions.empty()) { opts.repetitions = std::stoul(repetitions); }

  auto warmup = cmdLineParser.GetCmdOption("--warmup");
  if (!warmup.empty()) { opts.warmup = std::stoul(warmup); }

  auto papi_events = cmdLineParser.GetCmdOption("--papi_events");
  if (!papi_events.empty()) { opts.papi_events = papi_events; }
  // clang-format on

  std::cerr << "Configuration:" << "\n    input1:" << opts.input1
            << "\n    input2=" << opts.input2 << "\n    output=" << opts.output
            << "\n    validation=" << opts.validation
            << "\n    aggregate=" << opts.aggregate
            << "\n    repetitions=" << opts.repetitions
            << "\n    warmup=" << opts.warmup
            << "\n    papi_events=" << opts.papi_events << std::endl;
}

int main(int argc, char *argv[]) {
  ParseOptions(argc, argv);

  // Get problem sizes and alignment
  auto err = topology_init();
  if (err < 0) {
    fprintf(stderr, "Failed to initialize LIKWID's topology module\n");
    return EXIT_FAILURE;
  }
  CpuTopology_t topo = get_cpuTopology();

  // Initialize PAPI for performance measurement
  if (!opts.papi_events.empty()) {
    auto retval = PAPI_library_init(PAPI_VER_CURRENT);
    if (retval != PAPI_VER_CURRENT) {
      std::cerr << "Error initializing PAPI! " << PAPI_strerror(retval)
                << std::endl;
      return EXIT_FAILURE;
    }
    CHECK_PAPI_RETURN(PAPI_create_eventset(&papi_eventset));

    // Add specified comma-separated PAPI events to the event set
    std::stringstream ss(opts.papi_events);
    std::string event;
    while (std::getline(ss, event, ',')) {
      CHECK_PAPI_RETURN(PAPI_add_named_event(papi_eventset, event.c_str()));
      papi_nevents++;
    }
  }

  std::vector<size_t> problem_sizes;
  problem_sizes.push_back(topo->cacheLevels[0].size / sizeof(Particle) /
                          3); // Fits in L1 Cache
  problem_sizes.push_back(topo->cacheLevels[topo->numCacheLevels - 1].size /
                          sizeof(Particle)); // Does not fit in any cache
  size_t alignment = topo->cacheLevels[0].lineSize;

  // Get input1 data
  if (!opts.input1.empty()) {
    ReadData(opts.input1, *std::ranges::max_element(problem_sizes),
             input1_data);
  } else {
    std::cerr << "No input1 file specified. Exiting." << std::endl;
    return EXIT_FAILURE;
  }

  // Get input2 data
  if (!opts.input2.empty()) {
    ReadData(opts.input2, *std::ranges::max_element(problem_sizes),
             input2_data);
  } else {
    std::cerr << "No input2 file specified. Exiting." << std::endl;
    return EXIT_FAILURE;
  }

  // Read validation data if provided
  if (!opts.validation.empty()) {
    ParseValidationInfo(opts.validation);
  }

  // Check if output file exists to determine whether to write header
  bool write_header = true;
  if (std::filesystem::exists(opts.output)) {
    write_header = false;
  }

  // Determine output stream: file or standard output
  std::ofstream output_file(opts.output,
                            std::ios_base::app | std::ios_base::out);
  if (output_file.is_open()) {
    output = &output_file;
  } else {
    output = &std::cout;
  }

  if (write_header) {
    *output << "benchmark,container,problem_size,container_byte_size,time_"
               "unit";

    if (opts.aggregate) {
      // Write header for CSV if the output file does not already exist.
      *output << ",min_time,max_time,avg_time,stddev_time";

      std::stringstream ss(opts.papi_events);
      std::string event;
      while (std::getline(ss, event, ',')) {
        *output << ",min_" << event << ",max_" << event << ",avg_" << event
                << ",stddev_" << event;
      }

      *output << "\n";
    } else {
      *output << ",time" << (!opts.papi_events.empty() ? "," : "")
              << opts.papi_events << "\n";
    }
  }

  for (size_t n : problem_sizes) {
    // THIS IS GENERATED USING generate_datastructures.py
    RunAllBenchmarks<PartitionedContainer0123456>(n, alignment);
  }

  if (!opts.papi_events.empty()) {
    PAPI_cleanup_eventset(papi_eventset);
    PAPI_destroy_eventset(&papi_eventset);
  }
  return 0;
}
// END GENERATED CODE
