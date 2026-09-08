#ifndef UTILS_H
#define UTILS_H

#include <chrono>
#include <iostream>
#include <type_traits>
#include <vector>
#include <ranges>
#include <meta>

consteval auto nsdms(std::meta::info r) {
  return std::meta::nonstatic_data_members_of(r, std::meta::access_context::current());
}

inline size_t AlignSize(size_t size, size_t alignment) {
  return (size + alignment - 1) / alignment * alignment;
}

struct FileOpts {
  std::string input = "";      // Option "--input <string>"
  std::string validation = ""; // Option "--validation <string>"
  std::vector<double> factors = {0.25, 0.5, 0.9, 1, 1.1, 1.25, 2, 4}; // Option "--factors <string>"
  std::vector<int> cache_levels = {0, 1, 2}; // Option "--cache_levels <string>"
  std::vector<int> strides = {1}; // Option "--strides <string>"
};

struct ValidationInfo {
  std::string benchmark_name;
  size_t input_size;
  std::string validation_file;
};

// https://stackoverflow.com/questions/865668/parsing-command-line-arguments-in-c
class CmdLineParser {
public:
  CmdLineParser(int &argc, char **argv)
      : argc(argc), argv(argv), consumed(argc, false) {
    for (int i = 1; i < argc; ++i)
      this->tokens.push_back(std::string(argv[i]));
  }
  /* Look up an option and its value, marking both as consumed. */
  const std::string &GetCmdOption(const std::string &option) {
    auto itr = std::ranges::find(this->tokens, option);
    if (itr != this->tokens.end()) {
      MarkConsumed(itr);
      if (++itr != this->tokens.end()) {
        MarkConsumed(itr);
        return *itr;
      }
    }
    static const std::string empty_string("");
    return empty_string;
  }
  /// @author iain
  bool CmdOptionExists(const std::string &option) {
    auto itr = std::ranges::find(this->tokens, option);
    if (itr == this->tokens.end()) { return false; }
    MarkConsumed(itr);
    return true;
  }
  /* Drop every argument this parser consumed from argc/argv, so the remaining
     ones can be handed to another parser (e.g. google benchmark). */
  void RemoveParsedOptions() {
    int out = 1;
    for (int i = 1; i < argc; ++i) {
      if (!consumed[i]) { argv[out++] = argv[i]; }
    }
    for (int i = out; i < argc; ++i)
      argv[i] = nullptr;
    argc = out;
  }

private:
  void MarkConsumed(std::vector<std::string>::const_iterator itr) {
    // tokens[i] corresponds to argv[i + 1]
    consumed[std::distance(this->tokens.cbegin(), itr) + 1] = true;
  }

  int &argc;
  char **argv;
  std::vector<bool> consumed;
  std::vector<std::string> tokens;
};


/* Convert a time unit type to its string representation. */
template <typename Unit> std::string unit_to_string() {
  if constexpr (std::same_as<Unit, std::nano>) {
    return "ns";
  } else if constexpr (std::same_as<Unit, std::micro>) {
    return "us";
  } else if constexpr (std::same_as<Unit, std::milli>) {
    return "ms";
  } else if constexpr (std::same_as<Unit, std::ratio<1>>) {
    return "s";
  } else {
    return "unknown_unit";
  }
}


#endif // UTILS_H