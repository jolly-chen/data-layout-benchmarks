#ifndef UTILS_H
#define UTILS_H

#include "benchmarks.h"
#include <chrono>
#include <iostream>
#include <type_traits>
#include <vector>

inline size_t AlignSize(size_t size, size_t alignment) {
  return (size + alignment - 1) / alignment * alignment;
}

// https://stackoverflow.com/questions/865668/parsing-command-line-arguments-in-c
class CmdLineParser {
public:
  CmdLineParser(int &argc, char **argv) {
    for (int i = 1; i < argc; ++i)
      this->tokens.push_back(std::string(argv[i]));
  }
  const std::string &GetCmdOption(const std::string &option) const {
    std::vector<std::string>::const_iterator itr;
    itr = std::find(this->tokens.begin(), this->tokens.end(), option);
    if (itr != this->tokens.end() && ++itr != this->tokens.end()) {
      return *itr;
    }
    static const std::string empty_string("");
    return empty_string;
  }
  /// @author iain
  bool CmdOptionExists(const std::string &option) const {
    return std::find(this->tokens.begin(), this->tokens.end(), option) !=
           this->tokens.end();
  }

private:
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