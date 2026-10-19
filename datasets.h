#ifndef DATASETS_H
#define DATASETS_H

// Reading benchmark input data into std::vector<T> for any aggregate T, using
// reflection to map CSV columns to the data members of T.

#include <algorithm>
#include <any>
#include <charconv>
#include <cstddef>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <map>
#include <meta>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace datasets_detail {

template <class T>
constexpr auto members = std::define_static_array(
    std::meta::nonstatic_data_members_of(^^T, std::meta::access_context::unchecked()));

/* text without leading and trailing whitespace (including the '\r' of CRLF line endings). */
inline std::string_view Trim(std::string_view text) {
  auto first = text.find_first_not_of(" \t\r");
  if (first == std::string_view::npos) { return {}; }
  return text.substr(first, text.find_last_not_of(" \t\r") - first + 1);
}

/* Split line at commas, trimming whitespace around each token. */
inline std::vector<std::string_view> SplitCsvLine(std::string_view line) {
  std::vector<std::string_view> tokens;
  while (true) {
    auto comma = line.find(',');
    tokens.push_back(Trim(line.substr(0, comma)));
    if (comma == std::string_view::npos) { return tokens; }
    line.remove_prefix(comma + 1);
  }
}

/* Comma-separated names of the data members of T, for error messages. If
   readable_only is set, only members that can be read from a CSV token. */
template <class T>
std::string MemberNames(bool readable_only = false) {
  std::string names;
  template for (constexpr auto m : members<T>) {
    if (!readable_only || std::is_arithmetic_v<typename[:std::meta::type_of(m):]>) {
      if (!names.empty()) { names += ","; }
      names += std::meta::identifier_of(m);
    }
  }
  return names;
}

/* Index of the data member of T called name, or -1 if there is none. */
template <class T>
int MemberIndex(std::string_view name) {
  int i = 0;
  template for (constexpr auto m : members<T>) {
    if (std::meta::identifier_of(m) == name) { return i; }
    ++i;
  }
  return -1;
}

/* Whether the idx-th data member of T has a type that can be read from a CSV token. */
template <class T>
bool IsReadableMember(int idx) {
  int i = 0;
  template for (constexpr auto m : members<T>) {
    if (i++ == idx) { return std::is_arithmetic_v<typename[:std::meta::type_of(m):]>; }
  }
  return false;
}

/* Parse token into the idx-th data member of obj. Returns false if token is not a valid value. */
template <class T>
bool SetMember(T &obj, int idx, std::string_view token) {
  int i = 0;
  template for (constexpr auto m : members<T>) {
    if (i++ == idx) {
      using M = typename[:std::meta::type_of(m):];
      if constexpr (std::is_same_v<M, bool>) {
        if (token != "0" && token != "1") { return false; }
        obj.[:m:] = token == "1";
        return true;
      } else if constexpr (std::is_arithmetic_v<M>) {
        auto [end, ec] = std::from_chars(token.data(), token.data() + token.size(), obj.[:m:]);
        return ec == std::errc{} && end == token.data() + token.size();
      }
      return false;
    }
  }
  return false;
}

} // namespace datasets_detail

/* Read the first rows rows of the CSV file at path into a vector of T.
 *
 * The first line of the file must be a header row with the names of data
 * members of T, e.g. "pt,eta,phi,e". Each following line holds the values for
 * one element in the same order. Members of T without a column keep their
 * value-initialized value. */
template <class T>
std::vector<T> ReadCsv(const std::string &path, size_t rows) {
  using namespace datasets_detail;
  constexpr std::string_view type_name = std::meta::identifier_of(^^T);

  std::ifstream is(path);
  if (!is.is_open()) { throw std::runtime_error("Failed to open file " + path + " for reading"); }

  std::string line;
  if (!std::getline(is, line)) {
    throw std::runtime_error(std::format("File {} is empty, expected a header row", path));
  }
  // Skip the UTF-8 byte order mark that some programs write at the start of a file
  if (line.starts_with("\xEF\xBB\xBF")) { line.erase(0, 3); }

  std::vector<int> column_member;
  for (auto column : SplitCsvLine(line)) {
    int idx = MemberIndex<T>(column);
    if (idx < 0) {
      throw std::runtime_error(std::format(
          "Column '{}' in the header row of {} is not a data member of {}. Every dataset must "
          "start with a header row naming data members of {} ({}), e.g. '{}'",
          column, path, type_name, type_name, MemberNames<T>(), MemberNames<T>(true)));
    }
    if (!IsReadableMember<T>(idx)) {
      throw std::runtime_error(std::format(
          "Column '{}' in the header row of {} names a data member of {} with a non-arithmetic "
          "type, which cannot be read from a CSV file",
          column, path, type_name));
    }
    if (std::ranges::contains(column_member, idx)) {
      throw std::runtime_error(
          std::format("Column '{}' appears twice in the header row of {}", column, path));
    }
    column_member.push_back(idx);
  }

  std::vector<T> data(rows);
  for (size_t i = 0; i < rows; ++i) {
    if (!std::getline(is, line)) {
      throw std::runtime_error(std::format(
          "Not enough data in file {} for requested size {} (found {} rows after the header row)",
          path, rows, i));
    }

    auto tokens = SplitCsvLine(line);
    if (tokens.size() != column_member.size()) {
      throw std::runtime_error(std::format("Row {} of {} has {} values, but the header row has {} columns",
                                           i + 2, path, tokens.size(), column_member.size()));
    }
    for (size_t c = 0; c < tokens.size(); ++c) {
      if (!SetMember(data[i], column_member[c], tokens[c])) {
        throw std::runtime_error(
            std::format("Row {} of {}: '{}' is not a valid value", i + 2, path, tokens[c]));
      }
    }
  }

  return data;
}

/* Read every dataset listed in list_file (one path per line) with ReadCsv<T>.
 * Relative paths are relative to the directory of list_file. */
template <class T>
std::vector<std::vector<T>> LoadCsv(const std::string &list_file, size_t rows) {
  std::ifstream is(list_file);
  if (!is.is_open()) { throw std::runtime_error("Failed to open file " + list_file + " for reading"); }

  const auto dir = std::filesystem::path(list_file).parent_path();
  std::vector<std::vector<T>> data;
  for (std::string line; std::getline(is, line);) {
    std::filesystem::path path(datasets_detail::Trim(line));
    if (path.empty()) { continue; }
    data.push_back(ReadCsv<T>((path.is_relative() ? dir / path : path).string(), rows));
  }
  return data;
}

/* Input data for the benchmarks, one set of datasets per element type.
 *
 * Benchmarks first declare what they need with Reserve<T>(rows, files); Load()
 * then reads each type once, with the largest number of rows any benchmark
 * needs, and Get<T>() returns the data. */
class Datasets {
public:
  /* Use list_file for the datasets of the struct called type_name. */
  void SetListFile(std::string type_name, std::string list_file) {
    list_files[std::move(type_name)] = std::move(list_file);
  }

  /* Use list_file for the datasets of every struct without its own list file. */
  void SetDefaultListFile(std::string list_file) { default_list_file = std::move(list_file); }

  template <class T>
  void Reserve(size_t rows, size_t files) {
    auto &entry = entries[std::string(std::meta::identifier_of(^^T))];
    entry.rows = std::max(entry.rows, rows);
    entry.files = std::max(entry.files, files);
    entry.loader = [](const std::string &list_file, size_t n) -> std::any { return LoadCsv<T>(list_file, n); };
    entry.count = [](const std::any &data) { return std::any_cast<const std::vector<std::vector<T>> &>(data).size(); };
  }

  void Load() {
    // A list file for a struct no benchmark reads is most likely a typo
    for (const auto &[type_name, list_file] : list_files) {
      if (!entries.contains(type_name)) {
        std::string reserved;
        for (const auto &[name, entry] : entries) { reserved += (reserved.empty() ? "" : ", ") + name; }
        throw std::runtime_error(std::format("--input {}={}: no benchmark reads {}; the benchmarks read: {}",
                                             type_name, list_file, type_name, reserved));
      }
    }

    for (auto &[type_name, entry] : entries) {
      auto list_file = ListFile(type_name);
      entry.data = entry.loader(list_file, entry.rows);
      if (auto n = entry.count(entry.data); n < entry.files) {
        throw std::runtime_error(std::format("{} lists {} dataset(s) for {}, but the benchmarks need {}",
                                             list_file, n, type_name, entry.files));
      }
    }
  }

  template <class T>
  const std::vector<std::vector<T>> &Get() const {
    constexpr std::string_view type_name = std::meta::identifier_of(^^T);
    auto it = entries.find(std::string(type_name));
    if (it == entries.end()) {
      throw std::runtime_error(std::format("No datasets reserved for {}; call Reserve<{}>() before Load()",
                                           type_name, type_name));
    }
    if (!it->second.data.has_value()) {
      throw std::runtime_error(std::format("The datasets for {} are not loaded; call Load() first", type_name));
    }
    return std::any_cast<const std::vector<std::vector<T>> &>(it->second.data);
  }

private:
  struct Entry {
    size_t rows = 0;
    size_t files = 0;
    std::function<std::any(const std::string &, size_t)> loader;
    std::function<size_t(const std::any &)> count;
    std::any data;
  };

  std::string ListFile(const std::string &type_name) const {
    if (auto it = list_files.find(type_name); it != list_files.end()) { return it->second; }
    if (!default_list_file.empty()) { return default_list_file; }
    throw std::runtime_error(std::format("No input specified for {}. Use --input {}=<file> or --input <file>",
                                         type_name, type_name));
  }

  std::map<std::string, Entry> entries;
  std::map<std::string, std::string> list_files;
  std::string default_list_file;
};

#endif // DATASETS_H
