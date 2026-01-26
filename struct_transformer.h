// Compile with clang-p2996
#ifndef STRUCT_TRANSFORMER_H
#define STRUCT_TRANSFORMER_H

#include <array>
#include <cassert>
#include <experimental/meta>
#include <iostream>
#include <new>
#include <ranges>

template <auto... V> struct replicator_type {
  template <typename F>
  constexpr auto operator>>(F body) const -> decltype(auto) {
    return body.template operator()<V...>();
  }
};

template <auto... V> replicator_type<V...> replicator = {};

consteval auto
ExpandAll(std::span<std::meta::info const> r) -> std::meta::info {
  std::vector<std::meta::info> rv;
  for (std::meta::info i : r) {
    rv.push_back(reflect_constant(i));
  }
  return substitute(^^replicator, rv);
}

consteval auto nsdms(std::meta::info type) {
  return define_static_array(
      nonstatic_data_members_of(type, std::meta::access_context::current()));
}

//////////////

consteval auto SplitOp(std::vector<int> indices) {
  return define_static_array(indices);
}

/* Convert a list of SplitOps to a string. */
template <typename ...Ops>
std::string GetSplitOpsString() {
  auto get_static_array_string = [](std::span<const int> arr, size_t i) {
    std::string r = "";

    if (i != 0)
      r += "_";

    for (size_t i = 0; i < arr.size(); ++i) {
      r += std::to_string(arr[i]);
    }
    return r;
  };

  return [&]<size_t... Is>(std::index_sequence<Is...>) {
    return (get_static_array_string([:template_arguments_of(^^Ops)[0]:], Is) +
            ...);
  }(std::make_index_sequence<sizeof...(Ops)>());
}

/* Get the name of the container type as a string. */
template <typename T> std::string GetContainerName() {
  if constexpr (has_identifier(^^T)) {
    return std::string(identifier_of(^^T));
  } else {
    return T::to_string();
  }
}


consteval auto Mapping(std::vector<std::pair<size_t, size_t>> mapping) {
  return define_static_array(mapping);
}

/* Helper method to get a list of data_member_specs for members in
 * the struct S based on the given a list of indices
 */
template <typename S>
consteval auto GetMemberSpecs(std::span<const int> indices) {
  auto members =
      nonstatic_data_members_of(^^S, std::meta::access_context::unchecked());

  std::vector<std::meta::info> specs;
  for (auto index : indices) {
    auto mem_descr = data_member_spec(remove_cvref(type_of(members[index])),
                                      {.name = identifier_of(members[index])});
    specs.push_back(mem_descr);
  }
  return specs;
}

// clang-format off
/* Takes a struct In and generate a template specialization of struct Out
 * for each SplitOp provided, containing only the members of struct In, in the
 * order specified by the SplitOp.
 */
template <typename In, template <auto> typename Out, typename... SplitOps>
consteval void SplitStruct(SplitOps... ops) {
  (define_aggregate(substitute(^^Out, { std::meta::reflect_constant_array(ops)}),
                    GetMemberSpecs<In>(ops)),
   ...);
};
// clang-format on

/* Compute the aligned size for a given size and alignment. */
inline size_t AlignSize(size_t size, size_t alignment) {
  return (size + alignment - 1) / alignment * alignment;
}

///////////////////

/// A container that takes substructures with a subset of the members of ProxyRef in its
/// template parameters T..., and stores each substructure in a separate array.
/// The substructure arrays are stored contiguously in memory, aligned to the size of a cacheline.
template <typename ProxyRef, auto Mapping, typename... T> struct PartitionedContainerContiguous {
private:
  static_assert(nsdms(^^ProxyRef).size() == (... + nsdms(^^T).size()),
                "PartitionedContainerContiguous: Total number of members in substructures"
                "must equal number of members in ProxyRef");
  struct Partitions;
  consteval {
    define_aggregate(^^Partitions, { data_member_spec(substitute(^^std::span, {^^T}))... });
  }

  Partitions p;
  std::byte *storage;
  size_t n;

  public:
  PartitionedContainerContiguous(size_t n, size_t alignment) : n(n) {
    // Allocate each partition
    size_t total_size = (0 + ... + AlignSize(n * sizeof(T), alignment));
    storage = static_cast<std::byte*>(std::aligned_alloc(alignment, total_size));

    // Assign each partition to its location in the storage vector
    size_t offset = 0;
    template for (constexpr auto I : std::views::iota(0zu, sizeof...(T))) {
      using MemType = T...[I];
      constexpr auto m = nsdms(^^Partitions)[I];

      // https://learn.microsoft.com/en-us/cpp/standard-library/new-operators?view=msvc-170#op_new_arr
      p.[: m :] = std::span<MemType>(std::launder(reinterpret_cast<MemType*>(new (&storage[offset]) MemType[n])), n);
      offset += AlignSize(p.[:m:].size_bytes(), alignment);
    }
  }

  #pragma clang diagnostic ignored "-Wreturn-type"
  inline ProxyRef operator[](const size_t index) const {
    return [&]<size_t... Is>(std::index_sequence<Is...>) constexpr -> ProxyRef {
      constexpr auto partitions = nsdms(^^Partitions);
      return ProxyRef{ p.[: partitions[Mapping[Is].first] :][index]
                          .[: nsdms(template_arguments_of(type_of(partitions[Mapping[Is].first]))[0])
                              [Mapping[Is].second] :]... };
    }(std::make_index_sequence< nsdms(^^ProxyRef).size()>());
  }

  size_t size() const { return n; }

  static std::string to_string() {
    return std::string("PartitionedContainerContiguous") + GetSplitOpsString<T...>();
  }

  ~PartitionedContainerContiguous() {
    // Deallocate each partition
    template for (constexpr auto I : std::views::iota(0zu, sizeof...(T))) {
      using MemType = T...[I];
      constexpr auto m = nsdms(^^Partitions)[I];

      // Need to deallocate each element individually because it is constructed using placement new.
      for (size_t i = n - 1; i == 0; --i) {
        p.[:m:][i].~MemType();
      }
    }

    std::free(storage);
  }
};


/// A container that takes substructures with a subset of the members of ProxyRef in its
/// template parameters T..., and stores each substructure in a separate array.
/// The substructures are stored in separate arrays aligned to cacheline size,
/// in different memory locations.
template <typename ProxyRef, auto Mapping, typename... T> struct PartitionedContainer {
private:
  static_assert(nsdms(^^ProxyRef).size() == (... + nsdms(^^T).size()),
                "PartitionedContainer: Total number of members in partitions "
                "must equal number of members in ProxyRef");
  struct Partitions;
  consteval {
    define_aggregate(^^Partitions, { data_member_spec(add_pointer(^^T))... });
  }

  Partitions p;
  size_t n;

public:
  PartitionedContainer(size_t n, size_t alignment) : n(n) {
    // Allocate each partition
    template for (constexpr auto m : nsdms(^^Partitions)) {
      p.[: m :] = static_cast<typename[: type_of(m) :]>(
        std::aligned_alloc(alignment,
                           AlignSize(n * sizeof(typename[: remove_pointer(type_of(m)) :]), alignment)));
    }
  }

  #pragma clang diagnostic ignored "-Wreturn-type"
  inline ProxyRef operator[](const size_t index) const {
    return [&]<size_t... Is>(std::index_sequence<Is...>) constexpr -> ProxyRef {
      constexpr auto partitions = nsdms(^^Partitions);
      return ProxyRef{ p.[: partitions[Mapping[Is].first] :][index]
                          .[: nsdms(remove_pointer(type_of(partitions[Mapping[Is].first])))[Mapping[Is].second] :]... };
    }(std::make_index_sequence< nsdms(^^ProxyRef).size()>());
  }

  size_t size() const { return n; }

  static std::string to_string() {
    return std::string("PartitionedContainer") + GetSplitOpsString<T...>();
  }

  ~PartitionedContainer() {
    // Deallocate each partition
    template for (constexpr auto m : nsdms(^^Partitions)) {
      std::free(p.[: m :]);
    }
  }
};
#endif // STRUCT_TRANSFORMER_H