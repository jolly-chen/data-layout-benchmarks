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
expand_all(std::span<std::meta::info const> r) -> std::meta::info {
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

/* Helper method to get a list of data_member_specs for members in
 * the struct S based on the given a list of indices
 */
template <typename S>
consteval auto get_member_specs(std::span<const int> indices) {
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
                    get_member_specs<In>(ops)),
   ...);
};
// clang-format on

/* Compute the aligned size for a given size and alignment. */
size_t align_size(size_t size, size_t alignment) {
  return (size + alignment - 1) / alignment * alignment;
}

///////////////////

/// A container that takes partitions of the members of ProxyRef in its
/// template parameters T..., and stores each partition in a separate array.
/// The partitions are stored contiguously in memory, aligned to the size of a cacheline.
template <typename ProxyRef, typename... T> struct PartitionedContainer {
  static_assert(nsdms(^^ProxyRef).size() == (... + nsdms(^^T).size()),
                "PartitionedContainer: Total number of members in partitions "
                "must equal number of members in ProxyRef");
  struct Partitions;
  consteval {
    define_aggregate(^^Partitions, { data_member_spec(substitute(^^std::span, {^^T}))... });
  }

  alignas(64) std::vector<std::byte> storage;
  Partitions p;
  size_t n;

  PartitionedContainer(size_t n, size_t alignment) : n(n) {
    // Allocate each partition
    size_t total_size = (0 + ... + align_size(n * sizeof(T), alignment));
    storage.resize(total_size);

    // Assign each partition to its location in the storage vector
    size_t offset = 0;
    template for (constexpr auto I : std::views::iota(0zu, sizeof...(T))) {
      using MemType = T...[I];
      constexpr auto m = nsdms(^^Partitions)[I];

      // https://learn.microsoft.com/en-us/cpp/standard-library/new-operators?view=msvc-170#op_new_arr
      p.[: m :] = std::span<MemType>(std::launder(reinterpret_cast<MemType*>(new (&storage[offset]) MemType[n])), n);
      offset += align_size(p.[:m:].size_bytes(), alignment);
    }
  }

  inline ProxyRef operator[](size_t index) const {
#pragma clang diagnostic ignored "-Wreturn-type"
    auto get_arg = [index,
                    this]<std::meta::info Om, typename Out>() constexpr -> Out {
      template for (constexpr auto part : nsdms(^^Partitions)) {
        constexpr auto span_type = template_arguments_of(type_of(part))[0];
        template for (constexpr auto pm : nsdms(span_type)) {
          if constexpr (identifier_of(Om) == identifier_of(pm)) {
            return p.[:part:][index].[:pm:];
          }
        }
      }
    };

    return [:expand_all(nonstatic_data_members_of(^^ProxyRef, std::meta::access_context::current()) ):] >>
      [&]<auto... M> {
        return ProxyRef { get_arg.template operator()<M, typename[:type_of(M):]>()... };
    };
  }

  size_t size() const { return n; }

  ~PartitionedContainer() {
    // Deallocate each partition
    template for (constexpr auto I : std::views::iota(0zu, sizeof...(T))) {
      using MemType = T...[I];
      constexpr auto m = nsdms(^^Partitions)[I];

      // Need to deallocate each element individually because it is constructed using placement new.
      for (size_t i = n - 1; i == 0; --i) {
        p.[:m:][i].~MemType();
      }
    }
  }

  /* Get a string that contains the partitioning information stored in the
   * parameter pack T.  */
  static std::string get_partitions_string() {
    auto get_static_array_string = [](std::span<const int> arr, size_t i) {
      std::string r = "";

      if (i != 0)
        r += ",";

      r += "[";
      for (size_t i = 0; i < arr.size(); ++i) {
        r += std::to_string(arr[i]);
        if (i != arr.size() - 1) {
          r += ",";
        }
      }
      r += "]";
      return r;
    };

    auto get_splitops_string = [&]<size_t... Is>(std::index_sequence<Is...>) {
      return (get_static_array_string([:template_arguments_of(^^T)[0]:], Is) +
              ...);
    };

    std::string result = "[";
    result += get_splitops_string(std::make_index_sequence<sizeof...(T)>());
    return result + "]";
  }
};

#endif // STRUCT_TRANSFORMER_H