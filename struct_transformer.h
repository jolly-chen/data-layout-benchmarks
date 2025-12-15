// Compile with clang-p2996
#ifndef STRUCT_TRANSFORMER_H
#define STRUCT_TRANSFORMER_H

#include <array>
#include <cassert>
#include <experimental/meta>
#include <iostream>

consteval auto nsdms(std::meta::info type) {
  return define_static_array(nonstatic_data_members_of(type, std::meta::access_context::current()));
}

consteval auto SplitOp(std::vector<int> indices) {
  return define_static_array(indices);
}

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
template <typename In, template <auto> typename Out, typename... SplitOps>
consteval void SplitStruct(SplitOps... ops) {
  (define_aggregate(substitute(^^Out, { std::meta::reflect_constant_array(ops)}),
                    get_member_specs<In>(ops)),
   ...);
};
// clang-format on

template <typename Original, typename... T> struct PartitionedContainer {
  struct Partitions;
  consteval {
    define_aggregate(^^Partitions, { data_member_spec(add_pointer(^^T), { .name = identifier_of(nsdms(^^T)[0])})... });
  }

  Partitions p;
  size_t n;

  PartitionedContainer(size_t n) : n(n) {
    // Allocate each partition
    template for (constexpr auto &m : nsdms(^^Partitions)) {
      using MemType = typename[:remove_pointer(type_of(m)):];
      p.[:m:] = std::allocator<MemType>().allocate(n);
    }
  }

  inline Original operator[](size_t index) const {
    auto get_arg = [index, this]<std::meta::info om>() constexpr -> typename[: type_of(om) :] {
      template for (constexpr auto &part : nsdms(^^Partitions)) {
        template for (constexpr auto &pm : nsdms(remove_pointer(type_of(part)))) {
          if constexpr(identifier_of(om) == identifier_of(pm)) {
            return p.[: part :][index].[: pm :];
          }
        }
      }
    };

    auto construct_object = [&, this]<size_t... Is>(std::index_sequence<Is...>) constexpr -> Original {
      return Original{ get_arg.template operator()<nsdms(^^Original)[Is]>()... };
    };
    return construct_object(std::make_index_sequence<nsdms(^^Original).size()>{});

    // return { p.pt[index].pt, p.pt[index].eta, p.pt[index].phi, p.e[index].e };
  }

  size_t size() const { return n; }
};


#endif // STRUCT_TRANSFORMER_H