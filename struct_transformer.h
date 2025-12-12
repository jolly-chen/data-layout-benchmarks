// Compile with clang-p2996
#ifndef STRUCT_TRANSFORMER_H
#define STRUCT_TRANSFORMER_H

#include <array>
#include <cassert>
#include <experimental/meta>
#include <iostream>

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

// template <typename Original, typename P1, typename P2>
// struct ParitionedContainer {
//   P1 *p1;
//   P2 *p2;
//   size_t n;

//   ParitionedContainer(size_t n) : n(n) {
//     p1 = std::allocator<P1>().allocate(n);
//     p2 = std::allocator<P2>().allocate(n);
//   }

//   inline Original operator[](size_t index) const {
//     return {p1[index].pt, p1[index].eta, p1[index].phi, p2[index].e};
//   }

//   size_t size() const {
//     // Assuming both partitions have the same size
//     return n;
//   }
// };

// template <typename Original, typename T> struct PartitionedContainer {
//   struct Partitions;

//   consteval {
//     define_aggregate(^^Partitions, {data_member_spec(^^T)});
//   }

//   PartitionedContainer(size_t n) {}

//   // inline Original operator[](size_t index) { return Original{}; }
// };

#endif // STRUCT_TRANSFORMER_H