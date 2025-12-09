// Compile with clang-p2996
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
    auto mem_descr = data_member_spec(type_of(members[index]),
                                      {.name = identifier_of(members[index])});
    specs.push_back(mem_descr);
  }
  return specs;
}

struct Particle {
  int id;
  double pt, eta, phi, e;
  char charge;
  std::array<std::array<double, 3>, 3> posCovMatrix;
};

// template <auto Ts>
template <auto Members> struct TransformedParticle;

/////////////// consteval func version

// template <typename S, auto... SplitOps>
// consteval void TransformStruct() {
//   static_assert(
//       template_arguments_of(^^TransformStruct<S, SplitOps...>).size() ==
//       sizeof...(SplitOps) + 1);

//   (define_aggregate(^^TransformedParticle<SplitOps>,
//   get_member_specs<Particle>(SplitOps)), ...);
// };

/////////////// initializer list version

// template <typename S, typename... SplitOps>
// consteval void TransformStruct(std::initializer_list<SplitOps>... ops) {
//   (define_aggregate(
//        ^^TransformedParticle<std::define_static_array(std::vector{ops}).data()>,
//        get_member_specs<Particle>(ops)),
//    ...);
// };

/////////////// std::vector version

// template <typename S, typename... SplitOps>
//   requires requires { (std::same_as<SplitOps, std::vector<int>> && ...); }
// consteval void TransformStruct(SplitOps... ops) {
//   (define_aggregate(^^TransformedParticle<int>,
//                     get_member_specs<Particle>(ops)),
//    ...);
// };

/////////////// SplitOp version

template <typename S, typename... SplitOps>
consteval void TransformStruct(SplitOps... ops) {
  (define_aggregate(
       //  ^^TransformedParticle<reflect_constant(ops[std::make_index_sequence<ops.size()>{}])...>,
       //  ^^TransformedParticle<std::meta::reflect_constant_array(ops)>,
       substitute(^^TransformedParticle,
                  {
                      std::meta::reflect_constant_array(ops)}),
       get_member_specs<Particle>(ops)),
   ...);
};

/////////////// struct version

// template <typename S, typename... SplitOps>
// struct TransformStruct {
//   consteval {
//     static_assert(
//         template_arguments_of(^^TransformStruct<S, SplitOps...>).size() ==
//         sizeof...(SplitOps) + 1);

//     auto members = nonstatic_data_members_of(
//         ^^Particle, std::meta::access_context::unchecked());
//     (define_aggregate(^^TransformedParticle<SplitOps>, {}), ...);
//   }
// };

///////////////

consteval {
  // TransformStruct<Particle, SplitOp({0, 1, 2}), SplitOp({3, 4})>();
  // TransformStruct<Particle>(std::vector{0, 1, 2}, std::vector{3, 4});
  // TransformStruct<Particle>({0, 1, 2}, {3, 4});
  TransformStruct<Particle>(SplitOp({0, 1, 2}), SplitOp({3, 4}));
}

int main() {
  // TransformedParticle<reflec{1,2,3}> f;
  // TransformedParticle<SplitOp({0, 1, 2}).data()> f;
  // TransformStruct<Particle>({0, 1, 2}, {3, 4});

  // using tp = TransformStruct<Particle, SplitOp({0, 1, 2}), SplitOp({3, 4})> ;
  // TransformStruct<Particle>(std::vector{0, 1, 2}, std::vector{3, 4});
  // TransformStruct<Particle>(SplitOp({0, 1, 2}), SplitOp({3, 4}));

  TransformedParticle<SplitOp({0, 1, 2}).data()> f;
  f.id = 11;
  f.pt = 12;
  f.eta = 10;
  std::cout << "id: " << f.id << ", pt: " << f.pt << ", eta: " << f.eta << "\n";

  TransformedParticle<SplitOp({3, 4}).data()> g;
  g.phi = 1.5;
  g.e = 20.0;
  std::cout << "phi: " << g.phi << ", e: " << g.e << "\n";
}
