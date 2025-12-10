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

// clang-format off
template <typename In, template <auto> typename Out, typename... SplitOps>
consteval void TransformStruct(SplitOps... ops) {
  (define_aggregate(substitute(^^Out, { std::meta::reflect_constant_array(ops)}),
                    get_member_specs<In>(ops)),
   ...);
};
// clang-format on

///////////////

struct Particle {
  int id;
  double pt, eta, phi, e;
  char charge;
  std::array<std::array<double, 3>, 3> posCovMatrix;
};

template <auto... Members> struct TransformedParticle;

consteval void GenerateTransformedParticles() {

  TransformStruct<Particle, TransformedParticle>(SplitOp({0, 1, 2}),
                                                 SplitOp({3, 4}));
}

consteval { GenerateTransformedParticles(); }

int main() {
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
