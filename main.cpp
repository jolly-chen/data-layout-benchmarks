#include "benchmarks.h"
#include "datastructures.h"
#include <chrono>
#include <iostream>

using Clock = std::chrono::high_resolution_clock;

// template <typename T> void RunInvariantMassRandom() {
//   std::vector<T> v1(1000);
//   std::vector<T> v2(1000);
//   std::vector<double> results(1000);

//   auto start = Clock::now();
//   InvariantMassRandom(v1, v2, results);
//   auto end = Clock::now();

//   std::chrono::duration<double> elapsed = end - start;
//   std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";
// }

consteval auto nsdms(std::meta::info type) {
  return define_static_array(nonstatic_data_members_of(type, std::meta::access_context::current()));
}

template <typename Original, typename P1, typename P2> struct PContainer {
  struct Partitions {
    P1 *p1;
    P2 *p2;
  };
  Partitions p;
  size_t n;

  PContainer(size_t n) : n(n) {
    p.p1 = std::allocator<P1>().allocate(n);
    p.p2 = std::allocator<P2>().allocate(n);
  }

  inline Original operator[](size_t index) const {
    return {p.p1[index].pt, p.p1[index].eta, p.p1[index].phi, p.p2[index].e};
  }

  size_t size() const { return n; }
};

template <typename Original, typename... T> struct PartitionedContainer {
  struct Partitions;
  consteval {
    define_aggregate(^^Partitions, { data_member_spec(add_pointer(^^T))... });
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
    constexpr auto original_members = nsdms(^^Original);
    constexpr auto partitions = nsdms(^^Partitions);
    constexpr std::vector<std::meta::info> args;

    auto get_arg = [&](std::meta::info m) {
      template for (constexpr auto &p : partitions) {
        constexpr auto p_members = nsdms(remove_pointer(type_of(p)));
        template for (constexpr auto &pm : p_members) {
          if (identifier_of(m) == identifier_of(pm)) {
            return p.[:p:][index].[:pm:];
            // return 0;
          }
        }
      }
    };

    // template for (constexpr auto &om : original_members) {
    //   // Find the corresponding member in the partitions
    //   template for (constexpr auto &p : partitions) {
    //     constexpr auto p_members = nsdms(remove_pointer(type_of(p)));
    //     template for (constexpr auto &pm : p_members) {
    //       if (identifier_of(om) == identifier_of(pm)) {
    //         args.push_back(reflect_object(p.[:p:][index].[:pm:]));
    //       }
    //     }
    //   }
    // }


    auto construct_object = [&]<size_t... Is>(std::index_sequence<Is...>) -> Original {
        // return {[: args[Is] :]...};
        return {[: get_arg(original_members[Is]) :]...};
    };
    return construct_object(std::make_index_sequence<original_members.size()>{});
    // return {p.p1[index].pt, p.p1[index].eta, p.p1[index].phi, p.p2[index].e};
  }

  size_t size() const { return n; }
};

template <typename T> void RunInvariantMassRandomPartitioned() {
  PartitionedContainer<Particle, SubParticle<SplitOp({0, 1, 2}).data()>,
             SubParticle<SplitOp({3}).data()>>
      v1(10);
  PartitionedContainer<Particle, SubParticle<SplitOp({0, 1, 2}).data()>,
                       SubParticle<SplitOp({3}).data()>>
      v2(10);

  //   PartitionedContainer<Particle, SubParticle<SplitOp({0, 1, 2}).data()>,
  //                       SubParticle<SplitOp({3}).data()>>
  //       v1(1000), v2(1000);

    v1[0].pt = 0;

    // std::vector<double> results(1000);

    // auto start = Clock::now();
    // InvariantMassRandom(v1, v2, results);
    // auto end = Clock::now();

    // std::chrono::duration<double> elapsed = end - start;
    // std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";
}

int main() {
  //   RunInvariantMassRandom<Particle>();
  RunInvariantMassRandomPartitioned<Particle>();
  return 0;
}