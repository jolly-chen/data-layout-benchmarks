#ifndef DATASTRUCTURES_H
#define DATASTRUCTURES_H
#include "datastructures.h"
#include "struct_transformer.h"

struct Particle {
    double pt;
    double eta;
    double phi;
    double e;
};

struct ParticleRef {
    double& pt;
    double& eta;
    double& phi;
    double& e;
};

template <auto Members> struct SubParticle;

// Forward declarations of structures with a subset of Particle members
consteval { SplitStruct<Particle, SubParticle>(SplitOp({0})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({1})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({2})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({3})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({0, 1})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({1, 0})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({0, 2})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({2, 0})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({0, 3})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({3, 0})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({1, 2})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({2, 1})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({1, 3})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({3, 1})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({2, 3})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({3, 2})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({0, 1, 2})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({0, 2, 1})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({1, 0, 2})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({1, 2, 0})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({2, 0, 1})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({2, 1, 0})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({0, 1, 3})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({0, 3, 1})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({1, 0, 3})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({1, 3, 0})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({3, 0, 1})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({3, 1, 0})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({0, 2, 3})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({0, 3, 2})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({2, 0, 3})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({2, 3, 0})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({3, 0, 2})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({3, 2, 0})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({1, 2, 3})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({1, 3, 2})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({2, 1, 3})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({2, 3, 1})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({3, 1, 2})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({3, 2, 1})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({0, 1, 2, 3})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({0, 1, 3, 2})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({0, 2, 1, 3})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({0, 2, 3, 1})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({0, 3, 1, 2})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({0, 3, 2, 1})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({1, 0, 2, 3})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({1, 0, 3, 2})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({1, 2, 0, 3})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({1, 2, 3, 0})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({1, 3, 0, 2})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({1, 3, 2, 0})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({2, 0, 1, 3})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({2, 0, 3, 1})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({2, 1, 0, 3})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({2, 1, 3, 0})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({2, 3, 0, 1})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({2, 3, 1, 0})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({3, 0, 1, 2})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({3, 0, 2, 1})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({3, 1, 0, 2})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({3, 1, 2, 0})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({3, 2, 0, 1})); }
consteval { SplitStruct<Particle, SubParticle>(SplitOp({3, 2, 1, 0})); }

#endif // DATASTRUCTURES_H
