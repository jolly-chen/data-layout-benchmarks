#include "datastructures.h"
#include <iostream>

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

  TransformedParticle<SplitOp({0}).data()> h;
  h.id = 42;
  std::cout << "id: " << h.id << "\n";
}
