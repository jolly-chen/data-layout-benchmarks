import ROOT

from pathlib import Path

from .common import read_columns

NAME = "InvariantMassSequential"
NUM_INPUTS = 2


def generate(inputs, input_size):
    """
    Write the invariant masses of the pairs (inputs[0][i], inputs[1][i]). The results of
    InvariantMassRandom are compared against these too.
    """
    input1, input2 = inputs

    def four_vectors(path):
        pt, eta, phi, e = read_columns(path, ["pt", "eta", "phi", "e"], input_size)
        return [ROOT.Math.PtEtaPhiEVector(*v) for v in zip(pt, eta, phi, e)]

    pepe_1 = four_vectors(input1)
    pepe_2 = four_vectors(input2)
    # The number of elements, also without input_size, as main's validation list expects it
    n = min(len(pepe_1), len(pepe_2))
    pepe_1, pepe_2 = pepe_1[:n], pepe_2[:n]

    out_file = f"{Path(input1).parent}/{Path(input1).stem}_{Path(input2).stem}_{n}_{NAME}.validation"
    print(f"Generating {NAME} validation for {input1} and {input2} in {out_file}")

    px1 = ROOT.RVec["double"]([v.Px() for v in pepe_1])
    py1 = ROOT.RVec["double"]([v.Py() for v in pepe_1])
    pz1 = ROOT.RVec["double"]([v.Pz() for v in pepe_1])
    m1 = ROOT.RVec["double"]([v.M() for v in pepe_1])
    px2 = ROOT.RVec["double"]([w.Px() for w in pepe_2])
    py2 = ROOT.RVec["double"]([w.Py() for w in pepe_2])
    pz2 = ROOT.RVec["double"]([w.Pz() for w in pepe_2])
    m2 = ROOT.RVec["double"]([w.M() for w in pepe_2])

    im = ROOT.VecOps.InvariantMasses_PxPyPzM["double"](
        px1, py1, pz1, m1, px2, py2, pz2, m2
    )
    with open(out_file, "w") as validation_file:
        for s in im:
            validation_file.write(f"{s}\n")
