import ROOT

from pathlib import Path

from .common import read_columns

NAME = "DeltaR2Pairwise"
NUM_INPUTS = 1

# Must match kernels::deltar2_max_outer_size in benchmarks.h
MAX_OUTER_SIZE = 128


def generate(inputs, input_size):
    """
    Write DeltaR2 of the pairs (i, j) with i < min(MAX_OUTER_SIZE, input_size) and j > i,
    in the order the kernel computes them.
    """
    input1 = inputs[0]
    eta, phi = read_columns(input1, ["eta", "phi"], input_size)
    # The number of elements, also without input_size, as main's validation list expects it
    n = len(eta)

    out_file = f"{Path(input1).parent}/{Path(input1).stem}_{n}_{NAME}.validation"
    print(f"Generating {NAME} validation for {input1} in {out_file}")
    indices = [(i, j) for i in range(min(MAX_OUTER_SIZE, n)) for j in range(i + 1, n)]

    eta1 = ROOT.RVec["double"]([eta[i] for i, _ in indices])
    phi1 = ROOT.RVec["double"]([phi[i] for i, _ in indices])
    eta2 = ROOT.RVec["double"]([eta[j] for _, j in indices])
    phi2 = ROOT.RVec["double"]([phi[j] for _, j in indices])
    dr2_vec = ROOT.VecOps.DeltaR2["double"](eta1, eta2, phi1, phi2)
    with open(out_file, "w") as validation_file:
        for dr2 in dr2_vec:
            validation_file.write(f"{dr2}\n")
