import sys
import argparse
import scipy.stats
import numpy as np
import ROOT

from pathlib import Path
from itertools import product

def generate_random_vectors(ngen, seed):
    np.random.seed(seed)
    for i in range(ngen):
        pt = np.random.exponential(scale=10.0)
        eta = np.random.uniform(-5.0, 5.0)
        phi = np.random.rand() * 3.1415926535897931
        if i % 50 == 0:
            m = np.random.uniform(0.0, 10.0)
        else:
            m = scipy.stats.rel_breitwigner.rvs(1.0, 0.01)
        e = np.sqrt((m**2) + (pt * np.cosh(eta)) ** 2)
        yield [pt, eta, phi, e]


def generate_dataset(args):
    """
    Based on https://github.com/root-project/root/blob/654a2c8eadac3409a65dc017279666e1cd5ccb6d/math/mathcore/test/stress/VectorTest.h#L43
    """
    np.random.seed(args.seed)

    print(f"Generating dataset with {args.ngen} PtEtaPhiE vectors to {args.output}")
    with open(args.output, "w") as dataset_file:
        for pt, eta, phi, e in generate_random_vectors(args.ngen, args.seed):
            dataset_file.write(f"{pt},{eta},{phi},{e}\n")
            dataset_file.flush()


def generate_InvariantMassSequential_validation(input1, input2, max_results_size):
    out_file = f"{input1}.{input2}.InvariantMassSequential.validation"
    print(
        f"Generating InvariantMassSequential validation for {input1} and {input2} in {out_file}"
    )
    with open(input1, "r") as dataset1_file, open(input2, "r") as dataset2_file, open(
        out_file, "w"
    ) as validation_file:
        lines1 = dataset1_file.readlines()[-max_results_size:]
        lines2 = dataset2_file.readlines()[-max_results_size:]
        for line1, line2 in zip(lines1, lines2):
            pt1, eta1, phi1, e1 = map(float, line1.strip().split(","))
            pt2, eta2, phi2, e2 = map(float, line2.strip().split(","))
            v = ROOT.Math.PtEtaPhiEVector(pt1, eta1, phi1, e1)
            w = ROOT.Math.PtEtaPhiEVector(pt2, eta2, phi2, e2)
            validation_file.write(
                f"{ROOT.VecOps.InvariantMasses_PxPyPzM(v.Px(), v.Py(), v.Pz(), v.M(), w.Px(), w.Py(), w.Pz(), w.M())}\n"
            )
            validation_file.flush()


def generate_DeltaR2Pairwise_validation(input1, input2, max_results_size):
    out_file = f"{input1}.{input2}.DeltaR2Pairwise.validation"
    print(
        f"Generating DeltaR2Pairwise validation for {input1} and {input2} in {out_file}"
    )
    with open(input1, "r") as dataset1_file, open(input2, "r") as dataset2_file, open(out_file, "w") as validation_file:
        lines1 = dataset1_file.readlines()[-max_results_size:]
        lines2 = dataset2_file.readlines()[-max_results_size:]
        eta1 = ROOT.RVec["double"]()
        phi1 = ROOT.RVec["double"]()
        eta2 = ROOT.RVec["double"]()
        phi2 = ROOT.RVec["double"]()
        for line1 in lines1:
            for line2 in lines2:
                _, e1, p1, _ = map(float, lines1.strip().split(","))
                _, e2, p2, _ = map(float, lines2.strip().split(","))
                validation_file.write(f"{ROOT.VecOps.DeltaR2(phi1, eta1, phi2, eta2)}\n")
                validation_file.flush()


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Generate dataset of PtEtaPhiE vectors"
    )
    subparsers = parser.add_subparsers(dest="mode")

    gen_dataset = subparsers.add_parser("dataset", help="Generate dataset")
    gen_dataset.add_argument(
        "-n",
        "--ngen",
        type=int,
        default=int(10e5),
        help="Number of vectors to generate (default: 100000)",
    )
    gen_dataset.add_argument(
        "-o",
        "--output",
        type=str,
        default="dataset.csv",
        help="Output file name (default: dataset.csv)",
    )
    gen_dataset.add_argument(
        "--seed", type=int, default=123, help="Random seed (default: 123)"
    )

    gen_validation = subparsers.add_parser(
        "validation", help="Generate validation data"
    )
    gen_validation.add_argument(
        "-n",
        "--max_results_size",
        type=int,
        default=int(1048576),
        help="Maximum size of results (default: 1048576)",
    )
    gen_validation.add_argument("--DeltaR2Pairwise", nargs=2)
    gen_validation.add_argument("--InvariantMassSequential", nargs=2)
    gen_validation.add_argument("--InvariantMassRandom", nargs=2)

    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    if args.mode == "dataset":
        generate_dataset(args)
    elif args.mode == "validation":
        if args.DeltaR2Pairwise:
            input1 = args.DeltaR2Pairwise[0]
            input2 = args.DeltaR2Pairwise[1]
            generate_DeltaR2Pairwise_validation(input1, input2, args.max_results_size)
        elif args.InvariantMassSequential:
            input1 = args.InvariantMassSequential[0]
            input2 = args.InvariantMassSequential[1]
            generate_InvariantMassSequential_validation(
                input1, input2, args.max_results_size
            )
        elif args.InvariantMassRandom:
            input1 = args.InvariantMassRandom[0]
            input2 = args.InvariantMassRandom[1]
            # generate_InvariantMassRandom_validation(input1, input2, args.max_results_size)
