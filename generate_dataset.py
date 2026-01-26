import os
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


def generate_InvariantMassSequential_validation(
    input1, input2, input_size, max_results_size
):
    out_file = f"{os.path.dirname(input1)}/{Path(input1).stem}_{Path(input2).stem}_{input_size}_{max_results_size}_InvariantMassSequential.validation"
    print(
        f"Generating InvariantMassSequential validation for {input1} and {input2} in {out_file}"
    )
    with open(input1, "r") as dataset1_file, open(input2, "r") as dataset2_file, open(
        out_file, "w"
    ) as validation_file:
        # Only the last max_results_size lines are relevant
        lines1 = dataset1_file.readlines()[:input_size]
        lines2 = dataset2_file.readlines()[:input_size]
        pepe_1 = np.array(
            [
                ROOT.Math.PtEtaPhiEVector(*map(np.double, line.strip().split(",")))
                for line in lines1
            ]
        )
        pepe_2 = np.array(
            [
                ROOT.Math.PtEtaPhiEVector(*map(np.double, line.strip().split(",")))
                for line in lines2
            ]
        )

        # Get the index into the pairs indices at which the writing to the results array wraps around due to
        # the modulo operation on the results index in the benchmark code
        mod_split_idx = len(pepe_1) % max_results_size
        pepe_1 = np.concatenate(
            (pepe_1[-mod_split_idx:], pepe_1[-max_results_size:-mod_split_idx])
        )
        pepe_2 = np.concatenate(
            (pepe_2[-mod_split_idx:], pepe_2[-max_results_size:-mod_split_idx])
        )

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
        for s in im:
            validation_file.write(f"{s}\n")


def generate_DeltaR2Pairwise_validation(input1, input2, input_size, max_results_size):
    out_file = f"{os.path.dirname(input1)}/{Path(input1).stem}_{Path(input2).stem}_{input_size}_{max_results_size}_DeltaR2Pairwise.validation"
    print(
        f"Generating DeltaR2Pairwise validation for {input1} and {input2} in {out_file}"
    )
    with open(input1, "r") as dataset1_file, open(input2, "r") as dataset2_file, open(
        out_file, "w"
    ) as validation_file:
        lines1 = np.array(
            [
                list(map(np.double, line1.strip().split(",")))
                for line1 in dataset1_file.readlines()[:input_size]
            ]
        )
        lines2 = np.array(
            [
                list(map(np.double, line2.strip().split(",")))
                for line2 in dataset2_file.readlines()[:input_size]
            ]
        )
        max_outer_size = 128
        results_size = min(max_results_size, round(input_size * (input_size - 1) / 2))
        indices = [
            (i, j)
            for i in range(min(max_outer_size, input_size))
            for j in range(i + 1, input_size)
        ]

        # Get the index into the pairs indices at which the writing to the results array wraps around due to
        # the modulo operation on the results index in the benchmark code
        mod_split_idx = len(indices) % results_size

        indices = indices[-mod_split_idx:] + indices[-max_results_size:-mod_split_idx]
        eta1 = ROOT.RVec["double"]([lines1[i][1] for i, _ in indices])
        phi1 = ROOT.RVec["double"]([lines1[i][2] for i, _ in indices])
        eta2 = ROOT.RVec["double"]([lines2[j][1] for _, j in indices])
        phi2 = ROOT.RVec["double"]([lines2[j][2] for _, j in indices])
        dr2_vec = ROOT.VecOps.DeltaR2["double"](eta1, eta2, phi1, phi2)
        for dr2 in dr2_vec:
            validation_file.write(f"{dr2}\n")


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
        "--input_size",
        type=int,
        default=None,
        metavar=("<size>"),
        help="Maximum number of input entries to use (default: all)",
    )
    gen_validation.add_argument(
        "-m",
        "--max_results_size",
        type=int,
        default=int(65536),
        metavar=("<size>"),
        help="Maximum size of results (default: 65536)",
    )
    gen_validation.add_argument(
        "-i", "--input", type=str, help="Input dataset file", required=True
    )
    gen_validation.add_argument(
        "-i2",
        "--input2",
        type=str,
        help="Pass second input dataset file if different from the first",
        required=False,
    )
    gen_validation.add_argument(
        "-b",
        "--benchmark",
        nargs="+",
        choices=[
            "DeltaR2Pairwise",
            "InvariantMassSequential",
            # "InvariantMassRandom",
        ],
        help="Benchmark to generate validation for (DeltaR2Pairwise, InvariantMassSequential, InvariantMassRandom)",
        required=True,
    )

    args = parser.parse_args(args)
    if args.mode == "validation" and args.input2 is None:
        args.input2 = args.input
    return args


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    print(f"Configuration:")

    if args.mode == "dataset":
        print(
            f"\tMode: Dataset Generation\n\t-n: {args.ngen}\n\t-o: {args.output}\n\t--seed: {args.seed}"
        )
        generate_dataset(args)
    elif args.mode == "validation":
        print(
            f"\tMode: Validation Data Generation\n\t-i: {args.input}\n\t-i2: {args.input2}\n\t--input_size: {args.input_size}\n\t--max_results_size: {args.max_results_size}\n\t-b: {args.benchmark}"
        )
        if "DeltaR2Pairwise" in args.benchmark:
            generate_DeltaR2Pairwise_validation(
                args.input, args.input2, args.input_size, args.max_results_size
            )
        if "InvariantMassSequential" in args.benchmark:
            generate_InvariantMassSequential_validation(
                args.input, args.input2, args.input_size, args.max_results_size
            )
        # if "InvariantMassRandom" in args.benchmark:
        #     pass
        # generate_InvariantMassRandom_validation(args.input, args.input2, args.max_results_size)
