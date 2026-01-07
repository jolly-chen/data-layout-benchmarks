import sys
import argparse
import scipy.stats
import numpy as np
import ROOT

from pathlib import Path
from itertools import product

ROOT.EnableImplicitMT()

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
    out_file = f"{Path(input1).stem}_{Path(input2).stem}_{max_results_size}_InvariantMassSequential.validation"
    print(
        f"Generating InvariantMassSequential validation for {input1} and {input2} in {out_file}"
    )
    with open(input1, "r") as dataset1_file, open(input2, "r") as dataset2_file, open(
        out_file, "w"
    ) as validation_file:
        # Only the last max_results_size lines are relevant
        lines1 = dataset1_file.readlines()[-max_results_size:]
        lines2 = dataset2_file.readlines()[-max_results_size:]
        pepe_2 = [ROOT.Math.PtEtaPhiEVector(*map(np.double, line.strip().split(","))) for line in lines2]
        pepe_1 = [ROOT.Math.PtEtaPhiEVector(*map(np.double, line.strip().split(","))) for line in lines1]
        px1 = ROOT.RVec["double"]([v.Px() for v in pepe_1])
        py1 = ROOT.RVec["double"]([v.Py() for v in pepe_1])
        pz1 = ROOT.RVec["double"]([v.Pz() for v in pepe_1])
        m1 = ROOT.RVec["double"]([v.M() for v in pepe_1])
        px2 = ROOT.RVec["double"]([w.Px() for w in pepe_2])
        py2 = ROOT.RVec["double"]([w.Py() for w in pepe_2])
        pz2 = ROOT.RVec["double"]([w.Pz() for w in pepe_2])
        m2 = ROOT.RVec["double"]([w.M() for w in pepe_2])    
        
        validation_file.write("\n".join([str(im) for im in ROOT.VecOps.InvariantMasses_PxPyPzM["double"](px1, py1, pz1, m1, px2, py2, pz2, m2)]))


def generate_DeltaR2Pairwise_validation(input1, input2, max_results_size):
    out_file = f"{Path(input1).stem}_{Path(input2).stem}_{max_results_size}_DeltaR2Pairwise.validation"
    print(
        f"Generating DeltaR2Pairwise validation for {input1} and {input2} in {out_file}"
    )
    with open(input1, "r") as dataset1_file, open(input2, "r") as dataset2_file, open(
        out_file, "w"
    ) as validation_file:
        lines1 = np.array([list(map(np.double, line1.strip().split(","))) for line1 in dataset1_file.readlines()])
        lines2 = np.array([list(map(float, line1.strip().split(","))) for line1 in dataset2_file.readlines()])
        n_pairs = len(lines1) * len(lines2)
                
        # Only the last max_results_size pairwise combinations are relevant. 
        # Reverse the input arrays so the last combinations are generated first.
        # Need to use a list comprehension over the product generator to avoid 
        # creating the full cartesian product in memory which can be very large.
        combinations = np.array([p for p,i in zip(product(lines1[::-1], lines2[::-1]), range(n_pairs)) if i < max_results_size])
        eta1 = ROOT.VecOps.AsRVec(combinations[:, 0, 1].copy())
        phi1 = ROOT.VecOps.AsRVec(combinations[:, 0, 2].copy())
        eta2 = ROOT.VecOps.AsRVec(combinations[:, 1, 1].copy())
        phi2 = ROOT.VecOps.AsRVec(combinations[:, 1, 2].copy())
                
        validation_file.write(
            '\n'.join([str(dr2) for dr2 in ROOT.VecOps.DeltaR2["double"](phi1, eta1, phi2, eta2)])
        )      
        

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
        default=int(65536),
        metavar=("<size>"),
        help="Maximum size of results (default: 1048576)",
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

    if args.mode == "dataset":
        generate_dataset(args)
    elif args.mode == "validation":
        if "DeltaR2Pairwise" in args.benchmark:
            generate_DeltaR2Pairwise_validation(args.input, args.input2, args.max_results_size)
        if "InvariantMassSequential" in args.benchmark:
            generate_InvariantMassSequential_validation(args.input, args.input2, args.max_results_size)
        # if "InvariantMassRandom" in args.benchmark:
        #     pass
            # generate_InvariantMassRandom_validation(args.input, args.input2, args.max_results_size)
