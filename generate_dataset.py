import sys
import argparse
import importlib
import numpy as np

# Dataset generators, one per struct. Each yields rows with values for the columns it lists.
# Columns must be named like data members of the struct, since main.cpp maps the header row
# of a dataset to the members of the struct by name.


def generate_particles(ngen, seed):
    """
    Generate PtEtaPhiE vectors.
    Based on https://github.com/root-project/root/blob/654a2c8eadac3409a65dc017279666e1cd5ccb6d/math/mathcore/test/stress/VectorTest.h#L43
    """
    # Imported here, so that generating validation data does not need scipy
    import scipy.stats

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


# struct name -> (columns, generator)
DATASET_GENERATORS = {
    "Particle": (["pt", "eta", "phi", "e"], generate_particles),
}

# benchmark name -> module in validation/ with its reference implementation
VALIDATION_MODULES = {
    "InvariantMassSequential": "validation.invariant_mass",
    "DeltaR2Pairwise": "validation.delta_r2",
}


def generate_dataset(args):
    columns, generator = DATASET_GENERATORS[args.struct]

    print(f"Generating dataset with {args.ngen} {args.struct} elements to {args.output}")
    with open(args.output, "w") as dataset_file:
        dataset_file.write(",".join(columns) + "\n")
        for row in generator(args.ngen, args.seed):
            dataset_file.write(",".join(str(v) for v in row) + "\n")
            dataset_file.flush()


def generate_validation(args):
    for benchmark in args.benchmark:
        # Imported here, so that generating datasets does not need ROOT
        module = importlib.import_module(VALIDATION_MODULES[benchmark])
        inputs = [args.input1, args.input2 or args.input1][: module.NUM_INPUTS]
        module.generate(inputs, args.input_size)


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Generate benchmark input datasets and validation data"
    )
    subparsers = parser.add_subparsers(dest="mode")

    gen_dataset = subparsers.add_parser("dataset", help="Generate dataset")
    gen_dataset.add_argument(
        "-s",
        "--struct",
        choices=DATASET_GENERATORS.keys(),
        default="Particle",
        help="Struct to generate elements of (default: Particle)",
    )
    gen_dataset.add_argument(
        "-n",
        "--ngen",
        type=int,
        default=int(10e5),
        help="Number of elements to generate (default: 1000000)",
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
        "validation", help="Generate validation data (needs ROOT)"
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
        "-i1", "--input1", type=str, help="First input dataset file", required=True
    )
    gen_validation.add_argument(
        "-i2",
        "--input2",
        type=str,
        help="Second input dataset file (default: the first one)",
        required=False,
    )
    gen_validation.add_argument(
        "-b",
        "--benchmark",
        nargs="+",
        choices=VALIDATION_MODULES.keys(),
        help="Benchmark to generate validation for. InvariantMassRandom is validated against "
        "InvariantMassSequential.",
        required=True,
    )

    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    print(f"Configuration:")

    if args.mode == "dataset":
        print(
            f"\tMode: Dataset Generation\n\t--struct: {args.struct}\n\t-n: {args.ngen}\n\t-o: {args.output}\n\t--seed: {args.seed}"
        )
        generate_dataset(args)
    elif args.mode == "validation":
        print(
            f"\tMode: Validation Data Generation\n\t-i1: {args.input1}\n\t-i2: {args.input2}\n\t--input_size: {args.input_size}\n\t-b: {args.benchmark}"
        )
        generate_validation(args)
