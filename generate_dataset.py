import sys
import argparse
import importlib
from pathlib import Path

def find_dataset_modules():
    """
    Map struct names to the modules in dataset_generators/, one per struct.

    The struct name is derived from the file name (snake_case file name -> CamelCase struct,
    e.g. sub_particle.py -> SubParticle) instead of read from the module, so that listing
    the choices does not import every generator and its dependencies.
    """
    package_dir = Path(__file__).resolve().parent / "dataset_generators"
    return {
        "".join(part.capitalize() for part in path.stem.split("_")): f"dataset_generators.{path.stem}"
        for path in sorted(package_dir.glob("*.py"))
        # Skips __init__.py and private helper modules
        if not path.stem.startswith("_")
    }


# struct name -> module in dataset_generators/ with its generator
DATASET_MODULES = find_dataset_modules()

# benchmark name -> module in validation/ with its reference implementation
VALIDATION_MODULES = {
    "InvariantMassSequential": "validation.invariant_mass",
    "DeltaR2Pairwise": "validation.delta_r2",
}


def generate_dataset(args):
    # Imported here, so that generating validation data does not need numpy/scipy
    module = importlib.import_module(DATASET_MODULES[args.struct])

    print(f"Generating dataset with {args.ngen} {args.struct} elements to {args.output}")
    with open(args.output, "w") as dataset_file:
        dataset_file.write(",".join(module.COLUMNS) + "\n")
        for row in module.generate(args.ngen, args.seed):
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
        choices=DATASET_MODULES.keys(),
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
