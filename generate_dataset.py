import sys
import os
import argparse
import scipy.stats
import numpy as np


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

    with open(args.output, "w") as f:
        for pt, eta, phi, e in generate_random_vectors(args.ngen, args.seed):
            f.write(f"{pt},{eta},{phi},{e}\n")
            f.flush()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate dataset of PtEtaPhiE vectors"
    )
    parser.add_argument(
        "-n",
        "--ngen",
        type=int,
        default=int(10e5),
        help="Number of vectors to generate (default: 100000)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="dataset.csv",
        help="Output file name (default: dataset.csv)",
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="Random seed (default: 123)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_dataset(args)
