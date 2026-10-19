#!/usr/bin/env python3
"""
End-to-end test of the benchmark driver with a plugged-in struct and kernel.

Runs main.cpp built with the Particle kernels and the Cell stencil kernel
(cell_kernels.h), and fixed byte targets (TARGET_BYTES) so the problem sizes
do not depend on the machine, on the datasets in tests/data. Checks that
- every kernel is registered for every layout of its own struct, and no other,
- every run stays within its byte target,
- every Particle result validates against the reference data,
- a dataset without a header row is rejected.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

failures = 0


def check(cond, message):
    global failures
    if not cond:
        failures += 1
        print(f"\033[1;31mFAILED\033[0m {message}")


def layouts_in(header):
    """Codewords of the containers in a generated layouts_<Struct>.h."""
    return set(re.findall(r"^struct PartitionedContainer([0-9_]+) \{", header.read_text(), re.M))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", type=Path, required=True, help="main.cpp built with cell_kernels.h")
    parser.add_argument("--layouts_dir", type=Path, required=True, help="Directory with layouts_Particle.h and layouts_Cell.h")
    parser.add_argument("--data_dir", type=Path, required=True, help="tests/data")
    args = parser.parse_args()

    def run(*driver_args):
        # Run in tests/data, so the list files can be passed by name
        return subprocess.run([str(args.main.resolve()), *driver_args], capture_output=True, text=True, cwd=args.data_dir)

    particle_layouts = layouts_in(args.layouts_dir / "layouts_Particle.h")
    cell_layouts = layouts_in(args.layouts_dir / "layouts_Cell.h")
    inputs = ["--input", "Particle=particle_inputs", "--input", "Cell=cell_inputs"]

    # Registration: each kernel on all layouts of its own struct, once per target
    result = run(*inputs, "--benchmark_list_tests")
    check(result.returncode == 0, f"--benchmark_list_tests failed:\n{result.stderr}")
    names = result.stdout.split()
    registered = {}
    for name in names:
        kernel, layout = name.split("_PartitionedContainer")
        registered.setdefault(kernel, []).append(layout)
    expected = {
        "InvariantMassSequential": particle_layouts,
        "InvariantMassRandom": particle_layouts,
        "Stencil": cell_layouts,
    }
    check(set(registered) == set(expected), f"registered kernels {sorted(registered)}")
    for kernel, layouts in expected.items():
        found = registered.get(kernel, [])
        check(sorted(found) == sorted(list(layouts) * 2),
              f"{kernel} registered for {sorted(set(found))}, expected {sorted(layouts)} twice")

    # A short run of every benchmark
    result = run(*inputs, "--validation", "validation_files", "--benchmark_format=json", "--benchmark_min_time=1x")
    check(result.returncode == 0, f"benchmark run failed:\n{result.stderr}")
    check("VALIDATION FAILED" not in result.stderr, f"validation failed:\n{result.stderr}")
    benchmarks = json.loads(result.stdout)["benchmarks"] if result.returncode == 0 else []
    check(len(benchmarks) == len(names), f"{len(benchmarks)} results for {len(names)} benchmarks")
    for b in benchmarks:
        check(0 < b["problem_size"] and b["bytes_for_all"] <= b["target_bytes"],
              f"{b['name']}: n={b['problem_size']} uses {b['bytes_for_all']} of {b['target_bytes']} bytes")
        # The targets are small enough for the 1000 values of the reference
        if b["name"].startswith("InvariantMass"):
            check(b["validated"] == 1, f"{b['name']} with n={b['problem_size']} was not validated")

    # Datasets without a header row are rejected
    result = run("--input", "Particle=no_header_inputs", "--input", "Cell=cell_inputs", "--benchmark_list_tests")
    check(result.returncode != 0 and "header row" in result.stderr,
          f"dataset without header row not rejected: exit code {result.returncode}, stderr {result.stderr!r}")

    print(f"Checked {len(names)} benchmarks on {len(particle_layouts)} Particle and {len(cell_layouts)} Cell layouts")
    if failures == 0:
        print("\033[1;32mPASSED\033[0m")
    else:
        print(f"\033[1;31m{failures} check(s) FAILED\033[0m")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
