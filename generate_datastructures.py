from multiprocessing import Pool
from functools import partial
from itertools import permutations, chain, combinations
import numpy as np
import math
import os
import sys

data_members = [
    ("int", "id"),
    ("double", "pt"),
    ("double", "eta"),
    ("double", "phi"),
    ("double", "e"),
    ("char", "charge"),
    ("std::array<std::array<double, 3>, 3>", "posCovMatrix"),
]

struct_name_base = "Particle"


def generate_struct_definition(struct_name, members, type_modifier=""):
    lines = [f"struct {struct_name} {{"]
    for dtype, name in members:
        lines.append(f"    {dtype}{type_modifier} {name};")
    lines.append("};")
    return "\n".join(lines)


def generate_subsets(struct_name_base, members):
    "Subsequences of the iterable from shortest to longest."
    s = range(len(members))
    subsets = [
        list(p)
        for r in range(len(s) + 1)
        for c in combinations(s, r)
        for p in permutations(c)
        if c
    ]
    with open("datastructures.h", "w") as f:
        f.write("#ifndef DATASTRUCTURES_H\n")
        f.write("#define DATASTRUCTURES_H\n")
        f.write('#include "datastructures.h"\n')
        f.write('#include "struct_transformer.h"\n\n')
        f.write(generate_struct_definition(struct_name_base, members) + "\n\n")
        f.write(generate_struct_definition(f"{struct_name_base}Ref", members, "&") + "\n\n")
        f.write(f"template <auto Members> struct Sub{struct_name_base};\n\n")

        f.write(
            "// Forward declarations of structures with a subset of Particle members\n"
        )
        for subset in subsets:
            f.write(
                f"consteval {{ SplitStruct<{struct_name_base}, Sub{struct_name_base}>(SplitOp({{{', '.join(str(i) for i in subset)}}})); }}\n"
            )
        f.write("\n#endif // DATASTRUCTURES_H\n")


def convert_codeword_to_partitions(set, codeword):
    """
    Convert a codeword to list partitions of the set.
    """
    partitions = [[] for _ in range(len(codeword))]
    for i, c in enumerate(codeword):
        partitions[c - 1].append(i)
    return [p for p in partitions if p]


def generate_partitions(members):
    """
    Generates all the ways in which the members can be partitioned into seperate structs.
    Includes all permutations of members within each partition.

    Uses setpart1 in "Short Note: A Fast Iterative Algorithm for Generating Set Partitions"
    https://academic.oup.com/comjnl/article/32/3/281/331557

    """
    r = 0
    n = len(members)
    codeword = np.repeat(1, n + 1)
    n1 = n - 1
    g = np.repeat(1, n + 1)
    while r != 1:
        while r < n1:
            r += 1
            codeword[r] = 1
            g[r] = g[r - 1]

        for j in range(1, g[n1] + 2):
            codeword[n] = j
            partitions = convert_codeword_to_partitions(members, codeword[1:])
            if any(len(p) > 1 for p in partitions):
                for i, p in enumerate(partitions):
                    if len(p) > 1:
                        for perm in permutations(p):
                            partitions[i] = list(perm)
                            yield partitions
            else:
                yield partitions

        while codeword[r] > g[r - 1]:
            r -= 1

        codeword[r] += 1
        if codeword[r] > g[r]:
            g[r] = codeword[r]

def generate_partitioned_structs(struct_name_base, members):
    with open("main.cpp", "r") as f:
        lines = f.readlines()

    with open("main.cpp", "w") as f:
        main_start = [i for i, l in enumerate(lines) if "GetProblemSizes" in l][1]
        f.writelines(lines[: main_start + 1])

        f.write(f"\tconstexpr std::array containers = {{\n")
        partitions = generate_partitions(members)
        for i, p in enumerate(partitions):
            splitops = []
            for p in partition:
                splitops.append(f"Sub{struct_name_base}<SplitOp({{{', '.join(str(i) for i in p)}}}).data()>")

            f.write(f"\t\tRunAllBenchmarks<PartitionedContainer<{struct_name_base}Ref, {', '.join(splitops)}>>(n);\n")

            if i != 0: f.write(f",\n")
            f.write(f"\t\t^^PartitionedContainer<{struct_name_base}Ref, {', '.join(splitops)}>")

        f.write(f"\n\t}};\n\n")
        f.write("  RunAllBenchmarks<containers>(problem_sizes, alignment);\n")
        f.write("\t\n\treturn 0;\n}\n")
        f.write(f"// END GENERATED CODE\n")

if __name__ == "__main__":
    generate_subsets(struct_name_base, data_members)
    generate_partitioned_structs(struct_name_base, data_members)
