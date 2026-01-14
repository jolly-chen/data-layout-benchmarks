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

        for subset in subsets:
            subset_string = "".join(str(i) for i in subset)
            f.write(generate_struct_definition(f"Sub{struct_name_base}{subset_string}",
                                               [members[i] for i in subset]) + "\n")



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
            partition = convert_codeword_to_partitions(members, codeword[1:])
            for partition_perm in permutations(partition):
                partition_perm = list(partition_perm)
                if any(len(p) > 1 for p in partition_perm):
                    for i, p in enumerate(partition_perm):
                        if len(p) > 1:
                            for subset_perm in permutations(p):
                                partition_perm[i] = list(subset_perm)
                                yield partition_perm
                else:
                    yield partition_perm

        while codeword[r] > g[r - 1]:
            r -= 1

        codeword[r] += 1
        if codeword[r] > g[r]:
            g[r] = codeword[r]


def generate_partitioned_structs(struct_name_base, members):
    def subparticle_string(op):
        return f"Sub{struct_name_base}{''.join(str(m) for m in op)}"

    def define_partitions_struct(partition):
        s = "struct Partitions {\n"
        for si, subset in enumerate(partition):
            s += f"\t\tstd::span<{subparticle_string(subset)}> p{si};\n"
        s += "\t};"
        return s

    def assign_partitions(partition):
        s = "size_t offset = 0;\n"
        for si, subset in enumerate(partition):
            memtype = subparticle_string(subset)
            s += (
                f"\t\tp.p{si} = std::span<{memtype}>("
                + f"std::launder(reinterpret_cast<{memtype}*>(new (&storage[offset]) {memtype}[n])), n);\n"
            )
            s += f"\t\toffset += align_size(p.p{si}.size_bytes(), alignment);\n"
        return s

    def assign_proxyref(partition):
        mapping = [None] * len(members)
        for si, subset in enumerate(partition):
            for im, m in enumerate(subset):
                mapping[m] = [si, im]

        s = f"return {struct_name_base}Ref{{ {', '.join([f'p.p{si}[index].{members[m][1]}' for m, (si, im) in enumerate(mapping)])} }};"
        return s

    def deallocate_partitions(partition):
        s = "for (size_t i = n - 1; i == 0; --i) {\n"
        for si, subset in enumerate(partition):
            memtype = subparticle_string(subset)
            s += f"\t\t\tp.p{si}[i].~{memtype}();\n"
        s += "\t\t}"
        return s


    with open("datastructures.h", "a") as f:
        p_list = []

        for i, partition in enumerate(generate_partitions(members)):
            partition_string = "_".join(
                ["".join(str(m) for m in subset) for subset in partition]
            )

            if partition_string in p_list:
                # print(f"Duplicate partition {partition_string}, skipping.")
                continue
            p_list.append(partition_string)

            f.write(
                f"""
struct PartitionedContainer{partition_string} {{
    { define_partitions_struct(partition) }

    Partitions p;
    alignas(64) std::vector<std::byte> storage;
    size_t n;

    PartitionedContainer{partition_string}(size_t n, size_t alignment) : n(n) {{
        // Allocate each partition
        size_t total_size = 0 + { " + ".join([ f"align_size(n * sizeof({subparticle_string(subset)}), alignment)" for subset in partition ]) };
        storage.resize(total_size);

        // Assign each partition to its location in the storage vector
        { assign_partitions(partition) }
    }}

    inline {struct_name_base}Ref operator[](const size_t index) const {{
        { assign_proxyref(partition) }
    }}

    size_t size() const {{ return n; }}

    ~PartitionedContainer{partition_string}() {{
        // Deallocate each partition
        { deallocate_partitions(partition) }
    }}
}};
"""
            )

        f.write("\n#endif // DATASTRUCTURES_H\n")
        print(f"Generated {len(p_list)} partitioned data structures.")

    #########
    with open("main.cpp", "r") as f:
        lines = f.readlines()

    with open("main.cpp", "w") as f:
        main_start = [i for i, l in enumerate(lines) if "problem_sizes" in l][-1]
        f.writelines(lines[: main_start + 1])

        f.write(f"\t\t// THIS IS GENERATED USING generate_datastructures.py\n")
        for partition in p_list:
            f.write(
                f"\t\tRunAllBenchmarks<PartitionedContainer{partition}>(n, alignment);\n"
            )

        f.write("\t}\t\n\treturn 0;\n}\n")
        f.write(f"// END GENERATED CODE\n")

if __name__ == "__main__":
    generate_subsets(struct_name_base, data_members)
    generate_partitioned_structs(struct_name_base, data_members)
