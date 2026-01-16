from itertools import permutations
import numpy as np
import argparse
import sys


###########
# Helpers #
###########

def generate_struct_definition(struct_name, members, type_modifier=""):
    lines = [f"struct {struct_name} {{"]
    for dtype, name in members:
        lines.append(f"    {dtype}{type_modifier} {name};")
    lines.append("};")
    return "\n".join(lines)

def subparticle_string(op, struct_name_base):
    return f"Sub{struct_name_base}{''.join(str(m) for m in op)}"

def define_contiguous_partitions_struct(partition, struct_name_base):
    s = "struct Partitions {\n"
    for si, subset in enumerate(partition):
        s += f"\t\tstd::span<{subparticle_string(subset, struct_name_base)}> p{si};\n"
    s += "\t};"
    return s

def assign_contiguous_partitions(partition, struct_name_base):
    s = "size_t offset = 0;\n"
    for si, subset in enumerate(partition):
        memtype = subparticle_string(subset, struct_name_base)
        s += (
            f"\t\tp.p{si} = std::span<{memtype}>("
            + f"std::launder(reinterpret_cast<{memtype}*>(new (&storage[offset]) {memtype}[n])), n);\n"
        )
        s += f"\t\toffset += align_size(p.p{si}.size_bytes(), alignment);\n"
    return s

def deallocate_contiguous_partitions(partition, struct_name_base):
    s = "for (size_t i = n - 1; i == 0; --i) {\n"
    for si, subset in enumerate(partition):
        memtype = subparticle_string(subset, struct_name_base)
        s += f"\t\t\tp.p{si}[i].~{memtype}();\n"
    s += "\t\t}\n\n"
    s += "\t\tstd::free(storage);\n"
    return s

def define_partitions_struct(partition, struct_name_base):
    s = "struct Partitions {\n"
    for si, subset in enumerate(partition):
        s += f"\t\t{subparticle_string(subset, struct_name_base)} *p{si};\n"
    s += "\t};"
    return s

def assign_partitions(partition, struct_name_base):
    s = ""
    for si, subset in enumerate(partition):
        memtype = subparticle_string(subset, struct_name_base)
        if si != 0: s += "\n"
        s += (
            f"\t\tp.p{si} = static_cast<{memtype}*>(std::aligned_alloc(alignment, align_size(n * sizeof({memtype}), alignment)));"
        )
    return s

def deallocate_partitions(partition, struct_name_base):
    s = ""
    for si, subset in enumerate(partition):
        memtype = subparticle_string(subset, struct_name_base)
        if si != 0: s += "\n"
        s += f"\t\tstd::free(p.p{si});"
    return s

def assign_proxyref(members, partition, struct_name_base):
    mapping = [None] * len(members)
    for si, subset in enumerate(partition):
        for im, m in enumerate(subset):
            mapping[m] = [si, im]

    s = f"return {struct_name_base}Ref{{ {', '.join([f'p.p{si}[index].{members[m][1]}' for m, (si, im) in enumerate(mapping)])} }};"
    return s


def convert_codeword_to_partitions(codeword):
    """
    Convert a codeword to list partitions of the set.
    """
    partitions = [[] for _ in range(len(codeword))]
    for i, c in enumerate(codeword):
        partitions[c - 1].append(i)
    return [p for p in partitions if p]

###########
# Generators #
###########
def write_contiguous_partition(f, struct_name_base, partition_string, partition, members):
    f.write(f"""
struct PartitionedContainerContiguous{partition_string} {{
    { define_contiguous_partitions_struct(partition, struct_name_base) }

    Partitions p;
    std::byte *storage;
    size_t n;

    PartitionedContainerContiguous{partition_string}(size_t n, size_t alignment) : n(n) {{
        // Allocate each partition
        size_t total_size = 0 + { " + ".join([ f"align_size(n * sizeof({subparticle_string(subset, struct_name_base)}), alignment)" for subset in partition ]) };
        storage = static_cast<std::byte*>(std::aligned_alloc(alignment, total_size));

        // Assign each partition to its location in the storage vector
        { assign_contiguous_partitions(partition, struct_name_base) }
    }}

    inline {struct_name_base}Ref operator[](const size_t index) const {{
        { assign_proxyref(members, partition, struct_name_base) }
    }}

    size_t size() const {{ return n; }}

    ~PartitionedContainerContiguous{partition_string}() {{
        // Deallocate each partition
        { deallocate_contiguous_partitions(partition, struct_name_base) }
    }}
}};
""")

def write_partition(f, struct_name_base, partition_string, partition, members):
    f.write(f"""
struct PartitionedContainer{partition_string} {{
    { define_partitions_struct(partition, struct_name_base) }

  Partitions p;
  size_t n;

public:
    PartitionedContainer{partition_string}(size_t n, size_t alignment) : n(n) {{
        { assign_partitions(partition, struct_name_base) }
    }}

    inline {struct_name_base}Ref operator[](const size_t index) const {{
        { assign_proxyref(members, partition, struct_name_base) }
    }}

    size_t size() const {{ return n; }}

    ~PartitionedContainer{partition_string}() {{
        // Deallocate each partition
        { deallocate_partitions(partition, struct_name_base) }
    }}
}};
""")

def write_subsets(f, struct_name_base, members, subsets):
    f.write(generate_struct_definition(struct_name_base, members) + "\n\n")
    f.write(generate_struct_definition(f"{struct_name_base}Ref", members, "&") + "\n\n")

    for subset in subsets:
        subset_string = "".join(str(i) for i in subset)
        f.write(generate_struct_definition(f"Sub{struct_name_base}{subset_string}",
                                            [members[i] for i in subset]) + "\n")

def write_benchmarks(p_list, contiguous):
    with open("main.cpp", "r") as f:
        lines = f.readlines()

    with open("main.cpp", "w") as f:
        main_start = [i for i, l in enumerate(lines) if "problem_sizes" in l][-1]
        f.writelines(lines[: main_start + 1])

        f.write(f"\t\t// THIS IS GENERATED USING generate_datastructures.py\n")
        for partition in p_list:
            f.write(
                f"\t\tRunAllBenchmarks<PartitionedContainer{'Contiguous' if contiguous else ''}{partition}>(n, alignment);\n"
            )

        f.write("\t}\t\n\treturn 0;\n}\n")
        f.write(f"// END GENERATED CODE\n")

def generate_partitions(members, contiguous):
    """
    Generates all the ways in which the members can be partitioned into seperate structs.
    Includes all permutations of members within each partition.

    Uses setpart1 in "Short Note: A Fast Iterative Algorithm for Generating Set Partitions"
    https://academic.oup.com/comjnl/article/32/3/281/331557
    """
    def permute_elements_in_subset(partition):
        if any(len(p) > 1 for p in partition):
            for i, p in enumerate(partition):
                if len(p) > 1:
                    for subset_perm in permutations(p):
                        partition[i] = list(subset_perm)
                        yield partition
        else:
            yield partition

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
            partition = convert_codeword_to_partitions(codeword[1:])
            if contiguous:
                for partition_perm in permutations(partition):
                    partition_perm = list(partition_perm)
                    for permuted in permute_elements_in_subset(partition_perm):
                        yield permuted
            else:
                for permuted in permute_elements_in_subset(partition):
                        yield permuted

        while codeword[r] > g[r - 1]:
            r -= 1

        codeword[r] += 1
        if codeword[r] > g[r]:
            g[r] = codeword[r]


def generate_partitioned_structs(struct_name_base, members, start, end, contiguous):
    with open("datastructures.h", "w") as f:
        p_list = []
        s_list = []

        for partition in generate_partitions(members, contiguous):
            if end and len(p_list) >= end:
                break

            # Encode e.g., [[0,2],[1,3]] as "02_13"
            partition_string = "_".join(
                ["".join(str(m) for m in subset) for subset in partition]
            )

            # Skip duplicates
            if partition_string in p_list:
                # print(f"Skipping duplicate partition {partition_string}")
                continue
            p_list.append(partition_string)

            if len(p_list) < start:
                continue

            if contiguous:
                write_contiguous_partition(f, struct_name_base, partition_string, partition, members)
            else:
                write_partition(f, struct_name_base, partition_string, partition, members)

            for subset in partition:
                if subset not in s_list:
                    s_list.append(subset)

        print(f"Generated {len(p_list[start:])} partitioned data structures.")

    with open("datastructures.h", "r") as f:
        prev_lines = f.readlines()

    with open("datastructures.h", "w") as f:
        f.write(f"#ifndef DATASTRUCTURES_H\n")
        f.write(f"#define DATASTRUCTURES_H\n")
        f.write('#include "datastructures.h"\n')
        f.write('#include "struct_transformer.h"\n\n')

        write_subsets(f, struct_name_base, members, s_list)

        f.writelines(prev_lines)
        f.write(f"\n#endif // DATASTRUCTURES_H\n")

    write_benchmarks(p_list[start:], contiguous)


def generate_test_partitions():
    """
    Partitions used in test.cpp
    """
    struct_name_base = "S"
    members = [("int", "x"), ("double", "y"), ("float", "z"), ("char", "w")]

    with open("test.h", "w") as f:
        p_list = [[[0, 1], [2, 3]], [[0], [1], [2], [3]], [[0], [1], [2, 3]]]
        s_list = []

        for partition in p_list:
            partition_string = "_".join(
                ["".join(str(m) for m in subset) for subset in partition]
            )
            write_contiguous_partition(f, struct_name_base, partition_string, partition, members)
            write_partition(f, struct_name_base, partition_string, partition, members)

            for subset in partition:
                if subset not in s_list:
                    s_list.append(subset)

    with open("test.h", "r") as f:
        prev_lines = f.readlines()

    with open("test.h", "w") as f:
        f.write(f"#ifndef TEST_H\n")
        f.write(f"#define TEST_H\n")
        f.write('#include "struct_transformer.h"\n\n')

        write_subsets(f, struct_name_base, members, s_list)

        f.writelines(prev_lines)
        f.write(f"\n#endif // TEST_H\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, help="Set output directory", default=-1)
    parser.add_argument('--batch_num', type=int, help='Set input file(s)', default=0)
    parser.add_argument('--contiguous', action=argparse.BooleanOptionalAction, default=True, help='Generate contiguous partitioned structures')
    args = parser.parse_args()

    if args.batch_size == -1:
        start = 0
        end = None
    else:
        start = args.batch_num * args.batch_size
        end = start + args.batch_size

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

    generate_partitioned_structs(struct_name_base, data_members, start, end, args.contiguous)
    generate_test_partitions()
