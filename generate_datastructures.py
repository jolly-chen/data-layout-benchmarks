# !/usr/bin/env python3

# This script generates partitioned data structures for benchmarking different data layouts in main.cpp.
# It writes C++ struct definitions of containers that partition the members of a base struct into substructures,
# either in contiguous memory or separately allocated. The generated containers provide an
# Array of Structures (AoS)-like interface via proxy reference structs.

from itertools import permutations
import numpy as np
import argparse

###########
# Helpers #
###########


def generate_struct_definition(struct_name, members, type_modifier=""):
    """
    Get the C++ struct definition as a string.

    :param struct_name: Name of the struct
    :param members: List of tuples (data_type, member_name)
    :param type_modifier: Modifier to append to each data type (e.g., "&")
    """
    lines = [f"struct {struct_name} {{"]
    for dtype, name in members:
        lines.append(f"    {dtype}{type_modifier} {name};")
    lines.append("};")
    return "\n".join(lines)


def subparticle_string(op, struct_name_base, generator):
    """
    Get the subparticle struct name for a given split operation.

    :param op: List of member indices from the original struct that need to be included in the subparticle
    :param struct_name_base: Base name of the struct
    :param generator: 'python' or 'cpp' to determine subparticle string format
    """
    if generator == "cpp":
        return f"Sub{struct_name_base}<SplitOp({{{', '.join(str(i) for i in op)}}}).data()>"
    else:
        return f"Sub{struct_name_base}{''.join(str(m) for m in op)}"


def define_contiguous_partitions_struct(partition, struct_name_base, generator):
    """
    Get the string that defines a struct containing each partition as std::span members.

    :param partition: List of partitions, each a list of member indices
    :param struct_name_base: Base name of the struct
    :param generator: 'python' or 'cpp' to determine subparticle string format
    """
    s = "struct Partitions {\n"
    for si, subset in enumerate(partition):
        memtype = subparticle_string(subset, struct_name_base, generator)
        s += f"    std::span<{memtype}> p{si};\n"
    s += "  };"
    return s


def assign_contiguous_partitions(partition, struct_name_base, generator):
    """
    Get the string that assigns each partition to its location in the storage vector.

    :param partition: List of partitions, each a list of member indices
    :param struct_name_base: Base name of the struct
    :param generator: 'python' or 'cpp' to determine subparticle string format
    """
    s = "size_t offset = 0;\n"
    for si, subset in enumerate(partition):
        memtype = subparticle_string(subset, struct_name_base, generator)
        s += (
            f"    p.p{si} = std::span<{memtype}>("
            + f"std::launder(reinterpret_cast<{memtype}*>(new (&storage[offset]) {memtype}[n])), n);\n"
        )
        s += f"    offset += AlignSize(p.p{si}.size_bytes(), alignment);\n"
    return s


def deallocate_contiguous_partitions(partition, struct_name_base, generator):
    """
    Get the string that deallocates each partition from the storage vector.

    :param partition: List of partitions, each a list of member indices
    :param struct_name_base: Base name of the struct
    :param generator: 'python' or 'cpp' to determine subparticle string format
    """
    s = "for (size_t i = n - 1; i == 0; --i) {\n"
    for si, subset in enumerate(partition):
        memtype = subparticle_string(subset, struct_name_base, generator)
        s += f"      p.p{si}[i].~{memtype}();\n"
    s += "    }\n\n"
    s += "    std::free(storage);\n"
    return s


def define_partitions_struct(partition, struct_name_base, generator):
    """
    Get the string that defines a struct containing each partition as pointer members.

    :param partition: List of partitions, each a list of member indices
    :param struct_name_base: Base name of the struct
    :param generator: 'python' or 'cpp' to determine subparticle string format
    """
    s = "struct Partitions {\n"
    for si, subset in enumerate(partition):
        s += f"    {subparticle_string(subset, struct_name_base, generator)} *p{si};\n"
    s += "  };"
    return s


def assign_partitions(partition, struct_name_base, generator):
    """
    Get the string that allocates each partition separately, using an aligned allocator.

    :param partition: List of partitions, each a list of member indices
    :param struct_name_base: Base name of the struct
    :param generator: 'python' or 'cpp' to determine subparticle string format
    """
    s = ""
    for si, subset in enumerate(partition):
        memtype = subparticle_string(subset, struct_name_base, generator)
        if si != 0:
            s += "\n"
        s += f"    p.p{si} = static_cast<{memtype}*>(std::aligned_alloc(alignment, AlignSize(n * sizeof({memtype}), alignment)));"
    return s


def deallocate_partitions(partition, struct_name_base, generator):
    """
    Get the string that deallocates each partition separately.

    :param partition: List of partitions, each a list of member indices
    :param struct_name_base: Base name of the struct
    :param generator: 'python' or 'cpp' to determine subparticle string format
    """
    s = ""
    for si, subset in enumerate(partition):
        memtype = subparticle_string(subset, struct_name_base, generator)
        if si != 0:
            s += "\n"
        s += f"    std::free(p.p{si});"
    return s


def assign_proxyref(members, partition, struct_name_base):
    """
    Get the string that returns a proxy reference struct in operator[].

    :param members: List of tuples (data_type, member_name)
    :param partition: List of partitions, each a list of member indices
    :param struct_name_base: Base name of the struct
    """
    mapping = [None] * len(members)
    for si, subset in enumerate(partition):
        for im, m in enumerate(subset):
            mapping[m] = [si, im]

    s = f"return {struct_name_base}Ref{{ {', '.join([f'p.p{si}[index].{members[m][1]}' for m, (si, im) in enumerate(mapping)])} }};"
    return s


def convert_codeword_to_partitions(codeword):
    """
    Convert a codeword to list partitions of the set.

    :param codeword: List of integers representing the codeword
    """
    partitions = [[] for _ in range(len(codeword))]
    for i, c in enumerate(codeword):
        partitions[c - 1].append(i)
    return [p for p in partitions if p]


##############
# Generators #
##############


def generate_partitions(members, contiguous, only=None):
    """
    Generates all the ways in which the members can be partitioned into substructures.
    Includes all permutations of members within each partition.

    Uses setpart1 in "Short Note: A Fast Iterative Algorithm for Generating Set Partitions"
    https://academic.oup.com/comjnl/article/32/3/281/331557

    :param members: List of tuples (data_type, member_name) containing all members in the original structure
    :param contiguous: Whether to generate only contiguous partitions. If True, also permutes the order of partitions.
    :param only: List of codewords to only generate specific partitions
    Yields: List of partitions, each a list of member indices
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

    if only:
        for partition_string in only:
            try:
                partition = [
                    [int(o) for o in subset] for subset in partition_string.split("_")
                ]
                yield partition
            except Exception as e:
                print(f"Error parsing partition string '{partition_string}': {e}")
    else:
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


###########
# Writers #
###########


def write_contiguous_partition(
    f, struct_name_base, partition_string, partition, members, generator
):
    """
    Write a structure definition for a container that stores contiguous partitions, each with a subset
    of the data members in the original structure. The container provides a AoS-like interface
    by returning a proxy reference struct in operator[].

    :param f: File object to write to
    :param struct_name_base: Base name of the struct
    :param partition_string: String representation of the partition
    :param partition: List of partitions, each a list of member indices
    :param members: List of tuples (data_type, member_name) containing all members in the original struct
    :param generator: 'python' or 'cpp'
    """
    f.write(
        f"""
struct PartitionedContainerContiguous{partition_string} {{
    { define_contiguous_partitions_struct(partition, struct_name_base, generator) }
    Partitions p;
    std::byte *storage;
    size_t n;

    PartitionedContainerContiguous{partition_string}(size_t n, size_t alignment) : n(n) {{
        // Allocate each partition
        size_t total_size = 0 + { " + ".join([ f"AlignSize(n * sizeof({subparticle_string(subset, struct_name_base, generator)}), alignment)" for subset in partition ]) };
        storage = static_cast<std::byte*>(std::aligned_alloc(alignment, total_size));

        // Assign each partition to its location in the storage vector
        { assign_contiguous_partitions(partition, struct_name_base, generator) }
    }}

    inline {struct_name_base}Ref operator[](const size_t index) const {{
        { assign_proxyref(members, partition, struct_name_base) }
    }}

    size_t size() const {{ return n; }}

    ~PartitionedContainerContiguous{partition_string}() {{
        // Deallocate each partition
        { deallocate_contiguous_partitions(partition, struct_name_base, generator) }
    }}
}};
"""
    )


def write_partition(
    f, struct_name_base, partition_string, partition, members, generator
):
    """
    Write a structure definition for a container that stores partitions, not necessarily contiguous,
    each with a subset of the data members in the original structure. The container provides a
    AoS-like interface by returning a proxy reference struct in operator[].

    :param f: File object to write to
    :param struct_name_base: Base name of the struct
    :param partition_string: String representation of the partition
    :param partition: List of partitions, each a list of member indices
    :param members: List of tuples (data_type, member_name) containing all members in the original structure
    :param generator: 'python' or 'cpp'
    """
    f.write(
        f"""
struct PartitionedContainer{partition_string} {{
    { define_partitions_struct(partition, struct_name_base, generator) }

  Partitions p;
  size_t n;

public:
    PartitionedContainer{partition_string}(size_t n, size_t alignment) : n(n) {{
        { assign_partitions(partition, struct_name_base, generator) }
    }}

    inline {struct_name_base}Ref operator[](const size_t index) const {{
        { assign_proxyref(members, partition, struct_name_base) }
    }}

    size_t size() const {{ return n; }}

    ~PartitionedContainer{partition_string}() {{
        // Deallocate each partition
        { deallocate_partitions(partition, struct_name_base, generator) }
    }}
}};
"""
    )


def write_subsets(f, struct_name_base, members, subsets, generator):
    """
    Write struct definitions for all substructures used in the partitions.
    The substructures contain only a subset of the data members in the original structure.

    :param f: File object to write to
    :param struct_name_base: Base name of the struct
    :param members: List of tuples (data_type, member_name) containing all members in the original structure
    :param subsets: List of subsets, each a list of member indices
    :param generator: 'python' or 'cpp'
    """
    f.write(generate_struct_definition(struct_name_base, members) + "\n\n")
    f.write(generate_struct_definition(f"{struct_name_base}Ref", members, "&") + "\n\n")

    if generator == "cpp":
        f.write(f"template <auto Members> struct Sub{struct_name_base};\n\n")
        f.write(
            "// Forward declarations of structures with a subset of Particle members\n"
        )

        for subset in subsets:
            f.write(
                f"consteval {{ SplitStruct<{struct_name_base}, Sub{struct_name_base}>(SplitOp({{{', '.join(str(i) for i in subset)}}})); }}\n"
            )
    else:
        for subset in subsets:
            subset_string = "".join(str(i) for i in subset)
            f.write(
                generate_struct_definition(
                    f"Sub{struct_name_base}{subset_string}",
                    [members[i] for i in subset],
                )
                + "\n"
            )


def write_benchmarks(p_list, struct_name_base, members, contiguous, generator):
    """
    Write the benchmark invocations in main.cpp for all generated partitioned containers.

    :param p_list: List of partition strings
    :param struct_name_base: Base name of the struct
    :param members: List of tuples (data_type, member_name) containing all members in the original structure
    :param contiguous: Whether to use contiguous partitioned containers
    :param generator: 'python' or 'cpp'
    """
    with open("main.cpp", "r") as f:
        lines = f.readlines()

    with open("main.cpp", "w") as f:
        main_start = [i for i, l in enumerate(lines) if "problem_sizes" in l][-1]
        f.writelines(lines[: main_start + 1])

        f.write(f"    // THIS IS GENERATED USING generate_datastructures.py\n")
        if generator == "cpp":
            for p_string in p_list:
                partition = [[int(o) for o in subset] for subset in p_string.split("_")]
                mapping = len(members) * [None]
                splitops = ", ".join(
                    [
                        subparticle_string(subset, struct_name_base, generator)
                        for subset in partition
                    ]
                )

                for si, subsets in enumerate(partition):
                    for im, m in enumerate(subsets):
                        mapping[m] = f"{{{si}, {im}}}"

                f.write(
                    f"    RunAllBenchmarks<PartitionedContainer<{struct_name_base}Ref,"
                    + f" Mapping({{{', '.join(mapping)}}}).data(), {splitops}>>(n, alignment);\n"
                )
        else:
            for p_string in p_list:
                f.write(
                    f"    RunAllBenchmarks<PartitionedContainer{'Contiguous' if contiguous else ''}{p_string}>(n, alignment);\n"
                )

        f.write("  }\n\n")
        f.write("  PAPI_cleanup_eventset(papi_eventset);\n")
        f.write("  PAPI_destroy_eventset(&papi_eventset);\n")
        f.write("  return 0;\n}\n// END GENERATED CODE\n")


def write_partitioned_structs(
    struct_name_base, members, start, end, contiguous, generator, only=None
):
    """
    Generate partitioned data structures and write them to datastructures.h.
    Also write benchmark invocations to main.cpp.

    :param struct_name_base: Base name of the struct
    :param members: List of tuples (data_type, member_name) containing all members in the original structure
    :param start: Start index for batching
    :param end: End index for batching
    :param contiguous: Whether to generate contiguous partitioned structures
    :param generator: 'python' or 'cpp'
    :param only: List of codewords to only generate specific partitions
    """
    with open("datastructures.h", "w") as f:
        p_list = []
        s_list = []

        for partition in generate_partitions(members, contiguous, only):
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

            if generator == "python":
                if contiguous:
                    write_contiguous_partition(
                        f,
                        struct_name_base,
                        partition_string,
                        partition,
                        members,
                        generator,
                    )
                else:
                    write_partition(
                        f,
                        struct_name_base,
                        partition_string,
                        partition,
                        members,
                        generator,
                    )

            for subset in partition:
                if subset not in s_list:
                    s_list.append(subset)

        print(f"Generated {len(p_list[start:])} partitioned data structures.")

    with open("datastructures.h", "r") as f:
        prev_lines = f.readlines()

    with open("datastructures.h", "w") as f:
        f.write(f"#ifndef DATASTRUCTURES_H\n")
        f.write(f"#define DATASTRUCTURES_H\n")
        f.write("// THIS FILE IS GENERATED USING generate_datastructures.py\n")
        f.write('#include "datastructures.h"\n')
        f.write('#include "struct_transformer.h"\n\n')

        write_subsets(f, struct_name_base, members, s_list, generator)

        f.writelines(prev_lines)
        f.write(f"\n#endif // DATASTRUCTURES_H\n")

    write_benchmarks(p_list[start:], struct_name_base, members, contiguous, generator)


def write_test_partitions():
    """
    Generate test partitioned data structures used in the unit tests in test.cpp and write them to test.h.
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
            write_contiguous_partition(
                f, struct_name_base, partition_string, partition, members, "python"
            )
            write_partition(
                f, struct_name_base, partition_string, partition, members, "python"
            )

            for subset in partition:
                if subset not in s_list:
                    s_list.append(subset)

    with open("test.h", "r") as f:
        prev_lines = f.readlines()

    with open("test.h", "w") as f:
        f.write(f"#ifndef TEST_H\n")
        f.write(f"#define TEST_H\n")
        f.write("// THIS FILE IS GENERATED USING generate_datastructures.py\n")
        f.write('#include "struct_transformer.h"\n\n')

        write_subsets(f, struct_name_base, members, s_list, "python")

        f.writelines(prev_lines)
        f.write(f"\n#endif // TEST_H\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Set batch size for partition generation",
        default=-1,
    )
    parser.add_argument(
        "--batch_num",
        type=int,
        help="Set current batch number for partition generation",
        default=0,
    )
    parser.add_argument(
        "--contiguous",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Generate contiguous partitioned structures",
    )
    parser.add_argument(
        "--generator",
        type=str,
        choices=["python", "cpp"],
        default="python",
        help="Choose whether partitions are generated using Python or C++ Reflection",
    )
    parser.add_argument(
        "--only",
        type=str,
        nargs="*",
        help="Only generate partitions with these codewords (e.g., '01_23')",
    )
    args = parser.parse_args()

    print(
        f"Configuration:\n  batch_size={args.batch_size}\n  batch_num={args.batch_num}\n  contiguous={args.contiguous}\n  generator={args.generator}\n  only={args.only}"
    )

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

    write_partitioned_structs(
        struct_name_base,
        data_members,
        start,
        end,
        args.contiguous,
        args.generator,
        args.only,
    )
    write_test_partitions()
