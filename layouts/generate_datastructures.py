#!/usr/bin/env python3

# This script generates partitioned data structures for benchmarking different data layouts.
# It writes C++ struct definitions of containers that partition the members of a base struct into substructures,
# either in contiguous memory or separately allocated. The generated containers provide an
# Array of Structures (AoS)-like interface via proxy reference structs, and satisfy the LayoutOf
# concept in layout_helpers.h.
#
# Output (in --output_dir, default layouts/generated):
#   layouts_<Struct>.h  containers for the struct in the spec, in namespace layouts::<Struct>
#   layouts_index.h     includes all layouts_*.h in the directory
#   test.h              containers used by test.cpp

from itertools import permutations
from pathlib import Path
import itertools
import io
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


def substructure_string(op, struct_name_base):
    """
    Get the substructure struct name for a given split operation.

    :param op: List of member indices from the original struct that need to be included in the substructure
    :param struct_name_base: Base name of the struct
    """
    return f"Sub{struct_name_base}{''.join(str(m) for m in op)}"


def define_contiguous_partitions_struct(partition, struct_name_base):
    """
    Get the string that defines a struct containing each partition as std::span members.

    :param partition: List of partitions, each a list of member indices
    :param struct_name_base: Base name of the struct
    """
    s = ""
    for si, subset in enumerate(partition):
        memtype = substructure_string(subset, struct_name_base)
        s += f"    std::span<{memtype}> p{si};\n"
    return s


def assign_contiguous_partitions(partition, struct_name_base):
    """
    Get the string that assigns each partition to its location in the storage vector.

    :param partition: List of partitions, each a list of member indices
    :param struct_name_base: Base name of the struct
    """
    s = "        size_t offset = 0;\n"
    for si, subset in enumerate(partition):
        memtype = substructure_string(subset, struct_name_base)
        s += (
            f"        p{si} = std::span<{memtype}>("
            + f"std::launder(reinterpret_cast<{memtype}*>(new (&storage[offset]) {memtype}[n])), n);\n"
        )
        s += f"        offset += AlignSize(p{si}.size_bytes(), alignment);\n"
    return s


def deallocate_contiguous_partitions(partition, struct_name_base):
    """
    Get the string that deallocates each partition from the storage vector.

    :param partition: List of partitions, each a list of member indices
    :param struct_name_base: Base name of the struct
    """
    s = "        for (size_t i = 0; i < n; ++i) {\n"
    for si, subset in enumerate(partition):
        memtype = substructure_string(subset, struct_name_base)
        s += f"              p{si}[i].~{memtype}();\n"
    s += "        }\n\n"
    # s += "    std::free(storage);"
    return s


def define_partitions_struct(partition, struct_name_base):
    """
    Get the string that defines a struct containing each partition as pointer members.

    :param partition: List of partitions, each a list of member indices
    :param struct_name_base: Base name of the struct
    """
    s = ""
    for si, subset in enumerate(partition):
        s += f"    std::unique_ptr<{substructure_string(subset, struct_name_base)}[], StorageDeleter<{substructure_string(subset, struct_name_base)}>> p{si};\n"
    return s


def assign_partitions(partition, struct_name_base):
    """
    Get the string that allocates each partition separately, using an aligned allocator.

    :param partition: List of partitions, each a list of member indices
    :param struct_name_base: Base name of the struct
    """
    s = ""
    for si, subset in enumerate(partition):
        memtype = substructure_string(subset, struct_name_base)
        if si != 0:
            s += "\n"
        s += f"        p{si} = std::unique_ptr<{substructure_string(subset, struct_name_base)}[], StorageDeleter<{substructure_string(subset, struct_name_base)}>>(static_cast<{memtype}*>(std::aligned_alloc(alignment, AlignSize(n * sizeof({memtype}), alignment))), StorageDeleter<{substructure_string(subset, struct_name_base)}>());\n"
        s += f"        if (!p{si} && n != 0) {{ throw std::bad_alloc(); }}"
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

    s = f"return reference{{ {', '.join([f'p{si}[index].{members[m][1]}' for m, (si, im) in enumerate(mapping)])} }};"
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
        permutations_per_subset = [permutations(subset) for subset in partition]
        for permuted_partition in itertools.product(*permutations_per_subset):
            yield [list(subset) for subset in permuted_partition]

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


def bytes_for_all_expression(partition, struct_name_base):
    """
    Get the C++ expression for the number of bytes allocated for n elements, including the
    padding that aligns every partition to `alignment`.

    :param partition: List of partitions, each a list of member indices
    :param struct_name_base: Base name of the struct
    """
    return " + ".join(
        [
            f"AlignSize(n * sizeof({substructure_string(subset, struct_name_base)}), alignment)"
            for subset in partition
        ]
    )


def write_contiguous_partition(
    f, struct_name_base, partition_string, partition, members
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
    """
    f.write(
        f"""
struct PartitionedContainerContiguous{partition_string} {{
    using value_type = ::{struct_name_base};
    using reference = ::{struct_name_base}Ref;

{ define_contiguous_partitions_struct(partition, struct_name_base) }
    std::unique_ptr<std::byte[], StorageDeleter<std::byte>> storage;
    size_t n;

    static std::string to_string() {{ return "PartitionedContainerContiguous{partition_string}"; }}

    static size_t bytes_for_all(size_t n, size_t alignment) {{
        return { bytes_for_all_expression(partition, struct_name_base) };
    }}

    PartitionedContainerContiguous{partition_string}(size_t n, size_t alignment) : n(n) {{
        // Allocate each partition
        storage = std::unique_ptr<std::byte[], StorageDeleter<std::byte>>(static_cast<std::byte*>(std::aligned_alloc(alignment, bytes_for_all(n, alignment))), StorageDeleter<std::byte>());
        if (!storage && n != 0) {{ throw std::bad_alloc(); }}

        // Assign each partition to its location in the storage vector
{ assign_contiguous_partitions(partition, struct_name_base) }
    }}

    inline reference operator[](const size_t index) const {{
        { assign_proxyref(members, partition, struct_name_base) }
    }}

    size_t size() const {{ return n; }}

    ~PartitionedContainerContiguous{partition_string}() {{
        // Deallocate each partition
{ deallocate_contiguous_partitions(partition, struct_name_base) }
    }}
}};
"""
    )


def write_partition(f, struct_name_base, partition_string, partition, members):
    """
    Write a structure definition for a container that stores partitions, not necessarily contiguous,
    each with a subset of the data members in the original structure. The container provides a
    AoS-like interface by returning a proxy reference struct in operator[].

    :param f: File object to write to
    :param struct_name_base: Base name of the struct
    :param partition_string: String representation of the partition
    :param partition: List of partitions, each a list of member indices
    :param members: List of tuples (data_type, member_name) containing all members in the original structure
    """
    f.write(
        f"""
struct PartitionedContainer{partition_string} {{
    using value_type = ::{struct_name_base};
    using reference = ::{struct_name_base}Ref;

{ define_partitions_struct(partition, struct_name_base) }
    size_t n;

    static std::string to_string() {{ return "PartitionedContainer{partition_string}"; }}

    static size_t bytes_for_all(size_t n, size_t alignment) {{
        return { bytes_for_all_expression(partition, struct_name_base) };
    }}

    PartitionedContainer{partition_string}(size_t n, size_t alignment) : n(n) {{
{ assign_partitions(partition, struct_name_base) }
    }}

    inline reference operator[](const size_t index) const {{
        { assign_proxyref(members, partition, struct_name_base) }
    }}

    size_t size() const {{ return n; }}
}};
"""
    )


def write_reference_struct(f, struct_name_base, members):
    """
    Write the proxy reference struct returned by operator[] of the containers. It holds a
    reference to every member and can be assigned from a value of the original struct, so
    containers can be filled without knowing the member names.

    :param f: File object to write to
    :param struct_name_base: Base name of the struct
    :param members: List of tuples (data_type, member_name) containing all members in the original structure
    """
    lines = [f"struct {struct_name_base}Ref {{"]
    for dtype, name in members:
        lines.append(f"    {dtype}& {name};")
    lines.append("")
    lines.append(f"    {struct_name_base}Ref& operator=(const {struct_name_base}& other) {{")
    for _, name in members:
        lines.append(f"        {name} = other.{name};")
    lines.append("        return *this;")
    lines.append("    }")
    lines.append("};")
    f.write("\n".join(lines) + "\n\n")


def write_subsets(f, struct_name_base, members, subsets):
    """
    Write struct definitions for all substructures used in the partitions.
    The substructures contain only a subset of the data members in the original structure.

    :param f: File object to write to
    :param struct_name_base: Base name of the struct
    :param members: List of tuples (data_type, member_name) containing all members in the original structure
    :param subsets: List of subsets, each a list of member indices
    """
    f.write(generate_struct_definition(struct_name_base, members) + "\n\n")
    write_reference_struct(f, struct_name_base, members)

    for subset in subsets:
        subset_string = "".join(str(i) for i in subset)
        f.write(
            generate_struct_definition(
                f"Sub{struct_name_base}{subset_string}",
                [members[i] for i in subset],
            )
            + "\n"
        )


def write_header(path, guard, struct_name_base, members, subsets, containers):
    """
    Write a header with the original struct, its proxy reference struct, the substructures and
    the containers. The containers are placed in namespace layouts::<struct_name_base>, so headers
    for different structs can be included together and the driver can enumerate the containers
    of one struct by reflection.

    :param path: Output file
    :param guard: Include guard macro
    :param struct_name_base: Base name of the struct
    :param members: List of tuples (data_type, member_name) containing all members in the original structure
    :param subsets: List of subsets, each a list of member indices
    :param containers: String with the container definitions
    """
    with open(path, "w") as f:
        f.write(f"#ifndef {guard}\n")
        f.write(f"#define {guard}\n")
        f.write("// THIS FILE IS GENERATED USING layouts/generate_datastructures.py\n")
        f.write('#include "layout_helpers.h"\n\n')

        write_subsets(f, struct_name_base, members, subsets)

        f.write(f"\nnamespace layouts::{struct_name_base} {{\n")
        f.write(containers)
        f.write(f"}} // namespace layouts::{struct_name_base}\n")
        f.write(f"\n#endif // {guard}\n")


def write_index(output_dir):
    """
    Write layouts_index.h, which includes every layouts_<Struct>.h in output_dir. Delete a
    layouts_<Struct>.h to drop that struct.

    :param output_dir: Directory with the generated headers
    """
    structs = sorted(
        p.stem.removeprefix("layouts_")
        for p in output_dir.glob("layouts_*.h")
        if p.name != "layouts_index.h"
    )
    with open(output_dir / "layouts_index.h", "w") as f:
        f.write("#ifndef LAYOUTS_INDEX_H\n")
        f.write("#define LAYOUTS_INDEX_H\n")
        f.write("// THIS FILE IS GENERATED USING layouts/generate_datastructures.py\n")
        for s in structs:
            f.write(f'#include "layouts_{s}.h"\n')
        f.write("\n#endif // LAYOUTS_INDEX_H\n")
    print(f"Wrote {output_dir / 'layouts_index.h'} with structs: {', '.join(structs)}")


def write_partitioned_structs(
    output_dir, struct_name_base, members, start, end, contiguous, only=None, only_every=None
):
    """
    Generate partitioned data structures and write them to <output_dir>/layouts_<struct_name_base>.h.

    :param output_dir: Directory to write the generated header to
    :param struct_name_base: Base name of the struct
    :param members: List of tuples (data_type, member_name) containing all members in the original structure
    :param start: Start index for batching
    :param end: End index for batching
    :param contiguous: Whether to generate contiguous partitioned structures
    :param only: List of codewords to only generate specific partitions
    :param only_every: Only generate every n-th partition (always includes AoS and SoA)
    """
    aos_string = "".join(str(i) for i in range(len(members)))
    soa_string = "_".join(str(i) for i in range(len(members)))

    f = io.StringIO()
    p_list = []
    s_list = []

    for ip, partition in enumerate(generate_partitions(members, contiguous, only)):
        if end and len(p_list) >= end:
            break

        # Encode e.g., [[0,2],[1,3]] as "02_13"
        partition_string = "_".join(
            ["".join(str(m) for m in subset) for subset in partition]
        )

        if only_every and ip % only_every != 0 and \
            partition_string != aos_string and partition_string != soa_string:
            continue

        # Skip duplicates
        if partition_string in p_list:
            continue
        p_list.append(partition_string)

        # Skip the partitions of earlier batches
        if len(p_list) <= start:
            continue

        if contiguous:
            write_contiguous_partition(
                f, struct_name_base, partition_string, partition, members
            )
        else:
            write_partition(f, struct_name_base, partition_string, partition, members)

        for subset in partition:
            if subset not in s_list:
                s_list.append(subset)

    path = output_dir / f"layouts_{struct_name_base}.h"
    write_header(
        path,
        f"LAYOUTS_{struct_name_base.upper()}_H",
        struct_name_base,
        members,
        s_list,
        f.getvalue(),
    )
    print(f"Generated {len(p_list[start:])} partitioned data structures in {path}.")


def write_test_partitions(output_dir):
    """
    Generate test partitioned data structures used in the unit tests in test.cpp and write them
    to <output_dir>/test.h.

    :param output_dir: Directory to write test.h to
    """
    struct_name_base = "S"
    members = [("int", "x"), ("double", "y"), ("float", "z"), ("char", "w")]

    f = io.StringIO()
    p_list = [[[0, 1], [2, 3]], [[0], [1], [2], [3]], [[0], [1], [2, 3]]]
    s_list = []

    for partition in p_list:
        partition_string = "_".join(
            ["".join(str(m) for m in subset) for subset in partition]
        )
        write_contiguous_partition(
            f, struct_name_base, partition_string, partition, members
        )
        write_partition(f, struct_name_base, partition_string, partition, members)

        for subset in partition:
            if subset not in s_list:
                s_list.append(subset)

    write_header(
        output_dir / "test.h", "TEST_H", struct_name_base, members, s_list, f.getvalue()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_spec",
        type=str,
        help="Data member specification for which to generate data structures",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory for the generated headers (default: layouts/generated)",
        default=str(Path(__file__).resolve().parent / "generated"),
    )
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
        action='store_true',
        default=False,
        help="Generate contiguous partitioned structures",
    )
    parser.add_argument(
        "--only",
        type=str,
        nargs="*",
        help="Only generate partitions with these codewords (e.g., '01_23')",
    )
    parser.add_argument(
        "--only_every",
        type=int,
        help="Only generate every n-th partition (always includes AoS and SoA)",
    )
    args = parser.parse_args()

    print(
        f"Configuration:\n  output_dir={args.output_dir}\n  batch_size={args.batch_size}\n  batch_num={args.batch_num}\n  contiguous={args.contiguous}\n  only={args.only}\n  only_every={args.only_every}"
    )

    if args.batch_size == -1:
        start = 0
        end = None
    else:
        start = args.batch_num * args.batch_size
        end = start + args.batch_size

    with open(args.data_spec, "r") as f:
        lines = f.readlines()
        struct_name_base = lines[0].strip()
        data_members = [line.strip().split() for line in lines[1:] if line.strip()]
        print(data_members)

    # Member indices are joined without a separator in codewords and struct
    # names, so they must be single digits
    if len(data_members) > 10:
        parser.error(
            f"{args.data_spec} has {len(data_members)} members; data structures with more than 10 members are not supported yet"
        )
    # layouts_index.h is the index of the generated headers
    if struct_name_base == "index":
        parser.error(f"{args.data_spec}: the struct cannot be named 'index'")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_partitioned_structs(
        output_dir,
        struct_name_base,
        data_members,
        start,
        end,
        args.contiguous,
        args.only,
        args.only_every
    )
    write_index(output_dir)
    write_test_partitions(output_dir)
