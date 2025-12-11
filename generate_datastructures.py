from multiprocessing import Pool
from functools import partial
from itertools import permutations
import numpy as np
import math
import os
import sys

data_members = [
    ("int", "id"),
    ("float", "pt"),
    ("float", "eta"),
    ("float", "phi"),
    ("float", "e"),
    ("char", "charge"),
    ("std::array<std::array<double, 3>, 3>", "posCovMatrix"),
]

struct_name_base = "Particle"


def generate_struct_definition(struct_name, members):
    lines = [f"struct {struct_name} {{"]
    for dtype, name in members:
        lines.append(f"    {dtype} {name};")
    lines.append("};")
    return "\n".join(lines)


def B(n):
    """
    Compute the n-th Bell number
    https://en.wikipedia.org/wiki/Bell_number
    """
    bell = 0
    for k in range(n + 1):
        v = 0
        for i in range(k + 1):
            v += ((-1) ** (k - i)) * math.comb(k, i) * (i**n)
        bell += v // math.factorial(k)
    return bell


def D(n, r, d):
    if r == 2 and (d == 1):
        return B(n)

    if r == n and (1 <= d and d <= n - 1):
        return d + 1

    if r == n + 1 and (1 <= d and d <= n):
        return 1

    if (3 <= r and r <= n - 1) and (1 <= d and d <= r - 1):
        return d * D(n, r + 1, d) + D(n, r + 1, d + 1)


def unrank_partition(t, n):
    """Unrank the r-th partition of n elements into k non-empty subsets."""
    # This is a placeholder for the actual unranking algorithm.
    # Implementing this is non-trivial and requires combinatorial logic.
    codeword = np.repeat(1, n)

    if not 0 < t and t <= D(n, 2, 1):
        return codeword, 0

    codeword[0] = 1
    d = 1
    for r in range(2, n + 1):
        m = 0
        while t > m * D(n, r + 1, d):
            m += 1
        if m > d + 1:
            m = d + 1
        codeword[r - 1] = m
        t = t - (m - 1) * D(n, r + 1, d)
        if m > d:
            d = m
    return codeword, r


def convert_codeword_to_partitions(set, codeword):
    """
    Convert a codeword to list partitions of the set.
    """
    partitions = [[] for _ in range(len(codeword))]
    for i, c in enumerate(codeword):
        partitions[c - 1].append(set[i])
    return [p for p in partitions if p]


def generate_partitions(members, g, proc_num):
    """
    Generate all set partitions and permute elements in each subset.

    Based on "Parallel algorithms for generating subsets and set partitions" in
    Lecture Notes in Computer Science, vol 450: https://link.springer.com/chapter/10.1007/3-540-52921-7_57
    """
    # print(proc_num, g, members)

    n = len(members)
    t = proc_num * g + 1
    codeword, r = unrank_partition(t, n)
    # print(codeword)
    b = np.repeat(1, n)
    l = 0
    r = n
    j = 0
    max_codeword = 1

    for s in range(2, n):
        if codeword[s - 1] > max_codeword:
            max_codeword = codeword[s - 1]
        else:
            j += 1
            b[j - 1] = s

        while l != g and r != 1:
            while r < n - 1:
                r += 1
                codeword[r - 1] = 1
                j += 1
                b[j - 1] = r

            while codeword[n - 1] <= n - j and l != g:
                l += 1
                partitions = convert_codeword_to_partitions(members, codeword)
                if any(len(p) > 1 for p in partitions):
                    for i, p in enumerate(partitions):
                        if len(p) > 1:
                            for perm in permutations(p):
                                partitions[i] = list(perm)
                                print(partitions)
                else:
                    print(partitions)

                codeword[n - 1] += 1

            r = b[j - 1]
            codeword[r - 1] += 1
            codeword[n - 1] = 1
            if codeword[r - 1] > r - j:
                j -= 1


def generate_transformations(struct_name, members):
    with open("transformations.h", "w") as f:
        f.write("#ifndef TRANSFORMATIONS_H\n")
        f.write("#define TRANSFORMATIONS_H\n\n")

    pool_size = 1  # FIXME: parallel is not working yet; produces duplicates
    pool = Pool(processes=pool_size)
    g = np.ceil(B(len(members)) / float(pool_size)).astype(int)
    pool.map(partial(generate_partitions, members, g), range(0, pool_size))

    with open("transformations.h", "a") as f:
        f.write("\n#endif // TRANSFORMATIONS_H\n")


if __name__ == "__main__":
    generate_transformations(struct_name_base, range(3))
