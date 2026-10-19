# Data Layout Benchmarks
## Purpose
This benchmark suite enables systematic studies on the performance impact of different data layouts of an array of a (complex) data structure.

## Overview
The benchmark suite consists of 4 main parts, which only interact through small interfaces, so each can be replaced independently:
1. A **set of benchmark kernels** that perform computations on a data structure (`benchmarks.h`).
2. A **data layout generator** that generates different data layouts for a data structure given by a data specification (`layouts/`).
3. An **input parser** that reads datasets into any data structure, matching CSV columns to data members by name (`datasets.h`).
4. A **benchmark driver** that runs every kernel on every generated layout of the data structure the kernel works on (`main.cpp`).

For an array of a data structure with N data members, the data layout generator generates all possible data layouts by permuting the order of the data members and grouping them into arrays. The total number of generated layouts, $N_n$ is given by the recursive formula: $N_n = 2 \cdot (n-1) \cdot N_{n-1} - (n-1)\cdot(n-2)\cdot N_{n-2}$ with $N_0 = N_1 = 1$.

The driver finds the generated layouts by reflection, so it needs no changes when layouts or data structures are added. By default, each kernel runs with two problem sizes per layout: one where its input and output containers fit in the L1 cache, and one where they take up twice the last-level cache. The sizes are chosen in bytes, so all layouts of a kernel use the same amount of memory; the number of elements can differ between layouts because of the alignment padding of each partition. Results are written by [google benchmark](https://github.com/google/benchmark), e.g., as JSON with `--benchmark_format=json`.

## File Descriptions
- (1) `benchmarks.h`: Contains a set of kernels that perform computations on a particle data structure, representing common operations in high-energy physics, and one *benchmark descriptor* per kernel (namespace `bench`) that tells the driver how to set up, run and validate it. `bench::AllKernels` lists the benchmarks the driver runs.
- (2) `layouts/`: the data layout generator.
  - `generate_datastructures.py`
    - *input \<data spec file\>*: a file that specifies the data structure, with on the first line the structure's name and on each following line the type and name of a field in the data structure. See `layouts/specs/particle.spec` for an example.
    - *output* `layouts/generated/layouts_<Struct>.h`: the data structure, and one container per data layout in namespace `layouts::<Struct>`. Headers for several data structures can exist side by side.
    - *output* `layouts/generated/layouts_index.h`: includes every `layouts_<Struct>.h` in `layouts/generated/`. It is included by `main.cpp`. Delete a `layouts_<Struct>.h` and rerun the generator to drop a data structure.
    - *output* `layouts/generated/test.h`: the containers used by `test.cpp`.
  - `layout_helpers.h`: the helpers used by the generated containers; the `LayoutOf<C, T>` concept that every container must satisfy (any other layout implementation that satisfies it can be benchmarked too); and `ForEachLayout<T>(f)`, which calls `f` for every generated container of `T`. The driver uses `ForEachLayout` to find the layouts a kernel runs on.
  - `specs/`: data specifications.
- (3) `datasets.h`: reads CSV datasets into `std::vector<T>` for any data structure `T`, using reflection to map the columns in the header row to data members.
- (4) `main.cpp`: The driver that reads the input data, registers every benchmark in `bench::AllKernels` for every generated layout of its data structure, runs them, and validates the results.
  - *output*: one result per benchmark, layout and problem size. Its name is `<benchmark>_<layout>`, e.g., `InvariantMassSequential_PartitionedContainer0123456`, and its counters are `problem_size` (elements), `bytes_for_all` (bytes of all input and output containers), `target_bytes` (the cache-based size it was chosen for), and with `--validation`, `validated` (1 = matches the validation data, -1 = does not, 0 = no validation data for this size).
- `plot_results.py`: A script that contains several methods to plot the results of the benchmarks as a scatter plot or histogram. It reads an earlier CSV format of the results.

Miscellaneous:
- `generate_dataset.py`: generates input datasets (with a header row) and validation data for the benchmarks. See `--help` for more options on how to run the dataset generator. Generating validation data needs [ROOT](https://root.cern) with Python bindings; the reference implementations are in `validation/`, one module per kernel.
- `test.cpp`: A test file that verifies the correctness of the data structure generator, using the containers in `layouts/generated/test.h`.
- `run_in_batches.sh`: For data structures with many fields, the number of possible data layouts can be very large. This script compiles and runs the benchmarks in batches that contain a subset of possible data layouts to avoid running out of memory when compiling the generated data structures.
- `docker/Dockerfile`: A Dockerfile that sets up the environment for running the benchmarks in a docker container.

## Requirements
To compile the benchmarks, you need to have the following installed:
 - CMake
 - A C++ compiler with support for C++26 reflection ([P2996](https://wg21.link/p2996)), e.g., GCC 16.
 - [google benchmark](https://github.com/google/benchmark) (1.9.3 or newer), used to run the benchmarks and report results.
 - [Likwid](https://github.com/rrze-hpc/likwid), used to determine CPU cache size to automatically set the problem size of the benchmarks.

The containers must be generated before compiling (see below). The following commands are needed to compile:
```bash
cmake .
make
```
## How to Run
### Run with all generated data layouts
```
python3 layouts/generate_datastructures.py --data_spec layouts/specs/particle.spec
make
likwid-pin -C <core_num> ./main --input <input list file> [--validation <validation list file>] [google benchmark options]
```
Run `layouts/generate_datastructures.py --help` for more options on how to run the data structure generator. Run `./main --help` for more options on how to run the benchmarks. All other options are passed to google benchmark, e.g., `--benchmark_filter`, `--benchmark_repetitions` and `--benchmark_perf_counters` to gather hardware performance counters (this needs google benchmark built with `-DBENCHMARK_ENABLE_LIBPFM=ON` and [libpfm4](https://perfmon2.sourceforge.net/)); see its [user guide](https://github.com/google/benchmark/blob/main/docs/user_guide.md).

#### Input data
`--input` takes a *list file* with the path of one dataset per line; relative paths are relative to the directory of the list file. A benchmark that reads two arrays (e.g., `InvariantMassSequential`) reads the first dataset into its first container and the second into its second. Each dataset is a CSV file:
- The first row is a header with the names of data members of the data structure, e.g., `pt,eta,phi,e` for `Particle`. Columns can be in any order; data members without a column are zero-initialized. Only data members with arithmetic types can be read.
- Each following row holds the values of one element. A dataset needs at least as many rows as the largest problem size; `main` reports how many rows it needs if a dataset is too short.

For benchmarks on different data structures, give a list file per data structure with `--input <Struct>=<list file>`, e.g., `--input Particle=particle_files --input Cell=cell_files`. A list file without `<Struct>=` is used for every data structure that has no list file of its own. `main` exits with an error if no benchmark reads a `<Struct>` given this way, e.g., because of a typo.

The datasets can be generated using `generate_dataset.py dataset` or downloaded using the following links:
- https://cernbox.cern.ch/remote.php/dav/public-files/qqvSQoyp3Y4VFff/3m.zip (3 million particles)
- https://cernbox.cern.ch/remote.php/dav/public-files/qqvSQoyp3Y4VFff/3m_v2.zip (3 million particles, version 2)
- https://cernbox.cern.ch/remote.php/dav/public-files/qqvSQoyp3Y4VFff/10.zip (10 particles)

#### Validation data
`--validation` takes a list file with lines `<benchmark>,<input size>,<validation file>`, where relative paths are relative to the directory of the list file; `main` reads all validation files before running any benchmark and reports missing or malformed ones. The validation files can be generated with `generate_dataset.py validation`. Since the problem size can differ between layouts, validation data for a larger input size is also used for smaller ones for benchmarks whose result `i` only depends on element `i` (`InvariantMassSequential`, and `InvariantMassRandom`, which is validated against the `InvariantMassSequential` data). `DeltaR2Pairwise` (not registered by default) needs validation data for the exact input size.

We recommend pinning the benchmarks to a specific core using `likwid-pin` to avoid interference from other processes and to ensure that the benchmarks are run on the same core for all data layouts.

#### Run with specific data layouts
```
python3 layouts/generate_datastructures.py --data_spec layouts/specs/particle.spec --only layout1 layout2 ...
make
likwid-pin -C <core_num> ./main --input <input list file> [--validation <validation list file>] [google benchmark options]
```
Layouts are in the format of e.g., `0123456` for an AoS of a data structure with 7 fields, and `0_1_2_3_4_5_6` for an SoA.

## How to Modify
The benchmark suite is intended to be easily modifiable to allow for different benchmarks and data structures. Here are some guidelines on how to modify the benchmark suite:

### Adding a new kernel
1. Add the kernel to namespace `kernels` in `benchmarks.h` (or a header of its own, included by `benchmarks.h`). Write it as a template over the container type and access elements as `v[i].<member>`, so it compiles for every layout.
2. Add a benchmark descriptor to namespace `bench`: a struct that satisfies the concepts `bench::BenchmarkDescriptor` and `bench::BenchmarkFor` in `benchmarks.h`, which document each member it needs. See `bench::InvariantMassSequential` for an example. `value_type` selects the data structure whose layouts the kernel runs on, and `bytes_for_all` the memory it uses, from which the problem sizes are chosen.
3. Add the descriptor to `bench::AllKernels`.

`main.cpp` does not need to change.

### Adding a data structure
1. Create a new data specification file in `layouts/specs/`.
2. Run `python3 layouts/generate_datastructures.py --data_spec layouts/specs/<file>`. This writes `layouts/generated/layouts_<Struct>.h` next to the headers of the other data structures, and updates `layouts_index.h`.
3. Add kernels with `value_type = <Struct>` (see above), and give their datasets with `--input <Struct>=<list file>`.
