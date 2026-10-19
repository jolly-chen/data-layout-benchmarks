# Data Layout Benchmarks
## Purpose
This benchmark suite enables systematic studies on the performance impact of different data layouts of an array of a (complex) data structure.

## Overview
The benchmark suite consists of 3 main parts:
1. A **set of benchmark kernels** that perform computations on a data structure.
2. A **data layout generator** that generates different data layouts for a data structure given by a data specification.
3. A **main** file that calls the benchmarks with the generated data structures.

For an array of a data structure with N data members, the data layout generator generates all possible data layouts by permuting the order of the data members and grouping them into arrays. The total number of generated layouts, $N_n$ is given by the recursive formula: $N_n = 2 \cdot (n-1) \cdot N_{n-1} - (n-1)\cdot(n-2)\cdot N_{n-2}$ with $N_0 = N_1 = 1$.

The main file runs the benchmarks kernels for each generated data layout. By default, the kernels are run with two sizes of the input arrays: (1) smaller than L1 cache size and (2) larger than L3 cache size. The results are reported by [google benchmark](https://github.com/google/benchmark), on the console or, e.g., as JSON with `--benchmark_format=json`. `plot_results.py` contains plotting functions for an earlier CSV format of the results.

## File Descriptions
- (1) `benchmarks.h`: Contains  a set of kernels that perform computations on a particle data structure, representing common operations in high-energy physics.
- (2) `layouts/`: the data layout generator.
  - `generate_datastructures.py`
    - *input \<data spec file\>*: a file that specifies the data structure, with on the first line the structure's name and on each following line the type and name of a field in the data structure. See `layouts/specs/particle.spec` for an example.
    - *output* `layouts/generated/layouts_<Struct>.h`: the data structure, and one container per data layout in namespace `layouts::<Struct>`. Headers for several data structures can exist side by side.
    - *output* `layouts/generated/layouts_index.h`: includes every `layouts_<Struct>.h` in `layouts/generated/`. It is included by `main.cpp`. Delete a `layouts_<Struct>.h` and rerun the generator to drop a data structure.
    - *output* `layouts/generated/test.h`: the containers used by `test.cpp`.
  - `layout_helpers.h`: the helpers used by the generated containers; the `LayoutOf<C, T>` concept that every container must satisfy (any other layout implementation that satisfies it can be benchmarked too); and `ForEachLayout<T>(f)`, which calls `f` for every generated container of `T`. The driver uses `ForEachLayout` to find the layouts to run the benchmarks on.
  - `specs/`: data specifications.
- (3) `main.cpp`: The main file that reads initialization data, and calls the benchmarks for every generated layout of `Particle`, which it finds with `ForEachLayout`.
  - *output*: one google benchmark result per benchmark, layout and problem size, named `<benchmark>_<layout>`, with the counters `problem_size` (array size in elements) and `reference_size` (size of the proxy reference returned by the container).
- `plot_results.py`: A script that contains several methods to plot the results of the benchmarks as a scatter plot or histogram.

Miscellaneous:
- `generate_dataset.py`: generates initialization data for the benchmarks. See `--help` for more options on how to run the dataset generator.
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
Run `layouts/generate_datastructures.py --help` for more options on how to run the data structure generator. Run `./main --help` for more options on how to run the benchmarks.

The input files needed to run `main.cpp` can be generated using `generate_dataset.py` or downloaded using the following links:
- https://cernbox.cern.ch/remote.php/dav/public-files/qqvSQoyp3Y4VFff/3m.zip (3 million particles)
- https://cernbox.cern.ch/remote.php/dav/public-files/qqvSQoyp3Y4VFff/3m_v2.zip (3 million particles, version 2)
- https://cernbox.cern.ch/remote.php/dav/public-files/qqvSQoyp3Y4VFff/10.zip (10 particles)

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
To add a new kernel, add a new function to `benchmarks.h` that performs the desired computation on the data structure. Then, add a call to the new benchmark in `main.cpp` in the function `RunAllBenchmarks`.

### Changing the data structure
To change the data structure:
1. Create a new data specification file in `layouts/specs/`.
2. Run `python3 layouts/generate_datastructures.py --data_spec layouts/specs/<file>`. This writes `layouts/generated/layouts_<Struct>.h` next to the headers of the other data structures, and updates `layouts_index.h`. You can check `layouts_<Struct>.h` to verify that the new data structures have been generated as expected.
3. If needed, modify the benchmark kernels in `benchmarks.h` and the benchmarks in `main.cpp` to work with the new data structure. For example, if the new data structure has different field names or types, you will need to update the kernels accordingly.