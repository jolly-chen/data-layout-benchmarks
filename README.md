# Data Layout Benchmarks
## Purpose
This benchmark suite enables systematic studies on the performance impact of different data layouts of an array of a (complex) data structure.

## Overview
The benchmark suite consists of 3 main parts:
1. A **set of benchmark kernels** that perform computations on a data structure.
2. A **data layout generator** that generates different data layouts for a data strucutre given by a data specification.
3. A **main** file that calls the benchmarks with the generated data structures.

For an array of a data structure with N data members, the data layout generator generates all possible data layouts by permuting the order of the data members and grouping them into arrays. The total number of generated layouts, $N_n$ is given by the recursive formula: $N_n = 2 \cdot (n-1) \cdot N_{n-1} - (n-1)\cdot(n-2)\cdot N_{n-2}$ with $N_0 = N_1 = 1$.

The main file runs the benchmarks kernels for each generated data layout. By default, the kernels are run with two sizes of the input arrays: (1) smaller than L1 cache size and (2) larger than L3 cache size. The results are output in csv format results, which can be plotted using `plot_results.py` to visualize the performance impact of different data layouts.

## File Descriptions
- (1) `benchmarks.h`: Contains  a set of kernels that perform computations on a particle data structure, representing common operations in high-energy physics.
- (2) `generate_datastructures.py`:
  - *input \<data spec file\>*: a file that specifies data structure, with on the first line the structure's name and on following each line, the name and type of the fields in the data structure. See `particle.spec` for an example.
  - *output* `datastructures.h`: contains the generated data structures with different data layouts. This file is included by `main.cpp`.
  - *modifies* `main.cpp`: adds calls to the benchmarks in `benchmarks.h` with the generated data structures as input. The generated code is added between the comments `// BEGIN GENERATED CODE` and `// END GENERATED CODE`.
- (3) `main.cpp`: The main file that reads initialization data, calls the benchmarks with the generated data structures as input, and writes the results to stdout or a file.
  - *output*: results of the benchmarks in csv format: benchmark name, layout, array size in elements, data structure size in bytes, time unit, and one or more performance events.
- `plot_results.py`: A script that contains several methods to plot the results of the benchmarks as a scatter plot or histogram.

Miscellaneous:
- `generate_dataset.py`: generates initialization data for the benchmarks using `particle.spec` as input. See `--help` for more options on how to run the dataset generator.
- `test.cpp`: A test file that verifies the correctness of the data structure generator
  - `test.h`: Contains the data structures used in the test file, generated every time  `generate_datastructures.py` is run.
- `run_in_batches.sh`: For data structures with many fields, the number of possible data layouts can be very large. This script compiles and runs the benchmarks in batches that contain a subset of possible data layouts to avoid running out of memory when compiling the generated data structures.
- `docker/Dockerfile`: A Dockerfile that sets up the environment for running the benchmarks in a docker container.

## Requirements
To compile the benchmarks, you need to have the following installed:
 - CMake
 - [Likwid](https://github.com/rrze-hpc/likwid), used to determine CPU cache size to automatically set the problem size of the benchmarks.
 - [PAPI](https://github.com/icl-utk-edu/papi), used to gather performance events during the benchmarks.

The following commands are needed to compile:
```bash
cmake .
make
```
## How to Run
### Run with all generated data layouts
```
python3 generate_datastructures.py --data_spec <filename>
make
likwid-pin -C <core_num> ./main --input1 <filename> --input2 <filename> --output <filename> --papi_events "<event1,event2,...>"
```
Run `generate_datastructures.py --help` for more options on how to run the data structure generator. Run `./main --help` for more options on how to run the benchmarks.

The input files needed to run `main.cpp` can be generated using `generate_dataset.py` or downloaded using the following links:
- https://cernbox.cern.ch/remote.php/dav/public-files/qqvSQoyp3Y4VFff/3m.zip (3 million particles)
- https://cernbox.cern.ch/remote.php/dav/public-files/qqvSQoyp3Y4VFff/3m_v2.zip (3 million particles, version 2)
- https://cernbox.cern.ch/remote.php/dav/public-files/qqvSQoyp3Y4VFff/10.zip (10 particles)

The list of available papi events on your machine can be obtained by running `papi_avail` and `papi_native_avail` in the terminal.

We recommend pinning the benchmarks to a specific core using `likwid-pin` to avoid interference from other processes and to ensure that the benchmarks are run on the same core for all data layouts.

#### Run with specific data layouts
```
python3 generate_datastructures.py --data_spec <filename> --only "layout1 layout2 ..."
make
likwid-pin -C <core_num> ./main --input1 <filename> --input2 <filename> --output <filename> --papi_events "<event1,event2,...>"
```
Layouts are in the format of e.g., `0123456` for an AoS of a data structure with 7 fields, and `0_1_2_3_4_5_6` for an SoA.

## How to Modify
The benchmark suite is intended to be easily modifiable to allow for different benchmarks and data structures. Here are some guidelines on how to modify the benchmark suite:

### Adding a new kernel
To add a new kernel, add a new function to `benchmarks.h` that performs the desired computation on the data structure. Then, add a call to the new benchmark in `main.cpp` in the function `RunAllBenchmarks`.

### Changing the data structure
To change the data structure:
1. Create a new data specification file.
2. Rerun `generate_datastructures.py` with the new data specification file to generate the new data structures and update `main.cpp`. You can check `datastructures.h` to verify that the new data structures have been generated as expected.
3. If needed, modify the benchmark kernels in `benchmarks.h` to work with the new data structure. For example, if the new data structure has different field names or types, you will need to update the kernels accordingly.