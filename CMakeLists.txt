cmake_minimum_required(VERSION 3.16)

project(data_layout_benchmarks)

set(CMAKE_CXX_STANDARD 26)

add_compile_options(-O3 -Wall -Wextra -march=native -freflection)

# Find LIKWID
find_package(likwid CONFIG REQUIRED COMPONENTS marker)

if (vec)
    message(STATUS "Enabling vectorization flags")
    add_compile_options(-ftree-vectorize -ffast-math -fopt-info-vec-optimized -fopt-info-vec-missed)
else()
    message(STATUS "Disabling vectorization flags")
    add_compile_options(-fno-tree-vectorize -fno-fast-math)
endif()

find_package(benchmark REQUIRED)

# The benchmark driver: main.cpp, built with the layouts generated into
# layouts_dir. Pass KERNELS_HEADER/KERNELS as compile definitions to benchmark
# other kernels than bench::AllKernels.
function(add_benchmark_driver target layouts_dir)
    add_executable(${target} ${PROJECT_SOURCE_DIR}/main.cpp)
    target_include_directories(${target} BEFORE PRIVATE ${layouts_dir})
    target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/layouts
                               ${LIKWID_INCLUDE_DIRS})
    target_link_libraries(${target} PUBLIC benchmark::benchmark likwid::likwid)
endfunction()

add_benchmark_driver(main ${PROJECT_SOURCE_DIR}/layouts/generated)

# Tests of the framework itself, run with ctest. They generate their own
# layouts into the build directory.
option(BUILD_TESTS "Build the tests in tests/" OFF)
if (BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
