cmake_minimum_required(VERSION 3.16)

project(data_layout_benchmarks)

set(CMAKE_CXX_STANDARD 26)

add_compile_options(-O3 -Wall -Wextra -march=native -freflection)

# Find LIKWID
find_package(likwid CONFIG REQUIRED COMPONENTS marker)
include_directories(${LIKWID_INCLUDE_DIRS}) # if not set in likwid-config.cmake
link_libraries(likwid::likwid)

# Helpers and the containers written by layouts/generate_datastructures.py
include_directories(include layouts layouts/generated)

if (vec)
    message(STATUS "Enabling vectorization flags")
    add_compile_options(-ftree-vectorize -ffast-math -fopt-info-vec-optimized -fopt-info-vec-missed)
else()
    message(STATUS "Disabling vectorization flags")
    add_compile_options(-fno-tree-vectorize -fno-fast-math)
endif()

add_executable(test test.cpp)

find_package(benchmark REQUIRED)
add_executable(main main.cpp)
target_link_libraries(main PUBLIC benchmark::benchmark)
