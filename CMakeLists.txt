cmake_minimum_required(VERSION 3.16)

project(data_layout_benchmarks)

set(CMAKE_CXX_STANDARD 26)

add_compile_options(-g -Wall -Wextra -freflection-latest -stdlib=libc++)
add_link_options(-stdlib=libc++)

find_package(likwid CONFIG REQUIRED)
link_libraries(likwid::likwid)
include_directories(${LIKWID_INCLUDE_DIRS}) # if not set in likwid-config.cmake

# add_executable(test test.cpp)
# target_precompile_headers(test PUBLIC datastructures.h) # crashes the compiler?

add_executable(main main.cpp)
# target_precompile_headers(main PUBLIC datastructures.h) # crashes the compiler?
