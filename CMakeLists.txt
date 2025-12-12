cmake_minimum_required(VERSION 3.16)

project(data_layout_benchmarks)

set(CMAKE_CXX_STANDARD 26)

add_compile_options(-Wall -Wextra -Werror -freflection-latest -stdlib=libc++)
add_link_options(-stdlib=libc++)

add_executable(test test.cpp)
target_precompile_headers(test PUBLIC datastructures.h)

add_executable(main main.cpp)
target_precompile_headers(main REUSE_FROM test)