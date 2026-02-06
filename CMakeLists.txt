cmake_minimum_required(VERSION 3.16)

project(data_layout_benchmarks)

set(CMAKE_CXX_STANDARD 26)

add_compile_options(-O3 -Wall -Wextra -freflection-latest -stdlib=libc++)
add_link_options(-stdlib=libc++)

# Find LIKWID
find_package(likwid CONFIG REQUIRED COMPONENTS marker)
include_directories(${LIKWID_INCLUDE_DIRS}) # if not set in likwid-config.cmake
link_libraries(likwid::likwid)

# Find PAPI
find_path(PAPI_INCLUDE_DIR papi.h)
find_library(PAPI_LIBRARIES NAMES papi)

if (PAPI_INCLUDE_DIR)
    message(STATUS "PAPI include found at: ${PAPI_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "PAPI include directory (papi.h) not found.")
endif()

if (PAPI_LIBRARIES)
    message(STATUS "PAPI library found at: ${PAPI_LIBRARIES}")
else()
    message(FATAL_ERROR "PAPI library (libpapi.a or libpapi.so) not found.")
endif()

include_directories(${PAPI_INCLUDE_DIR} include)
link_libraries(${PAPI_LIBRARIES})

if (vec)
    message(STATUS "Enabling vectorization flags")
    add_compile_options(-march=native -ftree-vectorize -ffast-math)
endif()

add_executable(test test.cpp)
add_executable(main main.cpp)
