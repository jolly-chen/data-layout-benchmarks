# Tests of the framework: the layout generator, the input parser, the kernels on
# every layout, and plugging in another struct and kernel. Run with ctest.
#
# The tests generate their own layouts into the build directory, so they do not
# depend on the layouts in layouts/generated.

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(GENERATOR ${PROJECT_SOURCE_DIR}/layouts/generate_datastructures.py)
set(PARTICLE_SPEC ${PROJECT_SOURCE_DIR}/layouts/specs/particle.spec)
set(CELL_SPEC ${CMAKE_CURRENT_SOURCE_DIR}/plugin/cell.spec)

# Every 1000th Particle layout (plus AoS and SoA), and the test containers in test.h
set(TEST_LAYOUTS ${CMAKE_CURRENT_BINARY_DIR}/layouts)
add_custom_command(
    OUTPUT ${TEST_LAYOUTS}/layouts_Particle.h ${TEST_LAYOUTS}/layouts_index.h ${TEST_LAYOUTS}/test.h
    COMMAND ${Python3_EXECUTABLE} ${GENERATOR} --data_spec ${PARTICLE_SPEC} --only_every 1000 --output_dir ${TEST_LAYOUTS}
    DEPENDS ${GENERATOR} ${PARTICLE_SPEC}
    COMMENT "Generating Particle layouts for the tests")
add_custom_target(test_layouts_headers DEPENDS ${TEST_LAYOUTS}/layouts_index.h ${TEST_LAYOUTS}/test.h)

# Particle AoS and SoA next to every 10th Cell layout, for the plugin tests
set(PLUGIN_LAYOUTS ${CMAKE_CURRENT_BINARY_DIR}/plugin_layouts)
add_custom_command(
    OUTPUT ${PLUGIN_LAYOUTS}/layouts_Particle.h ${PLUGIN_LAYOUTS}/layouts_Cell.h ${PLUGIN_LAYOUTS}/layouts_index.h
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${PLUGIN_LAYOUTS}
    COMMAND ${Python3_EXECUTABLE} ${GENERATOR} --data_spec ${PARTICLE_SPEC} --only 0123456 0_1_2_3_4_5_6 --output_dir ${PLUGIN_LAYOUTS}
    COMMAND ${Python3_EXECUTABLE} ${GENERATOR} --data_spec ${CELL_SPEC} --only_every 10 --output_dir ${PLUGIN_LAYOUTS}
    DEPENDS ${GENERATOR} ${PARTICLE_SPEC} ${CELL_SPEC}
    COMMENT "Generating Particle and Cell layouts for the plugin tests")
add_custom_target(plugin_layouts_headers DEPENDS ${PLUGIN_LAYOUTS}/layouts_index.h)

function(add_framework_test target layouts_dir layouts_target)
    add_executable(${target} ${ARGN})
    add_dependencies(${target} ${layouts_target})
    target_include_directories(${target} BEFORE PRIVATE ${layouts_dir})
    target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/layouts)
    target_compile_definitions(${target} PRIVATE TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
    add_test(NAME ${target} COMMAND ${target})
endfunction()

# Containers store each partition contiguously and aligned
add_framework_test(test_layouts ${TEST_LAYOUTS} test_layouts_headers test_layouts.cpp)

# Layout enumeration, problem sizes, input parser, and kernel results on every layout
add_framework_test(test_framework ${TEST_LAYOUTS} test_layouts_headers test_framework.cpp)

# A Cell struct and a stencil kernel, plugged in without changing the framework
add_framework_test(test_plugin ${PLUGIN_LAYOUTS} plugin_layouts_headers plugin/test_plugin.cpp)
target_include_directories(test_plugin PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/plugin)

# The benchmark driver with the Cell kernel added to the Particle kernels, and
# fixed byte targets so the problem sizes, and the datasets they need, are the
# same on every machine
add_benchmark_driver(main_plugin ${PLUGIN_LAYOUTS})
add_dependencies(main_plugin plugin_layouts_headers)
target_include_directories(main_plugin PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/plugin)
target_compile_definitions(main_plugin PRIVATE KERNELS_HEADER="cell_kernels.h" KERNELS=plugin::Kernels
                           "TARGET_BYTES=16384,131072")
add_test(NAME test_plugin_driver
         COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/plugin/check_plugin.py
                 --main $<TARGET_FILE:main_plugin> --layouts_dir ${PLUGIN_LAYOUTS}
                 --data_dir ${CMAKE_CURRENT_SOURCE_DIR}/data)
