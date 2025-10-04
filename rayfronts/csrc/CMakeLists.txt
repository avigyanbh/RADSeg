# Follow https://nanobind.readthedocs.io/en/latest/building.html

cmake_minimum_required(VERSION 3.15...3.27)
project(rayfronts_cpp)

if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.10 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(rayfronts_cpp NOMINSIZE mapping/vdb_mapping.cpp)

# find_package(OpenVDB REQUIRED) FIXME
set(OPENVDB_INCLUDE_DIR "$ENV{CMAKE_INSTALL_PREFIX}/include/")
set(OPENVDB_LIBRARY_DIRS "$ENV{CMAKE_INSTALL_PREFIX}/lib")
set(OPENVDB_LIBRARIES "tbb" "openvdb")
target_include_directories(rayfronts_cpp PUBLIC ${OPENVDB_INCLUDE_DIR})
target_link_directories(rayfronts_cpp PUBLIC ${OPENVDB_LIBRARY_DIRS})
target_link_libraries(rayfronts_cpp PUBLIC ${OPENVDB_LIBRARIES})
