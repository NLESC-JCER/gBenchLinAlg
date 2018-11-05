#specify the minimum CMake version
cmake_minimum_required(VERSION 3.0.2)

# project Name
project(gBenchLA)

# CXX standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the output folder where your program will be created
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

# search for armadillo
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})
MESSAGE (STATUS "ARMADILLO PATH:" ${ARMADILLO_LIBRARIES})

# search for Eigen
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
include_directories(${EIGEN3_INCLUDE_DIR})
message(STATUS "EIGEN Include: " ${EIGEN3_INCLUDE_DIR})

# search for MPI
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
message(STATUS "MPI Include: " ${MPI_INCLUDE_PATH})

# search for Elemental
find_package(Elemental REQUIRED)
include_directories(${Elemental_INCLUDE_DIRS})
message(STATUS "Elemental Include: " ${Elemental_LIBRARIES})

# add the src dir
add_subdirectory(src/arma)
add_subdirectory(src/eigen)
add_subdirectory(src/elemental)
