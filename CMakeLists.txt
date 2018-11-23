#specify the minimum CMake version
cmake_minimum_required(VERSION 3.0.2)

# project Name
project(gBenchLA)

# required modules
include(CheckIncludeFile)
include(CheckIncludeFileCXX)
include(CheckIncludeFiles)
include(ExternalProject)

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

find_package(Boost 1.57.0 REQUIRED COMPONENTS program_options filesystem system timer)
include_directories(${Boost_INCLUDE_DIRS})
set (BOOST_CFLAGS_PKG "-I${Boost_INCLUDE_DIRS}")
set(BOOST_LIBS_PKG "-L${Boost_LIBRARY_DIRS}")
foreach(_blib ${Boost_LIBRARIES})
  string(REGEX REPLACE ".*/lib([^/]*)\\.[^.]*$" "-l\\1" _blib ${_blib})
  set(BOOST_LIBS_PKG "${BOOST_LIBS_PKG} ${_blib}")
endforeach(_blib)

# add the src dir
add_subdirectory(src/arma)
add_subdirectory(src/eigen)
add_subdirectory(src/eigen_mkl)
add_subdirectory(src/elemental)
add_subdirectory(src/elemental_mpispawn)
