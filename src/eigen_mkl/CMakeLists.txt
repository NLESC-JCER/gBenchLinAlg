

find_package(Threads REQUIRED)

# Add compiler flags
set(MKLROOT "/opt/intel/mkl")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -DEIGEN_USE_BLAS -DMKL_LP64 -m64 -I${MKLROOT}/include")
message(STATUS "CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS})

set(LIBMKL "-L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl")

file(GLOB_RECURSE ALL_BENCH_CPP *.cc)
foreach(ONE_BENCH_CPP ${ALL_BENCH_CPP})
	
	get_filename_component(ONE_BENCH_EXEC ${ONE_BENCH_CPP} NAME_WE)
	set(TARGET_NAME bench_${ONE_BENCH_EXEC})
	add_executable(${TARGET_NAME} ${ONE_BENCH_CPP})
	set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${ONE_BENCH_EXEC})
	target_link_libraries(${TARGET_NAME} Eigen3::Eigen ${CMAKE_THREAD_LIBS_INIT} ${LIBMKL} -lbenchmark)

endforeach()