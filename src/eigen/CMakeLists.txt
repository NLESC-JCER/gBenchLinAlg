

find_package(Threads REQUIRED)

# Add compiler flags

set(CXX_FLAGS "-O2 -Wall -fopenmp -march=native")
message(STATUS "CXX_FLAGS: " ${CXX_FLAGS})

file(GLOB_RECURSE ALL_BENCH_CPP *.cc)
foreach(ONE_BENCH_CPP ${ALL_BENCH_CPP})
	
	get_filename_component(ONE_BENCH_EXEC ${ONE_BENCH_CPP} NAME_WE)
	set(TARGET_NAME bench_${ONE_BENCH_EXEC})
	add_executable(${TARGET_NAME} ${ONE_BENCH_CPP})
	set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${ONE_BENCH_EXEC})
	target_compile_options(${TARGET_NAME}
	  PRIVATE
	  ${CXX_FLAGS})
	target_link_libraries(${TARGET_NAME} Eigen3::Eigen ${CMAKE_THREAD_LIBS_INIT} -lbenchmark)
endforeach()
