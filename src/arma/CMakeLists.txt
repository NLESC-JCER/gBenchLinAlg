

find_package(Threads REQUIRED)

# Add compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O2 -Wall -fopenmp")
message(STATUS "CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS})

file(GLOB_RECURSE ALL_BENCH_CPP *.cc)
foreach(ONE_BENCH_CPP ${ALL_BENCH_CPP})
	
	get_filename_component(ONE_BENCH_EXEC ${ONE_BENCH_CPP} NAME_WE)
	set(TARGET_NAME bench_${ONE_BENCH_EXEC})
	add_executable(${TARGET_NAME} ${ONE_BENCH_CPP})
	set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${ONE_BENCH_EXEC})
	target_link_libraries(${TARGET_NAME} benchmark ${ARMADILLO_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} -lbenchmark)

endforeach()