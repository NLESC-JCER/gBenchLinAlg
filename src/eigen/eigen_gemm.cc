#include <benchmark/benchmark.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <chrono>
#include <thread>
#include <omp.h>

// define eigen benchmark
typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> Mat;


void eigen_gemm(int size){
    Mat A = Mat::Random(size,size);
    Mat B = Mat::Random(size,size);
    Mat C = Mat::Zero(size,size);
    C = A*B;
}

static void GEMM(benchmark::State &state) {

	for (auto _ : state)
		eigen_gemm(state.range(0));
	
}

BENCHMARK(GEMM)->Arg(1024)->Arg(2048)->Arg(4096)->Arg(8192)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_MAIN();
