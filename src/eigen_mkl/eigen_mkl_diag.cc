#include <benchmark/benchmark.h>
#define EIGEN_USE_MKL_ALL
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <chrono>
#include <thread>
#include <omp.h>

// define eigen benchmark
typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> Mat;

void eigen_diag(int size){

    Mat A = Mat::Random(size,size);
    Eigen::SelfAdjointEigenSolver<Mat> ES(A);
}

static void DIAG(benchmark::State &state) {

	for (auto _ : state)
		eigen_diag(state.range(0));
	
}
BENCHMARK(DIAG)->Arg(1024)->Arg(2048)->Arg(4096)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_MAIN();
