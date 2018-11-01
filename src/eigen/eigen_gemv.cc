#include <benchmark/benchmark.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <chrono>
#include <thread>
#include <omp.h>

// define eigen benchmark
typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> Mat;
typedef Eigen::Matrix<double,Eigen::Dynamic,1> Vect;

void eigen_gemv(int size){

    Mat A = Mat::Random(size,size);
    Vect B = Vect::Random(size);
    Vect C = Vect::Zero(size);
    C = A*B;
}

static void GEMV(benchmark::State &state) {

	for (auto _ : state)
		eigen_gemv(state.range(0));
	
}
BENCHMARK(GEMV)->Arg(2048)->Arg(4096)->Arg(8196)->Arg(16384)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_MAIN();
