#include <benchmark/benchmark.h>
#include <armadillo>
#include <chrono>
#include <omp.h>


using namespace std;
using namespace arma;

void arma_diag(int size){

	mat A = randu<mat>(size,size);
    mat B = A + A.t();
    vec eigval;
    mat eigvec;

    eig_sym(eigval,eigvec,B);
}

static void DIAG(benchmark::State &state) {

	for (auto _ : state)
		arma_diag(state.range(0));
	
}
BENCHMARK(DIAG)->Arg(1024)->Arg(2048)->Arg(4096)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_MAIN();
