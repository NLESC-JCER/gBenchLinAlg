#include <benchmark/benchmark.h>
#include <armadillo>
#include <chrono>
#include <omp.h>


using namespace std;
using namespace arma;

void arma_gemv(int size){

	mat A = randu<mat>(size,size);
    vec B = randu<mat>(size);
    vec C = A*B;
}

static void GEMV(benchmark::State &state) {

	for (auto _ : state)
		arma_gemv(state.range(0));
	
}
BENCHMARK(GEMV)->Arg(2048)->Arg(4096)->Arg(8192)->Arg(16384)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_MAIN();
