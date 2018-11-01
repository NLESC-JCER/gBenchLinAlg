#include <benchmark/benchmark.h>
#include <armadillo>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace arma;

void arma_gemm(int size){

    mat A = randu<mat>(size,size);
    mat B = randu<mat>(size,size);
    mat C = A*B;
}

static void GEMM(benchmark::State &state) {

        for (auto _ : state)
                arma_gemm(state.range(0));

}

BENCHMARK(GEMM)->Arg(1024)->Arg(2048)->Arg(4096)->Arg(8192)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_MAIN();

