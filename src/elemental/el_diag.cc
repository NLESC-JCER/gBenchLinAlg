
#include<El.hpp>
#include<benchmark/benchmark.h>
#include<chrono>
#include<thread>


namespace {

	void elmpi_diag(El::Int size, const El::Grid& grid)
	{
		const El::Orientation ori = El::NORMAL;
		El::DistMatrix<double> A(grid), B(grid), C(grid);
		El::Uniform(A,size,size); 
		El::Zeros(B,size,size);
		El::Zeros(C,size,size);

		El::Transpose(A,B);
		El::Axpy(double(1.),A,B);

		El::DistMatrix<double,El::VR,El::STAR> w(grid);
		El::DistMatrix<double> Q(grid);

		El::HermitianEig(El::LOWER,B,w,Q);
	}

	void mpi_benchmark(benchmark::State &state) {

		double elapsed_second ;
		El::mpi::Comm comm = El::mpi::COMM_WORLD;
		
		const El::Grid grid(comm);
		const El::Int blocksize = 96;
		El::Int size;

		for (auto _ : state)
		{
			auto start = std::chrono::high_resolution_clock::now();
			size = state.range(0);
			elmpi_diag(size, grid);
			El::mpi::Barrier(comm);
			auto end = std::chrono::high_resolution_clock::now();

			auto const duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
			auto elapsed_second = duration.count();
			state.SetIterationTime(elapsed_second);
		}
	}
}
BENCHMARK(mpi_benchmark)->Arg(1024)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(mpi_benchmark)->Arg(2048)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(mpi_benchmark)->Arg(4096)->Unit(benchmark::kMillisecond)->UseRealTime();


// This reporter does nothing.
// We can use it to disable output from all but the root process
class NullReporter : public ::benchmark::BenchmarkReporter {
public:
  NullReporter() {}
  virtual bool ReportContext(const Context &) {return true;}
  virtual void ReportRuns(const std::vector<Run> &) {}
  virtual void Finalize() {}
};

// The main is rewritten to allow for MPI initializing and for selecting a
// reporter according to the process rank
int main(int argc, char **argv) {

  	El::Initialize(argc,argv);
	El::Environment env ( argc, argv);
	El::mpi::Comm comm = El::mpi::COMM_WORLD;
	const El::Int commRank = El::mpi::Rank(comm);
	const El::Int commSize = El::mpi::Size(comm);

	::benchmark::Initialize(&argc, argv);

	if(commRank == 0)
		// root process will use a reporter from the usual set provided by
		// ::benchmark
		::benchmark::RunSpecifiedBenchmarks();
	else {
		// reporting from other processes is disabled by passing a custom reporter
		NullReporter null;
		::benchmark::RunSpecifiedBenchmarks(&null);
	}

	El::Finalize();

	return 0;
}

