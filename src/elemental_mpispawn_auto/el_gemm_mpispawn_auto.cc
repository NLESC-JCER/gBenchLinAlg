#include <Eigen/Dense>
#include <Eigen/Core>
#include <El.hpp>
#include <benchmark/benchmark.h>
#include <iostream>
#include <string>
#include <mpi.h>
#include <chrono>
#include <thread>
#include <limits.h>
#include <unistd.h>
#include <experimental/filesystem>

using namespace std;
using std::cout;
using std::endl;


namespace {

    typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor> Mat;

    std::string  getExecpath()
    {
        char result[PATH_MAX];
        ssize_t count = readlink("/proc/self/exe",result, PATH_MAX);
        std::experimental::filesystem::path p = std::string(result, (count > 0) ? count : 0);

        return (std::string) p.parent_path() + "/";
    }

    void send_mat(Mat A, MPI_Comm workercomm)
    {
        // send metadata
        int mat_data[2] = { (int) A.rows(), (int) A.cols()};
        MPI_Bcast(&mat_data, 2, MPI_INT, MPI_ROOT, workercomm);

        // send the data
        int size = A.rows()*A.cols();
        MPI_Send(A.data(),size,MPI_DOUBLE, 0, 0, workercomm);

    }

    Mat recv_mat(int nrows, int ncols, MPI_Comm workercomm)
    {
        int size = nrows*ncols;
        double *localData;
        localData = (double *) calloc(size,sizeof(double));
        MPI_Recv(localData, size, MPI_DOUBLE, 0, 0, workercomm, MPI_STATUS_IGNORE);
        Eigen::Map<Mat> A(localData,nrows,ncols);
        return A;
    }


    Mat mpi_spawn_gemm(Mat A, Mat B, int num_workers)
    {
            MPI_Comm workercomm;

            std::string exec = getExecpath() + "el_gemm_mpispawn_auto_worker";
            const char *cstr = exec.c_str();

            MPI_Comm_spawn(cstr,MPI_ARGV_NULL, num_workers, MPI_INFO_NULL,
                            0, MPI_COMM_SELF, &workercomm, MPI_ERRCODES_IGNORE  );

            // send the first matrix
            //std::cout << "A : " <<std::endl << A << std::endl;
            send_mat(A,workercomm);

            // send the second matrix
            //std::cout << "B : " <<std::endl << B << std::endl;
            send_mat(B,workercomm);

            //gather all the matrices
            Mat C = recv_mat(A.rows(), B.cols(), workercomm);

            return C;
    }


    void mpi_benchmark(benchmark::State &state)
    {
        double elapsed_second ;
        El::Int size;
        int num_worker;
    
        for (auto _ : state)
        {

            auto start = std::chrono::high_resolution_clock::now();

            size = state.range(0);  
            num_worker = state.range(1);

            Mat A = Mat::Random(size,size);
            Mat B = Mat::Random(size,size);
            Mat C = mpi_spawn_gemm(A,B,num_worker);
            
            auto end = std::chrono::high_resolution_clock::now();

            auto const duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
            auto elapsed_second = duration.count();
            state.SetIterationTime(elapsed_second);
        }
    }

}
BENCHMARK(mpi_benchmark)->Args({512,4})->Unit(benchmark::kMillisecond)->UseRealTime();
// BENCHMARK(mpi_benchmark)->Args({1024,4})->Unit(benchmark::kMillisecond)->UseRealTime();
// BENCHMARK(mpi_benchmark)->Args({2048,4})->Unit(benchmark::kMillisecond)->UseRealTime();
// BENCHMARK(mpi_benchmark)->Args({4096,4})->Unit(benchmark::kMillisecond)->UseRealTime();
// BENCHMARK(mpi_benchmark)->Args({8192,4})->Unit(benchmark::kMillisecond)->UseRealTime();

// Blank reported for the slaves
class NullReporter : public ::benchmark::BenchmarkReporter {
public:
  NullReporter() {}
  virtual bool ReportContext(const Context &) {return true;}
  virtual void ReportRuns(const std::vector<Run> &) {}
  virtual void Finalize() {}
};


int main (int argc, char *argv[])
{

    El::Initialize(argc, argv);

    El::Environment env (argc, argv);
    El::mpi::Comm comm = El::mpi::COMM_WORLD;
    const El::Int commRank = El::mpi::Rank(comm);
    const El::Int commSize = El::mpi::Size(comm);

    ::benchmark::Initialize(&argc, argv);

    if (commRank == 0)
        ::benchmark::RunSpecifiedBenchmarks();
    else
    {
        NullReporter null;
        ::benchmark::RunSpecifiedBenchmarks(&null);
    }
            
    El::Finalize();
    return 0;
            
}
