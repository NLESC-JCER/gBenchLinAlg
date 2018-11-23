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
    typedef Eigen::Stride<Eigen::Dynamic,Eigen::Dynamic> Stride;
    typedef Eigen::Map<Mat,0,Stride> Map;
    typedef Eigen::Matrix<double,Eigen::Dynamic,1> Vect;
    typedef Eigen::Map<Vect> Mat2Vect;

    //typedef Eigen::Map<Mat> Vect2Mat;

    std::string  getExecpath()
    {
        char result[PATH_MAX];
        ssize_t count = readlink("/proc/self/exe",result, PATH_MAX);
        std::experimental::filesystem::path p = std::string(result, (count > 0) ? count : 0);

        return (std::string) p.parent_path() + "/";
    }

    struct ScatVect 
    {
        Vect v;
        Eigen::VectorXi sendcounts;
        Eigen::VectorXi displs;
    };


    int getRootFactor( int n ) {
        for( int t = sqrt(n); t > 0; t-- ) {
            if( n % t == 0 ) {
                return t;
            }
        }
        return 1;
    }

    ScatVect cycle_matrix(Mat A, int nProcRows, int nProcCols)
    {
        int shift;
        int nr, nc;

        ScatVect sv;
        sv.v = Vect(A.cols()*A.rows());
        sv.sendcounts = Eigen::VectorXi(nProcRows*nProcCols);
        sv.displs = Eigen::VectorXi(nProcRows*nProcCols);

        Mat _tmp;
        int pos = 0;
        int k = 0;
        for (int ipcol = 0; ipcol < nProcCols ; ipcol++)
        {
            for (int iprow = 0; iprow < nProcRows ; iprow++)
            {
                shift = iprow + ipcol * A.rows();

                nr = A.rows()/nProcRows + ( (A.rows()%nProcRows > iprow ) ? 1 : 0 );
                nc = A.cols()/nProcCols + ( (A.cols()%nProcCols > ipcol ) ? 1 : 0 );

                _tmp = Map(A.data()+shift, nr, nc, Stride(nProcCols*A.rows(),nProcRows));
                _tmp = Mat2Vect(_tmp.data(),_tmp.size(),1);  
                //std::cout << "_tmp : " << std::endl << _tmp << std::endl; 

                sv.v.segment(pos,_tmp.size()) = _tmp;
                sv.sendcounts[k] = _tmp.size();
                sv.displs[k] = pos;
                k++;   
                pos += _tmp.size();
            }
        }
        return sv;
    }

    Mat uncycle_matrix(Vect V, int nrows, int ncols, int nProcRows, int nProcCols)
    {

        int shift, pos=0;
        int nr, nc, vsize;
        Mat A = Mat::Zero(nrows,ncols);
        
        for (int ipcol = 0; ipcol < nProcCols ; ipcol++)
        {
            for (int iprow = 0; iprow < nProcRows ; iprow++)
            {

                shift = iprow + ipcol * nrows;

                nr = nrows/nProcRows + ( (nrows%nProcRows > iprow ) ? 1 : 0 );
                nc = ncols/nProcCols + ( (ncols%nProcCols > ipcol ) ? 1 : 0 );
                vsize = nr*nc;
                //std::cout << "chunk:\n" << Map(A.data() + shift, nr, nc, Stride(nProcCols*A.rows(),nProcRows)) << std::endl << std::endl;  
                //std::cout << "data in :\n" << Eigen::Map<Mat>(V.data()+pos,nr,nc) << std::endl;
                Map(A.data() + shift, nr, nc, Stride(nProcCols*A.rows(),nProcRows)) = Eigen::Map<Mat>(V.data()+pos,nr,nc);
                pos += vsize;

            }
        }
        //std::cout << "uncylced" << std::endl << A << std::endl;
        return A;
    }


    void send_scatterv_mat(Mat A, int nProcRows, int nProcCols, MPI_Comm workercomm)
    {
        // broadcast metadata
        int gridSize = nProcCols*nProcRows;
        int mat_data[3] = { (int) A.rows(), (int) A.cols(), gridSize};
        MPI_Bcast(&mat_data, 3, MPI_INT, MPI_ROOT, workercomm);

        // Scatterv data
        double *dA; // redundant for master
        ScatVect _SA = cycle_matrix(A,nProcRows,nProcCols);

        MPI_Bcast(_SA.sendcounts.data(), gridSize, MPI_INT, MPI_ROOT, workercomm);
        MPI_Bcast(_SA.displs.data(), gridSize, MPI_INT, MPI_ROOT, workercomm);

        MPI_Scatterv(_SA.v.data(), _SA.sendcounts.data(),_SA.displs.data(), MPI_DOUBLE, dA, \
                     0, MPI_DOUBLE, MPI_ROOT, workercomm);
    }

    Mat recv_gatherv_mat(int nrows, int ncols, int nProcRows, int nProcCols, MPI_Comm workercomm)
    {
        double *tmp;
        int *tmpi;
        int gridSize = nProcRows*nProcCols;

        // gather all the local sizes
        int *localSizes;
        localSizes = (int*) calloc(gridSize,sizeof(int));
        MPI_Gather(tmpi,0,MPI_INT, localSizes, 1, MPI_INT,MPI_ROOT,workercomm);

        // get the displs
        int *displs;
        displs = (int*) calloc(gridSize,sizeof(int));
        displs[0] = 0;

        // total size of the matrix
        int totalsize = localSizes[0];

        // compute the displacement vector and the total size
        for(int i=1; i<gridSize; i++)
        {
            displs[i] = displs[i-1]+localSizes[i];
            totalsize += localSizes[i];
        }
        
        // gather the matrix
        double *arr;
        arr = (double *)calloc(totalsize,sizeof(double));
        MPI_Gatherv(tmp,0,MPI_DOUBLE,arr,localSizes,displs,MPI_DOUBLE,MPI_ROOT,workercomm);

        // reorganize an eigen matrix
        Eigen::Map<Vect> _tmp(arr,nrows*ncols);
        Mat C = uncycle_matrix(_tmp,nrows,ncols,nProcRows,nProcCols);

        // free mem
        free(localSizes);
        free(displs);

        return C;
    }  


    Mat mpi_spawn_gemm(Mat A, Mat B, int num_workers)
    {

            MPI_Comm workercomm;

            std::string exec = getExecpath() + "el_gemm_mpispawn_worker";
            const char *cstr = exec.c_str();
            //std::cout << "path :  " << exec << std::endl;

            MPI_Comm_spawn(cstr ,MPI_ARGV_NULL, num_workers, MPI_INFO_NULL,
                            0, MPI_COMM_SELF, &workercomm, MPI_ERRCODES_IGNORE  );

            // evaluate grid dimension
            const int nProcRows = getRootFactor( num_workers );
            const int nProcCols = num_workers / nProcRows;
            const long int gridSize = nProcRows * nProcCols;
            //std::cout << " --> ROOT : " << nProcRows << "x" << nProcCols << std::endl;

            // send the first matrix
            //std::cout << "A : " <<std::endl << A << std::endl;
            send_scatterv_mat(A,nProcRows,nProcCols, workercomm);

            // send the second matrix
            //std::cout << "B : " <<std::endl << B << std::endl;
            send_scatterv_mat(B,nProcRows,nProcCols, workercomm);

            //gather all the matrices
            Mat C = recv_gatherv_mat(A.rows(), B.cols(), nProcRows, nProcCols, workercomm);

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
BENCHMARK(mpi_benchmark)->Args({4,4})->Unit(benchmark::kMillisecond)->UseRealTime();
// BENCHMARK(mpi_benchmark)->Arg(2048)->Unit(benchmark::kMillisecond)->UseRealTime();
// BENCHMARK(mpi_benchmark)->Arg(4096)->Unit(benchmark::kMillisecond)->UseRealTime();
// BENCHMARK(mpi_benchmark)->Arg(8192)->Unit(benchmark::kMillisecond)->UseRealTime();

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
