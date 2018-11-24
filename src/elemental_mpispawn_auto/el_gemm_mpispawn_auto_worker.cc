#include <mpi.h>
#include <iostream>
#include <El.hpp>

using namespace std;
using std::cout;
using std::endl;


typedef El::DistMatrix<double,El::MC,El::MR,El::ELEMENT> ElMat;
typedef El::DistMatrix<double,El::CIRC,El::CIRC,El::ELEMENT> ElMatCIRC;
El::Matrix<double> ElLocalMat;

ElMat recv_mat_queue(int rank, MPI_Comm parentcomm, const El::Grid& grid)
{

    // Receive the metadata
    int *mat_data;
    mat_data = (int*)calloc(2,sizeof(int));
    MPI_Bcast(mat_data, 2, MPI_INT, 0, parentcomm);

    // get the matrix size
    int nrows = mat_data[0];
    int ncols = mat_data[1];
    int size = nrows*ncols;

    // create the distributed matrix
    ElMat M(grid);
    El::Zeros(M,nrows,ncols);

    // ROOT Child receive the data
    // then QueueUp all the data of A
    if (rank == 0)
    {
        // receive the buffer
        double *localData;
        localData = (double *) calloc(size,sizeof(double));
        MPI_Recv(localData, size, MPI_DOUBLE, 0, 0, parentcomm, MPI_STATUS_IGNORE);

        // reserve and queue the data
        M.Reserve(size);
        int k = 0;
        for(int i=0; i<ncols; i++)
        {
            for (int j=0; j<nrows; j++)
            {
                M.QueueUpdate(j,i,localData[k]);
                k++;
            }
        }
    }

    // all process Process the Queue
    El::mpi::Barrier();
    M.ProcessQueues();
    return  M;
}

ElMat recv_mat_circ(int rank, MPI_Comm parentcomm, const El::Grid& grid)
{

    // Receive the metadata
    int *mat_data;
    mat_data = (int*)calloc(2,sizeof(int));
    MPI_Bcast(mat_data, 2, MPI_INT, 0, parentcomm);

    // get the matrix size
    int nrows = mat_data[0];
    int ncols = mat_data[1];
    int size = nrows*ncols;

    // create the distributed matrix

    ElMatCIRC M(grid);
    El::Zeros(M,nrows,ncols);

    // ROOT Child receive the data
    // then QueueUp all the data of A
    if (rank == 0)
    {
        // receive the buffer
        double *localData;
        localData = (double *) calloc(size,sizeof(double));
        MPI_Recv(localData, size, MPI_DOUBLE, 0, 0, parentcomm, MPI_STATUS_IGNORE);

        // reserve and queue the data
        M.Attach(nrows,ncols,grid,0,0,localData,nrows);
    }

    // all process Process the Queue
    El::mpi::Barrier();
    ElMat out(M);
    return  out;
}

void send_mat_queue(ElMat A, int rank, MPI_Comm parentcomm)
{
    double *localData;
    int size = A.Height()*A.Width();
    if (rank == 0)
    {
        
        localData = (double *) calloc(size,sizeof(double));
        A.ReservePulls(size);
        int k = 0;
        for(int i=0; i<A.Height(); i++)
        {
            for (int j=0; j< A.Width(); j++)
            {
                A.QueuePull(j,i);
                k++;
            }
        }
    }

    El::mpi::Barrier();
    A.ProcessPullQueue(localData);
    if (rank == 0)
    {
        MPI_Send(localData,size,MPI_DOUBLE, 0, 0, parentcomm);
        free(localData);
    }
    
}

void send_mat_circ(ElMat A, int rank, MPI_Comm parentcomm)
{

    int size = A.Height()*A.Width();
    ElMatCIRC out(A);

    // Process all queues
    El::mpi::Barrier();
    if (rank == 0)
    {
        MPI_Send(out.Buffer(),size,MPI_DOUBLE, 0, 0, parentcomm);
    }
}

int main(int argc, char *argv[])
{

    //MPI_Init(&argc, &argv);
    El::Initialize(argc,argv);

    MPI_Comm localcomm = MPI_COMM_WORLD;
    int workerRank;
    int workerSize;
    MPI_Comm_rank(localcomm,&workerRank);
    MPI_Comm_size(localcomm,&workerSize);

    MPI_Comm parentcomm;
    MPI_Comm_get_parent(&parentcomm);
    
    // make the elemental grid
    El::mpi::Comm comm = El::mpi::Comm(localcomm);
    const El::Grid grid(comm);

    // get the A matrix
    ElMat A = recv_mat_circ(workerRank, parentcomm, grid);
    //El::Print(A,"A Distributed matrix");
    
    // get the B matrix
    ElMat B = recv_mat_circ(workerRank, parentcomm, grid);
    //El::Print(B,"B Distributed matrix");

    // // create the C Matrix
    ElMat C(grid);
    //El::Zeros(C,A.Height(),B.Width());

    // perform the mult
    const El::Orientation ori = El::NORMAL;
    El::mpi::Barrier();
    El::Gemm(ori,ori,double(1.),A,B,double(0.),C);
    //El::Print(C,"C Distributed matrix");

    // send the result back to the parent
    send_mat_circ(C,workerRank,parentcomm);

    El::mpi::Barrier();
    MPI_Comm_free(&parentcomm);
    
    El::Finalize();
    return(0);
}