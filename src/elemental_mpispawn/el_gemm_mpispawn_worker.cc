#include <mpi.h>
#include <iostream>
#include <El.hpp>

using namespace std;
using std::cout;
using std::endl;

typedef El::DistMatrix<double,El::MC,El::MR,El::ELEMENT> ElMat;

void get_scatterv_mat(ElMat &Mat, double *localData, int rank, MPI_Comm parentcomm, const El::Grid& grid)
{

    // Receive the metadata
    int *mat_data;
    mat_data = (int*)calloc(3,sizeof(int));
    MPI_Bcast(mat_data, 3, MPI_INT, 0, parentcomm);
    
    // get the matrix size
    long int nrows = mat_data[0];
    long int ncols = mat_data[1];
    long int gridSize = mat_data[2];
    //std::cout << nrows << " " << ncols << " " << gridSize << std::endl;

    // receive the sendcount 
    int *sendcount;
    sendcount = (int*)calloc(gridSize,sizeof(int));
    MPI_Bcast(sendcount, gridSize, MPI_INT, 0, parentcomm);

    //receive the displacements
    int *displs;
    displs = (int*)calloc(gridSize,sizeof(int));
    MPI_Bcast(displs, gridSize, MPI_INT, 0, parentcomm);

    //for (int i =0; i < gridSize; i++)
    //    std::cout << sendcount[i] << "  " << displs[i] << std::endl;
    
    // get the max size
    int chunksize = sendcount[rank];

    // get the matrix
    double *A;              // redundant for worker
    //double *localData;
    localData = (double *) calloc(chunksize,sizeof(double));
    MPI_Scatterv(A, sendcount, displs, MPI_DOUBLE, localData, chunksize, MPI_DOUBLE, 0, parentcomm);    

    //for(int i=0; i < chunksize ; i++)
    //    std::cout << "Worker : " << rank << " - A : [" << i << "] : " << localData[i] <<std::endl;

    const El::Int localHeight = El::Length( nrows, grid.Row(), grid.Height() );
    const El::Int localWidth = El::Length( ncols, grid.Col(), grid.Width() );

    const int colAlign = 0;
    const int rowAlign = 0;

    Mat.Attach(nrows,ncols,grid,colAlign,rowAlign,localData,localHeight);
    //El::Print(Adist,"Element Distributed matrix");

    free(mat_data);
    free(sendcount);
    free(displs);
    //free(localData);

}

void send_gatherv_mat(ElMat Mat, MPI_Comm parentcomm)
{
    double *tmp; // redundant for sender
    int *tmpi;   // reundant for sender

    // gather all the localsize
    int localSize = Mat.LocalHeight()*Mat.LocalWidth();
    MPI_Gather(&localSize,1,MPI_INT,tmpi,0,MPI_INT,0,parentcomm);

    // send the matrix
    MPI_Gatherv(Mat.Buffer(), localSize, MPI_DOUBLE, tmp, 0, 0, MPI_DOUBLE, 0, parentcomm);
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
    ElMat A(grid);
    double *lA; 
    get_scatterv_mat(A,lA,workerRank, parentcomm, grid);
    //El::Print(A,"A Distributed matrix");
    
    // get the B matrix
    ElMat B(grid);
    double *lB;
    get_scatterv_mat(B,lB,workerRank, parentcomm, grid);
    //El::Print(B,"B Distributed matrix");

    // // create the C Matrix
    ElMat C(grid);
    El::Zeros(C,A.Height(),B.Width());

    // perform the mult
    const El::Orientation ori = El::NORMAL;
    El::mpi::Barrier();
    El::Gemm(ori,ori,double(1.),A,B,double(0.),C);
    //El::Print(C,"C Distributed matrix");

    // send the result back to the parent
    send_gatherv_mat(C,parentcomm);

	El::mpi::Barrier();
    MPI_Comm_free(&parentcomm);
    //MPI_Finalize();
    El::Finalize();
    return(0);
}