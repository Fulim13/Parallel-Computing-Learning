// run in cmd/powershell with mpiexec -n 4 01_Rank_NumReq_GetProcessorName
#include <iostream>
#include <mpi.h>

using namespace std;

int main(int args, char **argvs)
{
    cout << "Hello World" << endl;
    int rank = 0, numOfProcess = 0;

    MPI_Init(&args, &argvs);
    // The default communicator is called MPI_COMM_WORLD. It basically groups all the
    // processes when the program started. If you take a look at the example below,
    // you see a depiction of a program ran with five processes. Every process is
    // connected and can communicate inside this communicator.
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);         // process id
    MPI_Comm_size(MPI_COMM_WORLD, &numOfProcess); // number of thread/process

    // Single by single process
    if (rank == 0)
    {
        char name[MPI_MAX_PROCESSOR_NAME];
        int len;
        cout << "Hi there, from rank " << rank << endl;
        MPI_Get_processor_name(name, &len);
        cout << "PC name is " << name << endl;
    }

    cout << "Hello World from process rank(number) " << rank << " from " << numOfProcess << endl;
    MPI_Finalize();
    return 0;
}
