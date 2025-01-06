#include <stdlib.h>
#include <iostream>
#include <mpi.h>

using namespace std;

int main(int argc, char **argv)
{
    int rank, value, size;
    MPI_Status status = {0};

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    do
    {
        if (rank == 0)
        {
            printf("Please give a number: ");
            fflush(stdout);
            scanf("%d", &value);

            // Process 0 send the message
            MPI_Send(&value, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
        }
        else
        {
            // Receive from the previous rank/process ID
            MPI_Recv(&value, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &status);

            if (rank < size - 1)
                // All ranks, except the last, send the value to
                // their immediate neighbours
                MPI_Send(&value, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
        }
        printf("Process %d received %d (error code: %d)\n", rank, value,
               status.MPI_ERROR);
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);
    } while (value >= 0);

    MPI_Finalize();
    return 0;
}
