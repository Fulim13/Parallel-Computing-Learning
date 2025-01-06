#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    int rank, value;

    // Init MPI
    MPI_Init(&argc, &argv);

    // Get process ID (rank)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    do
    {

        if (rank == 0)
        {
            printf("Please give a number (negative number to terminate): ");
            fflush(stdout); // Force immediate printing
            scanf("%d", &value);
        }
        // Broadcast the input value to all the processes
        //  Root process usually 0 will send to all receiver, and receiver will receive the value variable value
        //  int MPI_Bcast( void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm )
        MPI_Bcast(&value, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Eveybody (receiver) will print the result
        printf("Process %d received %d\n", rank, value);
        fflush(stdout); // Force immediate printing

        // Synchronize the processes (wait for all the
        // processes to print their values before proceeding
        // to ask for the next value.)
        MPI_Barrier(MPI_COMM_WORLD);
        // Without barrier, process will run at different speed,
        // if the thread haven't print out the value before the thread 0, loop again, the prompt will generate firt, then other thread value then print out
    } while (value >= 0);

    printf("Process %d terminated.\n", rank);
    MPI_Finalize();

    return 0;
}
