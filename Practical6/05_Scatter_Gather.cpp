#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <assert.h>

using namespace std;

// Creates an array of random numbers. Each number has a value from 0 - 1
double *create_rand_nums(int num_elements)
{
    double *rand_nums = (double *)malloc(sizeof(double) * num_elements);
    assert(rand_nums != NULL); // Enough Memory or not
    int i;
    for (i = 0; i < num_elements; i++)
    {
        // Because divide the largest value, so you always get from the range from 0.0 to 1.0
        rand_nums[i] = (rand() / (double)RAND_MAX);
    }
    return rand_nums;
}

// Computes the average of an array of numbers
double compute_avg(double *array, int num_elements)
{
    double sum = 0.f;
    int i;
    for (i = 0; i < num_elements; i++)
    {
        sum += array[i];
    }
    return sum / num_elements;
}

// argc - count
// argv - vector
int main(int argc, char **argv)
{

    cout << argc << " inputs\n";
    for (int i = 0; i < argc; i++)
    {
        cout << "[" << i << "] = " << argv[i] << endl;
    }

    if (argc != 2)
    {
        // argv[0] - name of of program EG P6
        fprintf(stderr, "Usage: %s num_elements_per_proc\n", argv[0]);
        exit(1);
    }

    // argv[1] - number EG 6
    cout << "The num_elements_per_proc is (string) " << argv[1] << endl;

    // atoi - ascii to integer
    int num_elements_per_proc = atoi(argv[1]);
    cout << "The num_elements_per_proc is " << num_elements_per_proc << endl;

    // Seed the random number generator to get different results each time
    srand((unsigned int)time(NULL));

    // cout << "Done \n";
    // exit(0);

    int world_rank, world_size;
    MPI_Status status = {0};

    MPI_Init(&argc, &argv);

    // Create "world_rank" and set it to the process ID
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Create "world_size" and set it to number of MPI processes
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Create a random array of elements on the root process. Its total
    // size will be the number of elements per process times the number
    // of processes
    double *rand_nums = NULL;
    if (world_rank == 0)
    {
        rand_nums = create_rand_nums(num_elements_per_proc * world_size);
    }

    // For each process, create a buffer that will hold a subset of the entire
    // array
    double *sub_rand_nums = (double *)malloc(sizeof(double) * num_elements_per_proc);
    assert(sub_rand_nums != NULL); // Enough Memory or not

    // Scatter the random numbers from the root process to all processes in
    // the MPI world
    MPI_Scatter(rand_nums, num_elements_per_proc, MPI_DOUBLE, sub_rand_nums, num_elements_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Compute the average of your subset
    double sub_avg = compute_avg(sub_rand_nums, num_elements_per_proc);

    // Gather all partial averages down to the root process
    double *sub_avgs = NULL;
    if (world_rank == 0)
    {
        sub_avgs = (double *)malloc(sizeof(double) * world_size);
        assert(sub_avgs != NULL);
    }
    MPI_Gather(&sub_avg, 1, MPI_DOUBLE, sub_avgs, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Now that we have all of the partial averages on the root, compute the
    // total average of all numbers. Since we are assuming each process computed
    // an average across an equal amount of elements, this computation will
    // produce the correct answer.
    if (world_rank == 0)
    {
        double avg = compute_avg(sub_avgs, world_size);
        printf("Avg of all elements is %f\n", avg);
        // Compute the average across the original data for *comparison*
        double original_data_avg =
            compute_avg(rand_nums, num_elements_per_proc * world_size);
        printf("Avg computed across original data is %f\n", original_data_avg);
    }

    // Clean up
    if (world_rank == 0)
    {
        free(rand_nums);
        free(sub_avgs);
    }
    free(sub_rand_nums);

    MPI_Finalize();

    return 0;
}
