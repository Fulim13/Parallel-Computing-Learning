#include <iostream>
#include <omp.h>

double compute_pi_in_parallel_dynamically_change_num_threads()
{
    int i;
    double x, pi, sum = 0.0, step;
    double start_time, run_time;
    const int NUM_THREADS = 8;
    const int num_steps = 1000000; // Example number of steps

    step = 1.0 / (double)num_steps;

    start_time = omp_get_wtime();
    // Loop through different numbers of threads
    for (int num_threads = 1; num_threads <= NUM_THREADS; num_threads++)
    {
        sum = 0.0;                        // Reset sum for each number of threads
        omp_set_num_threads(num_threads); // Set the number of threads dynamically

// #pragma omp parallel for reduction(+:sum) private(x) num_threads(NUM_THREADS) // Set the number of threads
#pragma omp parallel for reduction(+ : sum) private(x)
        for (i = 1; i <= num_steps; i++)
        {
            x = (i - 0.5) * step;
            sum += 4.0 / (1.0 + x * x); // Sum all thread's results
        }

        // Calculate pi after the parallel region
        pi = step * sum;
        run_time = omp_get_wtime() - start_time;

        // Use a parallel block to retrieve the number of threads
        int actual_num_threads;
#pragma omp parallel
        {
            actual_num_threads = omp_get_num_threads();
        }

        printf("num_threads = %d\n", actual_num_threads);
        printf("Pi is %.16lf in %lf seconds and %d threads\n ", pi, run_time, actual_num_threads);
    }

    return run_time;
}

int main()
{
    compute_pi_in_parallel_dynamically_change_num_threads();
}
