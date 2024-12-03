#include <iostream>
#include <iomanip>
#include <omp.h>

const int NUM_THREADS = 16;
static long num_steps = 100000000;

using namespace std;

int compute_pi_using_partial_sum()
{
    int i;
    double step, pi = 0;
    double partial_sums[NUM_THREADS], sum = 0.0;
    step = 1.0 / (double)num_steps;

    omp_set_num_threads(NUM_THREADS);
    printf("Total OMP threads: %d\n", NUM_THREADS);

    double start_time = omp_get_wtime();
#pragma omp parallel
    {
        int i, id;
        double x;
        id = omp_get_thread_num();

        partial_sums[id] = 0.0;
        for (i = id; i < num_steps; i += NUM_THREADS)
        {
            x = (i + 0.5) * step;
            partial_sums[id] += 4.0 / (1.0 + x * x);
        }
    }
    for (i = 0, pi = 0.0; i < NUM_THREADS; i++)
    {
        sum += partial_sums[i];
    }
    pi = sum * step;
    double end_time = omp_get_wtime();

    cout << setprecision(16) << pi << endl;
    cout << "Work took " << (end_time - start_time) << " seconds\n";

    return 0;
}

int compute_pi_using_critical_region()
{
    int i;
    double step, pi = 0;
    double sum = 0.0;
    step = 1.0 / (double)num_steps;

    omp_set_num_threads(NUM_THREADS);
    printf("Total OMP threads: %d\n", NUM_THREADS);

    double start_time = omp_get_wtime();
#pragma omp parallel
    {
        int i, id;
        double x;
        id = omp_get_thread_num();

        // partial_sums[id] = 0.0;
        double partial_sum = 0.0;

        for (i = id; i < num_steps; i += NUM_THREADS)
        {
            x = (i + 0.5) * step;
            partial_sum += 4.0 / (1.0 + x * x);
        }

#pragma omp critical(partial_sum)
        {
            sum += partial_sum;
        }
    }

    // for (i = 0, pi = 0.0; i < NUM_THREADS; i++) {
    //     sum += partial_sums[i];
    // }
    pi = sum * step;
    double end_time = omp_get_wtime();

    cout << setprecision(16) << pi << endl;
    cout << "Work took " << (end_time - start_time) << " seconds\n";

    return 0;
}

int main()
{
    compute_pi_using_partial_sum();
    compute_pi_using_critical_region();
    return 0;
}
