#include <iostream>
#include <omp.h>

static long num_steps = 100000000;
double step;
double compute_pi()
{
    int i;
    double x, pi, sum = 0.0;
    double start_time, run_time;

    step = 1.0 / (double)num_steps;

    start_time = omp_get_wtime();

    for (i = 1; i <= num_steps; i++)
    {
        x = (i - 0.5) * step;
        sum = sum + 4.0 / (1.0 + x * x);
    }

    pi = step * sum;
    run_time = omp_get_wtime() - start_time;
    printf("Non-Parallel: pi with %ld steps is %.16lf in %lf seconds\n ", num_steps, pi, run_time);

    return run_time;
}

double compute_pi_in_parallel_v1()
{
    int i;
    double x, pi, sum = 0.0;
    double start_time, run_time;
    const int NUM_THREADS = 16;
    double partial_sum[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++)
    {
        partial_sum[i] = 0;
    }

    step = 1.0 / (double)num_steps;

    start_time = omp_get_wtime();

#pragma omp parallel num_threads(NUM_THREADS) private(x, i)
    {
        int id = omp_get_thread_num();
        for (i = id + 1; i <= num_steps; i += NUM_THREADS)
        {
            x = (i - 0.5) * step;
            partial_sum[id] += 4.0 / (1.0 + x * x);
        }
    }

    for (int i = 0; i < NUM_THREADS; i++)
    {
        sum += partial_sum[i];
    }

    pi = step * sum;
    run_time = omp_get_wtime() - start_time;
    printf("Version 1 (Parallel): pi with %ld steps is %.16lf in %lf seconds\n ", num_steps, pi, run_time);

    return run_time;
}

double compute_pi_in_parallel_v2()
{
    int i;
    double x, pi, sum = 0.0;
    double start_time, run_time;
    const int NUM_THREADS = 16;
    const int PAD_SIZES = 64;
    double partial_sum[NUM_THREADS][PAD_SIZES];

    for (int i = 0; i < NUM_THREADS; i++)
    {
        partial_sum[i][0] = 0;
    }

    step = 1.0 / (double)num_steps;

    start_time = omp_get_wtime();

#pragma omp parallel num_threads(NUM_THREADS) private(x, i)
    {
        int id = omp_get_thread_num();
        for (i = id + 1; i <= num_steps; i += NUM_THREADS)
        {
            x = (i - 0.5) * step;
            partial_sum[id][0] += 4.0 / (1.0 + x * x);
        }
    }

    for (int i = 0; i < NUM_THREADS; i++)
    {
        sum += partial_sum[i][0];
    }

    pi = step * sum;
    run_time = omp_get_wtime() - start_time;
    printf("Version 2 (Parallel - With Pad Size to Solve False Sharing): pi with %ld steps is %.16lf in %lf seconds\n ", num_steps, pi, run_time);

    return run_time;
}

double compute_pi_in_parallel_v3()
{
    int i;
    double x, pi, sum = 0.0, step;
    double start_time, run_time;
    const int NUM_THREADS = 16;
    const int num_steps = 1000000; // Example number of steps

    step = 1.0 / (double)num_steps;

    start_time = omp_get_wtime();

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

    printf("Version 3 (Parallel - For Reduction): pi with %d steps is %.16lf in %lf seconds\n ", num_steps, pi, run_time);

    return run_time;
}

int main()
{
    double ori_rt, mod_rt_v1, mod_rt_v2, mod_rt_v3;
    ori_rt = compute_pi();
    mod_rt_v1 = compute_pi_in_parallel_v1();
    mod_rt_v2 = compute_pi_in_parallel_v2();
    mod_rt_v3 = compute_pi_in_parallel_v3();
    std::cout << "The performance gain between non-parallel and parallel is " << ori_rt / mod_rt_v1 << std::endl;
    std::cout << "The performance gain between non-parallel and parallel (With Pad Size to Solve False Sharing) is " << ori_rt / mod_rt_v2 << std::endl;
    std::cout << "The performance gain between non-parallel and parallel(For + Reduction) is " << ori_rt / mod_rt_v3 << std::endl;

    return 0;
}
