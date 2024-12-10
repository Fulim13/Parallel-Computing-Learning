#include <iostream>
#include <cstdlib>
#include <omp.h>

using namespace std;

const long MAX = 100000000;
const int MAX_THREAD = 16;

double calculate_average_serial()
{
    double ave = 0.0, *A;
    int i;

    A = (double *)malloc(MAX * sizeof(double));
    if (A == NULL)
    {
        printf("Insufficient memory! Can't continue. Terminating the program abruptly.\n");
        return -1;
    }

    for (i = 0; i < MAX; i++)
    {
        A[i] = (double)i;
    }
    double start_time = omp_get_wtime();

    for (i = 0; i < MAX; i++)
    {
        ave += A[i];
    }

    ave = ave / MAX;

    double end_time = omp_get_wtime();

    printf("%f\n", ave);
    printf("Original work took %f seconds\n", end_time - start_time);
    free(A);
    return end_time - start_time;
}

double calculate_average_parallel_with_omp_critical_v1()
{
    double ave = 0.0, *A;
    int i;

    A = (double *)malloc(MAX * sizeof(double));
    if (A == NULL)
    {
        printf("Insufficient memory! Can't continue. Terminating the program abruptly.\n");
        return -1;
    }

    for (i = 0; i < MAX; i++)
    {
        A[i] = (double)i;
    }
    double start_time = omp_get_wtime();

#pragma omp parallel num_threads(MAX_THREAD)
    {
        int i;
        int id = omp_get_thread_num();

#pragma omp critical // other 15 thread have to wait outside
        for (i = id; i < MAX; i += MAX_THREAD)
        {
            ave += A[i];
        }
    }

    ave = ave / MAX;
    double end_time = omp_get_wtime();

    printf("%f\n", ave);
    printf("Modified1 work took %f seconds\n", end_time - start_time);
    free(A);
    return end_time - start_time;
}

double calculate_average_parallel_with_omp_critical_v2()
{
    double ave = 0.0, *A;
    int i;

    A = (double *)malloc(MAX * sizeof(double));
    if (A == NULL)
    {
        printf("Insufficient memory! Can't continue. Terminating the program abruptly.\n");
        return -1;
    }

    for (i = 0; i < MAX; i++)
    {
        A[i] = (double)i;
    }
    double start_time = omp_get_wtime();

#pragma omp parallel num_threads(MAX_THREAD)
    {
        int i;
        int id = omp_get_thread_num();

        for (i = id; i < MAX; i += MAX_THREAD)
        {
#pragma omp atomic // or critical // ave is shared so need to protected by critical region (protect by MAX times)
            ave += A[i];
        }
    }

    ave = ave / MAX;
    double end_time = omp_get_wtime();

    printf("%f\n", ave);
    printf("Modified 2 work took %f seconds\n", end_time - start_time);
    free(A);
    return end_time - start_time;
}

double calculate_average_parallel_with_omp_critical_v3()
{
    double ave = 0.0, *A;
    int i;

    A = (double *)malloc(MAX * sizeof(double));
    if (A == NULL)
    {
        printf("Insufficient memory! Can't continue. Terminating the program abruptly.\n");
        return -1;
    }

    for (i = 0; i < MAX; i++)
    {
        A[i] = (double)i;
    }
    double start_time = omp_get_wtime();

#pragma omp parallel num_threads(MAX_THREAD)
    {
        int i;
        int id = omp_get_thread_num();
        double partial_ave = 0.0;

        for (i = id; i < MAX; i += MAX_THREAD)
        {
            partial_ave += A[i];
        }

#pragma omp atomic // only 16 threads will go into this critical region (Number of threads access to critical region is reduced)
        ave += partial_ave;

        // #pragma omp critical(sum) // same as atomic but run faster
        //         ave += partial_ave;
    }

    ave = ave / MAX;
    double end_time = omp_get_wtime();

    printf("%f\n", ave);
    printf("Modified 3 work took %f seconds\n", end_time - start_time);
    free(A);
    return end_time - start_time;
}

double calculate_average_parallel_with_for_reduction()
{
    double ave = 0.0, *A;
    int i;

    A = (double *)malloc(MAX * sizeof(double));
    if (A == NULL)
    {
        printf("Insufficient memory! Can't continue. Terminating the program abruptly.\n");
        return -1;
    }

    for (i = 0; i < MAX; i++)
    {
        A[i] = (double)i;
    }
    double start_time = omp_get_wtime();

#pragma omp parallel for reduction(+ : ave)
    for (i = 0; i < MAX; i++)
    {
        ave += A[i];
    }

    ave = ave / MAX;

    double end_time = omp_get_wtime();

    printf("%f\n", ave);
    printf("Modified2 work took %f seconds\n", end_time - start_time);
    free(A);
    return end_time - start_time;
}

int main()
{
    double runTimeOriginal, runTimeModified1 = 0, runTimeModified2 = 0, runTimeModified3 = 0, runTimeModified4 = 0;

    runTimeOriginal = calculate_average_serial();
    // for (int i = 0; i < 10; i++)
    // {
    //     double res = calculate_average_parallel_with_omp_critical();
    //     if (i != 0)
    //     {
    //         runTimeModified1 += res;
    //     }
    // }
    // runTimeModified1 /= 10 - 1;

    // for (int i = 0; i < 10; i++)
    // {
    //     double res = calculate_average_parallel_with_for_reduction();
    //     if (i != 0)
    //     {
    //         runTimeModified2 += res;
    //     }
    // }
    // runTimeModified2 /= 10 - 1;

    runTimeModified1 = calculate_average_parallel_with_omp_critical_v1();
    runTimeModified2 = calculate_average_parallel_with_omp_critical_v2();
    runTimeModified3 = calculate_average_parallel_with_omp_critical_v3();
    runTimeModified4 = calculate_average_parallel_with_for_reduction();

    cout << "Performance gain runTimeOriginal/runTimeModified1 is " << runTimeOriginal / runTimeModified1 << endl;
    cout << "Performance gain runTimeOriginal/runTimeModified2 is " << runTimeOriginal / runTimeModified2 << endl;
    cout << "Performance gain runTimeOriginal/runTimeModified3 is " << runTimeOriginal / runTimeModified3 << endl;
    cout << "Performance gain runTimeOriginal/runTimeModified4 is " << runTimeOriginal / runTimeModified4 << endl;
}
