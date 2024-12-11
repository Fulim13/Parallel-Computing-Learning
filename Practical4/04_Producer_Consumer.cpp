#include <iostream>
#include <cstdlib>
#include <omp.h>

using namespace std;

const int N = 100000;

void fill_rand(int nval, double *A)
{
    for (int i = 0; i < nval; i++)
        A[i] = (double)rand();
}

double Sum_array(int nval, double *A)
{
    double sum = 0.0;
    for (int i = 0; i < nval; i++)
        sum = sum + A[i];
    return sum;
}

double producer_consumer_serial()
{
    double *A, sum, runtime;
    int flag = 0;

    A = (double *)malloc(N * sizeof(double));
    srand(100);

    runtime = omp_get_wtime();
    fill_rand(N, A);       // Producer: fill an array of data
    sum = Sum_array(N, A); // Consumer: sum the array
    runtime = omp_get_wtime() - runtime;

    printf(" In %lf seconds, The sum is %lf \n", runtime, sum);

    return runtime;
}

double producer_consumer_parallel()
{
    double *A, sum = 0.0, runtime;
    int flag = 0;

    A = (double *)malloc(N * sizeof(double));
    srand(100);

    runtime = omp_get_wtime();

#pragma omp parallel sections num_threads(2)
    {
#pragma omp section
        {
            int i = 0;
            // Producer: produce 1 random value and put it in A[i++]
            // until all values are produced
            while (i < N)
            {
                if (flag == 0)
                {
                    A[i] = (double)rand(); // produce a random value
                    i++;                   // move to the next index
                    flag = 1;              // signal the consumer
#pragma omp flush(flag)                    // Ensure visibility of flag
                }
            }
        }

#pragma omp section
        {
            int j = 0;
            // Consumer: sum the array when producer has placed data
            while (j < N)
            {
                if (flag == 1)
                {
                    sum += A[j]; // consume the value
                    j++;         // move to the next value
                    flag = 0;    // signal the producer
#pragma omp flush(flag)          // Ensure visibility of flag
                }
            }
        }
    }

    runtime = omp_get_wtime() - runtime;

    printf(" In %lf seconds, The sum is %lf \n", runtime, sum);

    return runtime;
}

int main()
{
    double ori = 0.0, mod1 = 0.0;

    ori = producer_consumer_serial();
    mod1 = producer_consumer_parallel();
    cout << "The performance gain (ori/mod1) is " << ori / mod1 << endl;
    return 0;
}
