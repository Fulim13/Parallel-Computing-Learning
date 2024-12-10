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

int producer_consumer_serial()
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

    return 0;
}

int producer_consumer_parallel()
{
    double *A, sum, runtime;
    int flag = 0;

    A = (double *)malloc(N * sizeof(double));
    srand(100);

    runtime = omp_get_wtime();

#pragma omp parallel sections num_threads(2)
    {
#pragma omp section
        {
            fill_rand(N, A); // Producer: fill an array of data
#pragma omp flush(A)
            flag = 1;
#pragma omp flush(flag)
        }

#pragma omp section
        {
            while (flag == 0)
            {
#pragma omp flush(flag)
            }
#pragma omp flush(A)
            sum = Sum_array(N, A); // Consumer: sum the array
        }
    }

    runtime = omp_get_wtime() - runtime;

    printf(" In %lf seconds, The sum is %lf \n", runtime, sum);

    return 0;
}

int main()
{
    double ori, mod1;

    ori = producer_consumer_serial();
    mod1 = producer_consumer_parallel();
    cout << "The performance gain (ori/mod1) is " << ori / mod1 << endl;
    return 0;
}
