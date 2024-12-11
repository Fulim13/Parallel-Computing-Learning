// P4Q4.cpp : Defines the entry point for the console application.
//

#include <stdio.h>
#include <omp.h>

float work1(int i)
{
    return 1.0 * i;
}

float work2(int i)
{
    return 2.0 * i;
}
float Sum_array(int nval, float *A)
{
    float sum = 0.0;
    for (int i = 0; i < nval; i++)
        sum = sum + A[i];
    return sum;
}

void atomic_example_no_atomic(float *x, float *y, int *index, int n)
{
    int i;

#pragma omp parallel for shared(x, y, index, n) num_threads(16)
    for (i = 0; i < n; i++)
    {
        x[index[i]] += work1(i);
        y[i] += work2(i);
    }
}

int atomic_exercise_no_atomic()
{
    float x[1000];
    float y[10000];
    int index[10000];
    float sum1, sum2;
    int i;

    for (i = 0; i < 10000; i++)
    {
        index[i] = i % 8;
        y[i] = 0.0;
    }
    for (i = 0; i < 1000; i++)
        x[i] = 0.0;
    atomic_example_no_atomic(x, y, index, 10000);

    sum1 = Sum_array(1000, x);
    sum2 = Sum_array(10000, y);

    printf(" The sum is %f and %f\n", sum1, sum2);
    return 0;
}

void atomic_example_with_atomic(float *x, float *y, int *index, int n)
{
    int i;

#pragma omp parallel for shared(x, y, index, n) num_threads(16)
    for (i = 0; i < n; i++)
    {
#pragma omp atomic
        x[index[i]] += work1(i);
        y[i] += work2(i);
    }
}

int atomic_exercise_with_atomic()
{
    float x[1000];
    float y[10000];
    int index[10000];
    float sum1, sum2;
    int i;

    for (i = 0; i < 10000; i++)
    {
        index[i] = i % 8;
        y[i] = 0.0;
    }
    for (i = 0; i < 1000; i++)
        x[i] = 0.0;
    atomic_example_with_atomic(x, y, index, 10000);

    sum1 = Sum_array(1000, x);
    sum2 = Sum_array(10000, y);

    printf(" The sum is %f and %f\n", sum1, sum2);
    return 0;
}

int main()
{
    atomic_exercise_no_atomic();
    atomic_exercise_with_atomic();
    return 0;
}
