#include <iostream>
#include <omp.h>

using namespace std;

int some_calculation()
{
    return 10;
}

const int MAX_THREADS = 16;

void barrier_exercise()
{
    int x[MAX_THREADS] = {0}, y[MAX_THREADS] = {0};

#pragma omp parallel num_threads(MAX_THREADS)
    {
        int mytid = omp_get_thread_num();
        x[mytid] = some_calculation();
#pragma omp barrier
        y[mytid] = x[mytid] + x[mytid + 1];
    }
    for (int i = 0; i < MAX_THREADS; i++)
        cout << "y[" << i << "] = " << y[i] << endl;
}

int main()
{
    barrier_exercise();
    return 0;
}
