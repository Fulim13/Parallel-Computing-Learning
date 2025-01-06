#include <stdio.h>
#include <omp.h>

int main()
{
    int a = 1;
    int b = 2;
    int c = 4;

    // Without firstprivate
    printf("Without firstprivate:\n");
#pragma omp parallel num_threads(4)
    {
        printf("Thread %d: Initial values: a=%d, b=%d, c=%d\n", omp_get_thread_num(), a, b, c);
        a = a + b;
        printf("Thread %d: Updated a=%d\n", omp_get_thread_num(), a);
    }
    printf("Outside parallel region: a=%d\n", a);

    // Reset a
    a = 1;

    // With firstprivate
    printf("\nWith firstprivate:\n");
#pragma omp parallel firstprivate(a) num_threads(4)
    {
        printf("Thread %d: Initial values: a=%d, b=%d, c=%d\n", omp_get_thread_num(), a, b, c);
        a = a + b;
        printf("Thread %d: Updated a=%d\n", omp_get_thread_num(), a);
    }
    printf("Outside parallel region: a=%d\n", a);

    return 0;
}
