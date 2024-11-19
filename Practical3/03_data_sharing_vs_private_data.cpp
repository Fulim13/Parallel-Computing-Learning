#include <iostream>
#include <omp.h>

void q1_data_sharing()
{
    int x = 5; // global

#pragma omp parallel
    {
        x = x + 1; // accessing global
        printf("shared: x is %d\n", x);
    }
}

void q1_no_data_sharing()
{
    int x = 5; // global

#pragma omp parallel
    {
        int x;
        x = 3;     // local
        x = x + 1; // accessing local
        printf("local: x is %d\n", x);
    }
    printf("global: x is %d\n", x);
}

void q1_no_data_sharing_v2()
{
    int x = 5; // global

#pragma omp parallel private(x) // create local x
    {
        x = 3;
        x = x + 1;
        printf("local: x is %d\n", x);
    }
    printf("global: x is %d\n", x);
}

void q1_no_data_sharing_v3()
{
    int x = 5;

#pragma omp parallel firstprivate(x)
    {
        x = x + 1;
        printf("local: x is %d\n", x);
    }
    printf("global: x is %d\n", x);
}

int main()
{
    q1_data_sharing();
    q1_no_data_sharing();
    q1_no_data_sharing_v2();
    q1_no_data_sharing_v3();
    return 0;
}
