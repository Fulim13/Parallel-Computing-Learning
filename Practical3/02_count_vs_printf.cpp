#include <iostream>
#include <omp.h>

void omp_demo_cout()
{
#pragma omp parallel
    {
        int id = omp_get_thread_num();
        std::cout << "Cout: Hello World!" << id << std::endl;
    }
}

void omp_demo_printf()
{
#pragma omp parallel
    {
        int id = omp_get_thread_num();
        printf("Printf: Hello World! %d\n", id);
    }
}

int main()
{
    omp_demo_cout();   // Not Thread Safe
    omp_demo_printf(); // Thread Safe
    return 0;
}
