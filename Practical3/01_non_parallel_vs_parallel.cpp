#include <iostream>
#include <omp.h>

void ordinary_demo()
{
    std::cout << "Normal Hello World!" << std::endl;
}

// Running parallel process, os will assign a thread id to each process
void omp_demo()
{
#pragma omp parallel
    { // this bracket must be on the next line, pragma does not understand {
        int id = omp_get_thread_num();
        std::cout << "Hello World!" << id << std::endl;
    }
}

int main()
{
    ordinary_demo();
    omp_demo();
    return 0;
}
