#include <iostream> //include <stdio.h>
#include <malloc.h>
#include "08_ParallelVectorAdd.cuh"
#include "08_SerialVectorAdd.h"

using namespace std;

#define SIZE 1024

void init_vector(int **a, int **b, int **c, int size)
{
    *a = (int *)malloc(size * sizeof(int));
    *b = (int *)malloc(size * sizeof(int));
    *c = (int *)malloc(size * sizeof(int));

    for (int i = 0; i < size; ++i)
    {
        // a =  [0 1 2 3 4 ... n]
        (*a)[i] = i; //(*a) - dereference first, before access to the element
        // b =  [0 1 2 3 4 ... n]
        (*b)[i] = i;
        // a+b=c [0 2 4 6 8 ... n]
        (*c)[i] = 0;
    }
}

int verify_result(int *c, int size)
{
    for (int i = 0; i < size; i++)
    {
        if (c[i] != 2 * i)
        {
            cout << "Error: c[" << i << "] = " << c[i] << " is incorrect; expecting " << 2 * i << endl;
            return 1;
        }
    }
    return 0;
}

void free_vectors(int *a, int *b, int *c)
{
    free(a);
    free(b);
    free(c);
}

void test_v1(int *a_ptr)
{
    // create 4 bytes of SIZE=1024 int block and return the first adddress
    a_ptr = (int *)malloc(SIZE * sizeof(int));
}

void test_v2(int **a_ptr_ptr)
{
    *a_ptr_ptr = (int *)malloc(SIZE * sizeof(int));
}

int main()
{
    int *a, *b, *c;

    test_v1(a);
    test_v2(&a);

    // Serial vector add
    init_vector(&a, &b, &c, SIZE);
    serial_vec_add(a, b, c, SIZE);
    for (int i = 0; i < 10; ++i)
    {
        printf("c[%d]=%d\n", i, c[i]);
    }
    verify_result(c, SIZE);
    free_vectors(a, b, c);

    // Parallel vector add
    init_vector(&a, &b, &c, SIZE);
    parallel_vec_add(a, b, c, SIZE);
    for (int i = 0; i < 10; ++i)
    {
        printf("c[%d]=%d\n", i, c[i]);
    }
    verify_result(c, SIZE);
    free_vectors(a, b, c);

    return 0;
}
