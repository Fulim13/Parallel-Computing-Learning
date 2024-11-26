#include <iostream>
#include "07_Cuda_Kernel.cuh"

int main()
{
    // Initialize arrays A, B, and C.
    double A[3], B[3], C[3];

    // Populate arrays A and B.
    A[0] = 5;
    A[1] = 8;
    A[2] = 3;
    B[0] = 7;
    B[1] = 6;
    B[2] = 4;

    // Sum array elements across ( C[0] = A[0] + B[0] ) into array C using CUDA.
    kernel(A, B, C, 3);

    // Print out result.
    // other process also need to use gpu memory, so put back to cpu memory
    std::cout << "C = " << C[0] << ", " << C[1] << ", " << C[2] << std::endl;
}
