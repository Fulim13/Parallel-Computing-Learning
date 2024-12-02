#include "cuda_runtime.h"
#include <device_launch_parameters.h>
#include "08_ParallelVectorAdd.cuh"

__global__ void vectorAdditionKernel(int *A, int *B, int *C, int arraySize)
{
    // Get thread ID.
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;

    // Check if thread is within array bounds.
    if (threadID < arraySize)
    {
        // Add a and b.
        C[threadID] = A[threadID] + B[threadID];
    }
}

/**
 * Wrapper function for the CUDA kernel function.
 * @param A Array A.
 * @param B Array B.
 * @param C Sum of array elements A and B directly across.
 * @param arraySize Size of arrays A, B, and C.
 */
int parallel_vec_add(int *A, int *B, int *C, int arraySize)
{

    // Initialize device pointers.
    int *d_A, *d_B, *d_C;

    // Allocate device memory.
    cudaMalloc((void **)&d_A, arraySize * sizeof(int));
    cudaMalloc((void **)&d_B, arraySize * sizeof(int));
    cudaMalloc((void **)&d_C, arraySize * sizeof(int));

    // Transfer arrays a and b to device.
    cudaMemcpy(d_A, A, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    // Calculate blocksize and gridsize.
    dim3 blockSize(512, 1, 1);
    dim3 gridSize(512 / arraySize + 1, 1);

    // Launch CUDA kernel.
    vectorAdditionKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, arraySize);

    // Copy result array c back to host memory.
    cudaMemcpy(C, d_C, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    // Free CUDA memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
