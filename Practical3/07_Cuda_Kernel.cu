#include "cuda_runtime.h"
#include "07_Cuda_Kernel.cuh"

// __global__ tell nvcc that this code is kernel not a function, that code translate to nvidia gpu machine code to run
__global__ void vectorAdditionKernel(double *A, double *B, double *C, int arraySize)
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
// for pc to run
void kernel(double *A, double *B, double *C, int arraySize)
{

    // Initialize device pointers.
    double *d_A, *d_B, *d_C;

    // Allocate device memory.
    cudaMalloc((void **)&d_A, arraySize * sizeof(double));
    cudaMalloc((void **)&d_B, arraySize * sizeof(double));
    cudaMalloc((void **)&d_C, arraySize * sizeof(double));

    // Transfer arrays a and b to device.
    cudaMemcpy(d_A, A, arraySize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, arraySize * sizeof(double), cudaMemcpyHostToDevice);

    // Calculate blocksize and gridsize.
    dim3 blockSize(512, 1, 1);             // 512 thread ,x,y, z
    dim3 gridSize(512 / arraySize + 1, 1); // grid two dimension

    // Launch CUDA kernel.
    vectorAdditionKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, arraySize);

    // Copy result array c back to host memory.
    cudaMemcpy(C, d_C, arraySize * sizeof(double), cudaMemcpyDeviceToHost);
}
