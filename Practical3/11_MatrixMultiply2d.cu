#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define BLOCKSIZE 16
const int MAX_DIM = 100;

__global__ void MatrixMulKernel(float *M, float *N, float *P, int Width)
{
    // Calculate the row index of the P element and M
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    // Calculate the column index of P and N
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((Row < Width) && (Col < Width))
    {
        float Pvalue = 0;
        // Each thread computes one element of the block sub-matrix
        for (int k = 0; k < Width; ++k)
        {
            Pvalue += M[Row * Width + k] * N[k * Width + Col];
        }
        P[Row * Width + Col] = Pvalue;
    }
}

int main()
{
    // Matrix dimensions
    int Width = MAX_DIM;

    // Allocating memory for host arrays
    float h_M[MAX_DIM][MAX_DIM], h_N[MAX_DIM][MAX_DIM], h_P[MAX_DIM][MAX_DIM];

    // Generating input arrays
    for (int i = 0; i < MAX_DIM; i++)
    {
        for (int j = 0; j < MAX_DIM; j++)
        {
            h_M[i][j] = (float)(rand() % 100);
            h_N[i][j] = (float)(rand() % 100);
        }
    }

    // Declaring device memory pointers
    float *d_M, *d_N, *d_P;

    // Allocating device memory
    cudaMalloc((void **)&d_M, MAX_DIM * MAX_DIM * sizeof(float));
    cudaMalloc((void **)&d_N, MAX_DIM * MAX_DIM * sizeof(float));
    cudaMalloc((void **)&d_P, MAX_DIM * MAX_DIM * sizeof(float));

    // Copying data from host to device
    cudaMemcpy(d_M, h_M, MAX_DIM * MAX_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, MAX_DIM * MAX_DIM * sizeof(float), cudaMemcpyHostToDevice);

    // Configuring the block size and grid size
    dim3 blockSize(BLOCKSIZE, BLOCKSIZE, 1);
    dim3 gridSize((Width + BLOCKSIZE - 1) / BLOCKSIZE, (Width + BLOCKSIZE - 1) / BLOCKSIZE, 1);

    // Calling the kernel
    MatrixMulKernel<<<gridSize, blockSize>>>(d_M, d_N, d_P, Width);

    // Transferring result from device to host
    cudaMemcpy(h_P, d_P, MAX_DIM * MAX_DIM * sizeof(float), cudaMemcpyDeviceToHost);

    // Checking correctness of answer
    int flag = 1;
    for (int i = 0; i < MAX_DIM; i++)
    {
        for (int j = 0; j < MAX_DIM; j++)
        {
            float expectedValue = 0;
            for (int k = 0; k < MAX_DIM; k++)
            {
                expectedValue += h_M[i][k] * h_N[k][j];
            }
            if (abs(h_P[i][j] - expectedValue) > 1e-5)
            {
                printf("Wrong value at (%d, %d)\n", i, j);
                printf("Expected value: %f, Found value: %f\n", expectedValue, h_P[i][j]);
                flag = 0;
                break;
            }
        }
        if (!flag)
            break;
    }
    if (flag)
        printf("The solution is correct\n");

    // Free allocated device memory
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);

    return 0;
}
