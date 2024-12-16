#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h> // new version of cuda, device is separated out

#define CEIL(a, b) ((a - 1) / b + 1)
const int BLOCKSIZE = 1024;

const int MAX_DIM = 100;
const int MAX_SIZE = MAX_DIM * MAX_DIM;
const int MAX_BYTES = MAX_SIZE * sizeof(float);

__global__ void matrix_mul(float *d_m1, float *d_m2, float *d_m3)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // guard extra thread that in final block that not inside GPU
    // our MAX_SIZE is 10,000, the problem need to 10,000 runs
    // and our available threads is number of threads(blockSize) * number of blocks(gridSize) is 1024 * 10 = 10240,
    // there is extra 240 threads in the last block that we don't want to include in calculation.
    // How does I count the gridSize = 10
    // The Formula: (MAX_SIZE + BLOCKSIZE) / BLOCKSIZE
    // Assumming MAX_SIZE = 10,000, BLOCKSIZE = 1024, the answer is 10.7656 and discarded to integer become 10
    if (idx >= MAX_SIZE)
        return;
    float tempsum = 0;
    int i = idx / MAX_DIM;
    int j = idx % MAX_DIM;
    for (int k = 0; k < MAX_DIM; k++)
    {
        tempsum += d_m1[i * MAX_DIM + k] * d_m2[j + k * MAX_DIM];
    }
    d_m3[idx] = tempsum;
}

int main()
{
    // allocating memory for host arrays
    float h_m1[MAX_SIZE], h_m2[MAX_SIZE], h_m3[MAX_SIZE];

    // generating input arrays
    for (int i = 0; i < MAX_SIZE; i++)
        h_m1[i] = (float)(rand() % 100);
    for (int i = 0; i < MAX_SIZE; i++)
        h_m2[i] = (float)(rand() % 100);

    // declaring device memory pointers
    float *d_m1, *d_m2, *d_m3;

    // allocating device memory
    cudaMalloc((void **)&d_m1, MAX_SIZE * sizeof(float));
    cudaMalloc((void **)&d_m2, MAX_SIZE * sizeof(float));
    cudaMalloc((void **)&d_m3, MAX_SIZE * sizeof(float));

    // copying data from host to device
    cudaMemcpy(d_m1, h_m1, MAX_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m2, h_m2, MAX_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // configure the blocksize and gridsize
    //  block size - the number of threads in a block
    dim3 blockSize(BLOCKSIZE, 1, 1); // x, y, z total up is 1024 threads which is the max threads we can use
    // grid size = the number of blocks in grid - Formula [(MAX_SIZE-1)/blocksize] + 1
    // Better Formula [MAX_SIZE + (blocksize -1)] / block_size
    dim3 gridSize((MAX_SIZE + BLOCKSIZE - 1) / BLOCKSIZE, 1);

    // calling kernel
    matrix_mul<<<gridSize, blockSize>>>(d_m1, d_m2, d_m3);

    // transferring result from device to host
    cudaMemcpy(h_m3, d_m3, MAX_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // checking correctness of answer
    int flag = 1;
    for (int i = 0; i < MAX_DIM; i++)
    {
        for (int j = 0; j < MAX_DIM; j++)
        {
            float tempsum = 0;
            for (int k = 0; k < MAX_DIM; k++)
            {
                tempsum += h_m1[i * MAX_DIM + k] * h_m2[j + k * MAX_DIM];
            }
            if (h_m3[i * MAX_DIM + j] != tempsum)
            {
                printf("wrong value at %d\n", i * MAX_DIM + j);
                printf("Expected value:%f, found value:%f\n", tempsum, h_m3[i * MAX_DIM + j]);
                flag = 0;
                break;
            }
        }
        if (flag == 0)
            break;
    }
    if (flag == 1)
        printf("The solution is correct\n");

    // free allocated device memory
    cudaFree(d_m1);
    cudaFree(d_m2);
    cudaFree(d_m3);
}
