#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <malloc.h>
#include <math.h>

#define THREADS_PER_BLOCK 128

void matrixMultiplyCPU(float *a, float *b, float *c, int width)
{
    float result;

    for (int row = 0; row < width; row++)
    {
        for (int col = 0; col < width; col++)
        {
            result = 0;
            for (int k = 0; k < width; k++)
            {
                result += a[row * width + k] * b[k * width + col];
            }
            c[row * width + col] = result;
        }
    }
}

__global__ void matrixMultiplySimple(float *a, float *b, float *c, int width)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    float result = 0;

    if (col < width && row < width)
    {
        for (int k = 0; k < width; k++)
        {
            result += a[row * width + k] * b[k * width + col];
        }
        c[row * width + col] = result;
    }
}

int main()
{
    int width = 200; // Define width of square matrix
    // Initialise grid and block variables
    int sqrtThreads = (int)sqrt(THREADS_PER_BLOCK);
    int nBlocks = width / sqrtThreads;
    if (width % sqrtThreads != 0)
    { // Add an extra block if necessary
        nBlocks++;
    }
    dim3 grid(nBlocks, nBlocks, 1);
    dim3 block(sqrtThreads, sqrtThreads, 1); // Max number of threads per block

    // Initialise host pointers (dynamically allocated memory) and device pointers
    // Note: _h is matrix for the host (CPU) and _d is matrix for device (GPU)
    float *a_h;
    float *b_h;
    float *c_h; // GPU (computed in parallel) results
    float *d_h; // CPU (computed in serial) results
    float *a_d; // Device memory for A matrix
    float *b_d; // Device memory for B matrix
    float *c_d; // Device memory for C matrix

    int size; // Number of bytes required by arrays

    // Create timer
    cudaEvent_t start;
    cudaEvent_t stop;
    float elapsed1, elapsed2; //, elapsed3;

    // Print out information about blocks and threads
    printf("Number of threads: %i (%ix%i)\n", block.x * block.y, block.x, block.y);
    printf("Number of blocks: %i (%ix%i)\n", grid.x * grid.y, grid.x, grid.y);

    // Dynamically allocate host memory
    size = width * width * sizeof(float);

    a_h = (float *)malloc(size);
    b_h = (float *)malloc(size);
    c_h = (float *)malloc(size);
    d_h = (float *)malloc(size);

    // Load host arrays with data
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < width; j++)
        {
            a_h[i * width + j] = (float)i;
            b_h[i * width + j] = (float)i;
        }
    }

    //@@ Allocate device memory for a_d, b_d and c_d matrices
    cudaMalloc((void **)&a_d, size);
    cudaMalloc((void **)&b_d, size);
    cudaMalloc((void **)&c_d, size);

    //@@ Copy host memory to device memory from a_h and, b_h to a_d and b_d
    cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);

    // Start timer for GPU
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    //@@ Launch kernel (GPU) passing in a_d, b_d, and c_d matrices
    matrixMultiplySimple<<<grid, block>>>(a_d, b_d, c_d, width);

    // Stop timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed1, start, stop);

    // Print execution time
    printf("Time to calculate results on GPU: %f ms\n", elapsed1);

    //@@ Copy results to host from c_d to c_h matrix
    cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost);

    // Start timer for CPU
    cudaEventRecord(start, 0);

    // Launch CPU code
    matrixMultiplyCPU(a_h, b_h, d_h, width);

    // Stop timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed2, start, stop);

    // Print execution time
    printf("Time to calculate results on CPU: %f ms\n", elapsed2);

    // Compare results
    for (int i = 0; i < width * width; i++)
    {
        if (fabs(c_h[i] - d_h[i]) > 1e-5)
        {
            printf("Error: CPU and GPU results do not match at index number %d\n", i);
            break;
        }
    }

    //@@ Free memory
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    free(a_h);
    free(b_h);
    free(c_h);
    free(d_h);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
