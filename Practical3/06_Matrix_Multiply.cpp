/*
**  PROGRAM: Matrix Multiply
**
**  PURPOSE: This is a simple matrix multiply program.
**           It will compute the product
**
**                C  = A * B
**
**           A and B are set to constant matrices so we
**           can make a quick test of the multiplication.
**
**
*/
#include <iostream>
#include <omp.h>
#ifdef APPLE
#include <stdlib.h>
#else
#include <malloc.h>
#endif

#define ORDER 1000
#define AVAL 3.0
#define BVAL 5.0
#define TOL 0.001 // tolerance

#define NUM_THREADS 16

using namespace std;

double multiply_matrices()
{
    int Ndim, Pdim, Mdim; /* A[N][P], B[P][M], C[N][M] */
    int i, j, k;
    double *A, *B, *C, cval, tmp, err, errsq;
    double dN, mflops;
    double start_time, run_time;

    Ndim = ORDER;
    Pdim = ORDER;
    Mdim = ORDER;

    // malloc allocate specify num of bytes (size of double is 8),
    // different processor may have different bits of double (current 32 bits), 8086 is 16 bits
    // malloc will return a adderss that need a pointer to point to the first bits in the allocated space
    A = (double *)malloc(Ndim * Pdim * sizeof(double));
    B = (double *)malloc(Pdim * Mdim * sizeof(double));
    C = (double *)malloc(Ndim * Mdim * sizeof(double));

    /* Initialize matrices */

    for (i = 0; i < Ndim; i++)
        for (j = 0; j < Pdim; j++)
            *(A + (i * Pdim + j)) = AVAL;
    // A[i * Pdim + j] = AVAL;

    for (i = 0; i < Pdim; i++)
        for (j = 0; j < Mdim; j++)
            *(B + (i * Mdim + j)) = BVAL;

    for (i = 0; i < Ndim; i++)
        for (j = 0; j < Mdim; j++)
            *(C + (i * Mdim + j)) = 0.0;

    /* Do the matrix product */

    start_time = omp_get_wtime();
    for (i = 0; i < Ndim; i++)
    {
        for (j = 0; j < Mdim; j++)
        {
            tmp = 0.0;
            for (k = 0; k < Pdim; k++)
            {
                /* C(i,j) = sum(over k) A(i,k) * B(k,j) */
                tmp += *(A + (i * Pdim + k)) * *(B + (k * Mdim + j));
            }
            *(C + (i * Mdim + j)) = tmp;
        }
    }
    /* Check the answer */

    run_time = omp_get_wtime() - start_time;

    printf(" Order %d multiplication in %f seconds \n", ORDER, run_time);

    dN = (double)ORDER;
    mflops = 2.0 * dN * dN * dN / (1000000.0 * run_time);

    printf(" Order %d multiplication at %f mflops\n", ORDER, mflops);

    cval = Pdim * AVAL * BVAL;
    errsq = 0.0;
    for (i = 0; i < Ndim; i++)
    {
        for (j = 0; j < Mdim; j++)
        {
            err = *(C + i * Mdim + j) - cval;
            errsq += err * err; // square it to make the error not balance each other //always have positive error
        }
    }

    // if not error, the error is 0
    if (errsq > TOL)
        printf("\n Errors in multiplication: %f", errsq);
    else
        printf("\n Hey, it worked");

    printf("\n all done \n");

    return run_time;
}

double multiply_matrices_in_parallel()
{
    int Ndim, Pdim, Mdim; /* A[N][P], B[P][M], C[N][M] */
    int i, j, k;
    double *A, *B, *C, cval, tmp, err, errsq;
    double dN, mflops;
    double start_time, run_time;

    Ndim = ORDER;
    Pdim = ORDER;
    Mdim = ORDER;

    // malloc allocate specify num of bytes (size of double is 8), different processor may have different bits of double (current 32 bits), 8086 is 16 bits
    // malloc will return a adderss that need a pointer to point to the first bits in the allocated space
    A = (double *)malloc(Ndim * Pdim * sizeof(double));
    B = (double *)malloc(Pdim * Mdim * sizeof(double));
    C = (double *)malloc(Ndim * Mdim * sizeof(double));

    /* Initialize matrices */

    for (i = 0; i < Ndim; i++)
        for (j = 0; j < Pdim; j++)
            *(A + (i * Pdim + j)) = AVAL;
    // A[i * Pdim + j] = AVAL;

    for (i = 0; i < Pdim; i++)
        for (j = 0; j < Mdim; j++)
            *(B + (i * Mdim + j)) = BVAL;

    for (i = 0; i < Ndim; i++)
        for (j = 0; j < Mdim; j++)
            *(C + (i * Mdim + j)) = 0.0;

    /* Do the matrix product */

    start_time = omp_get_wtime();

// i is private , because the pragma omp for already make it private
// all other variables is public
// Mdim, PDim, Ndim -> public (because we never write into these variables - only reading, no race condition)
// j -> private (all the threads handle it own i row, so each i row need to multiple each j column
// tmp -> private
// k -> private
// A, B -> public (never change the content in A and B)
// C -> public
// #pragma omp parallel for private(j, k, tmp) shared(A, B, C, Ndim, Pdim, Mdim)
#pragma omp parallel private(j, k, tmp) num_threads(NUM_THREADS)
    {
#pragma omp for
        for (i = 0; i < Ndim; i++)
        {
            for (j = 0; j < Mdim; j++)
            {
                tmp = 0.0;
                for (k = 0; k < Pdim; k++)
                {
                    /* C(i,j) = sum(over k) A(i,k) * B(k,j) */
                    tmp += *(A + (i * Pdim + k)) * *(B + (k * Mdim + j));
                }
                *(C + (i * Mdim + j)) = tmp;
            }
        }
    }

    run_time = omp_get_wtime() - start_time;

    printf(" Order %d multiplication in %f seconds \n", ORDER, run_time);

    dN = (double)ORDER;
    mflops = 2.0 * dN * dN * dN / (1000000.0 * run_time);

    printf(" Order %d multiplication at %f mflops\n", ORDER, mflops);

    /* Check the answer */
    cval = Pdim * AVAL * BVAL;
    errsq = 0.0;

#pragma omp parallel for private(i, j, err) shared(C, Ndim, Mdim, cval) reduction(+ : errsq)
    for (i = 0; i < Ndim; i++)
    {
        for (j = 0; j < Mdim; j++)
        {
            err = *(C + i * Mdim + j) - cval;
            errsq += err * err; // square it to make the error not balance each other //always have positive error
        }
    }

    // if not error, the error is 0
    if (errsq > TOL)
        printf("\n Errors in multiplication: %f", errsq);
    else
        printf("\n Hey, it worked");

    printf("\n all done \n");

    return run_time;
}

int main()
{
    double ori_rt, mod_rt;
    ori_rt = multiply_matrices();
    mod_rt = multiply_matrices_in_parallel();
    cout << "The performance gain is " << ori_rt / mod_rt << endl;
    return 0;
}
