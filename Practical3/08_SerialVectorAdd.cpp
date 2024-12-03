#include <stdio.h>

void VectorAdd(int *a, int *b, int *c, int n)
{
    int i;

    for (i = 0; i < n; ++i)
        c[i] = a[i] + b[i];
}

int serial_vec_add(int *a, int *b, int *c, int n)
{
    VectorAdd(a, b, c, n);

    return 0;
}
