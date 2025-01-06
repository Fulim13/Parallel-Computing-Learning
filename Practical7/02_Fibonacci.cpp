#include <stdlib.h>
#include <iostream>
#include <omp.h>

using namespace std;

#define N 5
#define FS 38

struct node
{
    int data;
    int fibdata;
    struct node *next;
};

int fib(int n)
{
    int x, y;
    if (n < 2)
    {
        return (n);
    }
    else
    {
        x = fib(n - 1);
        y = fib(n - 2);
        return (x + y);
    }
}

void processwork(struct node *p)
{
    int n;
    n = p->data;
    p->fibdata = fib(n);
}

struct node *init_list()
{
    int i;
    struct node *head = NULL;
    struct node *temp = NULL;
    struct node *p;

    head = (struct node *)malloc(sizeof(struct node));
    p = head;
    p->data = FS;   // dereferencing put into the node
    p->fibdata = 0; // deferencing put into the node
    for (i = 0; i < N - 1; i++)
    {
        temp = (struct node *)malloc(sizeof(struct node));
        p->next = temp;
        p = temp;
        p->data = FS + i + 1;
        p->fibdata = i + 1;
    }
    p->next = NULL;
    return head;
}

double calc_fib_serial()
{
    double start, end;
    struct node *p = NULL;
    struct node *temp = NULL;
    struct node *head = NULL;

    printf("Process linked list\n");
    printf("  Each linked list node will be processed by function 'processwork()'\n");
    printf("  Each ll node will compute %d fibonacci numbers beginning with %d\n", N, FS);

    p = init_list();
    head = p;

    start = omp_get_wtime();

    while (p != NULL)
    {
        processwork(p);
        p = p->next;
    }

    end = omp_get_wtime();
    p = head;
    while (p != NULL)
    {
        printf("%d : %d\n", p->data, p->fibdata);
        temp = p->next;
        free(p);
        p = temp;
    }
    free(p);

    printf("Compute Time: %f seconds\n", end - start);

    return end - start;
}

double calc_fib_parallel()
{
    double start, end;
    struct node *p = NULL;
    struct node *temp = NULL;
    struct node *head = NULL;
    struct node **arrayOfPointers;

    printf("Process linked list\n");
    printf("  Each linked list node will be processed by function 'processwork()'\n");
    printf("  Each ll node will compute %d fibonacci numbers beginning with %d\n", N, FS);

    p = init_list();
    head = p;

    arrayOfPointers = (struct node **)malloc(N * sizeof(struct node *));

    start = omp_get_wtime();

    int i = 0;
    p = head;
    while (p != NULL)
    {
        arrayOfPointers[i++] = p;
        p = p->next;
    }

#pragma omp parallel for
    for (i = 0; i < N; i++)
    {
        int id = i;
        processwork(arrayOfPointers[i]);
    }

    end = omp_get_wtime();
    p = head;
    while (p != NULL)
    {
        printf("%d : %d\n", p->data, p->fibdata);
        temp = p->next;
        free(p);
        p = temp;
    }
    free(p);

    printf("Compute Time: %f seconds\n", end - start);

    return end - start;
}

int main()
{
    double ori, modified;
    ori = calc_fib_serial();
    modified = calc_fib_parallel();

    cout << "The Performance gain is " << ori / modified << endl;
}
