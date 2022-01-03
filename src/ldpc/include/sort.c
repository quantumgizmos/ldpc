#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "sort.h"
#include "mod2sparse.h"



// Factorial function:
// http://www.rosettacode.org/wiki/Factorial#C




unsigned long long
ncr(int n, int k) {
    if (k > n) {
        return 0;
    }
    unsigned long long r = 1;
    for (unsigned long long d = 1; d <= k; ++d) {
        r *= n--;
        r /= d;
    }
    return r;
}

//https://stackoverflow.com/a/36714204
//Modified slighty to sort indices in ascending order.



int cmp(const void *a, const void *b)
{
    struct str *a1 = (struct str *)a;
    struct str *a2 = (struct str *)b;
    if ((*a1).value > (*a2).value)
        return 1;
    else if ((*a1).value < (*a2).value)
        return -1;
    else
        return 0;
}


void soft_decision_col_sort(double *soft_decisions,int *cols, int N){
    struct str *objects;
    objects=chk_alloc (N, sizeof *objects);
    for (int i = 0; i < N; i++)
    {
        objects[i].value = soft_decisions[i];
        objects[i].index = i;
    }
    //sort objects array according to value using qsort
    qsort(objects, N, sizeof(objects[0]), cmp);
    for (int i = 0; i < N; i++)
        cols[i]=objects[i].index;

    free(objects);

}


//https://stackoverflow.com/a/36714204
//Integer sort version Modified slighty to sort indices in ascending order.



int cmp_int(const void *a, const void *b)
{
    struct str_int *a1 = (struct str_int *)a;
    struct str_int *a2 = (struct str_int *)b;
    if ((*a1).value > (*a2).value)
        return 1;
    else if ((*a1).value < (*a2).value)
        return -1;
    else
        return 0;
}


void col_sort_int(int *integers,int *cols, int N){
    struct str_int *objects;
    objects=chk_alloc (N, sizeof *objects);
    for (int i = 0; i < N; i++)
    {
        objects[i].value = integers[i];
        objects[i].index = i;
    }
    //sort objects array according to value using qsort
    qsort(objects, N, sizeof(objects[0]), cmp_int);
    for (int i = 0; i < N; i++)
        cols[i]=objects[i].index;

    free(objects);

}