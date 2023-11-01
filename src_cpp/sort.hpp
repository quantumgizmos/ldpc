#ifndef SORT_H
#define SORT_H

namespace ldpc::sort{


struct str
{
    double value;
    int index;
};

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>

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


void soft_decision_col_sort(std::vector<double>& soft_decisions, std::vector<int>& cols, int N){
    struct str *objects;
    objects=new str[N];
    for (int i = 0; i < N; i++)
    {
        objects[i].value = soft_decisions[i];
        objects[i].index = i;
    }
    //sort objects array according to value using qsort
    qsort(objects, N, sizeof(objects[0]), cmp);
    for (int i = 0; i < N; i++)
        cols[i]=objects[i].index;

    delete[] objects;

}


//https://stackoverflow.com/a/36714204
//Integer sort version Modified slighty to sort indices in ascending order.

} //end namespace ldpc::sort


#endif