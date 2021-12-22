#ifndef SORT_H
#define SORT_H

struct str
{
    double value;
    int index;
};

struct str_int
{
    int value;
    int index;
};

unsigned long long ncr(int n, int k);
int cmp(const void *a, const void *b);
void soft_decision_col_sort(double *soft_decisions,int *cols, int N);
int cmp_int(const void *a, const void *b);
void col_sort_int(int *integers,int *cols, int N);

#endif