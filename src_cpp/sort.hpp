#ifndef SORT_H
#define SORT_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

namespace ldpc::sort {
    struct str {
        double value;
        int index;
    };

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



    int cmp(const void *a, const void *b) {
        auto *a1 = (struct str *) a;
        auto *a2 = (struct str *) b;
        if ((*a1).value > (*a2).value) {
            return 1;
        }
        if ((*a1).value < (*a2).value) {
            return -1;
        }
        return 0;
    }


    void soft_decision_col_sort(std::vector<double> &soft_decisions, std::vector<int> &cols, int N) {
        struct str *objects;
        objects = new str[N];
        for (int i = 0; i < N; i++) {
            objects[i].value = soft_decisions[i];
            objects[i].index = i;
        }
        //sort objects array according to value using qsort
        qsort(objects, N, sizeof(objects[0]), cmp);
        for (int i = 0; i < N; i++)
            cols[i] = objects[i].index;

        delete[] objects;

    }


//https://stackoverflow.com/a/36714204
//Integer sort version Modified slighty to sort indices in ascending order.

} //end namespace ldpc::sort


#endif