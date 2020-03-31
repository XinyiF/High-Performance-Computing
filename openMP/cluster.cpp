#include <cstdlib>
#include <iostream>
#include "cluster.h"

void cluster(const size_t n, const size_t t, const int *arr, const int *centers, int *dists) {
	size_t i;
    #pragma omp parallel num_threads(t)
    {
        unsigned int tid = omp_get_thread_num();
        #pragma omp for private(i) reduction(+:dists[tid])
        for (i = 0; i < n; i++) {
            dists[tid] += abs(arr[i] - centers[tid]);
        }
    }
}
