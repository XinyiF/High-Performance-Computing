#include "montecarlo.h"
#include <cstdlib>
#include <iostream>

using namespace std;
int montecarlo(const size_t n, const float *x, const float *y,
               const float radius) {
    size_t count = 0;
    size_t i = 0;
#pragma omp simd
    for (i = 0; i < n; i++) {
        if (x[i] * x[i] + y[i] * y[i] <= radius * radius) {
            count += 1;
        }
    }
    return (int)4 * count / n;
}
