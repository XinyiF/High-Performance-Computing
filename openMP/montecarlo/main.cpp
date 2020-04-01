#include "montecarlo.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <omp.h>
#include <ratio>
#include <string>
#include <time.h>

using std::cout;
using std::chrono::duration;
using std::chrono::high_resolution_clock;
using namespace std;

int main(int argc, const char *argv[]) {
    string N, T;
    if (argc > 1) {
        N = string(argv[1]);
        T = string(argv[2]);
    }
    size_t n = atoi(N.c_str());
    size_t t = atoi(T.c_str());
    float *x = new float[n];
    float *y = new float[n];
    const float r = 5.5;
    srand((unsigned)time(NULL));
    for (size_t i = 0; i < n; i++) {
        x[i] = rand() / (float)RAND_MAX * 2 * r - r;
        y[i] = rand() / (float)RAND_MAX * 2 * r - r;
    }
    omp_set_num_threads(t);
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;
    start = high_resolution_clock::now();
    int pi = montecarlo(n, x, y, r);
    end = high_resolution_clock::now();
    duration_sec =
    std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    cout << pi << endl;
    cout << duration_sec.count() << endl;
    delete[] x;
    delete[] y;
    return 0;
}
