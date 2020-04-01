#include "cluster.h"
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
    
    int *arr = new int[n];
    int *centers = new int[t];
    int *dists = new int[t];
    srand((unsigned)time(NULL));
    for (size_t i = 0; i < n; i++) {
        arr[i] = (rand() % (n + 1));
    }
    sort(arr, arr + n);
    for (size_t i = 1; i < t + 1; i++) {
        centers[i - 1] = (2 * i - 1) * n / (2 * t);
        dists[i - 1] = 0;
    }
    omp_set_num_threads(t);
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;
    start = high_resolution_clock::now();
    cluster(n, t, arr, centers, dists);
    end = high_resolution_clock::now();
    duration_sec =
    std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    int Max = -1;
    int index = -1;
    for (size_t i = 0; i < t; i++) {
        if (i == 0) {
            Max = dists[i];
            index = i;
        } else {
            if (dists[i] > Max) {
                Max = dists[i];
                index = i;
            }
        }
    }
    cout << Max << endl;
    cout << index << endl;
    cout << duration_sec.count() << endl;
    delete[] arr;
    delete[] centers;
    delete[] dists;
    return 0;
}

