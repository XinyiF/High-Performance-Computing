#include "count.cuh"
#include <cstdlib>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <time.h>
using namespace std;

int main(int argc, const char *argv[]) {
  string N;
  if (argc > 1) {
    N = string(argv[1]);
  }
  int n = atoi(N.c_str());
  thrust::host_vector<int> H(n);
  srand((unsigned)time(NULL));
  for (int i = 0; i < n; i++) {
    H[i] = (rand() % 101);
  }
  thrust::device_vector<int> d_in = H;
  thrust::device_vector<int> values(n);
  thrust::device_vector<int> counts(n);
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  count(d_in, values, counts);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // Get the elapsed time in milliseconds
  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  cout << values[values.size() - 1] << endl;
  cout << counts[counts.size() - 1] << endl;
  cout << ms << endl;
  return 0;
}
