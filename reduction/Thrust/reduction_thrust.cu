#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
using namespace std;

int main(int argc, const char *argv[]) {
  string N;
  if (argc > 1) {
    N = string(argv[1]);
  }
  unsigned int n = atoi(N.c_str());
  thrust::host_vector<int> H(n);
  for (unsigned int i = 0; i < n; i++) {
    H[i] = 1;
  }
  thrust::device_vector<int> D = H;
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  int sum = thrust::reduce(D.begin(), D.end());
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // Get the elapsed time in milliseconds
  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  cout << sum << endl;
  cout << ms << endl;
  return 0;
}