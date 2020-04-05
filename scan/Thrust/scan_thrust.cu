#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
using namespace std;

int main(int argc, const char *argv[]) {
  string N;
  if (argc > 1) {
    N = string(argv[1]);
  }
  unsigned int n = atoi(N.c_str());
  thrust::host_vector<float> H(n);
  for (unsigned int i = 0; i < n; i++) {
    H[i] = 1;
  }
  thrust::device_vector<float> D = H;
  thrust::device_vector<float> res(n);
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  thrust::exclusive_scan(D.begin(), D.end(), res.begin());
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // Get the elapsed time in milliseconds
  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  cout << res[n - 1] << endl;
  cout << ms << endl;
  return 0;
}