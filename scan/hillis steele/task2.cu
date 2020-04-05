#include "scan.cuh"
#include <iostream>
using namespace std;

int main(int argc, const char *argv[]) {
  string N, Threads;
  if (argc > 1) {
    N = string(argv[1]);
  }
  unsigned int n = atoi(N.c_str());
  float *in = new float[n];
  float *out = new float[n];
  for (unsigned int i = 0; i < n; i++) {
    in[i] = 1;
    out[i] = 0;
  }
  unsigned int threads_per_block = 1024;
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  scan(in, out, n, threads_per_block);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // Get the elapsed time in milliseconds
  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  cout << out[n - 1] << endl;
  cout << ms << endl;
  delete[] out;
  delete[] in;

  return 0;
}
