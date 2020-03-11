#include "matmul.cuh"
#include <iostream>
using namespace std;
int main(int argc, const char *argv[]) {
  string type, type1;
  if (argc > 1) {
    type = string(argv[1]);
    type1 = string(argv[2]);
  }
  size_t n = atoi(type.c_str());
  unsigned int threads_per_block = atoi(type1.c_str());
  float *A, *B, *C;
  cudaMallocManaged((float **)&A, n * n * sizeof(float));
  cudaMallocManaged((float **)&B, n * n * sizeof(float));
  cudaMallocManaged((float **)&C, n * n * sizeof(float));
  for (unsigned int i = 0; i < n * n; i++) {
    A[i] = 0.5;
    B[i] = 0.5;
  }
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  matmul(A, B, C, n, threads_per_block);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // Get the elapsed time in milliseconds
  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  cout << C[n * n - 1] << endl;
  cout << ms << endl;
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);

  return 0;
}

