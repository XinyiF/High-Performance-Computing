#include "mmul.h"
#include <cublas_v2.h>
#include <iostream>
using namespace std;
int main(int argc, const char *argv[]) {
  string N, r;
  if (argc > 1) {
    N = string(argv[1]);
    r = string(argv[2]);
  }
  int n = atoi(N.c_str());
  int n_test = atoi(r.c_str());
  float *A, *B, *C;
  cudaMallocManaged((float **)&A, n * n * sizeof(float));
  cudaMallocManaged((float **)&B, n * n * sizeof(float));
  cudaMallocManaged((float **)&C, n * n * sizeof(float));
  for (unsigned int i = 0; i < n * n; i++) {
    A[i] = i;
    B[i] = i;
    C[i] = 0;
  }

  // Create a handle for CUBLAS
  cublasHandle_t handle;
  cublasCreate(&handle);

  float avg = 0;
  for (int i = 0; i < n_test; i++) {
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    mmul(handle, A, B, C, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    avg += ms;
  }
  cout << avg / n_test << endl;

  // Destroy the handle
  cublasDestroy(handle);

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);

  return 0;
}