#include "vadd.cuh"
#include <iostream>
using namespace std;

int main(int argc, const char *argv[]) {
  string type;
  if (argc > 1) {
    type = string(argv[1]);
  }
  unsigned int n = atoi(type.c_str());
  float *a = new float[n];
  float *b = new float[n];
  float *dA, *dB;
  cudaMalloc((void **)&dA, n * sizeof(float));
  cudaMalloc((void **)&dB, n * sizeof(float));
  for (unsigned int i = 0; i < n; i++) {
    a[i] = 0.5;
    b[i] = 0.5;
  }
  cudaMemcpy(dA, a, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, b, n * sizeof(float), cudaMemcpyHostToDevice);

  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  vadd<<<(n + 511) / 512, 512>>>(dA, dB, n);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // Get the elapsed time in milliseconds
  float ms;
  cudaEventElapsedTime(&ms, start, stop);

  cudaMemcpy(a, dA, n * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(b, dB, n * sizeof(float), cudaMemcpyDeviceToHost);
  printf("%f\n", ms / 1000);
  printf("%f\n", b[0]);
  printf("%f\n", b[n - 1]);

  delete[] a;
  delete[] b;
  cudaFree(dA);
  cudaFree(dB);

  return 0;
}
