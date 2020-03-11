#include "vadd.cuh"
#include <iostream>
using namespace std;
int main(int argc, const char *argv[]) {
  string type;
  if (argc > 1) {
    type = string(argv[1]);
  }
  unsigned int n = atoi(type.c_str());
  float *dA, *dB;
  cudaMallocManaged((float **)&dA, n * sizeof(float));
  cudaMallocManaged((float **)&dB, n * sizeof(float));
  for (unsigned int i = 0; i < n; i++) {
    dA[i] = 0.5;
    dB[i] = 0.5;
  }

  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  vadd<<<(n + 1023) / 1024, 1024>>>(dA, dB, n);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // Get the elapsed time in milliseconds
  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  std::printf("%f\n", ms / 1000);
  std::printf("%f\n", dB[0]);
  std::printf("%f\n", dB[n - 1]);
  cudaFree(dA);
  cudaFree(dB);

  return 0;
}
