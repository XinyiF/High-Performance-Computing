#include "stencil.cuh"
#include <iostream>
using namespace std;

int main(int argc, const char *argv[]) {
  string N, r, Threads_per_block;
  if (argc > 1) {
    N = string(argv[1]);
    r = string(argv[2]);
    Threads_per_block = string(argv[3]);
  }
  unsigned int n = atoi(N.c_str());
  unsigned int R = atoi(r.c_str());
  unsigned int threads_per_block = atoi(Threads_per_block.c_str());
  float *image, *mask, *output;
  cudaMallocManaged((float **)&image, n * sizeof(float));
  cudaMallocManaged((float **)&output, n * sizeof(float));
  cudaMallocManaged((float **)&mask, (2 * R + 1) * sizeof(float));
  for (unsigned int i = 0; i < n; i++) {
    image[i] = i;
  }
  for (unsigned int i = 0; i < 2 * R + 1; i++) {
    mask[i] = 1;
  }
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  stencil(image, mask, output, n, R, threads_per_block);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // Get the elapsed time in milliseconds
  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  for (unsigned int i = 0; i < n; i++) {
    cout << output[i] << endl;
  }
  cout << "runtime is " << ms << endl;
  cudaFree(image);
  cudaFree(mask);
  cudaFree(output);

  return 0;
}
