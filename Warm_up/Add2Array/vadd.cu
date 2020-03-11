#include <iostream>
__global__ void vadd(const float *a, float *b, unsigned int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    b[i] = a[i] + b[i];
  }
}
