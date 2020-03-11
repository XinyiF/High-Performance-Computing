#include "matmul.cuh"
#include <iostream>
using namespace std;
__global__ void matmul_kernel(const float *A, const float *B, float *C,
                              size_t n) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n * n) {
    size_t row = i / n;
    size_t col = i % n;
    C[i]=0;
    for (size_t r = 0; r < n; r++) {
      C[i] += A[row * n + r] * B[r * n + col];
    }
  }
}
void matmul(const float *A, const float *B, float *C, size_t n,
            unsigned int threads_per_block) {
  matmul_kernel<<<((n * n) + threads_per_block - 1) / threads_per_block,
                  threads_per_block>>>(A, B, C, n);
  cudaDeviceSynchronize();
}

