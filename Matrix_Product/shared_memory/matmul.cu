#include "matmul.cuh"
#include <iostream>
using namespace std;
__global__ void matmul_kernel(const float *A, const float *B, float *C,
                              unsigned int n) {
  // tile size is blockDim.x*blockDim.y
  unsigned int global_tid = (blockIdx.y * blockDim.y + threadIdx.y) * n +
                            blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int block_tid = threadIdx.y * blockDim.x + threadIdx.x;
  unsigned int a_begin =
      blockIdx.y * blockDim.y * n; // the begin element of sub A
  unsigned int a_end = a_begin + n - 1;
  unsigned int b_begin = blockIdx.x * blockDim.x;
  float csub = 0;
  extern __shared__ float data[];
  for (unsigned int a = a_begin, b = b_begin; a < a_end;
       a += blockDim.x, b += blockDim.y * n) {
    // load subA and subB, each thread load an element of two sub matrix
    // if the index out of boundary, fill with 0
    if (a + threadIdx.y * n + threadIdx.x < n * n) {
      data[threadIdx.y * blockDim.y + threadIdx.x] =
          A[a + threadIdx.y * n + threadIdx.x];
    } else {
      data[threadIdx.y * blockDim.y + threadIdx.x] = 0;
    }
    if (b + threadIdx.y * n + threadIdx.x < n * n) {
      data[blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x] =
          B[b + threadIdx.y * n + threadIdx.x];
    } else {
      data[blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x] =
          0;
    }
    __syncthreads();
    // compute element of C of current thread
    for (unsigned int j = 0; j < blockDim.x; j++) {
      csub += data[threadIdx.y * blockDim.y + j] *
              data[blockDim.x * blockDim.y + j * blockDim.y + threadIdx.x];
    }
    __syncthreads();
  }
  if (global_tid < n * n) {
    C[global_tid] = csub;
  }
}

__host__ void matmul(const float *A, const float *B, float *C, unsigned int n,
                     unsigned int block_dim) {

  dim3 dimBlock(block_dim, block_dim);
  dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x,
               (n + dimBlock.y - 1) / dimBlock.y);
  matmul_kernel<<<dimGrid, dimBlock,
                  (2 * block_dim * block_dim) * sizeof(float)>>>(A, B, C, n);
  cudaDeviceSynchronize();
}
