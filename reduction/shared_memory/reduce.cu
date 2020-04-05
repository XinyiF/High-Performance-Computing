#include "reduce.cuh"
#include <iostream>
using namespace std;
__global__ void reduce_kernel(const int *g_idata, int *g_odata,
                              unsigned int n) {
  extern __shared__ int data[];
  unsigned int shared_tid = threadIdx.x;
  unsigned int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_tid < n) {
    data[shared_tid] = g_idata[global_tid];
  } else {
    data[shared_tid] = 0;
  }
  __syncthreads();
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (shared_tid < s) {
      data[shared_tid] += data[shared_tid + s];
    }
    __syncthreads();
  }
  g_odata[blockIdx.x] = data[0];
}

__host__ int reduce(const int *arr, unsigned int N,
                    unsigned int threads_per_block) {
  int *g_odata, *g_idata;
  int res = 0;
  int size = N;
  cudaMalloc((void **)&g_idata, size * sizeof(int));
  cudaMemcpy(g_idata, arr, size * sizeof(int), cudaMemcpyHostToDevice);
  while (size > 1) {
    cudaMalloc((void **)&g_odata, (size + threads_per_block - 1) /
                                      threads_per_block * sizeof(int));
    reduce_kernel<<<(size + threads_per_block - 1) / threads_per_block,
                    threads_per_block, threads_per_block * sizeof(int)>>>(
        g_idata, g_odata, size);

    size = (size + threads_per_block - 1) / threads_per_block;
    int *sum = new int[size];
    cudaMemcpy(sum, g_odata, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(g_odata);
    cudaFree(g_idata);
    cudaMalloc((void **)&g_idata, size * sizeof(int));
    cudaMemcpy(g_idata, sum, size * sizeof(int), cudaMemcpyHostToDevice);
    res = sum[0];
    free(sum);
  }
  return res;
  cudaDeviceSynchronize();
}