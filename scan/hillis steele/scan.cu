#include "scan.cuh"
#include <iostream>
#include <math.h>
using namespace std;
__global__ void scan_kernel(float *g_odata, float *g_idata, unsigned int n,
                            float *last_ele) {
  extern volatile __shared__ float data[];
  int thid = threadIdx.x;
  int g_tid = blockIdx.x * blockDim.x + threadIdx.x;
  int pout = 0, pin = 1;
  data[thid] = (thid == 0) ? 0 : g_idata[g_tid - 1];
  __syncthreads();
  for (int offset = 1; offset < blockDim.x; offset *= 2) {
    pout = 1 - pout;
    pin = 1 - pout;
    if (thid >= offset)
      data[pout * blockDim.x + thid] = data[pin * blockDim.x + thid] +
                                       data[pin * blockDim.x + thid - offset];
    else
      data[pout * blockDim.x + thid] = data[pin * blockDim.x + thid];
    __syncthreads();
  }
  if (g_tid < n) {
    g_odata[g_tid] = data[pout * blockDim.x + thid];
    if (thid == blockDim.x - 1) {
      last_ele[blockIdx.x] = data[thid] + g_idata[g_tid];
    }
  }
}
__global__ void helper_kernel(float *g_odata, unsigned int n,
                              unsigned int threads_per_block, float *last_ele) {
  extern __shared__ float temp[];
  int s_tid = threadIdx.x;
  int g_tid = blockIdx.x * blockDim.x + s_tid;
  if (g_tid < n) {
    temp[s_tid] = g_odata[g_tid];
  } else {
    temp[s_tid] = 0;
  }
  __syncthreads();
  if (blockIdx.x > 0) {
    for (int i = 0; i < blockIdx.x; i++) {
      temp[s_tid] += last_ele[i];
    }
    __syncthreads();
  }
  if (g_tid < n) {
    g_odata[g_tid] = temp[s_tid];
  }
}

__host__ void scan(const float *in, float *out, unsigned int n,
                   unsigned int threads_per_block) {
  float *g_odata, *g_idata, *last_ele;
  cudaMalloc((void **)&g_idata, n * sizeof(float));
  cudaMemcpy(g_idata, in, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&g_odata, n * sizeof(float));
  cudaMalloc((void **)&last_ele,
             (n + threads_per_block - 1) / threads_per_block * sizeof(float));
  scan_kernel<<<(n + threads_per_block - 1) / threads_per_block,
                threads_per_block, 2 * threads_per_block * sizeof(float)>>>(
      g_odata, g_idata, n, last_ele);
  if (n > threads_per_block) {
    helper_kernel<<<(n + threads_per_block - 1) / threads_per_block,
                    threads_per_block, threads_per_block * sizeof(float)>>>(
        g_odata, n, threads_per_block, last_ele);
  }
  cudaMemcpy(out, g_odata, n * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(g_idata);
  cudaFree(g_odata);
  cudaFree(last_ele);
  cudaDeviceSynchronize();
}
