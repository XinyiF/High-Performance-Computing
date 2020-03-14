#include "stencil.cuh"
#include <iostream>
using namespace std;

__global__ void stencil_kernel(const float *image, const float *mask,
                               float *output, unsigned int n, unsigned int R) {
  extern __shared__ float s[];
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int len = blockDim.x;
  float *s_image = &s[0];
  float *s_output = &s[len + 2 * R];
  float *s_mask = &s[2 * len + 2 * R];

  if (threadIdx.x < len / 2 + R) {
    if (threadIdx.x >= R)
      s_image[threadIdx.x] = image[blockIdx.x * blockDim.x + threadIdx.x - R];
    else
      s_image[threadIdx.x] = 0;
    if (len / 2 + R + threadIdx.x < R + blockDim.x)
      s_image[len / 2 + R + threadIdx.x] =
          image[blockIdx.x * blockDim.x + len / 2 + R + threadIdx.x - R];
    else
      s_image[len / 2 + R + threadIdx.x] = 0;
  }

  if (threadIdx.x < (2 * R + 1))
    s_mask[threadIdx.x] = mask[threadIdx.x];
  s_output[threadIdx.x] = 0;
  __syncthreads();
  if (i < n)
    for (int j = (int)(-R); j <= int(R); j++) {
      s_output[threadIdx.x] += s_image[threadIdx.x + j + R] * s_mask[R + j];
    }
  output[i] = s_output[threadIdx.x];
}

__host__ void stencil(const float *image, const float *mask, float *output,
                      unsigned int n, unsigned int R,
                      unsigned int threads_per_block) {

  stencil_kernel<<<(n + threads_per_block - 1) / threads_per_block,
                   threads_per_block,
                   (threads_per_block * 2 + 4 * R + 1) * sizeof(float)>>>(
      image, mask, output, n, R);
  cudaDeviceSynchronize();
}
