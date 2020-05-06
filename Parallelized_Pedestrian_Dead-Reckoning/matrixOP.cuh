#ifndef MOP_H
#define MOP_H
#include "KF.cuh"
#include <cublas_v2.h>

__device__ __host__ void mmul1(const float* A, const float* B, float* C, int m, int k, int n);
__device__ __host__ void mmul_ABA(cublasHandle_t handle, const float* A, const float* B, float* C, int m, int k, int n, float* T);
__device__ __host__ void madd(float* A, float* B, float* C, int m, int n);
__device__ __host__ void mcpy(float* dst, const float* src, int m, int n);
__device__ __host__ void minv(float* dst, const float* src, int m, int n);
__device__ __host__ void msub(float* A, float* B, float* C, int m, int n);
__device__ __host__ void eye(float* arr, int n);
__device__ void expQuat(float* Q, float* V);
__device__ void vecCopy(float* dst, float* src, int n);
__device__ void matCopy(float* dst, float* src, int m, int n, int m_start, int x, int n_start, int y, int alpha);
__device__ void to_skew(float* skew, float* V);
__device__ void cross(float* a, float* b, float* c);
#endif
