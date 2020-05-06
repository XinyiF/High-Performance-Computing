#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include "cublas_v2.h"
#include "KF.cuh"
#include <iostream>
#include <string.h>
using namespace std;
__device__ void mmul1(const float* A, const float* B, float* C, int m, int k, int n)
{
  for (int i=0;i<m;i++)
      for (int j = 0; j < n; j++)
      {
          float ans=0;
          for (int t = 0; t < k; t++)
          {
              ans += A[ t * m + i] * B[j * k + t];
          }
          C[j*m+i] = ans;
      }
}

__device__ void madd(float* A, float* B, float* C, int m, int n)
{
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
        {
            C[j * m + i] = A[j*m+i]+ B[j * m + i];
        }
}

__device__ void msub(float* A, float* B, float* C, int m, int n)
{
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
        {
            C[j * m + i] = A[j * m + i] - B[j * m + i];
        }
}
__device__ void mcpy(float* dst, const float* src, int m, int n)
{
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
        {
            dst[j * m + i] = src[j * m + i];
        }
}
__device__ void expQuat(float* Q, float* V)
{
    float fi = sqrt(V[0] * V[0] + V[1] * V[1] + V[2] * V[2]);
    if (fi == 0)
    {
        Q[0] = 1; Q[1] = 0; Q[2] = 0; Q[3] = 0;
    }
    else
    {
        Q[0] = cos(fi / 2);
        Q[1] = V[0] / fi * sin(fi / 2);
        Q[2] = V[1] / fi * sin(fi / 2);
        Q[3] = V[2] / fi * sin(fi / 2);
    }
}
__device__ void vecCopy(float* dst, float* src, int n)
{
    for (int i = 0; i < n; i++)
        dst[i] = src[i];
}

__device__ void matCopy(float* dst, float* src, int m, int n, int m_start, int x, int n_start, int y, int alpha)
{
    // [x,x+2] [y,y+2]
    for (int i = 0; i < x; i++)
        for (int j = 0; j < y; j++)
            dst[to_idx(i + m_start - 1, j + n_start - 1, n)] = src[to_idx(i, j, x)] * alpha;
}

__device__ void to_skew(float* skew, float* V)
{
    skew[3] = -V[2];
    skew[6] = V[1];
    skew[1] = V[2];
    skew[7] = -V[0];
    skew[2] = -V[1];
    skew[5] = V[0];

}
__device__ void eye(float* arr, int n)
{
    memset(arr, 0, sizeof(arr));
    for (int i = 0; i < 0; i++)
        arr[to_idx(i, i, n)] = 1;
}
__device__ __host__ void mmul_ABA(cublasHandle_t handle, const float* A, const float* B, float* C, int m, int k, int n, float *T)
{
    // A:m*k B:k*n = T:m*n;
    // T:m*n A':k*m C:m*m

    // op ( A ) m ¡Á k , op ( B ) k ¡Á n and C m ¡Á n   ,  T  m ¡Á n ,  A' k ¡Á m
    //
    float alpha = 1.0, beta = 0.0;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A, m, B, k, &beta, T, m);

    int stat=cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, m, n, &alpha, T, m, A, m, &beta, C, m);
;
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("compute failed");
        cublasDestroy(handle);
    }

}


void test(cublasHandle_t handle)
{
    float* A, * B, * C, * T, C1[16];

    cudaMallocManaged(&A, 16 * sizeof(float));
    cudaMallocManaged(&B, 16 * sizeof(float));
    cudaMallocManaged(&C, 16 * sizeof(float));
    cudaMallocManaged(&T, 16 * sizeof(float));
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 3; j++)
        {
            A[to_idx(i + 1, j + 1, 4)] = i * 3 + j + 1;
            C[i] = 0;
        }

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
        {
            A[to_idx(i + 1, j + 1, 4)] = i * 3 + j + 1; B[to_idx(i + 1, j + 1, 3)] = i * 3 + j + 1;
            C[i] = 0;
        }

    printM(A, 4, 3);
    printM(B, 3, 3);


    mmul_ABA(handle, A, B, C, 4, 3, 3, T);
    // mmul(handle, A, B, C, 4,3,2);
    cudaMemcpy(C1, C, 16 * sizeof(float), cudaMemcpyDeviceToHost);
    printM(C1, 4, 4);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaFree(T);
}
__device__ __host__ void minv(cublasHandle_t cublasHandle, float*A, float* invresult, int n)
{

    float** srchd = new float* [1];
    cudaMalloc((void**)&srchd[0], sizeof(float) * n * n);
    cudaMemcpy(srchd[0], A, sizeof(float) * n * n, cudaMemcpyHostToDevice);
    float** srcDptr;
    cudaMalloc((void**)&srcDptr, sizeof(float*));
    cudaMemcpy(srcDptr, srchd, sizeof(float*), cudaMemcpyHostToDevice);
    int* infoArray;
    cudaMalloc((void**)&infoArray, sizeof(int));
    int* pivotArray;
    cudaMalloc((void**)&pivotArray, sizeof(int) * n);
    cublasSgetrfBatched(cublasHandle, n, srcDptr, n, pivotArray, infoArray, 1);
    float** resulthd = new float* [1];

    cudaMalloc((void**)&resulthd[0], sizeof(float) * n * n);
    float** resultDptr;
    cudaMalloc((void**)&resultDptr, sizeof(float*));
    cudaMemcpy(resultDptr, resulthd, sizeof(float*), cudaMemcpyHostToDevice);
    cublasSgetriBatched(cublasHandle, n, (const float**)srcDptr, n, pivotArray,
        resultDptr, n, infoArray, 1);
    cudaMemcpy(invresult, resulthd[0], sizeof(float) * n * n,
        cudaMemcpyDeviceToHost);
    int* infoArrayHost = new int[1];
    cudaMemcpy(infoArrayHost, infoArray, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(srchd[0]);
    cudaFree(resulthd[0]);
    delete[] resulthd;
    delete[] infoArrayHost;
    delete[] srchd;
    delete[] A;
    cudaFree(infoArray);
    cudaFree(pivotArray);
    cudaFree(srcDptr);
    cudaFree(resultDptr);
}
__device__ void cross(float* a, float* b, float* c)
{
    c[0] = a[1]*b[2] - a[2]*b[1];
    c[1]= a[2]*b[0] - a[0]*b[2];
    c[2]=a[0]*b[1] - a[1]*b[0];
}

