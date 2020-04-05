#include "mmul.h"
#include <cublas_v2.h>
#include <iostream>

void mmul(cublasHandle_t handle, const float *A, const float *B, float *C,
          int n) {
  int lda = n, ldb = n, ldc = n;
  const float alf = 1;
  const float bet = 0;
  const float *alpha = &alf;
  const float *beta = &bet;

  // Do the actual multiplication
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, alpha, A, lda, B, ldb,
              beta, C, ldc);
  cudaDeviceSynchronize();
}