//Yisen 5/2020
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include "cublas_v2.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdio.h>
#include <cmath>
#include "KF.cuh"
#include "matrixOP.cuh"

using namespace std;
#define dt 0.005

//initialize matrix in Kalman filter
__host__ void init_KF(Filter* F, Hx* H, cublasHandle_t handle)
{
	F->PKF[to_idx(1, 1, 15)] = 0.1;
	F->PKF[to_idx(2, 2, 15)] = 0.1;
	F->PKF[to_idx(3, 3, 15)] = 0.1;
	F->PKF[to_idx(9, 9, 15)] = 0.5;

	F->QKF[to_idx(1, 1, 12)] = ((1e-3 * dt) * (1e-3 * dt) + (4.5e-3) * (4.5e-3) * dt);
	F->QKF[to_idx(2, 2, 12)] = ((1e-3 * dt) * (1e-3 * dt) + (4.5e-3) * (4.5e-3) * dt);
	F->QKF[to_idx(3, 3, 12)] = ((1e-3 * dt) * (1e-3 * dt) + (4.5e-3) * (4.5e-3) * dt);
	F->QKF[to_idx(4, 4, 12)] = (pow( (2.5*1e-3 * pi / 180 * dt), 2) + pow((7.4e-3 * pi / 180), 2) * dt) * 20;
	F->QKF[to_idx(5, 5, 12)] = (pow( (2.5*1e-3 * pi / 180 * dt), 2) + pow((7.4e-3 * pi / 180), 2) * dt) * 20;
	F->QKF[to_idx(6, 6, 12)] = (pow( (2.5*1e-3 * pi / 180 * dt), 2) + pow((7.4e-3 * pi / 180), 2) * dt) * 20;
	F->QKF[to_idx(7, 7, 12)] = pow((6e-5), 2) * dt;
	F->QKF[to_idx(8, 8, 12)] = pow((6e-5), 2) * dt;
	F->QKF[to_idx(9, 9, 12)] = pow((6e-5), 2) * dt;
	F->QKF[to_idx(10, 10, 12)] = pow((3e-4 * pi / 180), 2) * dt;
	F->QKF[to_idx(11, 11, 12)] = pow((3e-4 * pi / 180), 2) * dt;
	F->QKF[to_idx(12, 12, 12)] = pow((3e-4 * pi / 180), 2) * dt;

	F->RKF[ZUPT] = 1e-3;
	F->RKF[ZARU] = 5e-5;
	F->RKF[GPS] = 1e-1;
	F->RKF[ALTITUDE] = 1e-1;
	F->RKF[ELE] = 1e-1;

	F->Fi[to_idx(4, 1, 15)] = 1;
	F->Fi[to_idx(5, 2, 15)] = 1;
	F->Fi[to_idx(6, 3, 15)] = 1;
	F->Fi[to_idx(7, 4, 15)] = 1;
	F->Fi[to_idx(8, 5, 15)] = 1;
	F->Fi[to_idx(9, 6, 15)] = 1;
	F->Fi[to_idx(10, 7, 15)] = 1;
	F->Fi[to_idx(11, 8, 15)] = 1;
	F->Fi[to_idx(12, 9, 15)] = 1;
	F->Fi[to_idx(13, 10, 15)] = 1;
	F->Fi[to_idx(14, 11, 15)] = 1;
	F->Fi[to_idx(15, 12, 15)] = 1;

	float* T,*T1;
	cudaMallocManaged(&T, 15 * 15 * sizeof(float));
	cudaMallocManaged(&T1, 15 * 15 * sizeof(float));
	mmul_ABA(handle, F->Fi, F->QKF,T1 , 15, 12, 12, T);	//// F->QKFT= Fi*QKFT*Fi';
	cudaFree(T);
	cudaMemcpy(F->QKFT, T1, 15 * 15 * sizeof(float), cudaMemcpyDefault);
	
	int H_idx[16] = {0, 1 , 3 , 3 , 3 , 4 , 6 , 6 , 3 , 3 , 6 , 5 , 6 , 6 , 9 , 8};
	for (int i = 1; i < 16; i++)
	{
		H->H_size[i * 2] = H_idx[i];
		H->H_size[i * 2+1] = 16;
	}
	H->H_s[1] = 0;
	for (int i = 2; i < 16; i++)
	{
		H->H_s[i] = H->H_s[i-1] +H_idx[i-1]*16;
	}
	///////////////---------------------------------/////////////////////////////////////
	if (1 == 1)
	{
		float* Hx;
		Hx = &(H->Hx[(int)H->H_s[1]]);
		Hx[to_idx(1, 3, 1)] = 1;

		Hx = &(H->Hx[(int)H->H_s[2]]);
		Hx[to_idx(1, 1, 3)] = 1; Hx[to_idx(2, 2, 3)] = 1; Hx[to_idx(3, 3, 3)] = 1;

		Hx = &(H->Hx[(int)H->H_s[3]]);
		Hx[to_idx(1, 1, 3)] = 1; Hx[to_idx(2, 2, 3)] = 1; Hx[to_idx(3, 3, 3)] = 1;

		Hx = &(H->Hx[(int)H->H_s[4]]);
		Hx[to_idx(1, 14, 3)] = 1; Hx[to_idx(2, 15, 3)] = 1; Hx[to_idx(3, 16, 3)] = 1;

		Hx = &(H->Hx[(int)H->H_s[5]]);
		Hx[to_idx(1, 3, 4)] = 1; Hx[to_idx(2, 14, 4)] = 1; Hx[to_idx(3, 15, 4)] = 1; Hx[to_idx(4, 16, 4)] = 1;

		Hx = &(H->Hx[(int)H->H_s[6]]);
		Hx[to_idx(1, 1, 6)] = 1; Hx[to_idx(2, 2, 6)] = 1; Hx[to_idx(3, 3, 6)] = 1; 
		Hx[to_idx(4, 14, 6)] = 1; Hx[to_idx(5, 15, 6)] = 1; Hx[to_idx(6, 16, 6)] = 1;

		Hx = &(H->Hx[(int)H->H_s[7]]);
		Hx[to_idx(1, 1, 6)] = 1; Hx[to_idx(2, 2, 6)] = 1; Hx[to_idx(3, 3, 6)] = 1;
		Hx[to_idx(4, 14, 6)] = 1; Hx[to_idx(5, 15, 6)] = 1; Hx[to_idx(6, 16, 6)] = 1;

		Hx = &(H->Hx[(int)H->H_s[8]]);
		Hx[to_idx(1, 4, 3)] = 1; Hx[to_idx(2, 5, 3)] = 1; Hx[to_idx(3, 6, 3)] = 1;

		Hx = &(H->Hx[(int)H->H_s[9]]);
		Hx[to_idx(1, 3, 3)] = 1; Hx[to_idx(2, 4, 3)] = 1; Hx[to_idx(3, 5, 3)] = 1;

		Hx = &(H->Hx[(int)H->H_s[10]]);
		Hx[to_idx(1, 1, 6)] = 1; Hx[to_idx(2, 2, 6)] = 1; Hx[to_idx(3, 3, 6)] = 1; 
		Hx[to_idx(4, 4, 6)] = 1; Hx[to_idx(5, 5, 6)] = 1; Hx[to_idx(6, 6, 6)] = 1;

		Hx = &(H->Hx[(int)H->H_s[11]]);
		Hx[to_idx(1, 1, 5)] = 1; Hx[to_idx(2, 2, 5)] = 1; Hx[to_idx(3, 3, 5)] = 1; 
		Hx[to_idx(4, 4, 5)] = 1; Hx[to_idx(5, 5, 5)] = 1;

		Hx = &(H->Hx[(int)H->H_s[12]]);
		Hx[to_idx(1, 4, 6)] = 1; Hx[to_idx(2, 5, 6)] = 1; Hx[to_idx(3, 6, 6)] = 1; 
		Hx[to_idx(4, 14, 6)] = 1; Hx[to_idx(5, 15, 6)] = 1; Hx[to_idx(6, 16, 6)] = 1;

		Hx = &(H->Hx[(int)H->H_s[13]]);
		Hx[to_idx(1, 3, 6)] = 1; Hx[to_idx(2, 4, 6)] = 1; Hx[to_idx(3, 5, 6)] = 1;
		Hx[to_idx(4, 14, 6)] = 1; Hx[to_idx(5, 15, 6)] = 1; Hx[to_idx(6, 16, 6)] = 1;

		Hx = &(H->Hx[(int)H->H_s[14]]);
		Hx[to_idx(1, 1, 9)] = 1; Hx[to_idx(2, 2, 9)] = 1; Hx[to_idx(3, 3, 9)] = 1; 
		Hx[to_idx(4, 4, 9)] = 1; Hx[to_idx(5, 5, 9)] = 1; Hx[to_idx(6, 6, 9)] = 1;
		Hx[to_idx(7, 14, 9)] = 1; Hx[to_idx(8, 15, 9)] = 1; Hx[to_idx(9, 16, 9)] = 1;

		Hx = &(H->Hx[(int)H->H_s[15]]);
		Hx[to_idx(1, 1, 8)] = 1; Hx[to_idx(2, 2, 8)] = 1; Hx[to_idx(3, 3, 8)] = 1; 
		Hx[to_idx(4, 4, 8)] = 1; Hx[to_idx(5, 5, 8)] = 1;
		Hx[to_idx(6, 14, 8)] = 1; Hx[to_idx(7, 15, 8)] = 1; Hx[to_idx(8, 16, 8)] = 1;
	}

	int M_idx[16] = { 0,1 , 3 , 3 , 3 , 4 , 6 , 6 , 3 , 3 , 6 , 5 , 6 , 6 , 9 , 8 };
	for (int i = 1; i < 16; i++)
	{
		H->M_size[i * 2] = M_idx[i];
		H->M_size[i * 2 + 1] = M_idx[i];
	}
	H->M_s[1] = 0;
	for (int i = 2; i < 16; i++)
	{
		H->M_s[i] = H->M_s[i - 1] + M_idx[i - 1]* M_idx[i - 1];
	}
	if (1 == 1)
	{
		float* Mx;
		Mx = &(H->Mx[(int)H->M_s[1]]);
		Mx[to_idx(1, 1, 1)] = 0.1;

		Mx = &(H->Mx[(int)H->M_s[2]]);
		Mx[to_idx(1, 1, 3)] = 0.1; Mx[to_idx(2, 2,3)] = 0.1; Mx[to_idx(3, 3, 3)] = 0.1;

		Mx = &(H->Mx[(int)H->M_s[3]]);
		Mx[to_idx(1, 1, 3)] = 0.1; Mx[to_idx(2, 2,3)] = 0.1; Mx[to_idx(3, 3,3)] = 0.1;

		Mx = &(H->Mx[(int)H->M_s[4]]);
		Mx[to_idx(1, 1, 3)] = 5e-5; Mx[to_idx(2, 2,3)] = 5e-5; Mx[to_idx(3, 3, 3)] = 5e-5;

		Mx = &(H->Mx[(int)H->M_s[5]]);
		Mx[to_idx(1, 1, 4)] = 0.1; Mx[to_idx(2, 2, 4)] = 5e-5; Mx[to_idx(3, 3, 4)] = 5e-5; Mx[to_idx(4, 4, 4)] = 5e-5;

		Mx = &(H->Mx[(int)H->M_s[6]]);
		Mx[to_idx(1, 1, 6)] = 0.1; Mx[to_idx(2, 2,6)] = 0.1; Mx[to_idx(3, 3, 6)] = 0.1; 
		Mx[to_idx(4, 4, 6)] = 5e-5; Mx[to_idx(5, 5, 6)] = 5e-5; Mx[to_idx(6, 6, 6)] = 5e-5;
		
		Mx = &(H->Mx[(int)H->M_s[7]]);
		Mx[to_idx(1, 1, 6)] = 0.1; Mx[to_idx(2, 2, 6)] = 0.1; Mx[to_idx(3, 3, 6)] = 0.1;
		Mx[to_idx(4, 4, 6)] = 5e-5; Mx[to_idx(5, 5, 6)] = 5e-5; Mx[to_idx(6, 6, 6)] = 5e-5;
		
		Mx = &(H->Mx[(int)H->M_s[8]]);
		Mx[to_idx(1, 1, 3)] = 1e-3; Mx[to_idx(2, 2, 3)] = 1e-3; Mx[to_idx(3, 3, 3)] = 1e-3;

		Mx = &(H->Mx[(int)H->M_s[9]]);
		Mx[to_idx(1, 1, 3)] = 1e-1; Mx[to_idx(2, 2, 3)] = 1e-3; Mx[to_idx(3, 3, 3)] = 1e-3;

		Mx = &(H->Mx[(int)H->M_s[10]]);
		Mx[to_idx(1, 1, 6)] = 0.1; Mx[to_idx(2, 2, 6)] = 0.1; Mx[to_idx(3, 3, 6)] = 0.1;
		Mx[to_idx(4, 4, 6)] = 1e-3; Mx[to_idx(5, 5, 6)] = 1e-3; Mx[to_idx(6, 6, 6)] = 1e-3;

		Mx = &(H->Mx[(int)H->M_s[11]]);
		Mx[to_idx(1, 1, 5)] = 0.1; Mx[to_idx(2, 2, 5)] = 0.1; Mx[to_idx(3, 3, 5)] = 0.1;
		Mx[to_idx(4, 4, 5)] = 1e-3; Mx[to_idx(5, 5, 5)] = 1e-3; 
		
		Mx = &(H->Mx[(int)H->M_s[12]]);
		Mx[to_idx(1, 1, 6)] = 1e-3; Mx[to_idx(2, 2, 6)] = 1e-3; Mx[to_idx(3, 3, 6)] = 1e-3;
		Mx[to_idx(4, 4, 6)] = 5e-5; Mx[to_idx(5, 5, 6)] = 5e-5; Mx[to_idx(6, 6, 6)] = 5e-5;

		Mx = &(H->Mx[(int)H->M_s[13]]);
		Mx[to_idx(1, 1, 6)] = 0.1; Mx[to_idx(2, 2, 6)] = 1e-3; Mx[to_idx(3, 3, 6)] = 1e-3;
		Mx[to_idx(4, 4, 6)] = 5e-5; Mx[to_idx(5, 5, 6)] = 5e-5; Mx[to_idx(6, 6, 6)] = 5e-5;
		
		Mx = &(H->Mx[(int)H->M_s[14]]);
		Mx[to_idx(1, 1, 9)] = 0.1; Mx[to_idx(2, 2, 9)] = 0.1; Mx[to_idx(3, 3, 9)] = 0.1;
		Mx[to_idx(4, 4, 9)] = 1e-3; Mx[to_idx(5, 5, 9)] = 1e-3; Mx[to_idx(6, 6, 9)] = 1e-3;
		Mx[to_idx(7, 7, 9)] = 5e-5; Mx[to_idx(8, 8, 9)] = 5e-5; Mx[to_idx(9, 9, 9)] = 5e-5;

		Mx = &(H->Mx[(int)H->M_s[15]]);
		Mx[to_idx(1, 1, 8)] = 0.1; Mx[to_idx(2, 2, 8)] = 0.1; Mx[to_idx(3, 3, 8)] = 0.1;
		Mx[to_idx(4, 4, 8)] = 1e-3; Mx[to_idx(5, 5, 8)] = 1e-3;
		Mx[to_idx(6, 6, 8)] = 5e-5; Mx[to_idx(7, 7, 8)] = 5e-5; Mx[to_idx(8, 8, 8)] = 5e-5;
	}
}

__device__ void integrate(float* acc, float* gyro, float* P,
	float* V, float* Q, float* ABiasold, float* GBiasold)
{
	float RM[9];
	float a = Q[0]; float b = Q[1]; float c = Q[2]; float d = Q[3];
// Quaternion to Rotation matrix
	RM[0] = a * a + b * b - c * c - d * d;
	RM[3] = 2 * b * c - 2 * a * d;
	RM[6] = 2 * b * d + 2 * a * c;
	RM[1] = 2 * b * c + 2 * a * d;
	RM[4] = a * a - b * b + c * c - d * d;
	RM[7] = 2 * c * d - 2 * a * b;
	RM[2] = 2 * b * d - 2 * a * c;
	RM[5] = 2 * c * d + 2 * a * b;
	RM[8] = a * a - b * b - c * c + d * d;
	float vec1[3], vec2[3];
	vec1[0] = acc[0] - ABiasold[0];
	vec1[1] = acc[1] - ABiasold[1];
	vec1[2] = acc[2] - ABiasold[2];
	mmul1(RM, vec1, vec2, 3, 3, 1);
	vec1[0] = P[0] + V[0] * dt + 0.5 * (vec2[0] + 0) * dt * dt;
	vec1[1] = P[1] + V[1] * dt + 0.5 * (vec2[1] + 0) * dt * dt;
	vec1[2] = P[2] + V[2] * dt + 0.5 * (vec2[2] + g) * dt * dt;
	P[0] = vec1[0]; P[1] = vec1[1]; P[2] = vec1[2];

	vec1[0] = V[0] + (vec2[0] + 0) * dt;
	vec1[1] = V[1] + (vec2[1] + 0) * dt;
	vec1[2] = V[2] + (vec2[2] + g) * dt;
	V[0] = vec1[0]; V[1] = vec1[1]; V[2] = vec1[2];
	float Qe[4];
	vec1[0] = (gyro[0] - GBiasold[0]) * dt;
	vec1[1] = (gyro[1] - GBiasold[1]) * dt;
	vec1[2] = (gyro[2] - GBiasold[2]) * dt;
	expQuat(Qe, vec1);
	// Q*Qe
	a = Q[0] * Qe[0] + Q[1] * Qe[1] - Q[2] * Qe[2] - Q[3] * Qe[3];
	b = Q[0] * Qe[1] + Q[1] * Qe[0] + Q[2] * Qe[3] - Q[3] * Qe[2];
	c = Q[0] * Qe[2] - Q[1] * Qe[3] + Q[2] * Qe[0] + Q[3] * Qe[1];
	d = Q[0] * Qe[3] + Q[1] * Qe[2] - Q[2] * Qe[1] + Q[3] * Qe[0];

	Q[0] = a; Q[1] = b; Q[2] = c; Q[3] = d;


}

//get observation and measurement
void getHR(float* est, float* obs, int applyKFZUPT, int applyKFZARU, int applyKFGPS, int elevator, float* P, float* V, float* GBias, float* y)
{
	//y = [3]+[3]+[3]
	int n = 0;
	if (elevator == 1)
	{
		n = 3;
		vecCopy(est, P, 3);
		vecCopy(obs, y, 3);
	}
	if (applyKFGPS == 1)
	{
		n = 3;
		vecCopy(est, P, 3);
		vecCopy(obs, &y[1], 3);
	}
	if (applyKFZUPT == 1)
	{
		if (elevator == 1)
		{
			est[n] = V[0]; est[n + 1] = V[1];
			obs[n] = 0; obs[n + 1] = 0;
			n = 5;
		}
		else
		{
			est[n] = V[0]; est[n + 1] = V[n + 2]; est[n + 3] = V[2];
			obs[n] = 0; obs[n + 1] = 0; obs[n + 2] = 0;
			n = 6;
		}
	}
	if (applyKFZARU == 1)
	{
		est[n] = GBias[0]; est[n + 1] = GBias[1]; est[n + 2] = GBias[2];
		obs[n] = y[4]; obs[n + 1] = y[5]; obs[n + 2] = y[6];
	}
}

//quaternion to 3D rotation matrix 
__device__ void QtoRM(float* Q,float *RM)
{
	float a = Q[0]; float b = Q[1]; float c = Q[2]; float d = Q[3];
	RM[0] = a * a + b * b - c * c - d * d;
	RM[3] = 2 * b * c - 2 * a * d;
	RM[6] = 2 * b * d + 2 * a * c;
	RM[1] = 2 * b * c + 2 * a * d;
	RM[4] = a * a - b * b + c * c - d * d;
	RM[7] = 2 * c * d - 2 * a * b;
	RM[2] = 2 * b * d - 2 * a * c;
	RM[5] = 2 * c * d + 2 * a * b;
	RM[8] = a * a - b * b - c * c + d * d;
}

//core of Kalman filter
__device__ void ESKF(Filter* KF, Hx* H, float *y,float *acce, float *P, float *V, float* Q, 
	       float* ABias, float* GBias, int ZUPT1, int ZARU1, int GPS1, int elevator1,float *Fx,float *Xx, cublasHandle_t handle)
{
	float est[9];
	float obs[9];
	getHR(est,obs,ZUPT1,ZARU1,GPS1,elevator1,P,V,GBias,y);

	int n= elevator1 + GPS1 * 2 + ZARU1 * 4 + ZUPT1 * 8;
	float RM[9];
	QtoRM(Q, RM);

	float A[3];
	A[0] = acce[0] - ABias[0];
	A[1] = acce[1] - ABias[1];
	A[2] = acce[2] - ABias[2];
	
	float vec[3];
	float skew[9]; //col major

	mmul1(RM,A,vec,3,1,1);
	to_skew(skew, vec);

	matCopy(Fx, skew, 15, 15, 4,3, 7,3, -dt);
	matCopy(Fx, RM, 15, 15, 4, 3,10,3, -dt);
	matCopy(Fx, RM, 15, 15, 10,3, 10,3, -dt);

	float T[15 * 15],T1[15*15]; 
	mmul_ABA(handle, Fx, KF->PKF, T, 15, 15, 15, T1);
	madd(T, KF->QKFT, KF->PKF,15, 15);

	float R[4 * 3];
	R[0] = -Q[1]; R[4] = -Q[2]; R[8] = -Q[3];
	R[1] = Q[0]; R[5] = -Q[3]; R[9] = -Q[2];
	R[2] = -Q[3]; R[6] = -Q[0]; R[10] = -Q[1];
	R[3] = Q[2]; R[7] = -Q[1]; R[11] = -Q[0];

	int sizeH[2];
	sizeH[0] = H->H_size[2 * n]; sizeH[1] = H->H_size[2 * n+1];
	float HH[9*15];
	int indexH = H->H_s[n];
	mmul1(&(H->Hx[indexH]), Xx, HH, sizeH[0], sizeH[1],15);
	int size = sizeH[0];
	float K[9 * 9], ansK[9*9],tK[9*9],invK[9*9];
	// HH size*15   ,   PKF: 15*15
	mmul_ABA(handle, HH, KF->PKF, ansK, size, 15, 15, tK);
	int indexM = H->M_s[2 * n];
	madd(ansK, &(H->Mx[indexM]), tK, size, size);
	minv(tK, invK, size, size);
	float alpha = 1.0, beta = 0.0;
	//PKF 15*15 H': 15*size
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 15, size, 15, &alpha, KF->PKF, 15, HH, 15, &beta, tK, 15);
	// tK 15*size inv:size*size
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 15, size, size, &alpha, tK, 15, invK, size, &beta, K, size);
	float deltax[15];
	float dif[9];
	for (int i = 0; i < size; i++)
		dif[i] = obs[i] - est[i];
	//K:15*size
	mmul1(K, dif, deltax, 15, 1, size);
	matCopy(KF->dP, deltax, 3, 1, 1, 3, 1, 3, 1);
	matCopy(KF->dV, &deltax[3], 3, 1, 1, 3, 1, 3, 1);
	matCopy(KF->dtheta, &deltax[6], 3, 1, 1, 3, 1, 3, 1);
	matCopy(KF->dABias, &deltax[9], 3, 1, 1, 3, 1, 3, 1);
	matCopy(KF->dGBias, &deltax[12], 3, 1, 1, 3, 1, 3, 1);

	eye(T, 15);
	// K: 15*size HH:size*15
	mmul1(K, HH, T1,15,size,15);
	msub(T, T1, T, 15, 15);
	mmul1(T1, KF->PKF, T, 15, 15, 15);
	mcpy(KF->PKF, T, 15, 15);
	

}

//initialize quaternion
__device__ void init_Q(float *acce,float* Q)
{
	float grav = sqrt(acce[0] * acce[0] + acce[1] * acce[1] + acce[2] * acce[2]);
	float zx = acce[0] / grav;
	float zy = acce[1] / grav;
	float zz = acce[2] / grav;
	float xx, xy, xz;
	if (zx != 0)
	{
		xx = 0;
		xy = sqrt(1 / (1 + (zy * zy) / (zz * zz)));
		xz = -xy * zy / zz;
	}
	else
	{
		if (zy != 0)
		{
			xy = 0;
			xx = sqrt(1 / (1 + (zx * zx) / (zz * zz)));
			xz = -xx * zx / zz;
		}
		else
		{
			xz = 0;
			xx = sqrt(1 / (1 + (zx * zx) / (zy * zy)));
			xy = -xx * zx / zy;
		}
	}
	float v1[3] = { zx, zy, zz };
	float v2[3] = { zx, zy, zz };
	float y[3];
	cross(v1, v2,y);
	float R[9];

	R[0] = xx; R[3] = xy;  R[6] = xz;
	R[1] = y[0]; R[4] = y[1];  R[7] = y[2];
	R[2] = zx; R[5] = zy; R[8] = zz;
}
void printM(const float* a, int x, int y)
{
	float* arr;
	arr = (float*)malloc(x*y*sizeof(float));
	cudaMemcpy(arr, a, x*y * sizeof(float), cudaMemcpyDefault);
	for (int i = 0; i < x; i++)
	{
		for (int j = 0; j < y; j++)
			cout<<setw(7)<<arr[j * x + i]<<" ";
		printf("\n");
	}
	printf("\n");
	free(arr);
}

//used for check result
void print_KF(Filter* F, Hx* H)
{
	//cout << "PKF:\n";	printM(F->PKF, 15, 15);
	//cout << "QKF:\n";	printM(F->QKF, 12, 12);
	//cout << "Fi:\n";	printM(F->Fi, 15, 12);
	printf("QKFT:\n");	printM(F->QKFT, 15, 15);
	for (int i = 1; i < 16; i++)
	{
	//	cout << "Hx" << i << ":\n";
	//	cout << H->H_size[i * 2] << " " << H->H_size[i * 2 + 1] << "\n";
	//	printM(&(H->Hx[(H->H_s[i])]), H->H_size[i * 2], H->H_size[i * 2 + 1]);
		cout << "Mx" << i << ":\n";
		cout << H->M_size[i * 2] << " " << H->M_size[i * 2 + 1] << "\n";
		printM(&(H->Mx[(H->M_s[i])]), H->M_size[i * 2], H->M_size[i * 2 + 1]);
	}
}

//correct results based on output of Kalman filter
__device__ void correct(float *P, float *V, float *Q,float* ABias,float* GBias, Filter *KF)
{
	P[0] += KF->dP[0];
	P[1] += KF->dP[1];
	P[2] += KF->dP[2];

	V[0] += KF->dV[0];
	V[1] += KF->dV[1];
	V[2] += KF->dV[2];

	float tQ[4];
	expQuat(tQ, KF->dtheta);

	float a = Q[0] * tQ[0] + Q[1] * tQ[1] - Q[2] * tQ[2] - Q[3] * tQ[3];
	float b = Q[0] * tQ[1] + Q[1] * tQ[0] + Q[2] * tQ[3] - Q[3] * tQ[2];
	float c = Q[0] * tQ[2] - Q[1] * tQ[3] + Q[2] * tQ[0] + Q[3] * tQ[1];
	float d = Q[0] * tQ[3] + Q[1] * tQ[2] - Q[2] * tQ[1] + Q[3] * tQ[0];
	Q[0] = a; Q[1] = b; Q[2] = c; Q[3] = d;

	ABias[0] += KF->dABias[0];
	ABias[1] += KF->dABias[1];
	ABias[2] += KF->dABias[2];

	GBias[0] += KF->dGBias[0];
	GBias[1] += KF->dGBias[1];
	GBias[2] += KF->dGBias[2];
}
//error propagation when no update apply
__device__ void covariancePropagate(cublasHandle_t handle, Filter* KF, Hx* H, float *acce, 
	                    float *Q,float *ABias, float* Fx)
{
	float RM[9];
	QtoRM(Q, RM);
	float A[3];
	A[0] = acce[0] - ABias[0];
	A[1] = acce[1] - ABias[1];
	A[2] = acce[2] - ABias[2];
	float vec[3];
	float skew[9]; //col major
	mmul1(RM, A, vec, 3, 1, 1);
	to_skew(skew, vec);
	matCopy(Fx, skew, 15, 15, 4, 3, 7, 3, -dt);
	matCopy(Fx, RM, 15, 15, 4, 3, 10, 3, -dt);
	matCopy(Fx, RM, 15, 15, 10, 3, 10, 3, -dt);
	float T[15 * 15], T1[15 * 15]; 
	mmul_ABA(handle, Fx, KF->PKF, T, 15, 15, 15, T1);
	madd(T, KF->QKFT, KF->PKF, 15, 15);
}
__global__ void forward_integrate(cublasHandle_t handle, Filter* KF, Hx* H, int n, float* acc, float* gyro, float* GPSdata,
	float* baro, int* applyKF, int* applyZUPT, int* applyZARU, int* applyGPS, int* applyAltitude, float* pos, float* var)
{
	float P[3];
	float V[4];
	float Q[4];
	float ABias[3];
	float GBias[3];
	float Fx[15 * 15];
	float Xx[15 * 15];
	float y[7];
	init_Q(gyro, Q);
	for (int i = n; i >= 0; i--)
	{
		integrate(&acc[i * 3 * n], &gyro[i * 3 * n], P, V, Q, ABias, GBias);
		y[0] = baro[i]; y[1] = acc[3 * i]; y[2] = acc[3 * i + 1]; y[3] = acc[3 * i + 2];
		y[4] = gyro[3 * i]; y[5] = gyro[3 * i + 1]; y[6] = gyro[3 * i] + 2;
		if (applyKF[i] == 1)
		{
			ESKF(KF, H, y, &acc[i * 3 * n], &P[i * 3 * n], V, Q, ABias, GBias, applyZUPT[i * n], applyZARU[i * n], applyGPS[i * n], applyAltitude[i * n], Fx, Xx, handle);
			correct(P, V, Q, ABias, GBias, KF);
		}
		else
		{
			covariancePropagate(handle, KF, H, &acc[i * 3 * n], &Q[i * 3 * n], &ABias[i * 3 * n], Fx);
		}
		pos[i * 3] = P[0];
		pos[i * 3 + 1] = P[1];
		pos[i * 3 + 2] = P[2];
		var[i * 3] = KF->PKF[to_idx(1, 1, 15)];
		var[i * 3 + 1] = KF->PKF[to_idx(2, 2, 15)];
		var[i * 3 + 2] = KF->PKF[to_idx(3, 3, 15)];
	}
}

__global__ void backward_integrate(cublasHandle_t handle, Filter* KF, Hx* H, int n, float* acc, float* gyro, float* GPSdata,
	float* baro, int* applyKF, int* applyZUPT, int* applyZARU, int* applyGPS, int* applyAltitude, float* pos, float* var)
{
//	dt = -dt;
	float P[3];
	float V[4];
	float Q[4];
	float ABias[3];
	float GBias[3];
	float Fx[15 * 15];
	float Xx[15 * 15];
	float y[7];
	init_Q(gyro, Q);
	for (int i = n; i>=0; i--)
	{
		integrate(&acc[i * 3 * n], &gyro[i * 3 * n], P, V, Q, ABias, GBias);
		y[0] = baro[i]; y[1] = acc[3*i]; y[2] = acc[3 * i+1]; y[3] = acc[3 * i+2];
		y[4] = gyro[3*i]; y[5] = gyro[3 * i+1]; y[6] = gyro[3 * i]+2;
		if (applyKF[i] == 1)
		{
			ESKF(KF, H, y, &acc[i * 3 * n], &P[i * 3 * n], V, Q, ABias, GBias, applyZUPT[i * n], applyZARU[i * n], applyGPS[i * n], applyAltitude[i * n], Fx, Xx, handle);
			correct(P, V, Q, ABias, GBias, KF);
		}
		else
		{
			covariancePropagate(handle,  KF,  H, &acc[i*3*n], &Q[i*3*n],&ABias[i*3*n], Fx);
		}
		pos[i * 3] = P[0];
		pos[i * 3 + 1] = P[1];
		pos[i * 3 + 2] = P[2];
		var[i * 3] = KF->PKF[to_idx(1, 1, 15)];
		var[i * 3 + 1] = KF->PKF[to_idx(2, 2, 15)];
		var[i * 3 + 2] = KF->PKF[to_idx(3, 3, 15)];
	}
}

//smoother function
__global__ void pos_kernel(float* position1, float* position2, float* var1, float* var2,
	float* pos, int n) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < 3 * n) {
		pos[3*i] = (var2[i] * position1[i] + var1[i] * position2[i]) / (var1[i] + var2[i]);
		pos[3 * i+1] = (var2[3 * i + 1] * position1[3 * i + 1] + var1[3 * i + 1] * position2[3 * i + 1]) / (var1[3 * i + 1] + var2[3 * i + 1]);
		pos[3 * i+2] = (var2[3 * i + 2] * position1[3 * i + 2] + var1[3 * i + 2] * position2[3 * i + 2]) / (var1[3 * i + 2] + var2[3 * i + 2]);
    }
}