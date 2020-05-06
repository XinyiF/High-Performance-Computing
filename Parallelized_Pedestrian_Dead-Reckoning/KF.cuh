// Author: Nic Olsen
#ifndef KF_H
#define KF_H

#include <cublas_v2.h>
#include <string>
#include <fstream>
#include <iostream>
using namespace std;
#define CUDA_HD __host__ __device__
#define ZUPT 0
#define ZARU 1
#define GPS 2
#define ALTITUDE 3
#define ELE 4
#define to_idx(i,j,x) (j-1)*x+i-1 
#define pi 3.1415926
#define g -9.8027



typedef struct Filter_def
{
	float PKF[15 * 15];
	float QKF[12 * 12];
	float Fi[15 * 12];
	float QKFT[15 * 15];
	float RKF[5];
	float dP[3];
	float dV[3];
	float dtheta[3];
	float dABias[3];
	float dGBias[3];
} Filter;

typedef struct Hx_def{
	float Hx[(1 + 3 + 3 + 3 + 4 + 6 + 6 + 3 + 3 + 6 + 5 + 6 + 6 + 9 + 8) * 16];
	int H_s[16];
	int H_size[2 * 16];
	float Mx[1 + 9 + 9 + 9 + 16 + 36 + 36 + 9 + 9 + 36 + 25 + 36 + 36 + 81 + 64];
	int M_s[16];
	int M_size[2 * 16];
} Hx;


void mmul(cublasHandle_t handle, const float* A, const float* B, float* C, int n);
__host__ void init_KF(Filter* F, Hx* H, cublasHandle_t handle);
void print_KF(Filter* F, Hx* H);
void printM(const float* arr, int x, int y);
__device__ void getHR(float* est, float* obs, int applyKFZUPT, int applyKFZARU, int applyKFGPS, int elevator, float* P, float* V, float* GBias, float* y);
__device__ void ESKF(Filter* KF, Hx* H, float* y, float* acce, float* P, float* V, float* Q,
	float* ABias, float* GBias, int ZUPT1, int ZARU1, int GPS1, int elevator1, float* Fx, float* Xx, cublasHandle_t handle);
__global__ void forward_integrate(cublasHandle_t handle, Filter* KF, Hx* H, int n, const float* acc, const float* gyro, const float* GPSdata,
	const float* baro, const int* applyKF, const int* applyZUPT, const int* applyZARU, const int* applyGPS, const int* applyAltitude, float* pos, float* var);
__global__ void backward_integrate(cublasHandle_t handle, Filter* KF, Hx* H, int n, const float* acc, const float* gyro, const float* GPSdata,
	const float* baro, const int* applyKF, const int* applyZUPT, const int* applyZARU, const int* applyGPS, const int* applyAltitude, float* pos, float* var);
__device__ void init_Q(float* acce, float* Q);
__global__ void pos_kernel(float* position1, float* position2, float* var1, float* var2,
	float* pos, int n);

template<typename T>
int read_data(string filename, T* arr, int mode, int col)
{
	//myFile.open(filename, ios::in, 0);
	int count = 0;
	T x;
	string line;
	ifstream myfile(filename);
	if (myfile.is_open())
	{
		if (mode == 0)
		{
			while (getline(myfile, line))
			{
				count++;
			}
		}
		else
		{
			for (int i = 0; i < (mode * col); i++)
			{
				myfile >> x;
				arr[i] = x;
				count++;
				//cout << x << '\n';
			}
		}
		myfile.close();
	}


	if (mode == 0)
	{
		return count;
	}
	else
	{
		return (count == (mode*col));
	}
}
#endif
