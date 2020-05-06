#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include "KF.cuh"
#include "matrixOP.cuh"
#include "cublas_v2.h"
#include <math.h>
#include <iostream>


using std::cout;
int read_all_data(float* acc, float *gyro, float *gps, float *altitude,
int* applyKF, int* applyZUPT, int* applyZARU, int* applyALTITUDE);
int main(int argc, char** argv)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int n = 1;
    Filter* h_filter;
    cudaMallocManaged(&h_filter, sizeof(Filter));
    Hx* h_H;
    cudaMallocManaged(&h_H, sizeof(Hx));
    float* acc, * gyro, * gps, * altitude;
    int* applyKF, * applyZUPT, * applyZARU,*applyGPS, * applyALTITUDE;

////--------------------------reading data---------------------
    int data_points=read_all_data(acc, gyro, gps, altitude, applyKF, applyZUPT, applyZARU, applyALTITUDE);

    float* pos_fd, * var_fd, * pos_bk, * var_bk, * pos;
    cudaMallocManaged(&pos_fd, 3 * data_points * sizeof(float));
    cudaMallocManaged(&pos_bk, 3 * data_points * sizeof(float));
    cudaMallocManaged(&pos, 3 * data_points * sizeof(float));
    cudaMallocManaged(&var_fd, 3 * data_points * sizeof(float));
    cudaMallocManaged(&var_bk, 3 * data_points * sizeof(float));
///------------------------forward integral------------------------
    cudaEventRecord(start);

    init_KF(h_filter, h_H, handle);
    // print_KF(h_filter, h_H);

    cudaMallocManaged(&h_filter, sizeof(Filter));
    forward_integrate<<<1,1>>>(handle, h_filter, h_H, data_points, acc, gyro, gps, altitude, applyKF, applyZUPT, applyZARU, applyGPS, applyALTITUDE, pos_fd, var_fd);
    cudaDeviceSynchronize();
///------------------------backward integral------------------------
    init_KF(h_filter, h_H, handle);
    backward_integrate << <1, 1 >> > (handle, h_filter, h_H, data_points, acc, gyro, gps, altitude, applyKF, applyZUPT, applyZARU, applyGPS, applyALTITUDE, pos_bk, var_bk);
    cudaDeviceSynchronize();
///------------------------smoother------------------------
    pos_kernel <<<(n + 1023) / 1024, 1024 >> > (pos_fd, pos_bk, var_fd, var_bk, pos, data_points);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time;
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cout << time << "\n";

    cublasDestroy(handle);
    cudaFree(acc);
    cudaFree(gyro);
    cudaFree(gps);
    cudaFree(altitude);
    cudaFree(applyKF);
    cudaFree(applyZUPT);
    cudaFree(applyZARU);
    cudaFree(applyGPS);
    cudaFree(applyALTITUDE);
    cudaFree(pos_fd);
    cudaFree(pos_bk);
    cudaFree(pos);
    cudaFree(var_fd);
    cudaFree(var_bk);
    return 0;
}

int read_all_data(float* acc, float* gyro, float* gps, float* altitude,
    int* applyKF, int* applyZUPT, int* applyZARU, int* applyALTITUDE)
{
    string filename;
    int col,lines,res;
    int number;
    filename = "applyKF.txt"; col = 1;
    lines = read_data<int>(filename, NULL, 0, col);
    cudaMallocManaged(&applyKF, lines * col * sizeof(int));
    res = read_data<int>(filename, applyKF, lines, col);
    if (res != 1)
        cout << "read " << filename << " failed\n";
    else
        cout << "File " << filename << " loaded \n";
    number = lines;

    filename = "applyZUPT.txt"; col = 1;
    lines = read_data<int>(filename, NULL, 0, 3);
    cudaMallocManaged(&applyKF, lines * col * sizeof(int));
    res = read_data<int>(filename, applyKF, lines, 3);
    if (res != 1)
        cout << "read " << filename << " failed\n";
    else
        cout << "File " << filename << " loaded \n";

    filename = "applyZARU.txt"; col = 1;
    lines = read_data<int>(filename, NULL, 0, 3);
    cudaMallocManaged(&applyKF, lines * col * sizeof(int));
    res = read_data<int>(filename, applyKF, lines, 3);
    if (res != 1)
        cout << "read " << filename << " failed\n";
    else
        cout << "File " << filename << " loaded \n";

    filename = "applyALTITUDE.txt"; col = 1;
    lines = read_data<int>(filename, NULL, 0, 3);
    cudaMallocManaged(&applyKF, lines * col * sizeof(int));
    res = read_data<int>(filename, applyKF, lines, 3);
    if (res != 1)
        cout << "read " << filename << " failed\n";
    else
        cout << "File " << filename << " loaded \n";

    filename = "acce.txt"; col = 3;
    lines = read_data<int>(filename, NULL, 0, 3);
    cudaMallocManaged(&applyKF, lines * col * sizeof(int));
    res = read_data<int>(filename, applyKF, lines, 3);
    if (res != 1)
        cout << "read " << filename << " failed\n";
    else
        cout << "File " << filename << " loaded \n";

    filename = "gyro.txt"; col = 3;
    lines = read_data<int>(filename, NULL, 0, 3);
    cudaMallocManaged(&applyKF, lines * col * sizeof(int));
    res = read_data<int>(filename, applyKF, lines, 3);
    if (res != 1)
        cout << "read " << filename << " failed\n";
    else
        cout << "File " << filename << " loaded \n";

    filename = "gps.txt"; col = 3;
    lines = read_data<int>(filename, NULL, 0, 3);
    cudaMallocManaged(&applyKF, lines * col * sizeof(int));
    res = read_data<int>(filename, applyKF, lines, 3);
    if (res != 1)
        cout << "read " << filename << " failed\n";
    else
        cout << "File " << filename << " loaded \n";

    filename = "altitude.txt"; col = 1;
    lines = read_data<int>(filename, NULL, 0, 3);
    cudaMallocManaged(&applyKF, lines * col * sizeof(int));
    res = read_data<int>(filename, applyKF, lines, 3);
    if (res != 1)
        cout << "read " << filename << " failed\n";
    else
        cout << "File " << filename << " loaded \n";

    return number;
}