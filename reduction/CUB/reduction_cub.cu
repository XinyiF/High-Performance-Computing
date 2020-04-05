#include <cub/cub.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/util_allocator.cuh>
#include <iostream>
using namespace cub;
using namespace std;
CachingDeviceAllocator g_allocator(true);

int main(int argc, const char *argv[]) {
  string N, Threads;
  if (argc > 1) {
    N = string(argv[1]);
  }
  unsigned int n = atoi(N.c_str());
  int *h_in = new int[n];
  for (unsigned int i = 0; i < n; i++) {
    h_in[i] = 1;
  }
  // Set up device arrays
  int *d_in = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void **)&d_in, sizeof(int) * n));
  // Initialize device input
  CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(int) * n, cudaMemcpyHostToDevice));
  // Setup device output array
  int *d_sum = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void **)&d_sum, sizeof(int) * 1));
  // Request and allocate temporary storage
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  CubDebugExit(
      DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_sum, n));
  CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

  // Do the actual reduce operation
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  CubDebugExit(
      DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_sum, n));
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // Get the elapsed time in milliseconds
  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  int gpu_sum;
  CubDebugExit(
      cudaMemcpy(&gpu_sum, d_sum, sizeof(int) * 1, cudaMemcpyDeviceToHost));
  // Check for correctness
  cout << gpu_sum << endl;
  cout << ms << endl;

  // Cleanup
  if (d_in)
    CubDebugExit(g_allocator.DeviceFree(d_in));
  if (d_sum)
    CubDebugExit(g_allocator.DeviceFree(d_sum));
  if (d_temp_storage)
    CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
  delete[] h_in;

  return 0;
}
