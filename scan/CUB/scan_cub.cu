#include <cub/device/device_scan.cuh>
#include <cub/util_allocator.cuh>
#include <iostream>
using namespace cub;
using namespace std;
CachingDeviceAllocator g_allocator(true);

int main(int argc, const char *argv[]) {
  string N;
  if (argc > 1) {
    N = string(argv[1]);
  }
  int n = atoi(N.c_str());
  float *h_in = new float[n];
  float *cpu_out = new float[n];
  for (int i = 0; i < n; i++) {
    h_in[i] = 1.0;
    cpu_out[i] = 0;
  }
  // Set up device arrays
  float *d_in = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void **)&d_in, sizeof(float) * n));
  // Initialize device input
  CubDebugExit(
      cudaMemcpy(d_in, h_in, sizeof(float) * n, cudaMemcpyHostToDevice));
  // Setup device output array
  float *d_out = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void **)&d_out, sizeof(float) * n));
  // Request and allocate temporary storage
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  CubDebugExit(DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                        d_in, d_out, n));

  CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

  // Do the actual reduce operation
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  CubDebugExit(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                             d_in, d_out, n));
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // Get the elapsed time in milliseconds
  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  CubDebugExit(
      cudaMemcpy(cpu_out, d_out, sizeof(float) * n, cudaMemcpyDeviceToHost));
  // Check for correctness
  cout << cpu_out[n - 1] << endl;
  cout << ms << endl;
  // Cleanup
  if (d_in)
    CubDebugExit(g_allocator.DeviceFree(d_in));
  if (d_out)
    CubDebugExit(g_allocator.DeviceFree(d_out));
  if (d_temp_storage)
    CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
  delete[] h_in;
  delete[] cpu_out;
  return 0;
}
