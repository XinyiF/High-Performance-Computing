#include "reduce.cuh"
#include <iostream>
using namespace std;

int main(int argc, const char *argv[]) {
  string n, Threads;
  if (argc > 1) {
    n = string(argv[1]);
    Threads = string(argv[2]);
  }
  unsigned int N = atoi(n.c_str());
  unsigned int threads_per_block = atoi(Threads.c_str());

  int *arr = new int[N];
  for (unsigned int i = 0; i < N; i++) {
    arr[i] = 1;
  }
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  int sum = reduce(arr, N, threads_per_block);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // Get the elapsed time in milliseconds
  float ms;
  cudaEventElapsedTime(&ms, start, stop);

  cout << sum << endl;
  cout << ms << endl;
  delete[] arr;
  return 0;
}