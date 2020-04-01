#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <mpi.h>
#include <ratio>
#include <string>
#include <time.h>

using std::cout;
using std::chrono::duration;
using std::chrono::high_resolution_clock;
using namespace std;
int main(int argc, char *argv[]) {
  string N, T;
  if (argc > 1) {
    N = string(argv[1]);
  }
  size_t n = atoi(N.c_str());
  float *a = new float[n];
  float *b = new float[n];
  for (size_t i = 0; i < n; i++) {
    a[i] = 1;
    b[i] = 0;
  }
  int rank = 0, size = 0;
  MPI_Status status;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    high_resolution_clock::time_point start1;
    high_resolution_clock::time_point end1;
    duration<double, std::milli> duration_sec;
    start1 = high_resolution_clock::now();
    MPI_Send(&a[0], 1, MPI_FLOAT, 1, 0,
             MPI_COMM_WORLD); // send a[0] to process 1
    MPI_Recv(&a[0], 1, MPI_FLOAT, 1, 1, MPI_COMM_WORLD,
             &status); // receive b[0] from process 1
    end1 = high_resolution_clock::now();
    duration_sec =
        std::chrono::duration_cast<duration<double, std::milli>>(end1 - start1);
    float t1 = duration_sec.count();
    // cout << "in 0 b[0]=" << b[0] << endl;
    //  cout << "in 0 a[0]=" << a[0] << endl;
    } else if (rank == 1) {
    high_resolution_clock::time_point start2;
    high_resolution_clock::time_point end2;
    start2 = high_resolution_clock::now();

    MPI_Send(&b[0], 1, MPI_FLOAT, 0, 1, MPI_COMM_WORLD); // send b[0] to 0
    MPI_Recv(&b[0], 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD,
             &status); // receive a[0] from 0
    end2 = high_resolution_clock::now();
    duration_sec =
        std::chrono::duration_cast<duration<double, std::milli>>(end2 - start2);
    float t2 = duration_sec.count();
    // cout << "in 1 b[0]=" << b[0] << endl;
    //  cout << "in 1 a[0]=" << a[0] << endl;
  }
  MPI_Finalize();
  cout << t1+t2 << endl;
  return 0;
}
