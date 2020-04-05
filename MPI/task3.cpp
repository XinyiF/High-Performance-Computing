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
    
    high_resolution_clock::time_point start1;
    high_resolution_clock::time_point end1;
    duration<double, std::milli> duration_sec1;
    
    high_resolution_clock::time_point start2;
    high_resolution_clock::time_point end2;
    duration<double, std::milli> duration_sec2;
    float t1 = -1, t2 = -1, sum = 0;
    if (rank == 0) {
        start1 = high_resolution_clock::now();
        for (size_t i = 0; i < n; i++) {
            MPI_Send(&a[i], 1, MPI_FLOAT, 1, 0,
                     MPI_COMM_WORLD); // send a[i] to process 1
            MPI_Recv(&a[i], 1, MPI_FLOAT, 1, 1, MPI_COMM_WORLD,
                     &status); // receive b[i] from process 1
        }
        end1 = high_resolution_clock::now();
        duration_sec1 =
        std::chrono::duration_cast<duration<double, std::milli>>(end1 - start1);
        t1 = duration_sec1.count();
        MPI_Send(&t1, 1, MPI_FLOAT, 1, 0,
                 MPI_COMM_WORLD); // send t1 to process 1
    } else if (rank == 1) {
        
        start2 = high_resolution_clock::now();
        for (size_t i = 0; i < n; i++) {
            MPI_Send(&b[i], 1, MPI_FLOAT, 0, 1, MPI_COMM_WORLD); // send b[i] to 0
            MPI_Recv(&b[i], 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD,
                     &status); // receive a[i] from 0
        }
        end2 = high_resolution_clock::now();
        duration_sec2 =
        std::chrono::duration_cast<duration<double, std::milli>>(end2 - start2);
        MPI_Recv(&sum, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD,
                 &status); // receive t1 from process 0
        t2 = duration_sec2.count();
        sum += t2;
        cout << sum << endl;
    }
    MPI_Finalize();
    delete []a;
    delete []b;
    return 0;
}

