#include "count.cuh"
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
using namespace std;
void count(const thrust::device_vector<int> &d_in,
           thrust::device_vector<int> &values,
           thrust::device_vector<int> &counts) {
  thrust::device_vector<int> s = d_in;
  int size = d_in.size();
  thrust::sort(s.begin(), s.end());
  thrust::device_vector<int> value_d(size, 1);
  auto new_end = thrust::reduce_by_key(s.begin(), s.end(), value_d.begin(),
                                       values.begin(), counts.begin());
  counts.resize(new_end.second - counts.begin());
  values.resize(new_end.second - counts.begin());
}
