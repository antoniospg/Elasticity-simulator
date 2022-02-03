#include <algorithm>
#include "minMaxReduction.cuh"

using namespace std;

#define WP_SIZE 32

__device__ __inline__ int2 warpReduceMinMax(int2 val) {
#pragma unroll
  for (int offset = WP_SIZE / 2; offset > 0; offset /= 2) {
    val.x = max(val.x, __shfl_down_sync(-1, val.x, offset));
    val.y = min(val.y, __shfl_down_sync(-1, val.y, offset));
  }
  return val;
}

__global__ void blockReduceMinMax(int* g_data, int n_x, int n_y, int n_z,
                                  int2* g_ans) {
  int tid_block = (threadIdx.z * blockDim.y * blockDim.x +
                   threadIdx.y * blockDim.x + threadIdx.x);
  int lane = tid_block % WP_SIZE;
  int bid = (blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x +
             blockIdx.x);
  int wid = tid_block / WP_SIZE;
  int tid = tid_block + (blockDim.x * blockDim.y * blockDim.z) * bid;

  __shared__ int2 warpAns[32];

  int2 val = {g_data[tid], g_data[tid]};
  val = warpReduceMinMax(val);

  if (lane == 0) warpAns[wid] = val;

  __syncthreads();

  val = (tid_block < blockDim.x * blockDim.y * blockDim.z / WP_SIZE)
            ? warpAns[lane]
            : (int2){0, 1e9 + 1};

  if (wid == 0) val = warpReduceMinMax(val);
  if (tid_block == 0) g_ans[bid] = val;
}

