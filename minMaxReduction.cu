#include <stdio.h>

#include <algorithm>

#include "minMaxReduction.cuh"

using namespace std;

#define WP_SIZE 32

__device__ __inline__ int2 minMaxMC(cudaTextureObject_t tex, int3 init) {
  int2 min_max = {1e9, 0};
#pragma unroll
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++)
      for (int k = 0; k < 2; k++)
        min_max = {
            min(min_max.x, tex3D<int>(tex, init.x + i, init.y + j, init.z + k)),
            max(min_max.y,
                tex3D<int>(tex, init.x + i, init.y + j, init.z + k))};

  return min_max;
}

__device__ __inline__ int2 warpReduceMinMax(int2 val) {
#pragma unroll
  for (int offset = WP_SIZE / 2; offset > 0; offset /= 2) {
    val.x = min(val.x, __shfl_down_sync(-1, val.x, offset));
    val.y = max(val.y, __shfl_down_sync(-1, val.y, offset));
  }
  return val;
}

__global__ void blockReduceMinMax(cudaTextureObject_t tex, int n, int2* g_ans) {
  int tid_block = (threadIdx.z * blockDim.y * blockDim.x +
                   threadIdx.y * blockDim.x + threadIdx.x);
  int bid = (blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x +
             blockIdx.x);
  int tid = tid_block + (blockDim.x * blockDim.y * blockDim.z) * bid;

  if (tid >= n) return;

  int lane = tid_block % WP_SIZE;
  int wid = tid_block / WP_SIZE;

  int3 pos = {blockDim.x * blockIdx.x + threadIdx.x,
              blockDim.y * blockIdx.y + threadIdx.y,
              blockDim.z * blockIdx.z + threadIdx.z};

  __shared__ int2 warpAns[32];

  int2 val = minMaxMC(tex, pos);
  val = warpReduceMinMax(val);

  if (lane == 0) warpAns[wid] = val;

  __syncthreads();

  val = (tid_block < blockDim.x * blockDim.y * blockDim.z / WP_SIZE)
            ? warpAns[lane]
            : (int2){1e9, 0};

  if (wid == 0) val = warpReduceMinMax(val);
  if (tid_block == 0) g_ans[bid] = val;
}

