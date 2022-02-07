#include <stdio.h>

#include "getActiveBlocks.cuh"

#define WP_SIZE 32

__device__ __inline__ int warpReduceScan(int val, int laneid) {
#pragma unroll
  for (int offset = 1; offset < WP_SIZE; offset *= 2) {
    int y = __shfl_up_sync(-1, val, offset);
    if (laneid >= offset) val += y;
  }
  return val;
}

__global__ void getActiveBlocks(int2* g_blockMinMax, int n, int* g_ans,
                                int* numActiveBlocks) {
  int tid_block = threadIdx.x;
  int lane = tid_block % WP_SIZE;
  int bid = blockIdx.x;
  int wid = tid_block / WP_SIZE;
  int tid = tid_block + blockDim.x * bid;

  __shared__ int bTestPerWarp[32];

  bool non_empty_block =
      !(g_blockMinMax[tid].x == 0 & g_blockMinMax[tid].y == 0);

  int bTest = non_empty_block;
  bTest = warpReduceScan(bTest, lane);

  if (lane == 31) bTestPerWarp[wid] = bTest;
  __syncthreads();

  int val = (tid_block < blockDim.x / WP_SIZE) ? bTestPerWarp[lane] : 0;
  if (wid == 0) {
    val = warpReduceScan(val, lane);
    bTestPerWarp[lane] = val;
  }
  __syncthreads();

  bTest += (wid > 0) ? bTestPerWarp[wid - 1] : 0;

  if (tid_block == 0) {
    int block_sum = bTestPerWarp[31];

    for (int i = bid + 1; i < blockDim.x; i++)
      atomicAdd(numActiveBlocks + i, block_sum);
  }
  __syncthreads();

  bTest += numActiveBlocks[bid];
  if (non_empty_block) g_ans[bTest - 1] = tid;
}
