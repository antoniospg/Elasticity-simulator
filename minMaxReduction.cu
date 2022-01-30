#include <assert.h>

#include <algorithm>
#include <iostream>

#define WP_SIZE 32

using namespace std;

__device__ __inline__ int2 warpReduceMinMax(int2 val) {
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

int main() {
  int n_x = 128, n_y = 128, n_z = 128;
  int n = n_x * n_y * n_z;

  dim3 block_size = {8, 8, 8};
  dim3 grid_size = {(n_x + block_size.x - 1) / block_size.x,
                    (n_y + block_size.y - 1) / block_size.y,
                    (n_z + block_size.z - 1) / block_size.z};
  int num_blocks = grid_size.x * grid_size.y * grid_size.z;

  int* h_data = new int[n];
  int2* h_ans = new int2[num_blocks];
  for (int i = 0; i < n; i++) h_data[i] = 0;

  int off_x[2] = {1, 0}, off_y[2] = {1, 0}, off_z[2] = {1, 0};
  int non_empty_cubes[6] = {0, 8, 512, 1024, 1032, 16384};
  for (int x0 : non_empty_cubes)
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++)
        for (int k = 0; k < 2; k++)
          h_data[(x0 + off_x[i]) + 8 * (off_y[j]) +
                 8 * 8 * (off_z[k])] = x0 + 2;

  int* g_data;
  int2* g_ans;
  cudaMalloc(&g_data, n * sizeof(int));
  cudaMalloc(&g_ans, num_blocks * sizeof(int2));
  cudaMemcpy(g_data, h_data, n * sizeof(int), cudaMemcpyHostToDevice);

  blockReduceMinMax<<<grid_size, block_size>>>(g_data, n_x, n_y, n_z, g_ans);
  cudaMemcpy(h_ans, g_ans, num_blocks * sizeof(int2), cudaMemcpyDeviceToHost);

  for (int i = 0; i < 64; i++)
    cout << h_ans[i].x << " " << h_ans[i].y << " " << i << endl;
}
