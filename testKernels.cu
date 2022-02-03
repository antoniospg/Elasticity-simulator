
#include <assert.h>

#include <iostream>

#include "getActiveBlocks.cuh"
#include "minMaxReduction.cuh"

using namespace std;

int main() {
  int n_x = 128, n_y = 128, n_z = 128;
  int n = n_x * n_y * n_z;

  dim3 block_size = {8, 8, 8};
  dim3 grid_size = {(n_x + block_size.x - 1) / block_size.x,
                    (n_y + block_size.y - 1) / block_size.y,
                    (n_z + block_size.z - 1) / block_size.z};
  int num_blocks = grid_size.x * grid_size.y * grid_size.z;

  int* h_data = new int[n];
  int2* h_blockMinMax = new int2[num_blocks];
  int* g_data;
  int2* g_blockMinMax;
  int* g_h_activeBlkNum;
  int* g_numActiveBlocks;

  cudaMalloc(&g_data, n * sizeof(int));
  cudaMalloc(&g_blockMinMax, num_blocks * sizeof(int2));
  cudaMallocManaged(&g_h_activeBlkNum, num_blocks * sizeof(int));
  cudaMalloc(&g_numActiveBlocks, num_blocks * sizeof(int));

  for (int i = 0; i < n; i++) h_data[i] = 0;

  for (int i = 0; i < num_blocks; i++) g_h_activeBlkNum[i] = -1;

  int off_x[2] = {1, 0}, off_y[2] = {1, 0}, off_z[2] = {1, 0};
  int non_empty_cubes[6] = {0, 8, 512, 1024, 1032, 16384};

  for (int x0 : non_empty_cubes)
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++)
        for (int k = 0; k < 2; k++)
          h_data[(x0 + off_x[i]) + 8 * (off_y[j]) + 8 * 8 * (off_z[k])] =
              x0 + 100;

  cudaMemcpy(g_data, h_data, n * sizeof(int), cudaMemcpyHostToDevice);

  blockReduceMinMax<<<grid_size, block_size>>>(g_data, n_x, n_y, n_z,
                                               g_blockMinMax);

  int block_size2 = 128;
  int grid_size2 = (num_blocks + block_size2 - 1) / block_size2;
  getActiveBlocks<<<grid_size2, block_size2>>>(
      g_blockMinMax, num_blocks, g_h_activeBlkNum, g_numActiveBlocks);

  cudaDeviceSynchronize();

  cudaMemcpy(h_blockMinMax, g_blockMinMax, num_blocks * sizeof(int2),
             cudaMemcpyDeviceToHost);

  for (int i = 0; i < 257; i++)
    cout << g_h_activeBlkNum[i] << " " << h_blockMinMax[i].x << " "
         << h_blockMinMax[i].y << " " << i << endl;
}
