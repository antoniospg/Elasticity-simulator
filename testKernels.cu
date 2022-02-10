#include <assert.h>

#include <iostream>

#include "computeTex.cuh"
#include "errorHandling.cuh"
#include "getActiveBlocks.cuh"
#include "minMaxReduction.cuh"

#define WP_SIZE 32

__device__ __inline__ float3 interpolate3(uint3 pos1, uint3 pos2, int w1,
                                          int w2) {
  return float3{(float)(pos1.x * w1 + pos2.x * w2) / (w1 + w2),
                (float)(pos1.y * w1 + pos2.y * w2) / (w1 + w2),
                (float)(pos1.z * w1 + pos2.z * w2) / (w1 + w2)};
}

__device__ int3 sampleVolume(uint3 pos, volatile int* shem,
                             cudaTextureObject_t tex, float3* vertices) {
  int tid_block = threadIdx.x + blockDim.x * threadIdx.y +
                  blockDim.x * blockDim.y * threadIdx.z;

  // Neighbours in each direction
  int offsets[3] = {(threadIdx.x + 1) + blockDim.x * threadIdx.y +
                        blockDim.x * blockDim.y * threadIdx.z,
                    threadIdx.x + blockDim.x * (threadIdx.y + 1) +
                        blockDim.x * blockDim.y * threadIdx.z,
                    threadIdx.x + blockDim.x * threadIdx.y +
                        blockDim.x * blockDim.y * (threadIdx.z + 1)};

  bool bound_condition[3] = {threadIdx.x + 1 < blockDim.x,
                             threadIdx.y + 1 < blockDim.x,
                             threadIdx.z + 1 < blockDim.x};

  uint3 next_vertices[3] = {uint3{pos.x + 1, pos.y, pos.z},
                            uint3{pos.x, pos.y + 1, pos.z},
                            uint3{pos.x, pos.y, pos.z + 1}};

  int next_voxels[3] = {0, 0, 0};
  int num_vertices = 0;
  int3 indices = int3{0, 0, 0};

  // Check if vertex its out of boundaries
#pragma unroll
  for (size_t i = 0; i < 3; i++) {
    if (bound_condition[i])
      next_voxels[i] = shem[offsets[i]];
    else
      next_voxels[i] = tex3D<int>(tex, next_vertices[i].x, next_vertices[i].y,
                                  next_vertices[i].z);
  }

#pragma unroll
  for (size_t i = 0; i < 3; i++) {
    int w1 = shem[tid_block];
    int w2 = next_voxels[i];
    if (w1 == 0 && w2 == 0) continue;

    vertices[i] = interpolate3(pos, next_vertices[i], w1, w2);
    num_vertices++;
  }

  if (num_vertices == 1) indices.x = 1, indices.y = 0, indices.z = 0;
  if (num_vertices == 2) indices.x = 1, indices.y = 1, indices.z = 0;
  if (num_vertices == 3) indices.x = 1, indices.y = 1, indices.z = 1;

  return indices;
}

__global__ void generateTris(cudaTextureObject_t tex, int* activeBlocks,
                             int* numActiveBlocks) {
  int numBlk = *numActiveBlocks;
  int block_id = activeBlocks[blockIdx.x];
  int tid_block = threadIdx.x + blockDim.x * threadIdx.y +
                  blockDim.x * blockDim.y * threadIdx.z;
  int wid = tid_block / WP_SIZE;
  int lane = tid_block % WP_SIZE;

  int3 block_pos =
      int3{block_id % 16, (block_id / 16) % (16 * 16), block_id / (16 * 16)};
  uint3 pos = uint3{threadIdx.x + block_pos.x * blockDim.x,
                    threadIdx.y + block_pos.y * blockDim.y,
                    threadIdx.z + block_pos.z * blockDim.z};

  __shared__ int voxels[1024];
  voxels[tid_block] = tex3D<int>(tex, pos.x, pos.y, pos.z);
  __syncthreads();

  float3* vertices = new float3[3];

  int3 indices = sampleVolume(pos, voxels, tex, vertices);
}

using namespace std;

int main() {
  int num_points_x = 128, num_points_y = 128, num_points_z = 128;
  int num_points = num_points_x * num_points_y * num_points_z;
  int* h_data = new int[num_points];

  int off_x[2] = {1, 0}, off_y[2] = {1, 0}, off_z[2] = {1, 0};
  int non_empty_cubes[6] = {0, 8, 16, 32};

  for (int x0 : non_empty_cubes)
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++)
        for (int k = 0; k < 2; k++)
          h_data[(x0 + off_x[i]) + num_points_x * (off_y[j]) +
                 num_points_x * num_points_y * (off_z[k])] = x0 + 20;

  ComputeTex ct(h_data, num_points_x, num_points_y, num_points_z);

  int n_x = (num_points_x - 1), n_y = (num_points_y - 1),
      n_z = (num_points_y - 1);
  int n = n_x * n_y * n_z;

  dim3 block_size = {8, 8, 8};
  dim3 grid_size = {(n_x + block_size.x - 1) / block_size.x,
                    (n_y + block_size.y - 1) / block_size.y,
                    (n_z + block_size.z - 1) / block_size.z};
  int num_blocks = grid_size.x * grid_size.y * grid_size.z;

  int2* h_blockMinMax = new int2[num_blocks];
  int2* g_blockMinMax;
  int* g_h_activeBlkNum;
  int* g_numActiveBlocks;

  cudaMalloc(&g_blockMinMax, num_blocks * sizeof(int2));
  cudaMallocManaged(&g_h_activeBlkNum, num_blocks * sizeof(int));
  cudaMalloc(&g_numActiveBlocks, num_blocks * sizeof(int));

  for (int i = 0; i < num_blocks; i++) g_h_activeBlkNum[i] = -1;

  blockReduceMinMax<<<grid_size, block_size>>>(ct.texObj, n, g_blockMinMax);

  int block_size2 = 128;
  int grid_size2 = (num_blocks + block_size2 - 1) / block_size2;
  getActiveBlocks<<<grid_size2, block_size2>>>(
      g_blockMinMax, num_blocks, g_h_activeBlkNum, g_numActiveBlocks);

  int* d_numActiveBlk = g_numActiveBlocks + block_size2 - 1;

  cudaDeviceSynchronize();

  int numActiveBlk = 0;
  cudaMemcpy(&numActiveBlk, g_numActiveBlocks + block_size2 - 1, sizeof(int),
             cudaMemcpyDeviceToHost);

  cudaMemcpy(h_blockMinMax, g_blockMinMax, num_blocks * sizeof(int2),
             cudaMemcpyDeviceToHost);

  dim3 block_size3 = block_size;
  int num_blocks3 = block_size3.x * block_size.y + block_size.z;
  dim3 grid_size3 = {numActiveBlk};

  int* g_vertex_offset;
  cudaMalloc(&g_vertex_offset, 3 * n_x * n_y * n_z * sizeof(int));

  generateTris<<<grid_size3, block_size3>>>(ct.texObj, g_h_activeBlkNum,
                                            d_numActiveBlk);

  for (int i = 0; i < 8; i++)
    cout << g_h_activeBlkNum[i] << " " << h_blockMinMax[i].x << " "
         << h_blockMinMax[i].y << " " << i << endl;
}
