#include <stdio.h>

#include "constants.h"
#include "errorHandling.cuh"
#include "genTriangles.cuh"

__device__ __inline__ float3 genTriangles::interpolate3(uint3 pos1, uint3 pos2, int w1,
                                          int w2) {
  return float3{(float)(pos1.x * w1 + pos2.x * w2) / (w1 + w2),
                (float)(pos1.y * w1 + pos2.y * w2) / (w1 + w2),
                (float)(pos1.z * w1 + pos2.z * w2) / (w1 + w2)};
}

__device__ int3 genTriangles::sampleVolume_old(uint3 pos, volatile int* shem,
                                 cudaTextureObject_t tex, float3* vertices) {
  int tid_block = threadIdx.x + blockDim.x * threadIdx.y +
                  blockDim.x * blockDim.y * threadIdx.z;

  // Neighbours in each direction
  uint offsets[3] = {(threadIdx.x + 1) + blockDim.x * threadIdx.y +
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

__device__ bool3 genTriangles::get_active_edges(uint3 pos, volatile int* shem) {
  int tid_block = threadIdx.x + blockDim.x * threadIdx.y +
                  blockDim.x * blockDim.y * threadIdx.z;
  // Neighbours in each direction
  uint offsets[3] = {(threadIdx.x + 1) + blockDim.x * threadIdx.y +
                         blockDim.x * blockDim.y * threadIdx.z,
                     threadIdx.x + blockDim.x * (threadIdx.y - 1) +
                         blockDim.x * blockDim.y * threadIdx.z,
                     threadIdx.x + blockDim.x * threadIdx.y +
                         blockDim.x * blockDim.y * (threadIdx.z + 1)};

  bool xyz_edges[3] = {0, 0, 0};

#pragma unroll
  for (size_t i = 0; i < 3; i++) {
    if (shem[tid_block] == 0 || offsets[i] < 0 ||
        offsets[i] > blockDim.x * blockDim.y * blockDim.z)
      xyz_edges[i] = false;
    else
      xyz_edges[i] = shem[offsets[i]] != 0;
  }

  return bool3{xyz_edges[0], xyz_edges[1], xyz_edges[2]};
}

__device__ __inline__ bool genTriangles::get_neighbor_mapping(int edge,
                                                volatile bool3* shem) {
  int edge_offset[4] = {
      d_neighbourMappingTable[edge][0], d_neighbourMappingTable[edge][2],
      d_neighbourMappingTable[edge][1], d_neighbourMappingTable[edge][3]};

  int shem_offset = (threadIdx.x + edge_offset[0]) +
                    blockDim.x * (threadIdx.y + edge_offset[1]) +
                    blockDim.x * blockDim.y * (threadIdx.z + edge_offset[2]);

  if (edge_offset[3] == 0) return shem[shem_offset].x;
  if (edge_offset[3] == 1) return shem[shem_offset].y;
  if (edge_offset[3] == 2) return shem[shem_offset].z;
  return -1;
}

__global__ void genTriangles::generateTris(cudaTextureObject_t tex, int* activeBlocks,
                             int* numActiveBlocks) {
  uint numBlk = *numActiveBlocks;
  int block_id = activeBlocks[blockIdx.x];
  int tid_block = threadIdx.x + blockDim.x * threadIdx.y +
                  blockDim.x * blockDim.y * threadIdx.z;
  int wid = tid_block / WP_SIZE;
  int lane = tid_block % WP_SIZE;

  int3 block_pos =
      int3{block_id % 16, (block_id / 16) % (16 * 16), block_id / (16 * 16)};
  uint3 pos = uint3{threadIdx.x + block_pos.x * blockDim.x,
                    threadIdx.y + block_pos.y * blockDim.y + 1,
                    threadIdx.z + block_pos.z * blockDim.z};

  // Multi use shem to store voxel values
  __shared__ int voxels[1024];
  voxels[tid_block] = tex3D<int>(tex, pos.x, pos.y, pos.z);
  __syncthreads();

  bool3 xyz_edges = get_active_edges(pos, voxels);

  __shared__ bool3 activeEdges[1024];
  activeEdges[tid_block] = xyz_edges;
  __syncthreads();
}

void genTriangles::generateTrisWrapper(cudaTextureObject_t tex, int* activeBlocks,
                         int* numActiveBlocks, dim3 grid_size, dim3 block_size,
                         int isoVal) {
  gpuErrchk(cudaMemcpyToSymbol(d_isoVal, &isoVal, sizeof(int)));
  gpuErrchk(cudaMemcpyToSymbol(d_neighbourMappingTable, neighbourMappingTable,
                               12 * 4 * sizeof(int)));

  generateTris<<<grid_size, block_size>>>(tex, activeBlocks, numActiveBlocks);
}
