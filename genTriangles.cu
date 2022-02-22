#include <stdio.h>

#include "constants.h"
#include "errorHandling.cuh"
#include "genTriangles.cuh"

__device__ int genTriangles::getCubeidx(uint3 pos, volatile int* shem) {
  int tid_block = threadIdx.x + blockDim.x * threadIdx.y +
                  blockDim.x * blockDim.y * threadIdx.z;
  // Neighbours in each direction
  uint offsets[8] = {threadIdx.x + blockDim.x * threadIdx.y +
                         blockDim.x * blockDim.y * threadIdx.z,
                     (threadIdx.x + 1) + blockDim.x * threadIdx.y +
                         blockDim.x * blockDim.y * threadIdx.z,
                     (threadIdx.x + 1) + blockDim.x * threadIdx.y +
                         blockDim.x * blockDim.y * (threadIdx.z + 1),
                     threadIdx.x + blockDim.x * threadIdx.y +
                         blockDim.x * blockDim.y * (threadIdx.z + 1),
                     threadIdx.x + blockDim.x * (threadIdx.y + 1) +
                         blockDim.x * blockDim.y * threadIdx.z,
                     (threadIdx.x + 1) + blockDim.x * (threadIdx.y + 1) +
                         blockDim.x * blockDim.y * threadIdx.z,
                     (threadIdx.x + 1) + blockDim.x * (threadIdx.y + 1) +
                         blockDim.x * blockDim.y * (threadIdx.z + 1),
                     threadIdx.x + blockDim.x * (threadIdx.y + 1) +
                         blockDim.x * blockDim.y * (threadIdx.z + 1)};
  uint3 pos_offset[8] = {{pos.x, pos.y, pos.z},
                         {pos.x + 1, pos.y, pos.z},
                         {pos.x + 1, pos.y, pos.z + 1},
                         {pos.x, pos.y, pos.z + 1},
                         {pos.x, pos.y + 1, pos.z},
                         {pos.x + 1, pos.y + 1, pos.z},
                         {pos.x + 1, pos.y + 1, pos.z + 1},
                         {pos.x, pos.y + 1, pos.z + 1}};

  int cubeindex = 0;
  if (pos_offset[0].x >= blockDim.x || pos_offset[0].y >= blockDim.y ||
      pos_offset[0].z >= blockDim.z || shem[offsets[0]] < d_isoVal)
    cubeindex |= 1;
  else
    return 0;
  if (pos_offset[1].x >= blockDim.x || pos_offset[1].y >= blockDim.y ||
      pos_offset[1].z >= blockDim.z || shem[offsets[1]] < d_isoVal)
    cubeindex |= 2;
  if (pos_offset[2].x >= blockDim.x || pos_offset[2].y >= blockDim.y ||
      pos_offset[2].z >= blockDim.z || shem[offsets[2]] < d_isoVal)
    cubeindex |= 4;
  if (pos_offset[3].x >= blockDim.x || pos_offset[3].y >= blockDim.y ||
      pos_offset[3].z >= blockDim.z || shem[offsets[3]] < d_isoVal)
    cubeindex |= 8;
  if (pos_offset[4].x >= blockDim.x || pos_offset[4].y >= blockDim.y ||
      pos_offset[4].z >= blockDim.z || shem[offsets[4]] < d_isoVal)
    cubeindex |= 16;
  if (pos_offset[5].x >= blockDim.x || pos_offset[5].y >= blockDim.y ||
      pos_offset[5].z >= blockDim.z || shem[offsets[5]] < d_isoVal)
    cubeindex |= 32;
  if (pos_offset[6].x >= blockDim.x || pos_offset[6].y >= blockDim.y ||
      pos_offset[6].z >= blockDim.z || shem[offsets[6]] < d_isoVal)
    cubeindex |= 64;
  if (pos_offset[7].x >= blockDim.x || pos_offset[7].y >= blockDim.y ||
      pos_offset[7].z >= blockDim.z || shem[offsets[7]] < d_isoVal)
    cubeindex |= 128;

  return cubeindex;
}

__device__ __inline__ float3 genTriangles::lerpVertex(uint3 pos1, uint3 pos2,
                                                      int v1, int v2) {
  float3 vertex;
  float w = ((float)(d_isoVal - v1)) / (v2 - v1);

  vertex.x = pos1.x + w * (pos2.x - pos1.x);
  vertex.y = pos1.y + w * (pos2.y - pos1.y);
  vertex.z = pos1.z + w * (pos2.z - pos1.z);

  return vertex;
}

__device__ bool3 genTriangles::getVertex(uint3 pos, volatile int* shem,
                                         float3* vertices) {
  int offset[4] = {(threadIdx.x) + blockDim.x * (threadIdx.y) +
                       blockDim.x * blockDim.y * (threadIdx.z),
                   (threadIdx.x + 1) + blockDim.x * (threadIdx.y) +
                       blockDim.x * blockDim.y * (threadIdx.z),
                   (threadIdx.x) + blockDim.x * (threadIdx.y) +
                       blockDim.x * blockDim.y * (threadIdx.z + 1),
                   (threadIdx.x) + blockDim.x * (threadIdx.y + 1) +
                       blockDim.x * blockDim.y * (threadIdx.z)};

  uint3 pos_neigh[3] = {uint3{pos.x + 1, pos.y, pos.z},
                        uint3{pos.x, pos.y, pos.z + 1},
                        uint3{pos.x, pos.y + 1, pos.z}};

  if (shem[offset[0]] < d_isoVal) return bool3{0, 0, 0};

  int edge_offset[3] = {0, 1, 2};
  bool active_edge[3] = {0, 0, 0};

  for (size_t i = 0; i < 3; i++) {
    if (pos_neigh[i].x < blockDim.x && pos_neigh[i].y < blockDim.y &&
        pos_neigh[i].z < blockDim.z && shem[offset[i + 1]] < d_isoVal) {
      vertices[edge_offset[i]] =
          lerpVertex(pos, pos_neigh[i], shem[offset[0]], shem[offset[i + 1]]);

      active_edge[i] = true;
    } else if (pos_neigh[i].x >= blockDim.x || pos_neigh[i].y >= blockDim.y ||
               pos_neigh[i].z >= blockDim.z) {
      vertices[edge_offset[i]] =
          lerpVertex(pos, pos_neigh[i], shem[offset[0]], 0);

      active_edge[i] = true;
    } else
      active_edge[i] = false;
  }

  return bool3{active_edge[0], active_edge[1], active_edge[2]};
}

__device__ __inline__ int genTriangles::warpReduceScan(int val, int laneid) {
#pragma unroll
  for (int offset = 1; offset < WP_SIZE; offset *= 2) {
    int y = __shfl_up_sync(0xffffffff, val, offset);
    if (laneid >= offset) val += y;
  }
  return val;
}

__device__ int genTriangles::getVertexOffset(int nums) {
  int tid_block = (threadIdx.z * blockDim.y * blockDim.x +
                   threadIdx.y * blockDim.x + threadIdx.x);
  int bid = (blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x +
             blockIdx.x);
  int tid = tid_block + (blockDim.x * blockDim.y * blockDim.z) * bid;

  int lane = tid_block % WP_SIZE;
  int wid = tid_block / WP_SIZE;

  __shared__ int shem[32];
  nums = warpReduceScan(nums, lane);

  if (lane == 31) shem[wid] = nums;
  __syncthreads();

  int val = (tid_block < blockDim.x * blockDim.y * blockDim.z / WP_SIZE)
                ? shem[lane]
                : 0;
  if (wid == 0) {
    val = warpReduceScan(val, lane);
    shem[lane] = val;
  }
  __syncthreads();

  nums += (wid > 0) ? shem[wid - 1] : 0;

  return nums;
}

__device__ int genTriangles::borrowVertex(uint3 pos, int edge,
                                          volatile int3* shem) {
  int offset[4] = {
      d_neighbourMappingTable[edge][0], d_neighbourMappingTable[edge][1],
      d_neighbourMappingTable[edge][2], d_neighbourMappingTable[edge][3]};

  int shem_id = (threadIdx.x + offset[0]) +
                blockDim.x * (threadIdx.y + offset[1]) +
                blockDim.x * blockDim.y * (threadIdx.z + offset[2]);
  uint3 offset_pos =
      uint3{pos.x + offset[0], pos.y + offset[1], pos.z + offset[2]};

  if (offset_pos.x < blockDim.x && offset_pos.y < blockDim.y &&
      offset_pos.z < blockDim.z) {
    if (offset[3] == 0)
      return shem[shem_id].x;
    else if (offset[3] == 1)
      return shem[shem_id].z;
    else if (offset[3] == 2)
      return shem[shem_id].y;
    else
      return -1;
  } else
    return -1;
}

__global__ void genTriangles::generateTris(cudaTextureObject_t tex,
                                           int* activeBlocks,
                                           int* numActiveBlocks, uint3 nxyz,
                                           int* block_vertex_offset,
                                           int* block_index_offset,
                                           float3* vertices, int3* indices) {
  uint numBlk = *numActiveBlocks;
  int block_id = activeBlocks[blockIdx.x];
  int tid_block = threadIdx.x + blockDim.x * threadIdx.y +
                  blockDim.x * blockDim.y * threadIdx.z;
  int wid = tid_block / WP_SIZE;
  int lane = tid_block % WP_SIZE;

  uint3 block_pos =
      uint3{block_id % nxyz.x, (block_id / nxyz.x) % (nxyz.x * nxyz.y),
            block_id / (nxyz.x * nxyz.y)};
  uint3 pos = uint3{threadIdx.x + block_pos.x * blockDim.x,
                    threadIdx.y + block_pos.y * blockDim.y,
                    threadIdx.z + block_pos.z * blockDim.z};

  __shared__ int voxels[1024];
  voxels[tid_block] = tex3D<int>(tex, pos.x, pos.y, pos.z);
  __syncthreads();

  int cube_idx = getCubeidx(pos, voxels);
  float3 vertices_local[3];
  bool3 active_edge = getVertex(pos, voxels, vertices_local);

  int vertex_count = 0;
  if (active_edge.x) vertex_count++;
  if (active_edge.y) vertex_count++;
  if (active_edge.z) vertex_count++;

  int vertex_offset = getVertexOffset(vertex_count);
  int vertex_offset_next = vertex_offset;
  vertex_offset -= (vertex_count == 0 ? 0 : vertex_count);

  int3 vertex_block_id;
  vertex_block_id.x = active_edge.x + vertex_offset;
  vertex_block_id.y = active_edge.y + vertex_block_id.x;
  vertex_block_id.z = active_edge.z + vertex_block_id.y;

  __shared__ int3 vertices_block_id[1024];
  vertices_block_id[tid_block] = vertex_block_id;
  __syncthreads();

  int tris[18];
  int num_tris = 0;
  for (size_t i = 0; i < 18 && d_triTable[cube_idx][i] != -1; i++) {
    int tri_idx = borrowVertex(pos, d_triTable[cube_idx][i], vertices_block_id);
    tris[num_tris] = tri_idx;
    num_tris++;
  }

  int index_offset = getVertexOffset(num_tris);
  int index_offset_next = index_offset;
  index_offset -= (num_tris == 0 ? 0 : num_tris);

  // Global Offset
  if (tid_block == (blockDim.x * blockDim.y * blockDim.z - 1)) {
    int2 block_sum = {vertex_offset_next, index_offset_next};

    for (int i = blockIdx.x + 1; i < gridDim.x; i++) {
      atomicAdd(block_vertex_offset + i, block_sum.x);
      atomicAdd(block_index_offset + i, block_sum.y);
    }
  }

  // Write vertices to global memory
  if (active_edge.x)
    vertices[block_vertex_offset[blockIdx.x] + vertex_block_id.x - 1] =
        vertices_local[0];
  if (active_edge.y)
    vertices[block_vertex_offset[blockIdx.x] + vertex_block_id.y - 1] =
        vertices_local[1];
  if (active_edge.z)
    vertices[block_vertex_offset[blockIdx.x] + vertex_block_id.z - 1] =
        vertices_local[2];
}

void genTriangles::generateTrisWrapper(cudaTextureObject_t tex,
                                       int* activeBlocks, int* numActiveBlocks,
                                       dim3 grid_size, dim3 block_size,
                                       int isoVal, uint3 nxyz) {
  gpuErrchk(cudaMemcpyToSymbol(d_isoVal, &isoVal, sizeof(int)));
  gpuErrchk(cudaMemcpyToSymbol(d_neighbourMappingTable, neighbourMappingTable,
                               12 * 4 * sizeof(int)));
  gpuErrchk(cudaMemcpyToSymbol(d_edgeTable, edgeTable, 256 * sizeof(int)));
  gpuErrchk(cudaMemcpyToSymbol(d_triTable, triTable, 256 * 16 * sizeof(int)));

  // Global offset
  int* d_block_vertex_offset;
  int* d_block_index_offset;
  cudaMalloc(&d_block_vertex_offset, grid_size.x * sizeof(int));
  cudaMalloc(&d_block_index_offset, grid_size.x * sizeof(int));

  // store vertices / indices
  float3* d_vertices;
  int3* d_indices;
  cudaMalloc(&d_vertices, 128 * 128 * 128 * sizeof(float3));
  cudaMalloc(&d_indices, 128 * 128 * 128 * sizeof(int3));

  // Debug
  float3* h_vertices;
  int3* h_indices;
  h_vertices = (float3*)malloc(128 * 128 * 128 * sizeof(float3));
  h_indices = (int3*)malloc(128 * 128 * 128 * sizeof(int3));

  // Debug
  int* block_vertex_offset = (int*)malloc(grid_size.x * sizeof(int));
  int* block_index_offset = (int*)malloc(grid_size.x * sizeof(int));

  generateTris<<<grid_size, block_size>>>(
      tex, activeBlocks, numActiveBlocks, nxyz, d_block_vertex_offset,
      d_block_index_offset, d_vertices, d_indices);

  cudaMemcpy(block_vertex_offset, d_block_vertex_offset,
             grid_size.x * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(block_index_offset, d_block_index_offset,
             grid_size.x * sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < grid_size.x; i++) {
    printf("%d %d %d \n", block_vertex_offset[i], block_index_offset[i], i);
  }

  cudaMemcpy(h_vertices, d_vertices, 128 * 128 * 128 * sizeof(float3),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_indices, d_indices, 128 * 128 * 128 * sizeof(int3),
             cudaMemcpyDeviceToHost);

  printf("****************\n");
  for (int i = 0; i < 128; i++)
    printf("%f %f %f \n", h_vertices[i].x, h_vertices[i].y, h_vertices[i].z);

  cudaFree(d_block_vertex_offset);
  cudaFree(d_block_index_offset);
}
