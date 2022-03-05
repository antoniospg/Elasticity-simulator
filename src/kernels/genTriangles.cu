#include <assert.h>
#include <cooperative_groups.h>
#include <stdio.h>

#include <iostream>

#include "constants.h"
#include "genTriangles.cuh"

#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}

#define INF 1e5
using namespace std;

namespace cg = cooperative_groups;

__constant__ int d_neighbourMappingTable[12][4];
__constant__ int d_edgeTable[256];
__constant__ int d_triTable[256][16];
__constant__ int d_isoVal;
__device__ int my_block_count3_0 = 0;
__device__ int my_block_count3_1 = 0;

__device__ int genTriangles::getCubeidx(int3 pos, volatile int* shem) {
  int tid_block = threadIdx.x + blockDim.x * threadIdx.y +
                  blockDim.x * blockDim.y * threadIdx.z;

  pos = {(int)threadIdx.x, (int)threadIdx.y, (int)threadIdx.z};
  // Neighbours in each direction
  int offsets[8] = {threadIdx.x + blockDim.x * threadIdx.y +
                        blockDim.x * blockDim.y * threadIdx.z,
                    (threadIdx.x + 1) + blockDim.x * threadIdx.y +
                        blockDim.x * blockDim.y * threadIdx.z,
                    (threadIdx.x + 1) + blockDim.x * threadIdx.y +
                        blockDim.x * blockDim.y * (threadIdx.z + 1),
                    threadIdx.x + blockDim.x * threadIdx.y +
                        blockDim.x * blockDim.y * (threadIdx.z + 1),
                    threadIdx.x + blockDim.x * (threadIdx.y - 1) +
                        blockDim.x * blockDim.y * threadIdx.z,
                    (threadIdx.x + 1) + blockDim.x * (threadIdx.y - 1) +
                        blockDim.x * blockDim.y * threadIdx.z,
                    (threadIdx.x + 1) + blockDim.x * (threadIdx.y - 1) +
                        blockDim.x * blockDim.y * (threadIdx.z + 1),
                    threadIdx.x + blockDim.x * (threadIdx.y - 1) +
                        blockDim.x * blockDim.y * (threadIdx.z + 1)};
  int3 pos_offset[8] = {{pos.x, pos.y, pos.z},
                        {pos.x + 1, pos.y, pos.z},
                        {pos.x + 1, pos.y, pos.z + 1},
                        {pos.x, pos.y, pos.z + 1},
                        {pos.x, pos.y - 1, pos.z},
                        {pos.x + 1, pos.y - 1, pos.z},
                        {pos.x + 1, pos.y - 1, pos.z + 1},
                        {pos.x, pos.y - 1, pos.z + 1}};

  int cubeindex = 0, increment = 1;
  for (size_t i = 0; i < 8; i++) {
    if (pos_offset[i].x < blockDim.x && pos_offset[i].y >= 0 &&
        pos_offset[i].z < blockDim.z && shem[offsets[i]] < d_isoVal)
      cubeindex |= increment;
    // else if (pos_offset[i].x >= blockDim.x || pos_offset[i].y < 0 ||
    //          pos_offset[i].z >= blockDim.z)
    //   cubeindex |= (cubeindex & 1) * increment;
    increment *= 2;
  }

  return cubeindex;
}

__device__ __inline__ float3 genTriangles::lerpVertex(int3 pos1, int3 pos2,
                                                      int v1, int v2) {
  float3 vertex;
  float w = ((float)(d_isoVal - v1)) / (v2 - v1);

  vertex.x = pos1.x + w * (pos2.x - pos1.x);
  vertex.y = pos1.y + w * (pos2.y - pos1.y);
  vertex.z = pos1.z + w * (pos2.z - pos1.z);

  return vertex;
}

__device__ __inline__ float3 genTriangles::lerpVertex(float3 pos1, float3 pos2,
                                                      int v1, int v2) {
  float3 vertex;
  float w = ((float)(d_isoVal - v1)) / (v2 - v1);

  vertex.x = pos1.x + w * (pos2.x - pos1.x);
  vertex.y = pos1.y + w * (pos2.y - pos1.y);
  vertex.z = pos1.z + w * (pos2.z - pos1.z);

  return vertex;
}

__device__ bool3 genTriangles::getVertex(int3 pos, bool3& active_edges,
                                         volatile int* shem_voxel,
                                         volatile float3* shem_voxel_normals,
                                         float3* vertices, float3* normals) {
  int offset[4] = {(threadIdx.x) + blockDim.x * (threadIdx.y) +
                       blockDim.x * blockDim.y * (threadIdx.z),
                   (threadIdx.x + 1) + blockDim.x * (threadIdx.y) +
                       blockDim.x * blockDim.y * (threadIdx.z),
                   (threadIdx.x) + blockDim.x * (threadIdx.y - 1) +
                       blockDim.x * blockDim.y * (threadIdx.z),
                   (threadIdx.x) + blockDim.x * (threadIdx.y) +
                       blockDim.x * blockDim.y * (threadIdx.z + 1)};

  int3 check_offset[3] = {{threadIdx.x + 1, threadIdx.y, threadIdx.z},
                          {threadIdx.x, threadIdx.y - 1, threadIdx.z},
                          {threadIdx.x, threadIdx.y, threadIdx.z + 1}};

  int3 pos_neigh[3] = {int3{pos.x + 1, pos.y, pos.z},
                       int3{pos.x, pos.y - 1, pos.z},
                       int3{pos.x, pos.y, pos.z + 1}};

  bool active_edges_array[3] = {active_edges.x, active_edges.y, active_edges.z};

  for (size_t i = 0; i < 3; i++) {
    if (check_offset[i].x < blockDim.x && check_offset[i].y >= 0 &&
        check_offset[i].z < blockDim.z && active_edges_array[i]) {
      vertices[i] = lerpVertex(pos, pos_neigh[i], shem_voxel[offset[0]],
                               shem_voxel[offset[i + 1]]);

      normals[i] = lerpVertex(shem_voxel_normals[offset[0]],
                              shem_voxel_normals[offset[i + 1]],
                              shem_voxel[offset[0]], shem_voxel[offset[i + 1]]);

    } else if ((check_offset[i].x >= blockDim.x || check_offset[i].y < 0 ||
                check_offset[i].z >= blockDim.z) &&
               active_edges_array[i]) {
      if (i == 0) active_edges.x = 0;
      if (i == 1) active_edges.y = 0;
      if (i == 2) active_edges.z = 0;
    }
  }
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

__device__ int genTriangles::borrowVertex(int3 pos, int edge,
                                          volatile int3* vertices_block_id_shem,
                                          volatile bool3* active_edges_shem) {
  int tid_block = threadIdx.x + blockDim.x * threadIdx.y +
                  blockDim.x * blockDim.y * threadIdx.z;

  int offset[4] = {
      d_neighbourMappingTable[edge][0], d_neighbourMappingTable[edge][1],
      d_neighbourMappingTable[edge][2], d_neighbourMappingTable[edge][3]};

  int shem_id = (threadIdx.x + offset[0]) +
                blockDim.x * (threadIdx.y - offset[2]) +
                blockDim.x * blockDim.y * (threadIdx.z + offset[1]);
  int3 offset_pos = {threadIdx.x + offset[0], threadIdx.y - offset[2],
                     threadIdx.z + offset[1]};

  if (offset_pos.x < blockDim.x && offset_pos.y >= 0 &&
      offset_pos.z < blockDim.z) {
    if (offset[3] == 0) {
      if (active_edges_shem[shem_id].x == 0) return -1;
      return vertices_block_id_shem[shem_id].x;
    } else if (offset[3] == 1) {
      if (active_edges_shem[shem_id].z == 0) return -1;
      return vertices_block_id_shem[shem_id].z;
    } else if (offset[3] == 2) {
      if (active_edges_shem[shem_id].y == 0) return -1;
      return vertices_block_id_shem[shem_id].y;
    } else
      return -1;
  } else
    return -1;
}

__global__ void genTriangles::generateTris(
    cudaTextureObject_t tex, cudaTextureObject_t texNormal, int* activeBlocks,
    int* numActiveBlocks, dim3 grid_size, int* block_vertex_offset,
    int* block_index_offset, vert3* vertices, int3* indices) {
  uint numBlk = *numActiveBlocks;
  int block_id = activeBlocks[blockIdx.x];
  int tid_block = threadIdx.x + blockDim.x * threadIdx.y +
                  blockDim.x * blockDim.y * threadIdx.z;

  int3 block_pos = int3{block_id % (int)grid_size.x,
                        (block_id / (int)grid_size.x) % ((int)grid_size.y),
                        block_id / ((int)grid_size.x * (int)grid_size.y)};
  int3 pos = {threadIdx.x + block_pos.x * (blockDim.x - 1),
              threadIdx.y + block_pos.y * (blockDim.y - 1),
              threadIdx.z + block_pos.z * (blockDim.z - 1)};

  __shared__ unsigned int my_blockId;
  if (tid_block == 0) my_blockId = atomicAdd(&my_block_count3_0, 1);
  __syncthreads();

  __shared__ int voxels[1024];
  __shared__ float3 voxel_normals[1024];

  voxels[tid_block] = tex3D<int>(tex, pos.x, pos.y, pos.z);

  float4 voxel_normal4 = tex3D<float4>(texNormal, pos.x, pos.y, pos.z);
  float3 voxel_normal3 = {voxel_normal4.x, voxel_normal4.y, voxel_normal4.z};
  voxel_normals[tid_block] = voxel_normal3;

  __syncthreads();

  int cube_idx = getCubeidx(pos, voxels);
  float3 vertices_local[3];
  float3 normals_local[3];

  int active_hash = d_edgeTable[cube_idx];
  bool3 active_edge;
  active_edge.x = (active_hash & 1) == 1;
  active_edge.y = (active_hash & 256) == 256;
  active_edge.z = (active_hash & 8) == 8;

  getVertex(pos, active_edge, voxels, voxel_normals, vertices_local,
            normals_local);

  __shared__ bool3 active_edges_shem[1024];
  active_edges_shem[tid_block] = active_edge;
  __syncthreads();

  int vertex_count = 0;
  if (active_edge.x) vertex_count++;
  if (active_edge.y) vertex_count++;
  if (active_edge.z) vertex_count++;

  int vertex_offset = getVertexOffset(vertex_count);
  int vertex_offset_next = vertex_offset;
  vertex_offset -= vertex_count;

  int3 vertex_block_id;
  vertex_block_id.x = active_edge.x + vertex_offset;
  vertex_block_id.y = active_edge.y + vertex_block_id.x;
  vertex_block_id.z = active_edge.z + vertex_block_id.y;

  __shared__ int3 vertices_block_id[1024];
  vertices_block_id[tid_block] = vertex_block_id;
  __syncthreads();

  int tris[18];
  int num_tris = 0;
  for (size_t i = 0; i < 18 && d_triTable[cube_idx][i] != -1; i += 3) {
    int local_tris[3], tri_count = 0;

    for (size_t j = 0; j < 3; j++) {
      int tri_idx = borrowVertex(pos, d_triTable[cube_idx][i + j],
                                 vertices_block_id, active_edges_shem);
      if (tri_idx < 0) break;
      tri_count++;
      local_tris[j] = tri_idx;
    }
    if (tri_count == 3) {
      tris[num_tris] = local_tris[0];
      tris[num_tris + 1] = local_tris[1];
      tris[num_tris + 2] = local_tris[2];
      num_tris += 3;
    }
  }
  num_tris /= 3;

  int index_offset = getVertexOffset(num_tris);
  int index_offset_next = index_offset;
  index_offset -= num_tris;

  // Global offsets
  __shared__ int vertex_off;
  __shared__ int index_off;

  if (tid_block == blockDim.x * blockDim.y * blockDim.z - 1) {
    int2 block_sum = {vertex_offset_next, index_offset_next};

    vertex_off = atomicAdd(block_vertex_offset + gridDim.x, block_sum.x);
    index_off = atomicAdd(block_index_offset + gridDim.x, block_sum.y);
    // for (int i = blockIdx.x + 1; i < gridDim.x + 1; i++) {
    //  atomicAdd(block_vertex_offset + i, block_sum.x);
    //  atomicAdd(block_index_offset + i, block_sum.y);
    //}

    // atomicAdd(&my_block_count3_1, 1);
    // do {
    //  __threadfence();
    //} while (atomicAdd(&my_block_count3_1, 0) != my_block_count3_0);
  }
  __syncthreads();

  if (active_edge.x) {
    vertices[vertex_off + vertex_block_id.x - 1].pos = vertices_local[0];
    vertices[vertex_off + vertex_block_id.x - 1].normal = normals_local[0];
  }
  if (active_edge.y) {
    vertices[vertex_off + vertex_block_id.y - 1].pos = vertices_local[1];
    vertices[vertex_off + vertex_block_id.y - 1].normal = normals_local[1];
  }
  if (active_edge.z) {
    vertices[vertex_off + vertex_block_id.z - 1].pos = vertices_local[2];
    vertices[vertex_off + vertex_block_id.z - 1].normal = normals_local[2];
  }

  for (size_t i = 0; i < num_tris; i++) {
    indices[index_offset + index_off + i].x = vertex_off + tris[3 * i] - 1;
    indices[index_offset + index_off + i].y = vertex_off + tris[3 * i + 1] - 1;
    indices[index_offset + index_off + i].z = vertex_off + tris[3 * i + 2] - 1;
  }
}

__global__ void genTriangles::setGlobal() {
  my_block_count3_0 = 0;
  my_block_count3_1 = 0;
}

int2 genTriangles::generateTrisWrapper(
    cudaTextureObject_t tex, cudaTextureObject_t texNormal, int* activeBlocks,
    int* numActiveBlocks, dim3 grid_size3, dim3 block_size3, dim3 grid_size,
    int isoVal, uint3 nxyz, int* d_block_vertex_offset,
    int* d_block_index_offset, vert3* d_vertices, int3* d_indices) {
  cudaMemcpyToSymbol(d_isoVal, &isoVal, sizeof(int));
  cudaMemcpyToSymbol(d_neighbourMappingTable, neighbourMappingTable,
                     12 * 4 * sizeof(int));
  cudaMemcpyToSymbol(d_edgeTable, edgeTable, 256 * sizeof(int));
  cudaMemcpyToSymbol(d_triTable, triTable, 256 * 16 * sizeof(int));

  cudaMemset(d_block_vertex_offset, 0, (grid_size3.x + 1) * sizeof(int));
  cudaMemset(d_block_index_offset, 0, (grid_size3.x + 1) * sizeof(int));

  setGlobal<<<1, 1>>>();

  generateTris<<<grid_size3, block_size3>>>(
      tex, texNormal, activeBlocks, numActiveBlocks, grid_size,
      d_block_vertex_offset, d_block_index_offset, d_vertices, d_indices);

  int num_vertices = 0, num_indices = 0;
  cudaMemcpy(&num_vertices, d_block_vertex_offset + grid_size3.x, sizeof(int),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(&num_indices, d_block_index_offset + grid_size3.x, sizeof(int),
             cudaMemcpyDeviceToHost);

  return int2{num_vertices, num_indices};
}
