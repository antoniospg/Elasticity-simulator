#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <math.h>

#include <iostream>

#include "cuMesh.cuh"

using namespace std;

__global__ void deformVertices(float3* pos, int n) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

  if (x < n) {
    pos[x].x *= 1.0f;
    pos[x].y *= 1.0f;
    pos[x].z *= 1.0f;
  }
}

cuMesh::cuMesh(unsigned int VBO, unsigned int EBO) {
  mapVBO(VBO);
  mapEBO(EBO);
}

cuMesh::cuMesh() {}

cuMesh::~cuMesh() { deleteVBO_CUDA(); }

void cuMesh::mapVBO(unsigned int VBO) {
  vertices_g = nullptr;
  cudaGraphicsGLRegisterBuffer(&positionsVBO_CUDA, VBO,
                               cudaGraphicsMapFlagsWriteDiscard);
  cudaGraphicsMapResources(1, &positionsVBO_CUDA, 0);
}

void cuMesh::mapEBO(unsigned int EBO) {
  indices_g = nullptr;
  cudaGraphicsGLRegisterBuffer(&indicesEBO_CUDA, EBO,
                               cudaGraphicsMapFlagsWriteDiscard);
  cudaGraphicsMapResources(1, &indicesEBO_CUDA, 0);
}

void cuMesh::deleteVBO_CUDA() {
  cudaGraphicsUnmapResources(1, &indicesEBO_CUDA, 0);
  cudaGraphicsUnregisterResource(indicesEBO_CUDA);
  indices_g = nullptr;

  cudaGraphicsUnmapResources(1, &positionsVBO_CUDA, 0);
  cudaGraphicsUnregisterResource(positionsVBO_CUDA);
  vertices_g = nullptr;
}

void cuMesh::callKernel() {
  size_t num_bytes_v;
  cudaGraphicsResourceGetMappedPointer((void**)&vertices_g, &num_bytes_v,
                                       positionsVBO_CUDA);
  size_t num_vertices = num_bytes_v / (sizeof(float3));
  vertices_h = (float3*)malloc(num_bytes_v);
  cudaMemcpy(vertices_h, vertices_g, num_bytes_v, cudaMemcpyDeviceToHost);

  size_t num_bytes_i;
  cudaGraphicsResourceGetMappedPointer((void**)&indices_g, &num_bytes_i,
                                       indicesEBO_CUDA);
  size_t num_tri = num_bytes_i / (sizeof(int3));
  indices_h = (int3*)malloc(num_bytes_i);
  cudaMemcpy(indices_h, indices_g, num_bytes_i, cudaMemcpyDeviceToHost);

  dim3 dimBlock(16, 1, 1);
  dim3 dimGrid(ceil((float)num_vertices / dimBlock.x), 1, 1);

  deformVertices<<<dimGrid, dimBlock>>>(vertices_g, num_vertices);
}
