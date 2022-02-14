#include <cuda_gl_interop.h>
#include <glad/glad.h>

#include "cuMesh.cuh"

cuMesh::cuMesh(float3* h_vertices, int3* h_indices, size_t n_vertices,
               size_t n_indices) {
  cudaMemcpy(d_vertices, h_vertices, n_vertices * sizeof(float3),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_indices, h_indices, n_indices * sizeof(int3),
             cudaMemcpyHostToDevice);

  glGenBuffers(1, &VBO);
  glGenBuffers(1, &EBO);

  mapVBO();
  mapEBO();
}

cuMesh::cuMesh() {}

cuMesh::~cuMesh() { deleteVBO_CUDA(); }

void cuMesh::mapVBO() {
  cudaGraphicsGLRegisterBuffer(&positionsVBO_CUDA, VBO,
                               cudaGraphicsMapFlagsWriteDiscard);
  cudaGraphicsMapResources(1, &positionsVBO_CUDA, 0);
}

void cuMesh::mapEBO() {
  cudaGraphicsGLRegisterBuffer(&indicesEBO_CUDA, EBO,
                               cudaGraphicsMapFlagsWriteDiscard);
  cudaGraphicsMapResources(1, &indicesEBO_CUDA, 0);
}

void cuMesh::deleteVBO_CUDA() {
  cudaFree(d_indices);
  cudaGraphicsUnmapResources(1, &indicesEBO_CUDA, 0);
  cudaGraphicsUnregisterResource(indicesEBO_CUDA);

  cudaFree(d_vertices);
  cudaGraphicsUnmapResources(1, &positionsVBO_CUDA, 0);
  cudaGraphicsUnregisterResource(positionsVBO_CUDA);

  glDeleteBuffers(1, &VBO);
  glDeleteBuffers(1, &EBO);
}

