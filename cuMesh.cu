#include <glad/glad.h>
#include <cuda_gl_interop.h>

#include <iostream>

#include "cuMesh.cuh"

using namespace std;

cuMesh::cuMesh(float3* h_vertices, uint3* h_indices, size_t n_vertices,
               size_t n_indices) {
  glGenBuffers(1, &VBO);
  glGenBuffers(1, &EBO);
  glGenVertexArrays(1, &VAO);

  glBindVertexArray(VAO);

  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  uint size_VBO = n_vertices * sizeof(float3);
  glBufferData(GL_ARRAY_BUFFER, size_VBO, 0, GL_DYNAMIC_DRAW);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
  uint size_EBO = n_indices * sizeof(int3);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, size_EBO, 0, GL_DYNAMIC_DRAW);

  mapVBO();
  mapEBO();

  cudaMemcpy(d_vertices, h_vertices, n_vertices * sizeof(float3),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_indices, h_indices, n_indices * sizeof(uint3),
             cudaMemcpyHostToDevice);
}

cuMesh::cuMesh() {}

cuMesh::~cuMesh() { }

void cuMesh::mapVBO() {
  cudaGraphicsGLRegisterBuffer(&positionsVBO_CUDA, VBO,
                               cudaGraphicsMapFlagsWriteDiscard);
  cudaGraphicsMapResources(1, &positionsVBO_CUDA, 0);
  cudaGraphicsResourceGetMappedPointer((void**)&d_vertices, nullptr,
                                       positionsVBO_CUDA);
}

void cuMesh::mapEBO() {
  cudaGraphicsGLRegisterBuffer(&indicesEBO_CUDA, EBO,
                               cudaGraphicsMapFlagsWriteDiscard);
  cudaGraphicsMapResources(1, &indicesEBO_CUDA, 0);
  cudaGraphicsResourceGetMappedPointer((void**)&d_indices, nullptr,
                                       indicesEBO_CUDA);
}

void cuMesh::deleteVBO_CUDA() {
  cudaGraphicsUnmapResources(1, &indicesEBO_CUDA, 0);
  cudaGraphicsUnregisterResource(indicesEBO_CUDA);

  cudaGraphicsUnmapResources(1, &positionsVBO_CUDA, 0);
  cudaGraphicsUnregisterResource(positionsVBO_CUDA);

  glDeleteBuffers(1, &VBO);
  glDeleteBuffers(1, &EBO);
  glDeleteVertexArrays(1, &VAO);
}

