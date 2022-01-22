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

Chunks::Chunks(int3* indices_h, size_t n_i, float3* vertices_h, size_t n_v,
               size_t block_size) {
  // Inicializar links_h
  links_h = (int2*)malloc(2 * n_i * sizeof(int3));
  n_l = 3 * n_i;
  int i_link = 0;

  // Calcular links e map de vertice em link
  map<int, vector<int>> get_links;
  map<int, int> get_next_vertex;
  for (size_t i = 0; i < n_i; i++) {
    links_h[i_link] = {indices_h[i].x, indices_h[i].y};
    get_links[links_h[i_link].x].push_back(i_link);
    get_next_vertex[i_link] = links_h[i_link].y;
    i_link++;

    links_h[i_link] = {indices_h[i].y, indices_h[i].z};
    get_links[links_h[i_link].x].push_back(i_link);
    get_next_vertex[i_link] = links_h[i_link].y;
    i_link++;

    links_h[i_link] = {indices_h[i].z, indices_h[i].x};
    get_links[links_h[i_link].x].push_back(i_link);
    get_next_vertex[i_link] = links_h[i_link].y;
    i_link++;
  }

  // Colorir malha
  queue<int> q_vertex;
  set<int> visited;
  q_vertex.push(0);
  int first_color = 0;   // 0..3
  int second_color = 0;  // 0..5
  int count_links = 0;   // max = block_size

  while (!q_vertex.empty()) {
    int curr = q_vertex.front();
    q_vertex.pop();
    if (visited.find(curr) != visited.end()) continue;
    visited.insert(curr);

    for (auto link : get_links[curr]) {
      count_links++;
      colors[first_color].push_back(links_h[link]);
      q_vertex.push(get_next_vertex[link]);

      if (count_links == block_size) {
        first_color = (first_color + 1) % 4;
        count_links = 0;
      }
    }
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

  Chunks ch(indices_h, num_tri, vertices_h, num_vertices, 256);

  deformVertices<<<dimGrid, dimBlock>>>(vertices_g, num_vertices);
}
