#ifndef CU_MESH_CUH
#define CU_MESH_CUH

#include <array>
#include <cmath>
#include <map>
#include <queue>
#include <set>
#include <vector>
using namespace std;

class Chunks {
 public:
  array<vector<int2>, 4> colors;
  int3* indices_h;
  float3* vertices_h;
  int2* links_h;
  int n_l;

  Chunks(int3* indices_h, size_t num_indices, float3* vertices_h,
         size_t num_vertices, size_t block_size);
};

__global__ void deformVertices(float3* pos, int n);

class cuMesh {
 public:
  struct cudaGraphicsResource* positionsVBO_CUDA;
  struct cudaGraphicsResource* indicesEBO_CUDA;
  float3* vertices_g;
  int3* indices_g;
  float3* vertices_h;
  int3* indices_h;

  cuMesh(unsigned int VBO, unsigned int EBO);
  cuMesh();
  ~cuMesh();
  void mapVBO(unsigned int VBO);
  void mapEBO(unsigned int EBO);
  void deleteVBO_CUDA();
  void callKernel();
};

#endif
