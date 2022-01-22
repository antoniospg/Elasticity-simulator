#ifndef CU_MESH_CUH
#define CU_MESH_CUH

#include <array>
#include <cmath>
#include <map>
#include <queue>
#include <set>
#include <vector>
using namespace std;

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
