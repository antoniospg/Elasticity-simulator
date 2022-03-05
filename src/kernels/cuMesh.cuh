#ifndef CU_MESH_CUH
#define CU_MESH_CUH

struct vert3 {
  float3 pos;
  float3 normal;
};

class cuMesh {
 public:
  struct cudaGraphicsResource* positionsVBO_CUDA;
  struct cudaGraphicsResource* indicesEBO_CUDA;
  float3* d_vertices;
  int3* d_indices;
  uint VBO, EBO, VAO;

  cuMesh(vert3* vertices_in, int3* indices_in, size_t n_vertices,
         size_t n_indices, bool device_pointers);

  cuMesh();

  ~cuMesh();

  void mapVBO();
  void mapEBO();
  void deleteVBO_CUDA();
};

#endif
