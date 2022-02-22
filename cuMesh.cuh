#ifndef CU_MESH_CUH
#define CU_MESH_CUH

class cuMesh {
 public:
  struct cudaGraphicsResource* positionsVBO_CUDA;
  struct cudaGraphicsResource* indicesEBO_CUDA;
  float3* d_vertices;
  int3* d_indices;
  uint VBO, EBO, VAO;

  cuMesh(float3* h_vertices, uint3* h_indices, size_t n_vertices,
         size_t n_indices);
  cuMesh();
  ~cuMesh();
  void mapVBO();
  void mapEBO();
  void deleteVBO_CUDA();
};

#endif
