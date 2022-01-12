#ifndef CU_MESH_CUH
#define CU_MESH_CUH

namespace cuMesh {
__global__ void deformVertices(float3* pos, int n); 

void mapVBO(unsigned int VBO); 

void deleteVBO_CUDA();

void callKernel();
}

#endif

