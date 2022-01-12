#include "cuMesh.cuh"
#include <math.h>
#include <iostream>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

namespace cuMesh {
static struct cudaGraphicsResource* positionsVBO_CUDA;
static float3* pos_g;

__global__ void deformVertices(float3* pos, int n) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (x < n) {
    pos[x].x *= 1.02f;
    pos[x].y *= 1.02f;
    pos[x].z *= 1.02f; 
  }
}

void mapVBO(unsigned int VBO) {
  pos_g = nullptr;
  cudaGraphicsGLRegisterBuffer(&positionsVBO_CUDA, VBO,
                               cudaGraphicsMapFlagsWriteDiscard);
  cudaGraphicsMapResources(1, &positionsVBO_CUDA, 0);
}

void deleteVBO_CUDA() {
  cudaGraphicsUnmapResources(1, &positionsVBO_CUDA, 0);
  cudaGraphicsUnregisterResource(positionsVBO_CUDA);
  pos_g = nullptr;
}

void callKernel() {
  size_t num_bytes;
  cudaGraphicsResourceGetMappedPointer((void**)&pos_g, &num_bytes, positionsVBO_CUDA);

  dim3 dimBlock(16, 1, 1);
  dim3 dimGrid(ceil((float)num_bytes/12 /dimBlock.x), 1, 1);

  float3* pos_h; 
  cudaMallocHost(&pos_h, num_bytes);
  cudaMemcpy(pos_h, pos_g, num_bytes, cudaMemcpyDeviceToHost);
  
  std::cout << pos_h[0].x << " " << pos_h[0].y << " " << pos_h[0].z << std::endl;
  
  deformVertices<<<dimGrid, dimBlock>>>(pos_g, num_bytes/12);
}


}
