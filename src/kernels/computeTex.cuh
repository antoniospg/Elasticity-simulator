#ifndef COMPUTETEX_CUH
#define COMPUTETEX_CUH

namespace texKernels {

__global__ void getNormals(cudaTextureObject_t tex, int nx, int ny, int nz, float4* ans);
}

class ComputeTex {
 public:
  cudaTextureObject_t texObj;
  cudaArray_t cuArray;

  cudaTextureObject_t texObjNormal;
  cudaArray_t cuArrayNormal;

  int nx, ny, nz;

  ComputeTex(int* h_data, int nx, int ny, int nz);
  ComputeTex();
  ~ComputeTex();
};

#endif

