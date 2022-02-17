#ifndef COMPUTETEX_CUH
#define COMPUTETEX_CUH

class ComputeTex {
 public:
  cudaTextureObject_t texObj;
  cudaArray_t cuArray;
  int nx, ny, nz;

  ComputeTex(int* h_data, int nx, int ny, int nz);
  ComputeTex();
  ~ComputeTex();
};

#endif

