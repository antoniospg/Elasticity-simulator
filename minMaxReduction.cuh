#ifndef MINMAXREDUCTION_CUH
#define MINMAXREDUCTION_CUH

__device__ __inline__ int2 warpReduceMinMax(int2 val);

__global__ void blockReduceMinMax(cudaTextureObject_t tex, int n_z,
                                  int2* g_ans);

void blockReduceMinMaxWrapper(cudaTextureObject_t tex, int n_z, int2* g_ans,
                              dim3 grid_size, dim3 block_size);

#endif

