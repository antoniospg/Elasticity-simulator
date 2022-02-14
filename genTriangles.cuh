#ifndef GENTRIANGLES_CUH
#define GENTRIANGLES_CUH

typedef uchar3 bool3;
__constant__ int d_neighbourMappingTable[12][4];

__device__ __inline__ float3 interpolate3(uint3 pos1, uint3 pos2, int w1,
                                          int w2);

__device__ int3 sampleVolume_old(uint3 pos, volatile int* shem,
                                 cudaTextureObject_t tex, float3* vertices);

__device__ bool3 get_active_edges(uint3 pos, volatile int* shem);

__device__ __inline__ bool get_neighbor_mapping(int edge, volatile bool3* shem);

__global__ void generateTris(cudaTextureObject_t tex, int* activeBlocks,
                             int* numActiveBlocks);

void generateTrisWrapper(cudaTextureObject_t tex, int* activeBlocks,
                         int* numActiveBlocks, dim3 grid_size, dim3 block_size);

#endif
