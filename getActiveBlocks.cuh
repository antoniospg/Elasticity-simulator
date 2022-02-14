#ifndef GETACTIVEBLOCKS_CUH
#define GETACTIVEBLOCKS_CUH

__device__ __inline__ int warpReduceScan(int val, int laneid);

__global__ void getActiveBlocks(int2* g_blockMinMax, int n, int* g_ans,
                                int* numActiveBlocks);

void getActiveBlocksWrapper(int2* g_blockMinMax, int n, int* g_ans,
                            int* numActiveBlocks, dim3 grid_size,
                            dim3 block_size);

#endif

