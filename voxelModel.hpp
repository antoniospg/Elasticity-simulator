#ifndef VOXELMODEL_HPP
#define VOXELMODEL_HPP

#include <cuda_runtime.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "computeTex.cuh"
#include "constants.h"
#include "genTriangles.cuh"
#include "getActiveBlocks.cuh"
#include "mesh.hpp"
#include "minMaxReduction.cuh"
#include "voxelLoader.hpp"

using namespace std;

class VoxelModel {
 private:
  VoxelLoader vl;
  ComputeTex ct;

  uint n_x, n_y, n_z;
  dim3 block_size, grid_size;

  int2* d_blockMinMax;
  int* d_activeBlkNum;
  int* d_numActiveBlk;
  int* d_numActiveBlocks;
  int numActiveBlk;

 public:
  float3* d_vertices;
  int3* d_indices;

  VoxelModel(string path) : vl(path), ct(vl.pData, vl.n_x, vl.n_y, vl.n_z) {
    n_x = vl.n_x, n_y = vl.n_y, n_z = vl.n_z;
    uint n = n_x * n_y * n_z;

    // First Kernel Launch
    block_size = {8, 8, 8};
    grid_size = {(n_x + block_size.x - 1) / block_size.x,
                 (n_y + block_size.y - 1) / block_size.y,
                 (n_z + block_size.z - 1) / block_size.z};
    int num_blocks = grid_size.x * grid_size.y * grid_size.z;

    cudaMalloc(&d_blockMinMax, num_blocks * sizeof(int2));
    minMax::blockReduceMinMaxWrapper(ct.texObj, n, d_blockMinMax, grid_size,
                                     block_size);

#ifdef DEBUG1
    int2* h_minmax = new int2[num_blocks];
    int count = 0;
    cudaMemcpy(h_minmax, d_blockMinMax, num_blocks * sizeof(int2),
               cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < num_blocks; i++) {
      if (h_minmax[i].x != 0 && h_minmax[i].y != 0) count++;
    }
    cout << "count : " << count << endl;

#endif
    // Second Kernel Launch
    cudaMalloc(&d_activeBlkNum, num_blocks * sizeof(int));
    cudaMalloc(&d_numActiveBlocks, num_blocks * sizeof(int));
    cudaMemset(d_activeBlkNum, 0, num_blocks * sizeof(int));
    cudaMemset(d_numActiveBlocks, 0, num_blocks * sizeof(int));

    int block_size2 = 128;
    int grid_size2 = (num_blocks + block_size2 - 1) / block_size2;

    getActiveBlocks::getActiveBlocksWrapper(d_blockMinMax, num_blocks,
                                            d_activeBlkNum, d_numActiveBlocks,
                                            grid_size2, block_size2, 0);

    d_numActiveBlk = d_numActiveBlocks + grid_size2; 
    cudaMemcpy(&numActiveBlk, d_numActiveBlk, sizeof(int),
               cudaMemcpyDeviceToHost);
  }

  int2 generate(int isoVal) {
    int num_blocks = grid_size.x * grid_size.y * grid_size.z;

    // Third Kernel Launch
    dim3 block_size3 = block_size;
    int num_blocks3 = block_size3.x * block_size.y + block_size.z;
    dim3 grid_size3 = {numActiveBlk};
    uint3 nxyz = uint3{n_x, n_y, n_z};

    if (d_vertices != nullptr) cudaFree(d_vertices);
    if (d_indices != nullptr) cudaFree(d_indices);

    int2 nums = genTriangles::generateTrisWrapper(
        ct.texObj, d_activeBlkNum, d_numActiveBlk, grid_size3, block_size3,
        grid_size, isoVal, nxyz, &d_vertices, &d_indices);

    return nums;
  }

  void draw(Shader& sh, int isoVal) {
    int2 nums = generate(isoVal);

#ifdef DEBUG3
    float3* h_vertices = new float3[nums.x];
    int3* h_indices = new int3[nums.y];
    cudaMemcpy(h_vertices, d_vertices, nums.x * sizeof(float3),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_indices, d_indices, nums.y * sizeof(int3),
               cudaMemcpyDeviceToHost);

    ofstream f_debug("debug.obj");
    for (size_t i = 0; i < nums.x; i++) {
      f_debug << "v " << h_vertices[i].x << " " << h_vertices[i].y << " "
           << h_vertices[i].z << " " << endl;
    }
    f_debug << endl;
    for (size_t i = 0; i < nums.y; i++) {
      f_debug  << "f " << h_indices[i].x + 1 << " " << h_indices[i].y + 1 << " "
           << h_indices[i].z + 1 << " " << endl;
    }
    f_debug.close();
    exit(0);
#endif

    if (nums.x <= 0 || nums.y <= 0) return;
    Mesh* mesh = new Mesh(d_vertices, d_indices, nums.x, nums.y, true);
    mesh->render(sh);
    delete mesh;
  }

  ~VoxelModel() {
    cudaFree(d_blockMinMax);
    cudaFree(d_numActiveBlocks);
    cudaFree(d_activeBlkNum);
  }
};

#endif

