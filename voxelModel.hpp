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

  dim3 block_size3;
  dim3 grid_size3;
  uint3 nxyz;
  // Global offset
  int* d_block_vertex_offset;
  int* d_block_index_offset;

  // store vertices / indices
  vert3* d_vertices;
  int3* d_indices;

 public:
  VoxelModel(string path) : vl(path), ct(vl.pData, vl.n_x, vl.n_y, vl.n_z) {
    n_x = vl.n_x, n_y = vl.n_y, n_z = vl.n_z;
    uint n = n_x * n_y * n_z;

    // First Kernel Launch
    block_size = {8, 8, 8};
    grid_size = {(n_x + (block_size.x - 1) - 1) / (block_size.x - 1),
                 (n_y + (block_size.y - 1) - 1) / (block_size.y - 1),
                 (n_z + (block_size.z - 1) - 1) / (block_size.z - 1)};
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
      cout << h_minmax[i].x << " " << h_minmax[i].y << endl;
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

    // Third Kernel params
    block_size3 = block_size;
    grid_size3 = {numActiveBlk};
    nxyz = uint3{n_x, n_y, n_z};

    cudaMalloc(&d_block_vertex_offset, (grid_size3.x + 1) * sizeof(int));
    cudaMalloc(&d_block_index_offset, (grid_size3.x + 1) * sizeof(int));

    cudaMalloc(&d_vertices, nxyz.x * nxyz.y * nxyz.z * sizeof(vert3));
    cudaMalloc(&d_indices, nxyz.x * nxyz.y * nxyz.z * sizeof(int3));
  }

  int2 generate(int isoVal) {
    int num_blocks = grid_size.x * grid_size.y * grid_size.z;

    // Third Kernel Launch
    int2 nums = genTriangles::generateTrisWrapper(
        ct.texObj, ct.texObjNormal, d_activeBlkNum, d_numActiveBlk, grid_size3,
        block_size3, grid_size, isoVal, nxyz, d_block_vertex_offset,
        d_block_index_offset, d_vertices, d_indices);

    return nums;
  }

  void draw(Shader& sh, int isoVal) {
    int2 nums = generate(isoVal);

#ifdef DEBUG3
    ofstream fp("debug.obj");
    vert3* h_vertices = new vert3[nums.x];
    int3* h_indices = new int3[nums.y];
    cudaMemcpy(h_vertices, d_vertices, nums.x * sizeof(vert3),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_indices, d_indices, nums.y * sizeof(int3),
               cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < nums.x; i++) {
      cout << "v " << h_vertices[i].pos.x << " " << h_vertices[i].pos.y << " "
           << h_vertices[i].pos.z << " " << endl;
    }

    cout << endl;

    for (size_t i = 0; i < nums.y; i++) {
      cout << "f " << h_indices[i].x + 1 << " " << h_indices[i].y + 1 << " "
           << h_indices[i].z + 1 << " " << endl;
    }
    fp.close();
    cout << "##################" << endl;
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
    cudaFree(d_block_vertex_offset);
    cudaFree(d_block_index_offset);
    cudaFree(d_vertices);
    cudaFree(d_indices);
  }
};

#endif

