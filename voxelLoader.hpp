#ifndef VOXELLOADER_HPP
#define VOXELLOADER_HPP

#include <stdio.h>

#include <iostream>
#include <string>
#include <vector>

using namespace std;

class VoxelLoader {
 public:
  int* pData;
  int n_x, n_y, n_z;

  VoxelLoader(string path) {
    FILE* fp = fopen(path.c_str(), "rb");

    int vuSize[3];
    fread((void*)vuSize, 3, sizeof(int), fp);

    n_x = vuSize[0];
    n_y = vuSize[1];
    n_z = vuSize[2];

    int uCount = vuSize[0] * vuSize[1] * vuSize[2];
    pData = new int[uCount];
    fread((void*)pData, uCount, sizeof(int), fp);
    fclose(fp);
  }
};

#endif

