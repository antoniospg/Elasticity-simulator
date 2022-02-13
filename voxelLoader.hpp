#ifndef VOXELLOADER_HPP
#define VOXELLOADER_HPP

#include <stdio.h>

#include <iostream>
#include <string>
#include <vector>

using namespace std;

typedef unsigned short usint;

class VoxelLoader {
 public:
  unsigned short int* pData;
  usint n_x, n_y, n_z;

  VoxelLoader(string path) {
    FILE* fp = fopen(path.c_str(), "rb");

    unsigned short vuSize[3];
    fread((void*)vuSize, 3, sizeof(usint), fp);

    n_x = vuSize[0];
    n_y = vuSize[1];
    n_z = vuSize[2];

    usint uCount = vuSize[0] * vuSize[1] * vuSize[2];
    pData = new usint[uCount];
    fread((void*)pData, uCount, sizeof(usint), fp);
    fclose(fp);
  }
};

#endif

