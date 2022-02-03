#ifndef VOXELLOADER_HPP
#define VOXELLOADER_HPP

#include <stdio.h>

#include <iostream>
#include <string>
#include <vector>

using namespace std;

class VoxelLoader {
 public:
  unsigned short int* pData;
  int n_x, n_y, n_z;

  VoxelLoader(string path) {
    FILE* fp = fopen(path.c_str(), "rb");

    unsigned short* vuSize[3];
    fread((void*)vuSize, 3, sizeof(unsigned short), fp);

    n_x = int(vuSize[0]);
    n_y = int(vuSize[1]);
    n_z = int(vuSize[2]);

    int uCount = int(vuSize[0]) * int(vuSize[1]) * int(vuSize[2]);
    pData = new unsigned short[uCount];
    fread((void*)pData, uCount, sizeof(unsigned short), fp);
    fclose(fp);
  }
};

#endif

