#ifndef VOXELLOADER_HPP
#define VOXELLOADER_HPP

#define FIRST 
#include <stdio.h>

#include <iostream>
#include <string>
#include <vector>

using namespace std;

class VoxelLoader {
 public:
  int* pData;
  unsigned char* cData;
  unsigned short* sData;
  int n_x, n_y, n_z;

  VoxelLoader() {}

  VoxelLoader(string path) {
    FILE* fp = fopen(path.c_str(), "rb");

#ifdef FIRST
    // n_x = vuSize[0];
    // n_y = vuSize[1];
    // n_z = vuSize[2];
    n_x = 256;
    n_y = 256;
    n_z = 256;

    int uCount = n_x * n_z * n_z;
    pData = new int[uCount];
    cData = new unsigned char[uCount];
    fread((void*)cData, uCount, sizeof(char), fp);
    fclose(fp);

    for (size_t i = 0; i < uCount; i++) pData[i] = cData[i];
#endif

#ifdef SECOND
    int vuSize[3];
    fread((void*)vuSize, 3, sizeof(int), fp);

    n_x = vuSize[0];
    n_y = vuSize[1];
    n_z = vuSize[2];

    int uCount = n_x * n_z * n_z;
    pData = new int[uCount];
    fread((void*)pData, uCount, sizeof(int), fp);
    fclose(fp);
#endif

#ifdef THIRD
    unsigned short int vuSize[3];
    fread((void*)vuSize, 3, sizeof(unsigned short), fp);

    n_x = vuSize[0];
    n_y = vuSize[1];
    n_z = vuSize[2];

    n_x = 832;
    n_y = 600;
    n_z = 400;

    int uCount = n_x * n_z * n_z;
    sData = new unsigned short[uCount];
    pData = new int[uCount];
    fread((void*)sData, uCount, sizeof(unsigned short), fp);
    fclose(fp);

    for (size_t i = 0; i < uCount; i++) pData[i] = (unsigned int)sData[i];
#endif
  }
};

#endif

