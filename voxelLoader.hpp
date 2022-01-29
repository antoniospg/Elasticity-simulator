#include <stdio.h>

#include <iostream>
#include <string>
#include <vector>

using namespace std;

class VoxelLoader {
 private:
  vector<vector<vector<unsigned short>>> voxel_val;

 public:
  VoxelLoader(string path) {
    FILE* fp = fopen(path.c_str(), "rb");
    assert(fp != nullptr);

    unsigned short vuSize[3];
    fread((void*)vuSize, 3, sizeof(unsigned short), fp);

    voxel_val = vector<vector<vector<unsigned short>>>(
        vuSize[0], vector<vector<unsigned short>>(
                       vuSize[1], vector<unsigned short>(vuSize[2], -1)));

    int uCount = int(vuSize[0]) * int(vuSize[1]) * int(vuSize[2]);
    unsigned short* pData = new unsigned short[uCount];
    fread((void*)pData, uCount, sizeof(unsigned short), fp);
    fclose(fp);

    int count = 0;
    for (int i = 0; i < vuSize[0]; i++)
      for (int j = 0; j < vuSize[1]; j++)
        for (int k = 0; k < vuSize[2]; k++) {
          voxel_val[i][j][k] = pData[count];
          count++;
        }
  }
};
