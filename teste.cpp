#include <math.h>
#include <stdio.h>

#include <algorithm>
#include <iostream>

using namespace std;

int main() {
  FILE* fp = fopen("./stagbeetle.dat", "rb");

  unsigned short vuSize[3];
  fread((void*)vuSize, 3, sizeof(unsigned short), fp);

  int n_x = int(vuSize[0]);
  int n_y = int(vuSize[1]);
  int n_z = int(vuSize[2]);

  cout << n_x << " " << n_y << " " << n_z << endl;

  int uCount = int(vuSize[0]) * int(vuSize[1]) * int(vuSize[2]);
  unsigned short* pData = new unsigned short[uCount];
  fread((void*)pData, uCount, sizeof(unsigned short), fp);
  fclose(fp);

  int max_power =
      max(max(pow(2, (int)log2(n_x) + 1), pow(2, (int)log2(n_y) + 1)),
          pow(2, (int)log2(n_z) + 1));

  int inx = max_power, iny = max_power, inz = max_power;
  int* idata = new int[inx*iny*inz];

  for (size_t i = 0; i < uCount; i++) {
    idata[i] = pData[i];
  }
}
