#include <stdio.h>

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
}
