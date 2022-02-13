#include <stdlib.h>

#include <fstream>

using namespace std;

typedef unsigned short usint;

int main() {
  int nx = 16, ny = 16, nz = 16;
  usint* data = (usint*)malloc(nx * ny * nz * sizeof(usint));

  ofstream file("sphere.dat");
  file.write((char*)&nx, sizeof(usint));
  file.write((char*)&ny, sizeof(usint));
  file.write((char*)&nz, sizeof(usint));

  int idx = 0;
  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++)
      for (int k = 0; k < nz; k++) {
        uint val = i * i + j * j + k * k;
        file.write((char*)&val, sizeof(usint));
      }

  free(data);
  file.close();
}

