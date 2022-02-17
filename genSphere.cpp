#include <stdlib.h>

#include <fstream>
#include <iostream>

using namespace std;

int main() {
  int nx = 16, ny = 16, nz = 16;
  int* data = (int*)malloc(nx * ny * nz * sizeof(int));

  ofstream file("sphere.dat");
  file.write((char*)&nx, sizeof(int));
  file.write((char*)&ny, sizeof(int));
  file.write((char*)&nz, sizeof(int));

  int idx = 0;
  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++)
      for (int k = 0; k < nz; k++) {
        int val = i * i + j * j + k * k;
        data[k + nz * j + nz * ny * i] = val;
        file.write((char*)&val, sizeof(int));
      }

  for (int k = 0; k < nz; k++) {
    for (int i = 0; i < nx; i++) {
      for (int j = 0; j < ny; j++) {
        cout << data[k + nz * j + nz * ny * i] << " ";
      }
      cout << endl;
    }
    cout << "\n\n\n" << endl;
  }

  free(data);
  file.close();
}

