#include <assert.h>
#include <cuda_runtime.h>

#include <iostream>

#include "voxelModel.hpp"

using namespace std;

int main() { VoxelModel vm("sphere.dat", 100); }
