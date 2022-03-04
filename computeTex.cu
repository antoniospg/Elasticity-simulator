#include <iostream>

#include "computeTex.cuh"
using namespace std;

ComputeTex::ComputeTex() {
  nx = 0;
  ny = 0;
  nz = 0;
}

__global__ void texKernels::getNormals(cudaTextureObject_t tex, int nx, int ny,
                                       int nz, float4* ans) {
  int3 pos = {threadIdx.x + blockDim.x * blockIdx.x,
              threadIdx.y + blockDim.y * blockIdx.y,
              threadIdx.z + blockDim.z * blockIdx.z};
  int id = pos.x + nx * pos.y + nx * ny * pos.z;

  float4 dv = {0, 0, 0, 0};

  dv.x = tex3D<int>(tex, pos.x + 1, pos.y, pos.z) -
         tex3D<int>(tex, pos.x - 1, pos.y, pos.z);
  dv.x /= 2;

  dv.y = tex3D<int>(tex, pos.x, pos.y + 1, pos.z) -
         tex3D<int>(tex, pos.x, pos.y - 1, pos.z);
  dv.y /= 2;

  dv.z = tex3D<int>(tex, pos.x, pos.y, pos.z + 1) -
         tex3D<int>(tex, pos.x, pos.y, pos.z - 1);
  dv.z /= 2;

  dv.w = 1;

  ans[id] = dv;
}

ComputeTex ::ComputeTex(int* h_data, int nx, int ny, int nz) {
  // Volume size
  const cudaExtent volumeSize = make_cudaExtent(nx, ny, nz);

  // Allocate CUDA array in device memory
  cudaChannelFormatDesc channeldesc =
      cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned);
  cudaMalloc3DArray(&cuArray, &channeldesc, volumeSize);

  // copy data to 3d array
  cudaMemcpy3DParms copyParams = {0};
  copyParams.srcPtr =
      make_cudaPitchedPtr((void*)h_data, volumeSize.width * sizeof(int),
                          volumeSize.width, volumeSize.height);
  copyParams.dstArray = cuArray;
  copyParams.extent = volumeSize;
  copyParams.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&copyParams);

  // Specify texture
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = cuArray;

  // Specify texture object parameters
  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeBorder;
  texDesc.addressMode[1] = cudaAddressModeBorder;
  texDesc.addressMode[2] = cudaAddressModeBorder;
  texDesc.filterMode = cudaFilterModePoint;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;

  // Create texture object
  texObj = 0;
  cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

  // Create texture with normals
  float4* d_normal_data;
  cudaMalloc(&d_normal_data, nx * ny * nz * sizeof(float4));

  dim3 block_size = {8, 8, 8};
  dim3 grid_size;
  grid_size.x = (nx + block_size.x - 1) / block_size.x;
  grid_size.y = (ny + block_size.y - 1) / block_size.y;
  grid_size.z = (nz + block_size.z - 1) / block_size.z;

  texKernels::getNormals<<<grid_size, block_size>>>(texObj, nx, ny, nz,
                                                    d_normal_data);

  // NORMAL TEXTURE
  // Allocate CUDA array in device memory
  channeldesc =
      cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindSigned);
  cudaMalloc3DArray(&cuArrayNormal, &channeldesc, volumeSize);

  // copy data to 3d array
  copyParams = {0};
  copyParams.srcPtr = make_cudaPitchedPtr((void*)d_normal_data,
                                          volumeSize.width * sizeof(float4),
                                          volumeSize.width, volumeSize.height);
  copyParams.dstArray = cuArrayNormal;
  copyParams.extent = volumeSize;
  copyParams.kind = cudaMemcpyDeviceToDevice;
  cudaMemcpy3D(&copyParams);

  // Specify texture
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = cuArrayNormal;

  // Specify texture object parameters
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeWrap;
  texDesc.addressMode[1] = cudaAddressModeWrap;
  texDesc.addressMode[2] = cudaAddressModeWrap;
  texDesc.filterMode = cudaFilterModePoint;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;

  texObjNormal = 0;
  cudaCreateTextureObject(&texObjNormal, &resDesc, &texDesc, NULL);
  cudaFree(d_normal_data);
}

ComputeTex ::~ComputeTex() {
  if (nx == 0 && ny == 0 && nz == 0) return;
  // Destroy texture object
  cudaDestroyTextureObject(texObj);
  cudaDestroyTextureObject(texObjNormal);
  // Free device memory
  cudaFreeArray(cuArray);
  cudaFreeArray(cuArrayNormal);
}

