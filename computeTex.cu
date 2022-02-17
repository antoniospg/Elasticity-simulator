#include "computeTex.cuh"

ComputeTex::ComputeTex() {
  nx = 0;
  ny = 0;
  nz = 0;
}

ComputeTex ::ComputeTex(int *h_data, int nx, int ny, int nz) {
  // Volume size
  const cudaExtent volumeSize = make_cudaExtent(nx, ny, nz);

  // Allocate CUDA array in device memory
  cudaChannelFormatDesc channeldesc =
      cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned);
  cudaMalloc3DArray(&cuArray, &channeldesc, volumeSize);

  // copy data to 3d array
  cudaMemcpy3DParms copyParams = {0};
  copyParams.srcPtr =
      make_cudaPitchedPtr((void *)h_data, volumeSize.width * sizeof(int),
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
  texDesc.addressMode[0] = cudaAddressModeWrap;
  texDesc.addressMode[1] = cudaAddressModeWrap;
  texDesc.addressMode[2] = cudaAddressModeWrap;
  texDesc.filterMode = cudaFilterModePoint;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;

  // Create texture object
  texObj = 0;
  cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
}

ComputeTex ::~ComputeTex() {
  if (nx == 0 && ny == 0 && nz == 0) return;
  // Destroy texture object
  cudaDestroyTextureObject(texObj);
  // Free device memory
  cudaFreeArray(cuArray);
}

