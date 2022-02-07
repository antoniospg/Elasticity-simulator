#include <assert.h>

#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}

#define cdpErrchk(ans) \
  { cdpAssert((ans), __FILE__, __LINE__); }
__device__ void cdpAssert(cudaError_t code, const char *file, int line,
                          bool abort = true) {
  if (code != cudaSuccess) {
    printf("GPU kernel assert: %s %s %d\n", cudaGetErrorString(code), file,
           line);
    if (abort) assert(0);
  }
}
