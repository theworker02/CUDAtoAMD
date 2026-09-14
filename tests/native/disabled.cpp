#include "amd_blas.h"
#include "amd_runtime.h"
#include "amd_graph.h"
int main() {
  int count = 42;
  AmdModule module = nullptr;
  AmdBlasHandle blas = nullptr;
  AmdGraph graph=nullptr;
  return amdGraphBegin(nullptr,&graph)==AMD_ERROR_NOT_INITIALIZED && !graph &&
                 amdInit(0) == AMD_ERROR_NOT_INITIALIZED &&
                 amdGetDeviceCount(&count) == AMD_ERROR_NOT_INITIALIZED &&
                 count == 0 &&
                 amdModuleLoadFile(nullptr, "missing.hsaco", &module) ==
                     AMD_ERROR_NOT_INITIALIZED &&
                 amdBlasCreate(&blas) == AMD_ERROR_NOT_INITIALIZED && !blas &&
                 amdGetDeviceCount(nullptr) == AMD_ERROR_INVALID_VALUE
             ? 0
             : 1;
}
