#pragma once

#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "vector_types.h"

/* Device-language compatibility comes from HIP compilation, not host headers. */
#if defined(__HIPCC__) || defined(COMPATCUDA_ENABLE_HIP)
  #include <hip/hip_runtime.h>
#endif
