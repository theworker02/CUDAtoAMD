#pragma once
#include "cuda_runtime_api.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef int cufftHandle;
typedef int cufftType;
typedef struct {
  float x, y;
} cufftComplex;
typedef enum {
  CUFFT_SUCCESS = 0,
  CUFFT_INVALID_PLAN = 1,
  CUFFT_ALLOC_FAILED = 2,
  CUFFT_INVALID_TYPE = 3,
  CUFFT_INVALID_VALUE = 4,
  CUFFT_INTERNAL_ERROR = 5,
  CUFFT_NOT_SUPPORTED = 16
} cufftResult;
enum { CUFFT_C2C = 0x29, CUFFT_FORWARD = -1, CUFFT_INVERSE = 1 };
COMPATCUDA_API cufftResult cufftPlan1d(cufftHandle *plan, int length,
                                       cufftType type, int batch);
COMPATCUDA_API cufftResult cufftExecC2C(cufftHandle plan, cufftComplex *input,
                                        cufftComplex *output, int direction);
COMPATCUDA_API cufftResult cufftDestroy(cufftHandle plan);
#ifdef __cplusplus
}
#endif
