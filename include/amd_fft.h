#pragma once
#include "amd_runtime.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct AmdFftPlan_t* AmdFftPlan;
typedef struct AmdComplex32 { float real, imag; } AmdComplex32;
/* Contiguous, batched, 1D complex FP32 transforms. Inverse is unnormalized.
 * direction=-1 forward, +1 inverse. Input/output may be exactly equal for
 * in-place execution; otherwise they must not overlap. Keep buffers and stream
 * alive until synchronization. Destroy synchronizes the plan's device. */
AmdResult amdFftPlan1d(AmdDevice device,int length,int batch_count,AmdFftPlan* plan);
AmdResult amdFftExecC2C(AmdFftPlan plan,AmdStream stream,AmdDeviceAddr input,AmdDeviceAddr output,int direction);
AmdResult amdFftDestroy(AmdFftPlan plan);
#ifdef __cplusplus
}
#endif
