#pragma once
#include "amd_runtime.h"
#include <hip/hip_runtime_api.h>

namespace amd_native {
// Internal helpers accept live runtime-owned handles only.
bool device_index(AmdDevice device, int *index);
bool stream_device(AmdStream stream, int *index);
bool stream_handle(AmdStream stream, int device, hipStream_t *native);
class DeviceGuard {
  int previous_ = -1;

public:
  hipError_t status;
  explicit DeviceGuard(int device) : status(hipGetDevice(&previous_)) {
    if (status == hipSuccess && previous_ != device)
      status = hipSetDevice(device);
  }
  ~DeviceGuard() {
    if (previous_ >= 0)
      hipSetDevice(previous_);
  }
};
} // namespace amd_native
