#include <cuda_runtime.h>
#include <amd_runtime.h>
#include <amd_graph.h>
#include <amd_blas.h>
#include <amd_fft.h>
#include <amd_runtime.hpp>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <vector_functions.h>
#include <channel_descriptor.h>
#include <texture_types.h>
#include <surface_types.h>

int main() {
  const auto point = make_float3(1.0F, 2.0F, 3.0F);
  const auto extent = make_cudaExtent(1, 2, 3);
  const auto channel = cudaCreateChannelDesc<float>();
  dim3 grid(1, 1, 1);
  AmdDim3 native_grid = AMD_DIM3(1, 1, 1);
  return point.z == 3.0F && extent.depth == 3 && channel.x == 32 && grid.x == 1 && native_grid.x == 1 ? 0 : 1;
}
