#include <hip/hip_runtime.h>
// Portable FP32 reference kernels. Correctness-first; not tuned WMMA kernels.
extern "C" __global__ void amd_rmsnorm(const float *x, const float *weight,
                                       float *y, int rows, int cols,
                                       float epsilon) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows)
    return;
  const float *input = x + (size_t)row * cols;
  float sum = 0;
  for (int j = 0; j < cols; ++j)
    sum += input[j] * input[j];
  float scale = rsqrtf(sum / cols + epsilon);
  for (int j = 0; j < cols; ++j)
    y[(size_t)row * cols + j] = input[j] * scale * weight[j];
}
extern "C" __global__ void amd_softmax(const float *x, float *y, int rows,
                                       int cols) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows)
    return;
  const float *input = x + (size_t)row * cols;
  float maximum = -INFINITY;
  for (int j = 0; j < cols; ++j)
    maximum = fmaxf(maximum, input[j]);
  float sum = 0;
  for (int j = 0; j < cols; ++j)
    sum += expf(input[j] - maximum);
  for (int j = 0; j < cols; ++j)
    y[(size_t)row * cols + j] = expf(input[j] - maximum) / sum;
}
extern "C" __global__ void amd_swiglu(const float *gate, const float *value,
                                      float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float g = gate[i];
    float sigmoid = g >= 0 ? 1 / (1 + expf(-g)) : expf(g) / (1 + expf(g));
    y[i] = g * sigmoid * value[i];
  }
}
extern "C" __global__ void amd_rope(const float *x, const float *cosine,
                                    const float *sine, float *y, int rows,
                                    int cols) {
  int pair = blockIdx.x * blockDim.x + threadIdx.x;
  if (pair >= rows * (cols / 2))
    return;
  int i = 2 * pair;
  float a = x[i], b = x[i + 1], c = cosine[pair], s = sine[pair];
  y[i] = a * c - b * s;
  y[i + 1] = a * s + b * c;
}
