#pragma once

#include "cuda_runtime_api.h"

typedef struct cudaUUID_t { char bytes[16]; } cudaUUID_t;
typedef struct cudaPitchedPtr { void* ptr; size_t pitch; size_t xsize; size_t ysize; } cudaPitchedPtr;
typedef struct cudaExtent { size_t width; size_t height; size_t depth; } cudaExtent;
typedef struct cudaPos { size_t x; size_t y; size_t z; } cudaPos;
#ifdef __cplusplus
constexpr cudaExtent make_cudaExtent(size_t w, size_t h, size_t d) { return {w, h, d}; }
constexpr cudaPos make_cudaPos(size_t x, size_t y, size_t z) { return {x, y, z}; }
#else
static inline cudaExtent make_cudaExtent(size_t w, size_t h, size_t d) { cudaExtent v = {w, h, d}; return v; }
static inline cudaPos make_cudaPos(size_t x, size_t y, size_t z) { cudaPos v = {x, y, z}; return v; }
#endif
