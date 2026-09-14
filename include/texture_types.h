#pragma once

#include "channel_descriptor.h"

typedef unsigned long long cudaTextureObject_t;
enum cudaTextureAddressMode { cudaAddressModeWrap, cudaAddressModeClamp, cudaAddressModeMirror, cudaAddressModeBorder };
enum cudaTextureFilterMode { cudaFilterModePoint, cudaFilterModeLinear };
typedef struct cudaTextureDesc { cudaTextureAddressMode addressMode[3]; cudaTextureFilterMode filterMode; int normalizedCoords; } cudaTextureDesc;
