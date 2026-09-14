#pragma once

enum cudaChannelFormatKind { cudaChannelFormatKindSigned, cudaChannelFormatKindUnsigned, cudaChannelFormatKindFloat, cudaChannelFormatKindNone };
typedef struct cudaChannelFormatDesc { int x, y, z, w; cudaChannelFormatKind f; } cudaChannelFormatDesc;
template <typename T> inline cudaChannelFormatDesc cudaCreateChannelDesc();
template <> inline cudaChannelFormatDesc cudaCreateChannelDesc<float>() { return {32, 0, 0, 0, cudaChannelFormatKindFloat}; }
template <> inline cudaChannelFormatDesc cudaCreateChannelDesc<int>() { return {32, 0, 0, 0, cudaChannelFormatKindSigned}; }
template <> inline cudaChannelFormatDesc cudaCreateChannelDesc<unsigned int>() { return {32, 0, 0, 0, cudaChannelFormatKindUnsigned}; }
