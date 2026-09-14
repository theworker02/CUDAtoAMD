#pragma once

#include "vector_types.h"

constexpr int2 make_int2(int x, int y) { return {x, y}; }
constexpr uint2 make_uint2(unsigned int x, unsigned int y) { return {x, y}; }
constexpr float2 make_float2(float x, float y) { return {x, y}; }
constexpr float3 make_float3(float x, float y, float z) { return {x, y, z}; }
constexpr float4 make_float4(float x, float y, float z, float w) { return {x, y, z, w}; }
constexpr double2 make_double2(double x, double y) { return {x, y}; }
