#pragma once

/* Storage type; native BF16 math requires HIP/Clang lowering and capability checks. */
struct __nv_bfloat16 { unsigned short x; };
typedef __nv_bfloat16 nv_bfloat16;
