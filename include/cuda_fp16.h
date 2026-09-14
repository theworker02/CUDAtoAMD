#pragma once

/* Storage-only half type for source compatibility. Arithmetic lowering is a compiler concern. */
struct __half { unsigned short x; };
typedef __half half;
