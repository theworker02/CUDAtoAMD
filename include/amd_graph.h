#pragma once
#include "amd_runtime.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct AmdGraph_t *AmdGraph;
/* Thread-local single-stream capture. Preallocate all resources before begin.
 * End instantiates the executable. Destroy releases conservative resource pins.
 * This is a native HIP graph API, not the full CUDA graph API contract. */
AmdResult amdGraphBegin(AmdStream stream, AmdGraph *graph);
AmdResult amdGraphEnd(AmdGraph graph);
AmdResult amdGraphLaunch(AmdGraph graph, AmdStream stream);
AmdResult amdGraphDestroy(AmdGraph graph);
#ifdef __cplusplus
}
#endif
