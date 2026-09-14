#pragma once

/* Internal ABI v1: CUDA facades normalize into these backend responsibilities. */
namespace compat::abi {
enum class BackendKind { unavailable, hip };
enum class Operation { device, allocation, copy, stream, event, driver_context, driver_memory };
}
