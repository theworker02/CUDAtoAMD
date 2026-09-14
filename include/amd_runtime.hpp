#pragma once

#include "amd_runtime.h"
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace amd {
class RuntimeError final : public std::runtime_error {
public:
  explicit RuntimeError(AmdResult value)
      : std::runtime_error(amdGetErrorString(value)), value_(value) {}
  AmdResult status() const noexcept { return value_; }

private:
  AmdResult value_;
};
inline void check(AmdResult value) {
  if (value != AMD_SUCCESS)
    throw RuntimeError(value);
}
class Context {
public:
  explicit Context(uint32_t flags = 0) { check(amdInit(flags)); }
  static int device_count() {
    int value{};
    check(amdGetDeviceCount(&value));
    return value;
  }
};
class Device {
public:
  explicit Device(int index = 0) { check(amdGetDevice(&value_, index)); }
  ~Device() {
    if (value_)
      amdDeviceDestroy(value_);
  }
  Device(const Device &) = delete;
  Device &operator=(const Device &) = delete;
  Device(Device &&o) noexcept : value_(std::exchange(o.value_, nullptr)) {}
  AmdDevice raw() const noexcept { return value_; }
  AmdDeviceProps properties() const {
    AmdDeviceProps value{};
    check(amdGetDeviceProperties(value_, &value));
    return value;
  }

private:
  AmdDevice value_{};
};
class Stream {
  friend class DeviceBuffer;

public:
  explicit Stream(const Device &device) {
    AmdStream p{};
    check(amdStreamCreate(device.raw(), &p));
    value_ = {p, [](AmdStream s) noexcept {
                amdStreamSynchronize(s);
                amdStreamDestroy(s);
              }};
  }
  ~Stream() = default;
  Stream(const Stream &) = delete;
  Stream &operator=(const Stream &) = delete;
  Stream(Stream &&other) noexcept : value_(std::move(other.value_)) {}
  Stream &operator=(Stream &&other) noexcept {
    if (this != &other) {
      value_ = std::move(other.value_);
    }
    return *this;
  }
  void synchronize() const {
    if (!value_) throw RuntimeError(AMD_ERROR_INVALID_HANDLE);
    check(amdStreamSynchronize(raw()));
  }
  bool ready() const {
    if (!value_) throw RuntimeError(AMD_ERROR_INVALID_HANDLE);
    auto r = amdStreamQuery(raw());
    if (r == AMD_SUCCESS)
      return true;
    if (r == AMD_ERROR_NOT_READY)
      return false;
    check(r);
    return false;
  }
  AmdStream raw() const noexcept { return value_.get(); }

private:
  std::shared_ptr<AmdStream_t> value_;
};
class MemoryPool {
  friend class DeviceBuffer;

public:
  MemoryPool(const Device &device, size_t capacity = 0) {
    AmdMemPool p{};
    check(amdMemPoolCreate(device.raw(), capacity, &p));
    value_ = {p, [](AmdMemPool pool) noexcept { amdMemPoolDestroy(pool); }};
  }
  ~MemoryPool() = default;
  MemoryPool(const MemoryPool &) = delete;
  MemoryPool &operator=(const MemoryPool &) = delete;
  MemoryPool(MemoryPool &&o) noexcept : value_(std::move(o.value_)) {}
  AmdDeviceAddr allocate(size_t bytes, const Stream &s) {
    if (!value_ || !s.raw()) throw RuntimeError(AMD_ERROR_INVALID_HANDLE);
    AmdDeviceAddr p{};
    check(amdMemAllocAsync(&p, bytes, raw(), s.raw()));
    return p;
  }
  void free(AmdDeviceAddr p, const Stream &s) {
    if (!value_ || !s.raw()) throw RuntimeError(AMD_ERROR_INVALID_HANDLE);
    check(amdMemFreeAsync(p, raw(), s.raw()));
  }
  AmdMemPool raw() const noexcept { return value_.get(); }

private:
  std::shared_ptr<AmdMemPool_t> value_;
};
class DeviceBuffer {
public:
  DeviceBuffer(MemoryPool &p, size_t n, const Stream &s)
      : pool_(p.value_), stream_(s.value_), bytes_(n) {
    ptr_ = p.allocate(n, s);
  }
  ~DeviceBuffer() noexcept {
    if (ptr_) {
      amdMemFreeAsync(ptr_, pool_.get(), stream_.get());
      amdStreamSynchronize(stream_.get());
    }
  }
  void close() {
    if (ptr_) {
      check(amdMemFreeAsync(ptr_, pool_.get(), stream_.get()));
      ptr_ = 0;
    }
    check(amdStreamSynchronize(stream_.get()));
  }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer(DeviceBuffer &&o) noexcept
      : pool_(o.pool_), stream_(o.stream_), bytes_(o.bytes_),
        ptr_(std::exchange(o.ptr_, 0)) {}
  void copy_from(const void *src, size_t n) {
    if (!ptr_) throw RuntimeError(AMD_ERROR_INVALID_HANDLE);
    if (n > bytes_)
      throw std::out_of_range("host copy exceeds buffer");
    check(amdMemcpyHtoDAsync(ptr_, src, n, stream_.get()));
  }
  void copy_to(void *dst, size_t n) const {
    if (!ptr_) throw RuntimeError(AMD_ERROR_INVALID_HANDLE);
    if (n > bytes_)
      throw std::out_of_range("host copy exceeds buffer");
    check(amdMemcpyDtoHAsync(dst, ptr_, n, stream_.get()));
  }
  AmdDeviceAddr address() const noexcept { return ptr_; }

private:
  std::shared_ptr<AmdMemPool_t> pool_;
  std::shared_ptr<AmdStream_t> stream_;
  size_t bytes_;
  AmdDeviceAddr ptr_{};
};
class Module {
public:
  Module(const Device &d, const std::string &path) {
    check(amdModuleLoadFile(d.raw(), path.c_str(), &value_));
  }
  ~Module() {
    if (value_)
      amdModuleUnload(value_);
  }
  Module(const Module &) = delete;
  Module &operator=(const Module &) = delete;
  Module(Module &&other) noexcept : value_(std::exchange(other.value_, nullptr)) {}
  AmdFunction function(const std::string &name) const {
    AmdFunction f{};
    check(amdModuleGetFunction(&f, value_, name.c_str()));
    return f;
  }
  void launch(AmdFunction f, AmdDim3 grid, AmdDim3 block, uint32_t shared,
              const Stream &s, std::vector<void *> &args) const {
    if (!value_ || !s.raw()) throw RuntimeError(AMD_ERROR_INVALID_HANDLE);
    check(amdLaunchKernel(f, grid, block, shared, s.raw(), args.data()));
  }

private:
  AmdModule value_{};
};
} // namespace amd
