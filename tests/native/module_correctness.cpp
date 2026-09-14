#include "amd_runtime.hpp"
#include "cuda_runtime_api.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <type_traits>
static_assert(std::is_nothrow_destructible<amd::DeviceBuffer>::value);
static_assert(!std::is_copy_assignable<amd::Module>::value);
int main(int argc, char **argv) {
  if (argc != 2)
    return 1;
  try {
    amd::Context context;
    const int count = amd::Context::device_count();
    if (!count)
      return 77;
    // Exercise every detected device, not just the default device.
    for (int index = 0; index < count; ++index) {
      amd::Device device(index);
      if (std::string(device.properties().gcn_arch)
              .find(COMPATCUDA_TEST_ARCH) != 0)
        continue;
      if (count > 1 && cudaSetDevice((index + 1) % count) != cudaSuccess)
        return 10;
      amd::Stream stream(device);
      amd::MemoryPool pool(device, 1024 * 1024);
      constexpr int n = 257;
      std::vector<float> a(n), b(n), out(n, -12345);
      for (int i = 0; i < n; ++i) {
        a[i] = float(i) * 0.25f;
        b[i] = float(i % 7) - 3;
      }
      amd::DeviceBuffer da(pool, n * sizeof(float), stream),
          db(pool, n * sizeof(float), stream),
          dc(pool, n * sizeof(float), stream);
      // Buffers must remain valid after the wrapper owners move.
      amd::Stream moved_stream(std::move(stream));
      amd::MemoryPool moved_pool(std::move(pool));
      amd::MemoryPool wrong_pool(device);
      if (amdMemFreeAsync(da.address(), wrong_pool.raw(), moved_stream.raw()) !=
          AMD_ERROR_INVALID_VALUE)
        return 7;
      if (amdMemPoolDestroy(moved_pool.raw()) != AMD_ERROR_NOT_READY)
        return 8;
      da.copy_from(a.data(), a.size() * sizeof(float));
      db.copy_from(b.data(), b.size() * sizeof(float));
      amd::Module module(device, argv[1]);
      auto f = module.function("vector_add");
      auto pa = da.address(), pb = db.address(), pc = dc.address();
      int length = n;
      std::vector<void *> args{&pa, &pb, &pc, &length};
      module.launch(f, AMD_DIM3(3, 1, 1), AMD_DIM3(128, 1, 1), 0, moved_stream,
                    args);
      dc.copy_to(out.data(), out.size() * sizeof(float));
      moved_stream.synchronize();
      for (int i = 0; i < n; ++i)
        if (out[i] != a[i] + b[i])
          return 2;
      if (amdLaunchKernel(f, AMD_DIM3(0, 1, 1), AMD_DIM3(1, 1, 1), 0,
                          moved_stream.raw(),
                          args.data()) != AMD_ERROR_INVALID_VALUE)
        return 3;
      AmdModule invalid = nullptr;
      char bad[64]{};
      if (amdModuleLoadData(device.raw(), bad, sizeof(bad), &invalid) !=
              AMD_ERROR_INVALID_VALUE ||
          invalid)
        return 4;
      std::ifstream file(argv[1], std::ios::binary);
      std::vector<char> image((std::istreambuf_iterator<char>(file)), {});
      AmdModule from_data = nullptr;
      amd::check(amdModuleLoadData(device.raw(), image.data(), image.size(),
                                   &from_data));
      AmdModule truncated = nullptr;
      if (amdModuleLoadData(device.raw(), image.data(), 64, &truncated) !=
          AMD_ERROR_INVALID_VALUE)
        return 9;
      AmdFunction missing = nullptr;
      if (amdModuleGetFunction(&missing, from_data, "no_such_kernel") !=
              AMD_ERROR_SYMBOL_NOT_FOUND ||
          missing)
        return 5;
      AmdFunction from_data_function{};
      amd::check(
          amdModuleGetFunction(&from_data_function, from_data, "vector_add"));
      std::fill(out.begin(), out.end(), -999.0f);
      dc.copy_from(out.data(), out.size() * sizeof(float));
      amd::check(amdLaunchKernel(from_data_function, AMD_DIM3(3, 1, 1),
                                 AMD_DIM3(128, 1, 1), 0, moved_stream.raw(),
                                 args.data()));
      dc.copy_to(out.data(), out.size() * sizeof(float));
      moved_stream.synchronize();
      for (int i = 0; i < n; ++i)
        if (out[i] != a[i] + b[i])
          return 12;
      amd::check(amdModuleUnload(from_data));
      da.close();
      db.close();
      dc.close();
      if (amdMemFreeAsync(pa, moved_pool.raw(), moved_stream.raw()) !=
          AMD_ERROR_INVALID_VALUE)
        return 11;
      std::unique_ptr<amd::DeviceBuffer> surviving_buffer;
      {
        amd::Stream local_stream(device);
        amd::MemoryPool local_pool(device);
        surviving_buffer =
            std::make_unique<amd::DeviceBuffer>(local_pool, 16, local_stream);
      }
      // Pool/stream wrapper objects have now gone out of scope.
      surviving_buffer->close();
      std::cout << "vector_add: 257 exact results on "
                << device.properties().gcn_arch << "\n";
      return 0;
    }
    return 77;
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 6;
  }
}
