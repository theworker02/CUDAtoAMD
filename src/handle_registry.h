#pragma once
#include <memory>
#include <mutex>
#include <new>
#include <thread>
#include <unordered_map>

namespace amd_handles {
struct Entry {
  void *pointer;
  const void *type;
  void (*destroy)(void *);
  bool live;
  size_t pins = 0;
  Entry(void *p, const void *t, void (*d)(void *))
      : pointer(p), type(t), destroy(d), live(true) {}
  ~Entry() { destroy(pointer); }
};
inline std::recursive_mutex mutex;
inline std::unordered_map<const void *, std::unique_ptr<Entry>> entries;
inline bool capturing = false;
inline std::thread::id capture_thread;
inline bool pinned(const void *p) {
  auto it = entries.find(p);
  return it != entries.end() && it->second->pins;
}
template <class T> inline const void *tag() {
  static const int value = 0;
  return &value;
}
template <class T> bool valid(T *p) {
  auto it = entries.find(p);
  return it != entries.end() && it->second->live &&
         it->second->type == tag<T>();
}
template <class T> T *track(T *p) noexcept {
  if (!p)
    return nullptr;
  std::unique_ptr<T> pending(p);
  try {
    // Tombstones prevent address reuse from making a stale handle valid again.
    // Cap total metadata; fail allocation rather than grow without bound.
    if (entries.size() >= 1000000)
      return nullptr;
    auto entry = std::make_unique<Entry>(
        p, tag<T>(), [](void *v) { delete static_cast<T *>(v); });
    pending.release();
    entries.emplace(p, std::move(entry));
    return p;
  } catch (...) {
    return nullptr;
  }
}
template <class T> void retire(T *p) {
  if (p)
    entries.at(p)->live = false;
}
} // namespace amd_handles
