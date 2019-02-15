/*
 *
 */

#ifndef BLOCK_ALLOCATOR_H_U74O6
#define BLOCK_ALLOCATOR_H_U74O6

#include <exception>
#include <memory>
#include <mutex>
#include <tuple>
#include <vector>

#include <zisa/config.hpp>

namespace zisa {

template <class T>
class block_allocator;

template <class T>
class locked_ptr {
public:
  locked_ptr() : data(nullptr), index(0), allocator(nullptr) {}

  locked_ptr(T *const data,
             int_t index,
             std::shared_ptr<block_allocator<T>> allocator)
      : data(data), index(index), allocator(std::move(allocator)) {}

  locked_ptr(const locked_ptr &other) = delete;

  locked_ptr(locked_ptr &&other) { *this = std::move(other); }

  ~locked_ptr() {
    if (data != nullptr && allocator != nullptr) {
      allocator->release(*this);
    }
  }

  void operator=(const locked_ptr &other) = delete;
  void operator=(locked_ptr &&other) {

    data = other.data;
    other.data = nullptr;

    index = other.index;
    other.index = 0;

    allocator = std::move(other.allocator);
    other.allocator = nullptr;
  }

  T *operator->() { return data; }
  T const *operator->() const { return data; }

  T &operator*() { return *data; }
  const T &operator*() const { return *data; }

private:
  T *data;

  int_t index;
  std::shared_ptr<block_allocator<T>> allocator;

  friend class block_allocator<T>;
};

template <class T>
class block_allocator
    : public std::enable_shared_from_this<block_allocator<T>> {
public:
  block_allocator(int_t max_elements) {
    free_list.reserve(max_elements);
    large_block.reserve(max_elements);
  }

  ~block_allocator() {
    for (auto &ptr : large_block) {
      delete ptr;
    }
  }

  template <class... Args>
  locked_ptr<T> allocate(Args &&... args) {
    auto [index, allocated] = acquire();

    if (!allocated) {
      large_block[index] = new T(std::forward<Args &&>(args)...);
    }

    return locked_ptr<T>(large_block[index], index, this->shared_from_this());
  }

  std::tuple<int_t, bool> acquire() {
    auto lock = std::lock_guard(mutex);

    if (free_list.empty()) {

      if (large_block.size() == large_block.capacity()) {
        throw std::length_error("Exceeding capacity of the block_allocator.");
      }

      large_block.push_back(nullptr);
      return std::tuple(large_block.size() - 1, false);
    }

    auto index = free_list.back();
    free_list.pop_back();
    return std::tuple(index, true);
  }

  void release(const locked_ptr<T> &ptr) {
    auto lock = std::lock_guard(mutex);
    free_list.push_back(ptr.index);
  }

private:
  std::vector<int_t> free_list;
  std::vector<T *> large_block;
  std::mutex mutex;
};

} // namespace zisa

#endif /* end of include guard */
