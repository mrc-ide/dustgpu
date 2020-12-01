#ifndef DUST_CONTAINERS_CUH
#define DUST_CONTAINERS_CUH

#include <cstdint>
#include <cstddef>
#include <cstdlib> // malloc
#include <cstring> // memcpy

namespace dust {

// TODO: eventually we could consider making data a fixed length
// array and templating the size. Not really sure how much better
// this would be

// CUDA: change to cudaMalloc, cudaMemcpy
// CUDA: Add memcpy methods to pull back to host
// CUDA: Use cudaGetSymbolAddress() when doing the move on __host__
template <typename T>
class DeviceArray {
public:
  // Default constructor
  DeviceArray() : data_(nullptr), size_(0), stride_(1) {}
  // Constructor to allocate empty memory
  DeviceArray(size_t size, size_t stride) : size_(size), stride_(stride) {
    // DEBUG
    printf("malloc of size %lu\n", size_ * sizeof(T));
    data_ = (T*) std::malloc(size_ * sizeof(T));
    if (!data_) {
      throw std::runtime_error("malloc failed");
    }
    // DEBUG
    printf("memset (constructor)\n");
    std::memset(data_, 0, size_);
  }
  // Constructor from vector
  DeviceArray(std::vector<T>& data, size_t stride)
    : size_(data.size()),
      stride_(stride) {
    // DEBUG
    printf("malloc of size %lu\n", size_ * sizeof(T));
    data_ = (T*) std::malloc(size_ * sizeof(T));
    if (!data_) {
      throw std::runtime_error("malloc failed");
    }
    // DEBUG
    printf("memcpy (constructor)\n");
    std::memcpy(data_, data.data(), size_ * sizeof(T));
  }
  // TODO: should we just '= delete' the rule of five methods below?
  // Copy
  DeviceArray(const DeviceArray& other)
    : size_(other.size_),
      stride_(other.stride_) {
      // DEBUG
      printf("memcpy (copy)\n");
      std::memcpy(data_, other.data_, size_ * sizeof(T));
  }
  // Copy assign
  DeviceArray& operator=(const DeviceArray& other) {
    if (this != &other) {
      std::free(data_);
      size_ = other.size_;
      stride_ = other.stride_;
      // DEBUG
      printf("memcpy (copy assign)\n");
      std::memcpy(data_, other.data_, size_ * sizeof(T));
    }
    return *this;
  }
  // Move
  DeviceArray(DeviceArray&& other) : data_(nullptr), size_(0), stride_(1) {
    data_ = other.data_;
    size_ = other.size_;
    stride_ = other.stride_;
    other.data_ = nullptr;
    other.size_ = 0;
    other.stride_ = 1;
  }
  // Move assign
  DeviceArray& operator=(DeviceArray&& other) {
    if (this != &other) {
      // DEBUG
      printf("free (move assign)\n");
      std::free(data_);
      data_ = other.data_;
      size_ = other.size_;
      stride_ = other.stride_;
      other.data_ = nullptr;
      other.size_ = 0;
      other.stride_ = 1;
    }
    return *this;
  }
  ~DeviceArray() {
    // DEBUG
    printf("free (destructor)\n");
    std::free(data_);
  }
  void getArray(std::vector<T>& dst) const {
    // DEBUG
    printf("memcpy (D->H)\n");
    std::memcpy(dst.data(), data_, size_ * sizeof(T));
  }
  void setArray(std::vector<T>& src) {
    size_ = src.size();
    // DEBUG
    printf("memcpy (H->D)\n");
    std::memcpy(data_, src.data(), size_ * sizeof(T));
  }
  T* data() {
    return data_;
  }
  size_t size() const {
    return size_;
  }
  size_t stride() const {
    return stride_;
  }
private:
  T* data_;

  // CUDA: these need to be malloc'd, or passed as args to the kernel
  // OR, as they aren't actually used by this class any more, we could
  // just get rid of them and keep in the interleaved class instead
  size_t size_;
  size_t stride_;
};

// CUDA: mark all class methods below as __device__ (maybe also __host__)
// The class from before, which is a light wrapper around a pointer
// This can be used within a kernel with copying memory
// Issue is that there's no way to tell whether the data being referred to
// has been freed or not, so a segfault could well be on the cards.
// tbh the previous implementation also had this issue, accessing a pointer
// through [] it does not control
template <typename T>
class interleaved {
public:
  interleaved(DeviceArray<T>& data, size_t offset)
    : data_(data.data() + offset),
      stride_(data.stride()) {}
  interleaved(T* data, size_t stride)
    : data_(data),
      stride_(stride) {}
  // I feel like there might be some way to define these with inheritance
  // but not sure as this would be the base class, and it would take the child
  // class in the constructor
  T& operator[](size_t i) {
    return data_[i * stride_];
  }
  const T& operator[](size_t i) const {
    return data_[i * stride_];
  }
  interleaved<T> operator+(size_t by) {
    return interleaved(data_ + by * stride_, stride_);
  }
  const interleaved<T> operator+(size_t by) const {
    return interleaved(data_ + by * stride_, stride_);
  }
private:
  T* data_;
  size_t stride_;
};

}

#endif