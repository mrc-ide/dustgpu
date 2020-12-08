#ifndef DUST_CONTAINERS_CUH
#define DUST_CONTAINERS_CUH

#include <cstdint>
#include <cstddef>
#include <cstdlib> // malloc
#include <cstring> // memcpy
#include <vector>

#include "cuda.cuh"

namespace dust {

// This manages memory on the device, allocating, copying and freeing
// Include as a class member, use getArray and setArray to copy between
// host and device.
template <typename T>
class DeviceArray {
public:
  // Default constructor
  DeviceArray() : data_(nullptr), size_(0) {}
  // Constructor to allocate empty memory
  DeviceArray(const size_t size) : size_(size) {
#ifdef __NVCC__
    CUDA_CALL(cudaMalloc((void**)&data_, size_ * sizeof(T)));
    CUDA_CALL(cudaMemset(data_, 0, size_ * sizeof(T)));
#else
    data_ = (T*) std::malloc(size_ * sizeof(T));
    if (!data_) {
      cpp11::stop("malloc failed");
    }
    std::memset(data_, 0, size_ * sizeof(T));
#endif
  }
  // Constructor from vector
  DeviceArray(const std::vector<T>& data)
    : size_(data.size()) {
#ifdef __NVCC__
    CUDA_CALL(cudaMalloc((void**)&data_, size_ * sizeof(T)));
    CUDA_CALL(cudaMemcpy(data_, data.data(), size_ * sizeof(T),
                         cudaMemcpyDefault));
#else
    data_ = (T*) std::malloc(size_ * sizeof(T));
    if (!data_) {
      cpp11::stop("malloc failed");
    }
    std::memcpy(data_, data.data(), size_ * sizeof(T));
#endif
  }
  // Copy
  DeviceArray(const DeviceArray& other)
    : size_(other.size_) {
#ifdef __NVCC__
    CUDA_CALL(cudaMemcpy(data_, other.data_, size_ * sizeof(T),
                         cudaMemcpyDefault));
#else
    std::memcpy(data_, other.data_, size_ * sizeof(T));
#endif
  }
  // Copy assign
  DeviceArray& operator=(const DeviceArray& other) {
    if (this != &other) {
      size_ = other.size_;
#ifdef __NVCC__
      CUDA_CALL(cudaFree(data_));
      CUDA_CALL(cudaMemcpy(data_, other.data_, size_ * sizeof(T),
                           cudaMemcpyDefault));
#else
      std::free(data_);
      std::memcpy(data_, other.data_, size_ * sizeof(T));
#endif
    }
    return *this;
  }
  // Move
  DeviceArray(DeviceArray&& other) : data_(nullptr), size_(0) {
    data_ = other.data_;
    size_ = other.size_;
    other.data_ = nullptr;
    other.size_ = 0;
  }
  // Move assign
  DeviceArray& operator=(DeviceArray&& other) {
    if (this != &other) {
#ifdef __NVCC__
      CUDA_CALL(cudaFree(data_));
#else
      std::free(data_);
#endif
      data_ = other.data_;
      size_ = other.size_;
      other.data_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }
  ~DeviceArray() {
#ifdef __NVCC__
    CUDA_CALL(cudaFree(data_));
#else
    std::free(data_);
#endif
  }
  void getArray(std::vector<T>& dst) const {
    if (dst.size() > size_) {
      cpp11::stop("Tried D->H copy with device array (%i) shorter than host array (%i)\n",
                  size_, dst.size());
    }
#ifdef __NVCC__
    CUDA_CALL(cudaMemcpy(dst.data(), data_, dst.size() * sizeof(T),
                         cudaMemcpyDefault));
#else
    std::memcpy(dst.data(), data_, dst.size() * sizeof(T));
#endif
  }
  void setArray(const std::vector<T>& src) {
    if (src.size() > size_) {
      cpp11::stop("Tried H->D copy with host array (%i) longer than device array (%i)\n",
                  src.size(), size_);
    } else {
      size_ = src.size();
    }
#ifdef __NVCC__
    CUDA_CALL(cudaMemcpy(data_, src.data(), size_ * sizeof(T),
                         cudaMemcpyDefault));
#else
    std::memcpy(data_, src.data(), size_ * sizeof(T));
#endif
  }
  T* data() {
    return data_;
  }
  size_t size() const {
    return size_;
  }
private:
  T* data_;
  size_t size_;
};

// Specialisation of the above for void* memory needed by some cub functions
// Construct once and use set_size() to modify
template <>
class DeviceArray<void> {
public:
  // Default constructor
  DeviceArray() : data_(nullptr), size_(0) {}
  // Constructor to allocate empty memory
  DeviceArray(const size_t size) : size_(size) {
    if (size_ > 0) {
#ifdef __NVCC__
      CUDA_CALL(cudaMalloc((void**)&data_, size_));
#else
      data_ = (void*) std::malloc(size_);
      if (!data_) {
        cpp11::stop("malloc failed");
      }
#endif
    }
  }
  ~DeviceArray() {
#ifdef __NVCC__
    CUDA_CALL(cudaFree(data_));
#else
    std::free(data_);
#endif
  }
  void set_size(size_t size) {
    size_ = size;
#ifdef __NVCC__
    CUDA_CALL(cudaFree(data_));
    if (size_ > 0) {
      CUDA_CALL(cudaMalloc((void**)&data_, size_));
    }
#else
    std::free(data_);
    if (size_ > 0) {
      data_ = (void*) std::malloc(size_);
      if (!data_) {
        cpp11::stop("malloc failed");
      }
    }
#endif
  }
  void* data() {
    return data_;
  }
  size_t size() const {
    return size_;
  }
  size_t& size() {
    return size_;
  }
private:
  DeviceArray ( const DeviceArray<void> & ) = delete;
  DeviceArray ( DeviceArray<void> && ) = delete;

  void* data_;
  size_t size_;
};

// The class from before, which is a light wrapper around a pointer
// This can be used within a kernel with copying memory
// Issue is that there's no way to tell whether the data being referred to
// has been freed or not, so a segfault could well be on the cards.
// tbh the previous implementation also had this issue, accessing a pointer
// through [] it does not control
template <typename T>
class interleaved {
public:
  DEVICE interleaved(DeviceArray<T>& data, size_t offset, size_t stride)
    : data_(data.data() + offset),
      stride_(stride) {}
  DEVICE interleaved(T* data, size_t stride)
    : data_(data),
      stride_(stride) {}
  DEVICE T& operator[](size_t i) {
    return data_[i * stride_];
  }
  DEVICE const T& operator[](size_t i) const {
    return data_[i * stride_];
  }
  DEVICE interleaved<T> operator+(size_t by) {
    return interleaved(data_ + by * stride_, stride_);
  }
  DEVICE const interleaved<T> operator+(size_t by) const {
    return interleaved(data_ + by * stride_, stride_);
  }
private:
  T* data_;
  size_t stride_;
};

}

#endif