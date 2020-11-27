#ifndef DUST_DUST_HPP
#define DUST_DUST_HPP

#include <dust/rng.hpp>

#include <algorithm>
#include <sstream>
#include <utility>
#include <cstdlib> // malloc
#include <cstring> // memcpy - remove this when CUDA done
#ifdef _OPENMP
#include <omp.h>
#endif

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
    data_ = (T*) malloc(size_ * sizeof(T));
    if (!data_) {
      throw std::runtime_error("malloc failed");
    }
    // DEBUG
    printf("memset (constructor)\n");
    memset(data_, 0, size_);
  }
  // Constructor from vector
  DeviceArray(std::vector<T>& data, size_t stride)
    : size_(data.size()),
      stride_(stride) {
    // DEBUG
    printf("malloc of size %lu\n", size_ * sizeof(T));
    data_ = (T*) malloc(size_ * sizeof(T));
    if (!data_) {
      throw std::runtime_error("malloc failed");
    }
    // DEBUG
    printf("memcpy (constructor)\n");
    memcpy(data_, data.data(), size_ * sizeof(T));
  }
  // TODO: should we just '= delete' the rule of five methods below?
  // Copy
  DeviceArray(const DeviceArray& other)
    : size_(other.size_),
      stride_(other.stride_) {
      // DEBUG
      printf("memcpy (copy)\n");
      memcpy(data_, other.data_, size_ * sizeof(T));
  }
  // Copy assign
  DeviceArray& operator=(const DeviceArray& other) {
    if (this != &other) {
      free(data_);
      size_ = other.size_;
      stride_ = other.stride_;
      // DEBUG
      printf("memcpy (copy assign)\n");
      memcpy(data_, other.data_, size_ * sizeof(T));
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
      free(data_);
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
    free(data_);
  }
  void getArray(std::vector<T>& dst) const {
    // DEBUG
    printf("memcpy (D->H)\n");
    memcpy(dst.data(), data_, size_ * sizeof(T));
  }
  void setArray(std::vector<T>& src) {
    size_ = src.size();
    // DEBUG
    printf("memcpy (H->D)\n");
    memcpy(data_, src.data(), size_ * sizeof(T));
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

// NB: these functions expect pointers for dest and src,
// so if applying to a vector use .data().
// If the vector is passed in this will still compile due to the template,
// but as not passed by ref will update a copy not the vector itself
template <typename T, typename U, typename Enable = void>
size_t destride_copy(T dest, U src, size_t at, size_t stride) {
  for (size_t i = 0; at < src.size(); ++i, at += stride) {
    dest[i] = src[at];
  }
  return at;
}

template <typename T, typename U, typename Enable = void>
size_t stride_copy(T dest, U src, size_t at, size_t stride) {
  for (size_t i = 0; i < src.size(); ++i, at += stride) {
    dest[at] = src[i];
  }
  return at;
}

template <typename T>
size_t stride_copy(T dest, int src, size_t at, size_t stride) {
  dest[at] = src;
  return at + stride;
}

template <typename T>
size_t stride_copy(T dest, double src, size_t at, size_t stride) {
  dest[at] = src;
  return at + stride;
}

// Alternative (protoype - definition in model file)
template <typename T>
void update_device(size_t step,
             const interleaved<typename T::real_t> state,
             interleaved<int> internal_int,
             interleaved<typename T::real_t> internal_real,
             dust::rng_state_t<typename T::real_t> rng_state,
             interleaved<typename T::real_t> state_next);

// This will become the __global__ kernel
template <typename real_t, typename T>
void run_particles(size_t step_from, size_t step_to, size_t n_particles,
                   DeviceArray<real_t>& state,
                   DeviceArray<real_t>& state_next,
                   DeviceArray<int>& internal_int,
                   DeviceArray<real_t>& internal_real,
                   dust::pRNG<real_t>& rng_state) {
  // DEBUG
  printf("RNG in\n");
  std::vector<uint64_t> rng_in = rng_state.raw_state();
  for (auto rng_bin = rng_in.begin(); rng_bin != rng_in.end(); rng_bin++) {
    std::cout << *rng_bin << std::endl;
  }

  // int p_idx = blockIdx.x * blockDim.x + threadIdx.x;
  // if (p_idx < n_particles) {
  for (size_t i = 0; i < n_particles; ++i) {
    interleaved<real_t> p_state(state, i);
    interleaved<real_t> p_state_next(state_next, i);
    interleaved<int> p_internal_int(internal_int, i);
    interleaved<real_t> p_internal_real(internal_real, i);
    dust::rng_state_t<real_t> p_rng = rng_state[i];
    for (int curr_step = step_from; curr_step < step_to; ++curr_step) {
      update_device<T>(curr_step,
                       p_state,
                       p_internal_int,
                       p_internal_real,
                       p_rng,
                       p_state_next);

      // CUDA: This move may need a __device__ class explictly defined?
      interleaved<real_t> tmp = p_state;
      p_state = p_state_next;
      p_state_next = tmp;

      // DEBUG
      for (int j = 0; j < state.size(); j++) {
        printf("%.1f ", p_state[j]);
      }
      printf("\n");
    }
  }

  // DEBUG
  printf("RNG out\n");
  std::vector<uint64_t> rng_out = rng_state.raw_state();
  for (auto rng_bin = rng_out.begin(); rng_bin != rng_out.end(); rng_bin++) {
    std::cout << *rng_bin << std::endl;
  }
}

template <typename T>
class Particle {
public:
  typedef typename T::init_t init_t;
  typedef typename T::real_t real_t;

  Particle(init_t data, size_t step) :
    _model(data),
    _step(step),
    _y(_model.initial(_step)),
    _y_swap(_model.size()) {
  }

  void run(const size_t step_end, dust::rng_state_t<real_t> rng_state) {
    while (_step < step_end) {
      _model.update(_step, _y.data(), rng_state, _y_swap.data());
      _step++;
      std::swap(_y, _y_swap);
      // DEBUG
      for (int j = 0; j < size(); j++) {
        printf("%.1f ", _y[j]);
      }
      printf("\n");
    }
  }

  void state(const std::vector<size_t>& index,
             typename std::vector<real_t>::iterator end_state) const {
    for (size_t i = 0; i < index.size(); ++i) {
      *(end_state + i) = _y[index[i]];
    }
  }

  void state_full(typename std::vector<real_t>::iterator end_state) const {
    for (size_t i = 0; i < _y.size(); ++i) {
      *(end_state + i) = _y[i];
    }
  }

  size_t size() const {
    return _y.size();
  }

  size_t step() const {
    return _step;
  }

  void swap() {
    std::swap(_y, _y_swap);
  }

  void set_step(const size_t step) {
    _step = step;
  }

  void set_state(const Particle<T>& other) {
    _y_swap = other._y;
  }

  void set_state(typename std::vector<real_t>::const_iterator state) {
    for (size_t i = 0; i < _y.size(); ++i, ++state) {
      _y[i] = *state;
    }
  }

  size_t size_internal_real() const {
    return _model.size_internal_real();
  }

  size_t size_internal_int() const {
    return _model.size_internal_int();
  }

  void internal_real(real_t * dest, size_t stride) const {
    _model.internal_real(dest, stride);
  }

  void internal_int(int * dest, size_t stride) const {
    _model.internal_int(dest, stride);
  }

private:
  T _model;
  size_t _step;

  std::vector<real_t> _y;
  std::vector<real_t> _y_swap;
};

// There are a few objects here that when written mark host/device
// memory as stale. Methods which access them then check, and update
// if necessary
// At the moment hand-written, but these objects could be wrapped in
// another class to automate this?
template <typename T>
class Dust {
public:
  typedef typename T::init_t init_t;
  typedef typename T::real_t real_t;

  Dust(const init_t data, const size_t step, const size_t n_particles,
       const size_t n_threads, const std::vector<uint64_t>& seed) :
    _n_threads(n_threads),
    _rng(n_particles, seed),
    _stale_host(false),
    _stale_device(true) {
    initialise(data, step, n_particles);
  }

  void reset(const init_t data, const size_t step) {
    const size_t n_particles = _particles.size();
    _stale_device = true;
    _stale_host = false;
    initialise(data, step, n_particles);
  }

  // It's the callee's responsibility to ensure that index is in
  // range [0, n-1]
  void set_index(const std::vector<size_t>& index) {
    _index = index;
  }

  // It's the callee's responsibility to ensure this is the correct length:
  //
  // * if is_matrix is false then state must be length n_state_full()
  //   and all particles get the state
  // * if is_matrix is true, state must be length (n_state_full() *
  //   n_particles()) and every particle gets a different state.
  void set_state(const std::vector<real_t>& state, bool is_matrix) {
    _stale_device = true;
    const size_t n_particles = _particles.size();
    const size_t n_state = n_state_full();
    auto it = state.begin();
    for (size_t i = 0; i < n_particles; ++i) {
      _particles[i].set_state(it);
      if (is_matrix) {
        it += n_state;
      }
    }
  }

  void set_step(const size_t step) {
    const size_t n_particles = _particles.size();
    for (size_t i = 0; i < n_particles; ++i) {
      _particles[i].set_step(step);
    }
  }

  // TODO: what to do with this one? Currently calls run,
  // but should also support run_device
  void set_step(const std::vector<size_t>& step) {
    const size_t n_particles = _particles.size();
    for (size_t i = 0; i < n_particles; ++i) {
      _particles[i].set_step(step[i]);
    }
    const auto r = std::minmax_element(step.begin(), step.end());
    if (*r.second > *r.first) {
      run(*r.second);
    }
  }

  void run(const size_t step_end) {
    refresh_host();
    _stale_device = true;

    // DEBUG
    printf("RNG in\n");
    std::vector<uint64_t> rng_in = _rng.raw_state();
    for (auto rng_bin = rng_in.begin(); rng_bin != rng_in.end(); rng_bin++) {
      std::cout << *rng_bin << std::endl;
    }

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(_n_threads)
#endif
    for (size_t i = 0; i < _particles.size(); ++i) {
      _particles[i].run(step_end, _rng.state(i));
    }

    // DEBUG
    printf("RNG out\n");
    std::vector<uint64_t> rng_out = _rng.raw_state();
    for (auto rng_bin = rng_out.begin(); rng_bin != rng_out.end(); rng_bin++) {
      std::cout << *rng_bin << std::endl;
    }
  }

  void run_device(const size_t step_end) {
    // Using same name here as in the prototype for simplicity.
    //
    // In order to make the implementation here as easy to think about
    // as possible, we'll create massive vectors for all the bits used
    // here. We'll want to think about that generally when getting the
    // GPU version really working as if we can avoid doing an
    // extraction here and maintain state on the device as much as
    // possible, things will likely be faster and easier. The
    // initialisation below is far from the most efficient, but should
    // work for now.
    //
    // We need to modify here on return:
    //
    // state
    // rng_state
    //
    // as these are required to continue on with the model.
    refresh_device();
    _stale_host = true;

    run_particles<real_t, T>(step(), step_end, _particles.size(),
                  _yi, _yi_next, _internal_int, _internal_real,
                  _rng);

    // In the inner loop, the swap will keep the locally scoped interleaved variables
    // updated, but the interleaved variables passed in have not yet been updated.
    // If an even number of steps have been run state will have been swapped back into
    // the original place, but an on odd number of steps the passed variables
    // need to be swapped.
    if ((step_end - step()) % 2) {
      std::swap(_yi, _yi_next);
    }
  }

  void state(std::vector<real_t>& end_state) {
    refresh_host();
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(_n_threads)
#endif
    for (size_t i = 0; i < _particles.size(); ++i) {
      _particles[i].state(_index, end_state.begin() + i * _index.size());
    }
  }

  void state(std::vector<size_t> index,
             std::vector<real_t>& end_state) {
    refresh_host();
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(_n_threads)
#endif
    for (size_t i = 0; i < _particles.size(); ++i) {
      _particles[i].state(index, end_state.begin() + i * index.size());
    }
  }

  void state_full(std::vector<real_t>& end_state) {
    const size_t n = n_state_full();
    refresh_host();
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(_n_threads)
#endif
    for (size_t i = 0; i < _particles.size(); ++i) {
      _particles[i].state_full(end_state.begin() + i * n);
    }
  }

  // There are two obvious ways of reordering; we can construct a
  // completely new set of particles, like
  //
  //   std::vector<Particle<T>> next;
  //   for (auto const& i: index) {
  //     next.push_back(_particles[i]);
  //   }
  //   _particles = next;
  //
  // but this seems like a lot of churn.  The other way is to treat it
  // like a slightly weird state update where we swap around the
  // contents of the particle state (uses the set_state() and swap()
  // methods on particles).

  // CUDA: run this, or a scatter kernel, depending on which
  // variables are stale
  void reorder(const std::vector<size_t>& index) {
    _stale_device = true;
    for (size_t i = 0; i < _particles.size(); ++i) {
      size_t j = index[i];
      _particles[i].set_state(_particles[j]);
    }
    for (auto& p : _particles) {
      p.swap();
    }
  }

  size_t n_particles() const {
    return _particles.size();
  }

  size_t n_state() const {
    return _index.size();
  }

  size_t n_state_full() const {
    return _particles.front().size();
  }

  size_t step() const {
    return _particles.front().step();
  }

  std::vector<uint64_t> rng_state() {
    return _rng.export_state();
  }

  void set_rng_state(const std::vector<uint64_t>& rng_state) {
    _rng.import_state(rng_state);
  }

  // NOTE: it only makes sense to expose long_jump, and not jump,
  // because each rng stream is one jump away from the next.
  void rng_long_jump() {
    _rng.long_jump();
  }

  // New things
  size_t size_internal_real() const {
    return _particles.front().size_internal_real();
  }

  size_t size_internal_int() const {
    return _particles.front().size_internal_int();
  }

  std::vector<real_t> internal_real() const {
    std::vector<real_t> ret(size_internal_real());
    _internal_real.getArray(ret);
    return(ret);
  }

  std::vector<int> internal_int() const {
    std::vector<int> ret(size_internal_int());
    _internal_int.getArray(ret);
    return(ret);
  }

private:
  std::vector<size_t> _index;
  const size_t _n_threads;
  // CUDA: this needs copying to and from device
  // CUDA: needs a putRNG/getRNG within kernel
  dust::pRNG<real_t> _rng;
  std::vector<Particle<T>> _particles;

  // New things for device support
  bool _stale_host, _stale_device;
  DeviceArray<real_t> _yi, _yi_next, _internal_real;
  DeviceArray<int> _internal_int;

  void initialise(const init_t data, const size_t step,
                  const size_t n_particles) {
    _particles.clear();
    _particles.reserve(n_particles);
    for (size_t i = 0; i < n_particles; ++i) {
      _particles.push_back(Particle<T>(data, step));
    }

    const size_t n = n_state_full();
    _index.clear();
    _index.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      _index.push_back(i);
    }

    // Set internal state (on device)
    size_t int_len = size_internal_int();
    size_t real_len = size_internal_real();
    const size_t stride = n_particles;
    std::vector<int> int_vec(int_len * stride);
    std::vector<real_t> real_vec(real_len * stride);
    int* int_data = int_vec.data();
    real_t* real_data = real_vec.data();
    for (size_t i = 0; i < n_particles; ++i) {
      _particles[i].internal_int(int_data + i, stride);
      _particles[i].internal_real(real_data + i, stride);
    }
    _internal_int = DeviceArray<int>(int_vec, stride);
    _internal_real = DeviceArray<real_t>(real_vec, stride);

    // Set state (on device)
    _yi = DeviceArray<real_t>(n * n_particles, n_particles);
    _yi_next = DeviceArray<real_t>(n * n_particles, n_particles);
  }

  // CUDA: eventually we need to have more refined methods here:
  // CUDA:  in-device scatter kernel to support shuffling
  // CUDA:  host driven scatter?
  void refresh_device() {
    if (_stale_device) {
      const size_t np = n_particles(), ny = n_state_full();
      std::vector<real_t> y_tmp(ny); // Individual particle state
      std::vector<real_t> y(np * ny); // Interleaved state of all particles
#ifdef _OPENMP
      #pragma omp parallel for schedule(static) num_threads(_n_threads)
#endif
      for (size_t i = 0; i < np; ++i) {
        _particles[i].state_full(y_tmp.begin());
        stride_copy(y.data(), y_tmp, i, np);
      }
      _yi.setArray(y); // H -> D
      _stale_device = false;
    }
  }

  // CUDA: eventually we need to have more refined methods here:
  // CUDA:  cub::deviceselect to get just some state
  //        (e.g. refresh_host_partial called by state
  //              refresh_host called by state_full)
  void refresh_host() {
    if (_stale_host) {
      const size_t np = n_particles(), ny = n_state_full();
      std::vector<real_t> y_tmp(ny); // Individual particle state
      std::vector<real_t> y(np * ny); // Interleaved state of all particles
      _yi.getArray(y); // D -> H
      for (size_t i = 0; i < np; ++i) {
        destride_copy(y_tmp.data(), y, i, np);
        _particles[i].set_state(y_tmp.begin());
      }
      _stale_host = false;
    }
  }
};

// NB: not needed on GPU for now
template <typename T>
std::vector<typename T::real_t>
dust_simulate(const std::vector<size_t> steps,
              const std::vector<typename T::init_t> data,
              const std::vector<typename T::real_t> state,
              const std::vector<size_t> index,
              const size_t n_threads,
              const std::vector<uint64_t>& seed) {
  typedef typename T::real_t real_t;
  const size_t n_state_return = index.size();
  const size_t n_particles = data.size();
  std::vector<Particle<T>> particles;
  particles.reserve(n_particles);
  for (size_t i = 0; i < n_particles; ++i) {
    particles.push_back(Particle<T>(data[i], steps[0]));
    if (i > 0 && particles.back().size() != particles.front().size()) {
      std::stringstream msg;
      msg << "Particles have different state sizes: particle " << i + 1 <<
        " had length " << particles.front().size() << " but expected " <<
        particles.back().size();
      throw std::invalid_argument(msg.str());
    }
  }
  const size_t n_state_full = particles.front().size();

  dust::pRNG<real_t> rng(n_particles, seed);
  std::vector<real_t> ret(n_particles * n_state_return * steps.size());
  size_t n_steps = steps.size();

#ifdef _OPENMP
  #pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
  for (size_t i = 0; i < particles.size(); ++i) {
    particles[i].set_state(state.begin() + n_state_full * i);
    for (size_t t = 0; t < n_steps; ++t) {
      particles[i].run(steps[t], rng.state(i));
      size_t offset = t * n_state_return * n_particles + i * n_state_return;
      particles[i].state(index, ret.begin() + offset);
    }
  }

  return ret;
}

#endif
