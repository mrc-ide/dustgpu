#ifndef DUST_DUST_HPP
#define DUST_DUST_HPP

#include <dust/rng.hpp>

#include <algorithm>
#include <sstream>
#include <utility>
#ifdef _OPENMP
#include <omp.h>
#endif

// NB: these functions expect pointers for dest and src,
// so if applying to a vector use .data().
// If the vector is passed in this will still compile due to the template,
// but as not passed by ref will update a copy not the vector itself
template <typename T, typename U, typename Enable = void>
size_t destride_copy(T dest, U src, size_t at, size_t stride) {
  size_t i;
  for (i = 0; at < src.size(); ++i, at += stride) {
    dest[i] = src[at];
  }
  return i;
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

template <typename T>
size_t stride_copy(T dest, uint64_t src, size_t at, size_t stride) {
  dest[at] = src;
  return at + stride;
}

// Alternative (protoype - definition in model file)
template <typename T>
DEVICE void update_device(size_t step,
             const dust::interleaved<typename T::real_t> state,
             dust::interleaved<int> internal_int,
             dust::interleaved<typename T::real_t> internal_real,
             dust::device_rng_state_t<typename T::real_t>& rng_state,
             dust::interleaved<typename T::real_t> state_next);

// __global__ for shuffling particles
template<typename real_t>
KERNEL void device_scatter(int* scatter_index,
                           real_t* state,
                           real_t* scatter_state,
                           size_t state_size) {
#ifdef __NVCC__
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < state_size) {
#else
  for (size_t i = 0; i < state_size; ++i) {
#endif
    scatter_state[i] = state[scatter_index[i]];
  }
}

// __global__ for running the model
template <typename real_t, typename T>
KERNEL void run_particles(size_t step_from, size_t step_to, size_t n_particles,
                          real_t* state,
                          real_t* state_next,
                          int* internal_int,
                          real_t* internal_real,
                          uint64_t* rng_state) {
#ifdef __NVCC__
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_particles) {
#else
  for (size_t i = 0; i < n_particles; ++i) {
#endif
    dust::interleaved<real_t> p_state(state + i, n_particles);
    dust::interleaved<real_t> p_state_next(state_next + i, n_particles);
    dust::interleaved<int> p_internal_int(internal_int + i, n_particles);
    dust::interleaved<real_t> p_internal_real(internal_real + i, n_particles);
    dust::interleaved<uint64_t> p_rng(rng_state + i, n_particles);

    dust::device_rng_state_t<real_t> rng_block = dust::loadRNG<real_t>(p_rng);
    for (int curr_step = step_from; curr_step < step_to; ++curr_step) {
      update_device<T>(curr_step,
                       p_state,
                       p_internal_int,
                       p_internal_real,
                       rng_block,
                       p_state_next);
#ifdef __NVCC__
      __syncwarp();
#endif

      dust::interleaved<real_t> tmp = p_state;
      p_state = p_state_next;
      p_state_next = tmp;
    }
    dust::putRNG(rng_block, p_rng);
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
       const size_t n_threads, const std::vector<uint64_t>& seed,
       const int device_id) :
    _n_threads(n_threads),
    _rng(n_particles, seed),
    _stale_host(false),
    _stale_device(true),
    _device_id(device_id) {
#ifdef __NVCC__
    CUDA_CALL(cudaSetDevice(_device_id));
    cudaProfilerStart();
#endif
    initialise(data, step, n_particles);
  }

#ifdef __NVCC__
  ~Dust() {
    cudaProfilerStop();
  }
#endif

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
    set_device_index();
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

  // Host run only, as device run assumes all particles are running forward
  // the same number of steps
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
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(_n_threads)
#endif
    for (size_t i = 0; i < _particles.size(); ++i) {
      _particles[i].run(step_end, _rng.state(i));
    }
  }

  void run_device(const size_t step_end) {
    // Using same name here as in the prototype for simplicity.
    refresh_device();
    _stale_host = true;

#ifdef __NVCC__
    const size_t blockSize = 128;
    const size_t blockCount = (_particles.size() + blockSize - 1) / blockSize;
    run_particles<real_t, T><<<blockCount, blockSize>>>(
                  step(), step_end, _particles.size(),
                  _yi.data(), _yi_next.data(),
                  _internal_int.data(), _internal_real.data(),
                  _rngi.data());
    cudaDeviceSynchronize();
#else
    run_particles<real_t, T>(step(), step_end, _particles.size(),
                  _yi.data(), _yi_next.data(),
                  _internal_int.data(), _internal_real.data(),
                  _rngi.data());
#endif
    // In the inner loop, the swap will keep the locally scoped interleaved variables
    // updated, but the interleaved variables passed in have not yet been updated.
    // If an even number of steps have been run state will have been swapped back into
    // the original place, but an on odd number of steps the passed variables
    // need to be swapped.
    if ((step_end - step()) % 2) {
      std::swap(_yi, _yi_next);
    }
    set_step(step_end);
  }

  void state(std::vector<real_t>& end_state) {
      size_t np = _particles.size();
      size_t index_size = _index.size();
    if (_stale_host) {
#ifdef __NVCC__
      // Run the selection and copy items back
      cub::DeviceSelect::Flagged(_select_tmp.data(),
                                 _select_tmp.size(),
                                 _yi.data(),
                                 _indexi.data(),
                                 _yi_selected.data(),
                                 _num_selected.data(),
                                 _yi.size());
      std::vector<real_t> yi_selected(np * index_size);
      _yi_selected.getArray(yi_selected);

#ifdef _OPENMP
      #pragma omp parallel for schedule(static) num_threads(_n_threads)
#endif
      for (size_t i = 0; i < np; ++i) {
        destride_copy(end_state.data() + i * index_size, yi_selected, i, np);
      }
#else
      refresh_host();
#endif
    }
    // This would be better as an else, but the ifdefs are clearer this way
    if (!_stale_host) {
#ifdef _OPENMP
      #pragma omp parallel for schedule(static) num_threads(_n_threads)
#endif
      for (size_t i = 0; i < np; ++i) {
        _particles[i].state(_index, end_state.begin() + i * index_size);
      }
    }
  }

  // Assuming this is less frequently called than the above
  // With a passed index. We could send this to device and use the method above
  // but may be quicker just to copy the whole state in that case
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
  void reorder(const std::vector<size_t>& index) {
    if (_stale_host) {
      size_t n_particles = _particles.size();
      size_t n_state = n_state_full();

      // e.g. 4 particles with 3 states ABC stored on device as
      // [1_A, 2_A, 3_A, 4_A, 1_B, 2_B, 3_B, 4_B, 1_C, 2_C, 3_C, 4_C]
      // e.g. index [3, 1, 3, 2] with would be
      // [3_A, 1_A, 3_A, 2_A, 3_B, 1_B, 3_B, 2_B, 3_C, 1_C, 3_C, 2_C] interleaved
      // i.e. input repeated n_state_full times, plus a strided offset
      // [3, 1, 3, 2, 3 + 4, 1 + 4, 3 + 4, 2 + 4, 3 + 8, 1 + 8, 3 + 8, 2 + 8]
      // [3, 1, 3, 2, 7, 5, 7, 6, 11, 9, 11, 10]
      std::vector<int> scatter_state(n_state * n_particles);
#ifdef _OPENMP
      #pragma omp parallel for schedule(static) num_threads(_n_threads)
#endif
      for (size_t i = 0; i < n_state; ++i) {
        for (size_t j = 0; j < n_particles; ++j) {
          scatter_state[i * n_particles + j] = index[j] + i * n_particles;
        }
      }
      _scatter_index.setArray(scatter_state);
#ifdef __NVCC__
      const size_t blockSize = 32;
      const size_t blockCount = (scatter_state.size() + blockSize - 1) / blockSize;
      device_scatter<real_t><<<blockCount, blockSize>>>(
        _scatter_index.data(),
        _yi.data(),
        _yi_next.data(),
        scatter_state.size());
      cudaDeviceSynchronize();
#else
      device_scatter<real_t>(_scatter_index.data(),
                             _yi.data(),
                             _yi_next.data(),
                             scatter_state.size());
#endif
      std::swap(_yi, _yi_next);
    } else {
      _stale_device = true;
      for (size_t i = 0; i < _particles.size(); ++i) {
        size_t j = index[i];
        _particles[i].set_state(_particles[j]);
      }
      for (auto& p : _particles) {
        p.swap();
      }
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
    refresh_host();
    return _rng.export_state();
  }

  void set_rng_state(const std::vector<uint64_t>& rng_state) {
    _stale_device = true;
    _rng.import_state(rng_state);
  }

  // NOTE: it only makes sense to expose long_jump, and not jump,
  // because each rng stream is one jump away from the next.
  void rng_long_jump() {
    refresh_host();
    _stale_device = true;
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

  int device_id() const {
    return _device_id;
  }

private:
  // delete move and copy to avoid accidentally using them
  Dust ( const Dust & ) = delete;
  Dust ( Dust && ) = delete;

  std::vector<size_t> _index;
  const size_t _n_threads;
  dust::pRNG<real_t> _rng;
  std::vector<Particle<T>> _particles;

  // New things for device support
  int _device_id;
  bool _stale_host, _stale_device;
  dust::DeviceArray<real_t> _yi, _yi_next, _internal_real;
  dust::DeviceArray<int> _internal_int;
  dust::DeviceArray<uint64_t> _rngi;

  // For the index on the device
  dust::DeviceArray<char> _indexi;
  dust::DeviceArray<real_t> _yi_selected;
  dust::DeviceArray<void> _select_tmp;
  dust::DeviceArray<int> _num_selected, _scatter_index;

  void initialise(const init_t data, const size_t step,
                  const size_t n_particles) {
    _particles.clear();
    _particles.reserve(n_particles);
    for (size_t i = 0; i < n_particles; ++i) {
      _particles.push_back(Particle<T>(data, step));
    }

    // Set the index
    const size_t n = n_state_full();
    _index.clear();
    _index.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      _index.push_back(i);
    }
    // std::vector<bool> is specialised and cannot be used here
    std::vector<char> device_index(n * n_particles, 1);
    _indexi = dust::DeviceArray<char>(device_index);
    _num_selected = dust::DeviceArray<int>(1);

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
    _internal_int = dust::DeviceArray<int>(int_vec);
    _internal_real = dust::DeviceArray<real_t>(real_vec);

    // Allocate memory for state + rng (on device)
    _yi = dust::DeviceArray<real_t>(n * n_particles);
    _yi_next = dust::DeviceArray<real_t>(n * n_particles);
    _yi_selected = dust::DeviceArray<real_t>(n * n_particles);
    size_t rng_len = dust::rng_state_t<real_t>::size();
    _rngi = dust::DeviceArray<uint64_t>(rng_len * n_particles);
    _scatter_index = dust::DeviceArray<int>(n * n_particles);
#ifdef __NVCC__
    set_cub_tmp();
#endif
  }

  void refresh_device() {
    if (_stale_device) {
      const size_t np = n_particles(), ny = n_state_full();
      const size_t rng_len = dust::rng_state_t<real_t>::size();
      std::vector<real_t> y_tmp(ny); // Individual particle state
      std::vector<real_t> y(np * ny); // Interleaved state of all particles
      std::vector<uint64_t> rng(np * rng_len); // Interleaved RNG state
#ifdef _OPENMP
      #pragma omp parallel for schedule(static) num_threads(_n_threads)
#endif
      for (size_t i = 0; i < np; ++i) {
        // Interleave state
        _particles[i].state_full(y_tmp.begin());
        stride_copy(y.data(), y_tmp, i, np);

        // Interleave RNG state
        dust::rng_state_t<real_t> p_rng = _rng[i];
        size_t rng_offset = i;
        for (size_t j = 0; j < rng_len; ++j) {
          rng_offset = stride_copy(rng.data(), p_rng[j], rng_offset, np);
        }
      }
      // H -> D copies
      _yi.setArray(y);
      _rngi.setArray(rng);
      _stale_device = false;
    }
  }

  // TODO: could have RNG refresh/state refresh as separate functions
  // Although RNG is basically part of state, so maybe this makes sense
  void refresh_host() {
    if (_stale_host) {
      const size_t np = n_particles(), ny = n_state_full();
      const size_t rng_len = dust::rng_state_t<real_t>::size();
      std::vector<real_t> y_tmp(ny); // Individual particle state
      std::vector<real_t> y(np * ny); // Interleaved state of all particles
      std::vector<uint64_t> rngi(np * rng_len); // Interleaved RNG state
      std::vector<uint64_t> rng(np * rng_len); //  Deinterleaved RNG state
      // D -> H copies
      _yi.getArray(y);
      _rngi.getArray(rngi);
#ifdef _OPENMP
      #pragma omp parallel for schedule(static) num_threads(_n_threads)
#endif
      for (size_t i = 0; i < np; ++i) {
        destride_copy(y_tmp.data(), y, i, np);
        _particles[i].set_state(y_tmp.begin());

        // Destride RNG
        for (size_t j = 0; j < rng_len; ++j) {
          rng[i * np + j] = rngi[i + j * np];
        }
      }
      _rng.import_state(rng);
      _stale_host = false;
    }
  }

  // Sets a boolean index on the device marking the elements of interleaved
  // state to pull down when requested
  void set_device_index() {
    size_t n_particles = _particles.size();
    std::vector<char> bool_idx(n_state_full() * n_particles, 0);
    // e.g. 4 particles with 3 states ABC stored on device as
    // [1_A, 2_A, 3_A, 4_A, 1_B, 2_B, 3_B, 4_B, 1_C, 2_C, 3_C, 4_C]
    // e.g. index [1, 3] would be
    // [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1] bool index on interleaved state
    // i.e. initialise to zero and copy 1 np times, at each offset given in index
    for (auto idx_pos = _index.cbegin(); idx_pos != _index.cend(); idx_pos++) {
      std::fill_n(bool_idx.begin() + (*idx_pos * n_particles), n_particles, 1);
    }
    _indexi.setArray(bool_idx);
#ifdef __NVCC__
    set_cub_tmp();
#endif
  }

#ifdef __NVCC__
  void set_cub_tmp() {
    // Free the array before running cub function below
    size_t tmp_bytes = 0;
    _select_tmp.set_size(tmp_bytes);
    cub::DeviceSelect::Flagged(_select_tmp.data(),
                               tmp_bytes,
                               _yi.data(),
                               _indexi.data(),
                               _yi_selected.data(),
                               _num_selected.data(),
                               _yi.size());
    _select_tmp.set_size(tmp_bytes);
  }
#endif
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
