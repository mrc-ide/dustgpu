#ifndef DUST_DUST_HPP
#define DUST_DUST_HPP

#include <dust/rng.hpp>

#include <algorithm>
#include <sstream>
#include <utility>
#ifdef _OPENMP
#include <omp.h>
#endif

template <typename T>
class interleaved {
public:
  interleaved(T* data, size_t stride) : data_(data), stride_(stride) {}
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

template <typename T>
class Dust {
public:
  typedef typename T::init_t init_t;
  typedef typename T::real_t real_t;

  Dust(const init_t data, const size_t step, const size_t n_particles,
       const size_t n_threads, const std::vector<uint64_t>& seed) :
    _n_threads(n_threads),
    _rng(n_particles, seed) {
    initialise(data, step, n_particles);
  }

  void reset(const init_t data, const size_t step) {
    const size_t n_particles = _particles.size();
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
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(_n_threads)
#endif
    for (size_t i = 0; i < _particles.size(); ++i) {
      _particles[i].run(step_end, _rng.state(i));
    }
  }

  void run2(const size_t step_end) {
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

    const size_t np = n_particles(), ny = n_state_full();
    std::vector<real_t> y(np * ny);
    std::vector<real_t> y_next(np * ny);
    std::vector<real_t_ y_tmp(ny);
    for (size_t i = 0; i < np; ++i) {
      interleaved<real_y> yi(y.data() + i, np);
      _particles[i].state_full(y_tmp.begin());
      for (size_t j = 0; j < ny; ++j) {
        yi[j] = y_tmp[j];
      }
    }

    run_particles(step(), step_end, _particles.size(),
                  y, y_next, internal_int(), internal_real(),
                  _rng.state(0));

    for (size_t i = 0; i < np; ++i) {
      interleaved<real_y> yi(y.data() + i, np);
      for (size_t j = 0; j < ny; ++j) {
        y_tmp[j] = yi[j];
      }
      _particles[i].set_state(y_tmp.begin());
    }
  }

  void state(std::vector<real_t>& end_state) const {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(_n_threads)
#endif
    for (size_t i = 0; i < _particles.size(); ++i) {
      _particles[i].state(_index, end_state.begin() + i * _index.size());
    }
  }

  void state(std::vector<size_t> index,
             std::vector<real_t>& end_state) const {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(_n_threads)
#endif
    for (size_t i = 0; i < _particles.size(); ++i) {
      _particles[i].state(index, end_state.begin() + i * index.size());
    }
  }

  void state_full(std::vector<real_t>& end_state) const {
    const size_t n = n_state_full();
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
    const size_t len = size_internal_real();
    const size_t stride = n_particles();
    std::vector<real_t> ret(len * stride);
    real_t* data = ret.data();
    for (size_t i = 0; i < n_particles(); ++i) {
      _particles[i].internal_real(data + i, stride);
    }
    return ret;
  }

  std::vector<int> internal_int() const {
    const size_t len = size_internal_int();
    const size_t stride = n_particles();
    std::vector<int> ret(len * stride);
    int* data = ret.data();
    for (size_t i = 0; i < n_particles(); ++i) {
      _particles[i].internal_int(data + i, stride);
    }
    return ret;
  }

private:
  std::vector<size_t> _index;
  const size_t _n_threads;
  dust::pRNG<real_t> _rng;
  std::vector<Particle<T>> _particles;

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
  }
};


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

// Alternative
template <typename T>
void update2(size_t step,
             const interleaved<typename T::real_t> state,
             interleaved<int> internal_int,
             interleaved<typename T::real_t> internal_real,
             dust::rng_state_t<typenme T::real_t> rng_state,
             interleaved<typename T::real_t> state_next);

template <typename real_t>
void run_particles(size_t step_from, size_t step_to, size_t n_particles,
                   std::vector<real_t>& state,
                   std::vector<real_t>& state_next,
                   std::vector<int>& internal_int,
                   std::vector<real_t>& internal_real,
                   dust::rng_state_t<real_t> rng_state) {
  for (size_t i = 0; i < n_particles; ++i) {
    while (_step < step_end) {
      // perhaps this should be a static method of the model? that
      // might be easier to deal with
      update2<model>
      _model.update(_step, _y.data(), rng_state, _y_swap.data());
      _step++;
      std::swap(_y, _y_swap);
    }
  }

  // We only *have* to do this on odd numbers of steps, but it is a
  // tiny cost compared to everything else.
  //
  // TODO: add some explanation as to why this is needed!
  std::copy(state_next.begin(), state_next.end(), state.begin());
}

#endif
