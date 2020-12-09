#ifndef DUST_RNG_HPP
#define DUST_RNG_HPP

#include <algorithm>
#include "containers.cuh"
#include "xoshiro.hpp"
#include "distr/binomial.hpp"
#include "distr/normal.hpp"
#include "distr/poisson.hpp"
#include "distr/uniform.hpp"

namespace dust {

// This is just a container class for state
template <typename T>
class pRNG {
public:
  pRNG(const size_t n, const std::vector<uint64_t>& seed) :
    n_(n), state_(n * rng_state_t<T>::size()) {
    auto len = rng_state_t<T>::size();
    auto n_seed = seed.size() / len;
    auto state_it = state_.begin();
    for (size_t i = 0; i < n; ++i) {
      if (i < n_seed) {
        std::copy_n(seed.begin() + i * len, len, state_it);
        state_it += len;
      } else {
        rng_state_t<T> prev = state(i - 1);
        for (size_t j = 0; j < len; ++j) {
          *state_it = prev[j];
          state_it++;
        }
        xoshiro_jump(state(i));
      }
    }
  }

  size_t size() const {
    return n_;
  }

  void jump() {
    for (size_t i = 0; i < n_; ++i) {
      xoshiro_jump(state(i));
    }
  }

  void long_jump() {
    for (size_t i = 0; i < n_; ++i) {
      xoshiro_long_jump(state(i));
    }
  }

  // Access an individual rng_state by constructing the small struct
  // that contains a pointer to the memory, offset as needed, and our
  // stride.
  rng_state_t<T> state(size_t i) {
    return rng_state_t<T>{state_.data() + i * rng_state_t<T>::size()};
  }

  // Possibly nicer way of doing the above
  rng_state_t<T> operator[](size_t i) {
    return rng_state_t<T>{state_.data() + i * rng_state_t<T>::size()};
  }

  std::vector<uint64_t> export_state() {
    return state_;
  }

  void import_state(const std::vector<uint64_t>& state) {
    std::copy(state.begin(), state.end(), state_.begin());
  }

private:
  const size_t n_;
  std::vector<uint64_t> state_;
};

// Read state from global memory
template <typename T>
DEVICE device_rng_state_t<T>
  loadRNG(dust::interleaved<uint64_t>& full_rng_state) {
          device_rng_state_t<T> rng_state;
  for (int i = 0; i < device_rng_state_t<T>::size(); i++) {
    rng_state.s[i] = full_rng_state[i];
  }
  return rng_state;
}

// Write state into global memory
template <typename T>
DEVICE void putRNG(device_rng_state_t<T>& rng_state,
                   dust::interleaved<uint64_t>& full_rng_state) {
  for (int i = 0; i < device_rng_state_t<T>::size(); i++) {
    full_rng_state[i] = rng_state.s[i];
  }
}

}

#endif
