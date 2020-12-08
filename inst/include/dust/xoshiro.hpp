#ifndef DUST_XOSHIRO_HPP
#define DUST_XOSHIRO_HPP

#include <cstdint>
#include <cstddef>
#include <vector>
#include <limits>

#include "cuda.cuh"

// This is derived from http://prng.di.unimi.it/xoshiro256starstar.c
// and http://prng.di.unimi.it/splitmix64.c, copies of which are
// included in the package (in inst/rng in the source package). The
// original code is CC0 licenced but was written by David Blackman and
// Sebastiano Vigna.
//
// MD5 (splitmix64.c) = 7e38529aa7bb36624ae4a9d6808e7e3f
// MD5 (xoshiro256starstar.c) = 05f9ecd49bbed98304d982313c91d0f6

namespace dust {

// Data for multiple related RNG streams is stored together in a large
// array, and we will be passed a structure that can easily index into
// this. Each RNG state has 4 elements and we put all the first
// elements together, then all the second and so forth.
//
// All the xoshiro code does is index into the array, so we provide an
// operator for that.
template <typename T>
struct rng_state_t {
  using real_t = T;
  static size_t size() {
    return 4;
  }
  uint64_t* s;
  uint64_t& operator[](size_t i) {
    return s[i];
  }
};

// State for the GPU, which needs to compile to four integers,
// not a pointer to global memory
// TODO: Should these be classes with inheritance instead?
template <typename T>
struct device_rng_state_t {
  using real_t = T;
  static DEVICE size_t size() {
    return 4;
  }
  uint64_t s[4];
  DEVICE uint64_t& operator[](size_t i) {
    return s[i];
  }
};

static inline HOSTDEVICE uint64_t rotl(const uint64_t x, int k) {
  return (x << k) | (x >> (64 - k));
}

// This is the core generator (next() in the original C code)
inline HOSTDEVICE uint64_t xoshiro_next(uint64_t * state) {
  const uint64_t result = rotl(state[1] * 5, 7) * 9;

  const uint64_t t = state[1] << 17;

  state[2] ^= state[0];
  state[3] ^= state[1];
  state[1] ^= state[2];
  state[0] ^= state[3];

  state[2] ^= t;

  state[3] = rotl(state[3], 45);

  return result;
}

template <typename T>
inline HOST uint64_t xoshiro_next(rng_state_t<T> state) {
  return xoshiro_next(state.s);
}

template <typename T>
inline DEVICE uint64_t xoshiro_next(device_rng_state_t<T>& state) {
  return xoshiro_next(state.s);
}

inline HOST uint64_t splitmix64(uint64_t seed) {
  uint64_t z = (seed += 0x9e3779b97f4a7c15);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
  z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
  return z ^ (z >> 31);
}

template <typename T>
inline HOST std::vector<uint64_t> xoshiro_initial_seed(uint64_t seed) {
  // normal brain: for i in 1:4
  // advanced brain: -funroll-loops
  // galaxy brain:
  std::vector<uint64_t> state(T::size());
  state[0] = splitmix64(seed);
  state[1] = splitmix64(state[0]);
  state[2] = splitmix64(state[1]);
  state[3] = splitmix64(state[2]);
  return state;
}

/* NB: jump functions not defined for device_rng_state_t as they do not take
   a reference. See template specialisations for xoshiro_next to see how to
   add support, if needed */

/* This is the jump function for the generator. It is equivalent
   to 2^128 calls to next(); it can be used to generate 2^128
   non-overlapping subsequences for parallel computations. */
template <typename T>
inline HOST void xoshiro_jump(rng_state_t<T> state) {
  static const uint64_t JUMP[] = { 0x180ec6d33cfd0aba, 0xd5a61266f0c9392c,
                                   0xa9582618e03fc9aa, 0x39abdc4529b1661c };

  uint64_t s0 = 0;
  uint64_t s1 = 0;
  uint64_t s2 = 0;
  uint64_t s3 = 0;
  for (long unsigned int i = 0; i < sizeof JUMP / sizeof *JUMP; i++) {
    for (int b = 0; b < 64; b++) {
      if (JUMP[i] & UINT64_C(1) << b) {
        s0 ^= state[0];
        s1 ^= state[1];
        s2 ^= state[2];
        s3 ^= state[3];
      }
      xoshiro_next(state);
    }
  }

  state[0] = s0;
  state[1] = s1;
  state[2] = s2;
  state[3] = s3;
}

/* This is the long-jump function for the generator. It is equivalent to
   2^192 calls to next(); it can be used to generate 2^64 starting points,
   from each of which jump() will generate 2^64 non-overlapping
   subsequences for parallel distributed computations. */
template <typename T>
inline HOST void xoshiro_long_jump(rng_state_t<T> state) {
  static const uint64_t LONG_JUMP[] =
    { 0x76e15d3efefdcbbf, 0xc5004e441c522fb3,
      0x77710069854ee241, 0x39109bb02acbe635 };

  uint64_t s0 = 0;
  uint64_t s1 = 0;
  uint64_t s2 = 0;
  uint64_t s3 = 0;
  for (long unsigned int i = 0; i < sizeof LONG_JUMP / sizeof *LONG_JUMP; i++) {
    for (int b = 0; b < 64; b++) {
      if (LONG_JUMP[i] & UINT64_C(1) << b) {
        s0 ^= state[0];
        s1 ^= state[1];
        s2 ^= state[2];
        s3 ^= state[3];
      }
      xoshiro_next(state);
    }
  }

  state[0] = s0;
  state[1] = s1;
  state[2] = s2;
  state[3] = s3;
}

template <typename T, typename U = typename T::real_t>
HOST U unif_rand(T& state) {
  printf("generic host\n");
  const uint64_t value = xoshiro_next(state);
  return U(value) / U(std::numeric_limits<uint64_t>::max());
}

// Template specialisations for the device
#ifdef __NVCC__
DEVICE double unif_rand(device_rng_state_t<double>& state) {
  const uint64_t value = xoshiro_next(state);
  // 18446744073709551616.0 == __ull2double_rn(UINT64_MAX)
  double rand = (__ddiv_rn(__ull2double_rn(value), 18446744073709551616.0));
  return rand;
}

DEVICE float unif_rand(device_rng_state_t<float>& state) {
  const uint64_t value = xoshiro_next(state);
  float rand = (__fdiv_rn(__ull2float_rn(value), 18446744073709551616.0f));
  return rand;
}
#endif

}

#endif
