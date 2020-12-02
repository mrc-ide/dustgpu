#ifndef DUST_DISTR_NORMAL_HPP
#define DUST_DISTR_NORMAL_HPP

#include <cmath>
#include <cfloat>

namespace dust {
namespace distr {

template <typename T>
HOSTDEVICE inline typename T::real_t box_muller(T& rng_state) {
  // This function implements the Box-Muller transform:
  // http://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform#Basic_form
  // Do not send a really small number to log().
  using real_t = typename T::real_t;
#ifdef __NVCC__
  const real_t two_pi = 2 * M_PI;
  // Assumes float for now - use DBL_EPSILON if real_t == double
  const real_t epsilon = FLT_EPSILON;
#else
  constexpr real_t epsilon = std::numeric_limits<real_t>::epsilon();
  constexpr real_t two_pi = 2 * M_PI;
#endif

  real_t u1, u2;
  do {
    u1 = dust::unif_rand(rng_state);
    u2 = dust::unif_rand(rng_state);
  } while (u1 <= epsilon);

  return std::sqrt(-2.0 * std::log(u1)) * std::cos(two_pi * u2);
}

// The type declarations for mean and sd are ugly but prevent the
// compiler complaining about conflicting inferred types for real_t
template <typename T>
HOSTDEVICE typename T::real_t rnorm(T& rng_state,
                                    typename T::real_t mean,
                                    typename T::real_t sd) {
  typename T::real_t z = box_muller<T>(rng_state);
  return z * sd + mean;
}

}
}

#endif
