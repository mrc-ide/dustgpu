#ifndef DUST_DISTR_UNIFORM_HPP
#define DUST_DISTR_UNIFORM_HPP

namespace dust {
namespace distr {

template <typename T>
HD typename T::real_t runif(T& rng_state,
             typename T::real_t min,
             typename T::real_t max) {
  return dust::unif_rand<T>(rng_state) * (max - min) + min;
}

}
}

#endif
