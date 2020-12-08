#ifndef DUST_DISTR_UNIFORM_HPP
#define DUST_DISTR_UNIFORM_HPP

namespace dust {
namespace distr {

__nv_exec_check_disable__
template <typename T>
HOSTDEVICE typename T::real_t runif(T& rng_state,
             typename T::real_t min,
             typename T::real_t max) {
  return dust::unif_rand(rng_state) * (max - min) + min;
}

}
}

#endif
