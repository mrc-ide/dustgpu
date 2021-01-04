#ifndef DUST_DISTR_BINOMIAL_HPP
#define DUST_DISTR_BINOMIAL_HPP

#include <cmath>

#ifdef __NVCC__
  // Exact function for this table is
  // std::lgamma(k + 1) - (std::log(std::sqrt(2 * M_PI)) +
  //    (k + 0.5) * std::log(k + 1) - (k + 1))
__constant__ float constkTailValues[] =
  {0.08106146679532733f,
   0.041340695955409457f,
   0.027677925684998161f,
   0.020790672103765173f,
   0.016644691189821259f,
   0.013876128823071099f,
   0.011896709945892425f,
   0.010411265261970115f,
   0.009255462182705898f,
   0.0083305634333594725f,
   0.0075736754879489609f,
   0.0069428401072073598f,
   0.006408994188007f,
   0.0059513701127578145f,
   0.0055547335519605667f,
   0.0052076559196052585f};
#endif

namespace dust {
namespace distr {

// Faster version of pow(x, n) for integer 'n' by using
// "exponentiation by squaring"
// https://en.wikipedia.org/wiki/Exponentiation_by_squaring
template <typename T>
HOSTDEVICE T fast_pow(T x, int n) {
  T pow = 1.0;
  if (n != 0) {
    while (true) {
      if(n & 01) {
        pow *= x;
      }
      if(n >>= 1) {
        x *= x;
      } else {
        break;
      }
    }
  }
  return pow;
}

// Binomial random numbers via inversion (for low np only!). Draw a
// random number from U(0, 1) and find the 'n' up the distribution
// (given p) that corresponds to this
__nv_exec_check_disable__
template <typename T>
inline HOSTDEVICE typename T::real_t binomial_inversion(T& rng_state,
                                                        int n,
                                                        typename T::real_t p) {
  using real_t = typename T::real_t;
  real_t u = dust::unif_rand(rng_state);

  // This is equivalent to qbinom(u, n, p)
  const real_t q = 1 - p;
  const real_t r = p / q;
  const real_t g = r * (n + 1);
  real_t f = fast_pow(q, n);
  int k = 0;
  while (u >= f) {
    u -= f;
    k++;
    f *= (g / k - r);
  }

  return k;
}

template <typename real_t>
inline HOSTDEVICE real_t stirling_approx_tail(real_t k) {
#ifndef __CUDA_ARCH__
  static real_t kTailValues[] = {0.0810614667953272,  0.0413406959554092,
                                 0.0276779256849983,  0.02079067210376509,
                                 0.0166446911898211,  0.0138761288230707,
                                 0.0118967099458917,  0.0104112652619720,
                                 0.00925546218271273, 0.00833056343336287};
#endif
  real_t tail;
  if (k <= 15) {
#ifndef __CUDA_ARCH__
    tail = kTailValues[static_cast<int>(k)];
#else
    tail = constkTailValues[static_cast<int>(k)];
#endif
  } else {
    real_t kp1sq = (k + 1) * (k + 1);
    tail = (1.0 / 12 - (1.0 / 360 - 1.0 / 1260 / kp1sq) / kp1sq) / (k + 1);
  }
  return tail;
}

// https://www.tandfonline.com/doi/abs/10.1080/00949659308811496
__nv_exec_check_disable__
template <typename T>
inline HOSTDEVICE double btrs(T& rng_state, typename T::real_t n,
                              typename T::real_t p) {
  // This is spq in the paper.
  using real_t = typename T::real_t;
  const real_t stddev = std::sqrt(n * p * (1 - p));

  // Other coefficients for Transformed Rejection sampling.
  const real_t b = 1.15 + 2.53 * stddev;
  const real_t a = -0.0873 + 0.0248 * b + 0.01 * p;
  const real_t c = n * p + 0.5;
  const real_t v_r = 0.92 - 4.2 / b;
  const real_t r = p / (1 - p);

  const real_t alpha = (2.83 + 5.1 / b) * stddev;
  const real_t m = std::floor((n + 1) * p);

  real_t draw;
  while (true) {
    real_t u = dust::unif_rand(rng_state);
    real_t v = dust::unif_rand(rng_state);
    u = u - 0.5;
    real_t us = 0.5 - std::fabs(u);
    real_t k = std::floor((2 * a / us + b) * u + c);

    // Region for which the box is tight, and we
    // can return our calculated value This should happen
    // 0.86 * v_r times. In the limit as n * p is large,
    // the acceptance rate converges to ~79% (and in the lower
    // regime it is ~24%).
    if (us >= 0.07 && v <= v_r) {
      draw = k;
      break;
    }
    // Reject non-sensical answers.
    if (k < 0 || k > n) {
      continue;
    }

    // This deviates from Hormann's BRTS algorithm, as there is a log missing.
    // For all (u, v) pairs outside of the bounding box, this calculates the
    // transformed-reject ratio.
    v = std::log(v * alpha / (a / (us * us) + b));
    real_t upperbound =
      ((m + 0.5) * std::log((m + 1) / (r * (n - m + 1))) +
       (n + 1) * std::log((n - m + 1) / (n - k + 1)) +
       (k + 0.5) * std::log(r * (n - k + 1) / (k + 1)) +
       stirling_approx_tail(m) + stirling_approx_tail(n - m) -
       stirling_approx_tail(k) - stirling_approx_tail(n - k));
    if (v <= upperbound) {
      draw = k;
      break;
    }
  }
  return draw;
}

template <typename T>
HOSTDEVICE int rbinom(T& rng_state, int n,
                      typename T::real_t p) {
  int draw;

  // Early exit:
  if (n == 0 || p == 0) {
    return 0;
  }
  if (p == 1) {
    return n;
  }

  // TODO: Should control for this too, but not really clear what we
  // need to do to safely deal.
  /*
    if (n < 0 || p < 0 || p > 1) {
    return NaN;
    }
  */

  typename T::real_t q = p;
  if (p > 0.5) {
    q = 1 - q;
  }

  if (n * q >= 10) {
    draw = static_cast<int>(btrs(rng_state, n, q));
  } else {
    draw = static_cast<int>(binomial_inversion(rng_state, n, q));
  }

  if (p > 0.5) {
    draw = n - draw;
  }

  return draw;
}

}
}

#endif
