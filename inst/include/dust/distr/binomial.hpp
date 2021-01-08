#ifndef DUST_DISTR_BINOMIAL_HPP
#define DUST_DISTR_BINOMIAL_HPP

#include <cmath>

#ifdef __NVCC__
  // Exact function for this table is
  // std::lgamma(k + 1) - (std::log(std::sqrt(2 * M_PI)) +
  //    (k + 0.5) * std::log(k + 1) - (k + 1))
__constant__ float constkTailValues[] = {
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
                                  0.0052076559196052585f,
                                  0.0049013959484298653f,
                                  0.0046291537493345913f,
                                  0.0043855602492257617f,
                                  0.0041663196919898837f,
                                  0.0039679542186377148f,
                                  0.0037876180684577321f,
                                  0.0036229602246820036f,
                                  0.0034720213829828594f,
                                  0.0033331556367386383f,
                                  0.0032049702280616543f,
                                  0.0030862786826162392f,
                                  0.0029760639835672009f,
                                  0.0028734493623545632f,
                                  0.0027776749297458991f,
                                  0.0026880788285268409f,
                                  0.0026040819192587605f,
                                  0.0025251752497723601f,
                                  0.0024509097354297182f,
                                  0.0023808876082398456f,
                                  0.0023147552905129487f,
                                  0.0022521974243261411f,
                                  0.0021929318432967193f,
                                  0.0021367053177385742f,
                                  0.0020832899382980941f,
                                  0.0020324800282622846f,
                                  0.0019840894972418255f,
                                  0.0019379495639952893f,
                                  0.0018939067895757944f,
                                  0.0018518213729947774f,
                                  0.0018115656687029968f,
                                  0.0017730228939285553f,
                                  0.0017360859968675868f,
                                  0.0017006566641839527f,
                                  0.0016666444469990438f,
                                  0.001633965989896069f,
                                  0.0016025443491400893f,
                                  0.0015723083877219324f,
                                  0.0015431922375341856f,
                                  0.0015151348208348736f,
                                  0.0014880794221880933f,
                                  0.0014619733060499129f,
                                  0.001436767373576231f,
                                  0.0014124158545030241f,
                                  0.0013888760298357283f,
                                  0.0013661079815676658f,
                                  0.0013440743670685151f,
                                  0.0013227402145616907f,
                                  0.0013020727377295316f,
                                  0.001282041167968373f,
                                  0.0012626166012807971f,
                                  0.0012437718593503178f,
                                  0.0012254813623826522f,
                                  0.0012077210134293637f,
                                  0.0011904680924885724f,
                                  0.0011737011595869262f,
                                  0.0011573999656775413f,
                                  0.0011415453712970702f,
                                  0.0011261192715892321f,
                                  0.001111104527126372f,
                                  0.0010964849005574706f,
                                  0.0010822449980310012f,
                                  0.0010683702151936814f,
                                  0.0010548466869408912f,
                                  0.0010416612416292992f,
                                  0.0010288013577337551f,
                                  0.0010162551247958618f,
                                  0.0010040112064189088f,
                                  0.00099205880559338766f,
                                  0.0009803876338878581f,
                                  0.00096898788103771949f,
                                  0.00095785018794458665f,
                                  0.00094696562098306458f,
                                  0.00093632564789913886f,
                                  0.00092592211569808569f,
                                  0.00091574722972609379f,
                                  0.00090579353434350196f,
                                  0.00089605389439384453f,
                                  0.00088652147843504281f,
                                  0.00087718974270956096f,
                                  0.00086805241602405658f,
                                  0.00085910348576589968f,
                                  0.0008503371848291863f,
                                  0.00084174797910918642f,
                                  0.00083333055562206937f,
                                  0.00082507981221624505f,
                                  0.00081699084654474063f,
                                  0.00080905894668603651f,
                                  0.00080127958193543236f,
                                  0.00079364839422169098f,
                                  0.00078616118980789906f,
                                  0.00077881393195866622f,
                                  0.00077160273326626339f,
                                  0.00076452384899994286f,
                                  0.00075757367056894509f,
                                  0.00075074871966762657f,
                                  0.00074404564190899691f,
                                  0.00073746120165196771f,
                                  0.00073099227711281856f,
                                  0.00072463585468085512f,
                                  0.00071838902505305668f,
                                  0.0007122489779476382f,
                                  0.00070621299857975828f,
                                  0.0007002784636256365f,
                                  0.00069444283690245356f,
                                  0.00068870366612827638f,
                                  0.00068305857956829641f,
                                  0.0006775052823400074f,
                                  0.00067204155374156471f,
                                  0.00066666524440961439f,
                                  0.00066137427273815774f,
                                  0.00065616662294587513f,
                                  0.00065104034212026818f,
                                  0.00064599353800076642f};
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
inline HOSTDEVICE typename T::real_t binomial_inversion(T& rng_statef,
                                                        int nf,
                                                        typename T::real_t p) {
  using real_t = typename T::real_t;
#ifdef __NVCC__
  // Assumes float for now - use DBL_EPSILON if real_t == double
  const real_t epsilon = FLT_EPSILON;
#else
  constexpr real_t epsilon = std::numeric_limits<real_t>::epsilon();
#endif
  real_t u = dust::unif_rand(rng_state);

  // This is equivalent to qbinom(u, n, p)
  const real_t q = 1 - p;
  const real_t r = p / q;
  const real_t g = r * (n + 1);
  real_t f = fast_pow(q, n);
  int k = 0;
  while (u >= f && f > epsilon) {
    u -= f;
    k++;
    f *= (g / k - r);
  }

  return k;
}

inline HOSTDEVICE double stirling_approx_tail(double k) {
  static double kTailValues[] = {0.0810614667953272,  0.0413406959554092,
                                 0.0276779256849983,  0.02079067210376509,
                                 0.0166446911898211,  0.0138761288230707,
                                 0.0118967099458917,  0.0104112652619720,
                                 0.00925546218271273, 0.00833056343336287,
                                 0.0075736754879489609, 0.0069428401072073598,
                                 0.006408994188007, 0.0059513701127578145,
                                 0.0055547335519605667, 0.0052076559196052585};
  double tail;
  if (k <= 15) {
    tail = kTailValues[static_cast<int>(k)];
  } else {
    double kp1sq = (k + 1) * (k + 1);
    tail = (1.0 / 12 - (1.0 / 360 - 1.0 / 1260 / kp1sq) / kp1sq) / (k + 1);
  }
  return tail;
}

inline HOSTDEVICE float stirling_approx_tail(float k) {
#ifndef __CUDA_ARCH__
  static float kTailValues[] = {0.0810614667953272f,  0.0413406959554092f,
                                 0.0276779256849983f,  0.02079067210376509f,
                                 0.0166446911898211f,  0.0138761288230707f,
                                 0.0118967099458917f,  0.0104112652619720f,
                                 0.00925546218271273f, 0.00833056343336287f,
                                 0.0075736754879489609f, 0.0069428401072073598f,
                                 0.006408994188007f, 0.0059513701127578145f,
                                 0.0055547335519605667f, 0.0052076559196052585f};
#endif
  float tail;
#ifndef __CUDA_ARCH__
  if (k <= 15) {
    tail = kTailValues[static_cast<int>(k)];
#else
  if (k <= 127) {
    tail = constkTailValues[__float2int_rn(k)];
#endif
  } else {
    float kp1sq = (k + 1) * (k + 1);
    tail = (1.0f / 12 - (1.0f / 360 - 1.0f / 1260 / kp1sq) / kp1sq) / (k + 1);
  }
  return tail;
}

// https://www.tandfonline.com/doi/abs/10.1080/00949659308811496
__nv_exec_check_disable__
template <typename T>
inline HOSTDEVICE double btrs(T& rng_state, typename T::real_t nf,
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
    // 0.86 * v_r times. In the limit as n * p is largef,
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
HOSTDEVICE int rbinom(T& rng_state, int nf,
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
