// Generated by odin.dust (version 0.0.10) - do not edit
template <typename T>
T odin_sum1(const T * x, size_t from, size_t to);
template <typename real_t, typename int_t>
real_t odin_sum2(const real_t * x, int_t from_i, int_t to_i, int_t from_j, int_t to_j, int_t dim_x_1);
class sir {
public:
  typedef int int_t;
  typedef double real_t;
  struct init_t {
    real_t beta;
    int_t dim_I;
    int_t dim_lambda;
    int_t dim_m;
    int_t dim_m_1;
    int_t dim_m_2;
    int_t dim_n_IR;
    int_t dim_n_SI;
    int_t dim_p_SI;
    int_t dim_R;
    int_t dim_S;
    int_t dim_s_ij;
    int_t dim_s_ij_1;
    int_t dim_s_ij_2;
    real_t dt;
    real_t gamma;
    real_t I_ini;
    std::vector<real_t> initial_I;
    std::vector<real_t> initial_R;
    std::vector<real_t> initial_S;
    real_t initial_time;
    std::vector<real_t> lambda;
    std::vector<real_t> m;
    int_t N_age;
    std::vector<real_t> n_IR;
    std::vector<real_t> n_SI;
    int_t offset_variable_I;
    int_t offset_variable_R;
    real_t p_IR;
    std::vector<real_t> p_SI;
    std::vector<real_t> s_ij;
    real_t S_ini;
  };
  sir(const init_t& data): internal(data) {
  }
  size_t size() {
    return 1 + internal.dim_S + internal.dim_I + internal.dim_R;
  }
  std::vector<real_t> initial(size_t step) {
    std::vector<real_t> state(1 + internal.dim_S + internal.dim_I + internal.dim_R);
    state[0] = internal.initial_time;
    std::copy(internal.initial_S.begin(), internal.initial_S.end(), state.begin() + 1);
    std::copy(internal.initial_I.begin(), internal.initial_I.end(), state.begin() + internal.offset_variable_I);
    std::copy(internal.initial_R.begin(), internal.initial_R.end(), state.begin() + internal.offset_variable_R);
    return state;
  }
  #ifdef __NVCC__
  __device__
  #endif
  void update(size_t step, const real_t * state, dust::rng_state_t<real_t> rng_state, real_t * state_next) {
    const real_t * S = state + 1;
    const real_t * I = state + internal.offset_variable_I;
    const real_t * R = state + internal.offset_variable_R;
    real_t N = odin_sum1(S, 0, internal.dim_S) + odin_sum1(I, 0, internal.dim_I) + odin_sum1(R, 0, internal.dim_R);
    state_next[0] = (step + 1) * internal.dt;
    for (int_t i = 1; i <= internal.dim_n_IR; ++i) {
      internal.n_IR[i - 1] = dust::distr::rbinom(rng_state, std::round(I[i - 1]), internal.p_IR);
    }
    for (int_t i = 1; i <= internal.dim_s_ij_1; ++i) {
      for (int_t j = 1; j <= internal.dim_s_ij_2; ++j) {
        internal.s_ij[i - 1 + internal.dim_s_ij_1 * (j - 1)] = internal.m[internal.dim_m_1 * (j - 1) + i - 1] * I[i - 1];
      }
    }
    for (int_t i = 1; i <= internal.dim_R; ++i) {
      state_next[internal.offset_variable_R + i - 1] = R[i - 1] + internal.n_IR[i - 1];
    }
    for (int_t i = 1; i <= internal.dim_lambda; ++i) {
      internal.lambda[i - 1] = internal.beta / (real_t) N * odin_sum2(internal.s_ij.data(), i - 1, i, 0, internal.dim_s_ij_2, internal.dim_s_ij_1);
    }
    for (int_t i = 1; i <= internal.dim_p_SI; ++i) {
      internal.p_SI[i - 1] = 1 - std::exp(- internal.lambda[i - 1] * internal.dt);
    }
    for (int_t i = 1; i <= internal.dim_n_SI; ++i) {
      internal.n_SI[i - 1] = dust::distr::rbinom(rng_state, std::round(S[i - 1]), internal.p_SI[i - 1]);
    }
    for (int_t i = 1; i <= internal.dim_I; ++i) {
      state_next[internal.offset_variable_I + i - 1] = I[i - 1] + internal.n_SI[i - 1] - internal.n_IR[i - 1];
    }
    for (int_t i = 1; i <= internal.dim_S; ++i) {
      state_next[1 + i - 1] = S[i - 1] - internal.n_SI[i - 1];
    }
  }
private:
  init_t internal;
};
template <typename real_t, typename int_t>
real_t odin_sum2(const real_t * x, int_t from_i, int_t to_i, int_t from_j, int_t to_j, int_t dim_x_1) {
  real_t tot = 0.0;
  for (int_t j = from_j; j < to_j; ++j) {
    int_t jj = j * dim_x_1;
    for (int_t i = from_i; i < to_i; ++i) {
      tot += x[i + jj];
    }
  }
  return tot;
}
#include <array>
#include <cpp11/R.hpp>
#include <cpp11/sexp.hpp>
#include <cpp11/doubles.hpp>
#include <cpp11/integers.hpp>
#include <cpp11/list.hpp>
#include <cpp11/strings.hpp>
#include <vector>

// These would be nice to make constexpr but the way that NA values
// are defined in R's include files do not allow it.
template <typename T>
inline T na_value();

template <>
inline int na_value<int>() {
  return NA_INTEGER;
}

template <>
inline double na_value<double>() {
  return NA_REAL;
}

template <typename T>
inline bool is_na(T x);

template <>
inline bool is_na(int x) {
  return x == NA_INTEGER;
}

template <>
inline bool is_na(double x) {
  return ISNA(x);
}

inline size_t object_length(cpp11::sexp x) {
  return ::Rf_xlength(x);
}

template <typename T>
void user_check_value(T value, const char *name, T min, T max) {
  if (ISNA(value)) {
    cpp11::stop("'%s' must not be NA", name);
  }
  if (min != na_value<T>() && value < min) {
    cpp11::stop("Expected '%s' to be at least %g", name, (double) min);
  }
  if (max != na_value<T>() && value > max) {
    cpp11::stop("Expected '%s' to be at most %g", name, (double) max);
  }
}

template <typename T>
void user_check_array_value(const std::vector<T>& value, const char *name,
                            T min, T max) {
  for (auto& x : value) {
    user_check_value(x, name, min, max);
  }
}

inline size_t user_get_array_rank(cpp11::sexp x) {
  if (!::Rf_isArray(x)) {
    return 1;
  } else {
    cpp11::integers dim = cpp11::as_cpp<cpp11::integers>(x.attr("dim"));
    return dim.size();
  }
}

template <size_t N>
void user_check_array_rank(cpp11::sexp x, const char *name) {
  size_t rank = user_get_array_rank(x);
  if (rank != N) {
    if (N == 1) {
      cpp11::stop("Expected a vector for '%s'", name);
    } else if (N == 2) {
      cpp11::stop("Expected a matrix for '%s'", name);
    } else {
      cpp11::stop("Expected an array of rank %d for '%s'", N, name);
    }
  }
}

template <size_t N>
void user_check_array_dim(cpp11::sexp x, const char *name,
                          const std::array<int, N>& dim_expected) {
  cpp11::integers dim = cpp11::as_cpp<cpp11::integers>(x.attr("dim"));
  for (size_t i = 0; i < N; ++i) {
    if (dim[(int)i] != dim_expected[i]) {
      Rf_error("Incorrect size of dimension %d of '%s' (expected %d)",
               i + 1, name, dim_expected[i]);
    }
  }
}

template <>
inline void user_check_array_dim<1>(cpp11::sexp x, const char *name,
                                    const std::array<int, 1>& dim_expected) {
  if ((int)object_length(x) != dim_expected[0]) {
    cpp11::stop("Expected length %d value for '%s'", dim_expected[0], name);
  }
}

template <size_t N>
void user_set_array_dim(cpp11::sexp x, const char *name,
                        std::array<int, N>& dim) {
  cpp11::integers dim_given = cpp11::as_cpp<cpp11::integers>(x.attr("dim"));
  std::copy(dim_given.begin(), dim_given.end(), dim.begin());
}

template <>
inline void user_set_array_dim<1>(cpp11::sexp x, const char *name,
                                  std::array<int, 1>& dim) {
  dim[0] = object_length(x);
}

template <typename T>
T user_get_scalar(cpp11::list user, const char *name,
                  const T previous, T min, T max) {
  T ret = previous;
  cpp11::sexp x = user[name];
  if (x != R_NilValue) {
    if (object_length(x) != 1) {
      cpp11::stop("Expected a scalar numeric for '%s'", name);
    }
    // TODO: when we're getting out an integer this is a bit too relaxed
    if (TYPEOF(x) == REALSXP) {
      ret = cpp11::as_cpp<T>(x);
    } else if (TYPEOF(x) == INTSXP) {
      ret = cpp11::as_cpp<T>(x);
    } else {
      cpp11::stop("Expected a numeric value for %s", name);
    }
  }

  if (is_na(ret)) {
    cpp11::stop("Expected a value for '%s'", name);
  }
  user_check_value<T>(ret, name, min, max);
  return ret;
}

// This is not actually really enough to work generally as there's an
// issue with what to do with checking previous, min and max against
// NA_REAL -- which is not going to be the correct value for float
// rather than double.  Further, this is not extendable to the vector
// cases because we hit issues around partial template specification.
//
// We can make the latter go away by replacing std::array<T, N> with
// std::vector<T> - the cost is not great.  But the NA issues remain
// and will require further thought. However, this template
// specialisation and the tests that use it ensure that the core code
// generation is at least compatible with floats.
//
// See #6
template <>
inline float user_get_scalar<float>(cpp11::list user, const char *name,
                                    const float previous, float min, float max) {
  double value = user_get_scalar<double>(user, name, previous, min, max);
  return static_cast<float>(value);
}

template <typename T, size_t N>
std::vector<T> user_get_array_fixed(cpp11::list user, const char *name,
                                    const std::vector<T> previous,
                                    const std::array<int, N>& dim,
                                    T min, T max) {
  cpp11::sexp x = user[name];
  if (x == R_NilValue) {
    if (previous.size() == 0) {
      cpp11::stop("Expected a value for '%s'", name);
    }
    return previous;
  }

  user_check_array_rank<N>(x, name);
  user_check_array_dim<N>(x, name, dim);

  std::vector<T> ret = cpp11::as_cpp<std::vector<T>>(x);
  user_check_array_value(ret, name, min, max);

  return ret;
}

template <typename T, size_t N>
std::vector<T> user_get_array_variable(cpp11::list user, const char *name,
                                       std::vector<T> previous,
                                       std::array<int, N>& dim,
                                       T min, T max) {
  cpp11::sexp x = user[name];
  if (x == R_NilValue) {
    if (previous.size() == 0) {
      cpp11::stop("Expected a value for '%s'", name);
    }
    return previous;
  }

  user_check_array_rank<N>(x, name);
  user_set_array_dim<N>(x, name, dim);

  std::vector<T> ret = cpp11::as_cpp<std::vector<T>>(x);
  user_check_array_value(ret, name, min, max);

  return ret;
}

// This is sum with inclusive "from", exclusive "to", following the
// same function in odin
template <typename T>
#ifdef __NVCC__
__host__ __device__
#endif
T odin_sum1(const T * x, size_t from, size_t to) {
  T tot = 0.0;
  for (size_t i = from; i < to; ++i) {
    tot += x[i];
  }
  return tot;
}


inline cpp11::writable::integers integer_sequence(size_t from, size_t len) {
  cpp11::writable::integers ret(len);
  int* data = INTEGER(ret);
  for (size_t i = 0, j = from; i < len; ++i, ++j) {
    data[i] = j;
  }
  return ret;
}
template<>
sir::init_t dust_data<sir>(cpp11::list user) {
  typedef typename sir::real_t real_t;
  typedef typename sir::int_t int_t;
  sir::init_t internal;
  internal.initial_time = 0;
  internal.I_ini = NA_REAL;
  internal.N_age = NA_INTEGER;
  internal.beta = 0.20000000000000001;
  internal.dt = 1;
  internal.gamma = 0.10000000000000001;
  internal.S_ini = 1000;
  internal.beta = user_get_scalar<real_t>(user, "beta", internal.beta, NA_REAL, NA_REAL);
  internal.dt = user_get_scalar<real_t>(user, "dt", internal.dt, NA_REAL, NA_REAL);
  internal.gamma = user_get_scalar<real_t>(user, "gamma", internal.gamma, NA_REAL, NA_REAL);
  internal.I_ini = user_get_scalar<real_t>(user, "I_ini", internal.I_ini, NA_REAL, NA_REAL);
  internal.N_age = user_get_scalar<int_t>(user, "N_age", internal.N_age, NA_REAL, NA_REAL);
  internal.S_ini = user_get_scalar<real_t>(user, "S_ini", internal.S_ini, NA_REAL, NA_REAL);
  internal.dim_I = internal.N_age;
  internal.dim_lambda = internal.N_age;
  internal.dim_m_1 = internal.N_age;
  internal.dim_m_2 = internal.N_age;
  internal.dim_n_IR = internal.N_age;
  internal.dim_n_SI = internal.N_age;
  internal.dim_p_SI = internal.N_age;
  internal.dim_R = internal.N_age;
  internal.dim_S = internal.N_age;
  internal.dim_s_ij_1 = internal.N_age;
  internal.dim_s_ij_2 = internal.N_age;
  internal.p_IR = 1 - std::exp(- internal.gamma * internal.dt);
  internal.initial_I = std::vector<real_t>(internal.dim_I);
  internal.initial_R = std::vector<real_t>(internal.dim_R);
  internal.initial_S = std::vector<real_t>(internal.dim_S);
  internal.lambda = std::vector<real_t>(internal.dim_lambda);
  internal.n_IR = std::vector<real_t>(internal.dim_n_IR);
  internal.n_SI = std::vector<real_t>(internal.dim_n_SI);
  internal.p_SI = std::vector<real_t>(internal.dim_p_SI);
  internal.dim_m = internal.dim_m_1 * internal.dim_m_2;
  internal.dim_s_ij = internal.dim_s_ij_1 * internal.dim_s_ij_2;
  for (int_t i = 1; i <= internal.dim_I; ++i) {
    internal.initial_I[i - 1] = internal.I_ini;
  }
  for (int_t i = 1; i <= internal.dim_R; ++i) {
    internal.initial_R[i - 1] = 0;
  }
  for (int_t i = 1; i <= internal.dim_S; ++i) {
    internal.initial_S[i - 1] = internal.S_ini;
  }
  internal.offset_variable_I = 1 + internal.dim_S;
  internal.offset_variable_R = 1 + internal.dim_S + internal.dim_I;
  internal.s_ij = std::vector<real_t>(internal.dim_s_ij);
  internal.m = user_get_array_fixed<real_t, 2>(user, "m", internal.m, {internal.dim_m_1, internal.dim_m_2}, NA_REAL, NA_REAL);
  return internal;
}
template <>
cpp11::sexp dust_info<sir>(const sir::init_t& internal) {
  cpp11::writable::strings nms({"time", "S", "I", "R"});
  cpp11::writable::list dim(4);
  dim[0] = cpp11::writable::integers({1});
  dim[1] = cpp11::writable::integers({internal.dim_S});
  dim[2] = cpp11::writable::integers({internal.dim_I});
  dim[3] = cpp11::writable::integers({internal.dim_R});
  dim.names() = nms;
  cpp11::writable::list index(4);
  index[0] = cpp11::writable::integers({1});
  index[1] = integer_sequence(2, internal.dim_S);
  index[2] = integer_sequence(internal.offset_variable_I + 1, internal.dim_I);
  index[3] = integer_sequence(internal.offset_variable_R + 1, internal.dim_R);
  index.names() = nms;
  size_t len = internal.offset_variable_R + internal.dim_R;
  using namespace cpp11::literals;
  return cpp11::writable::list({
           "dim"_nm = dim,
           "len"_nm = len,
           "index"_nm = index});
}
