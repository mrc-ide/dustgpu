#include <iostream>
#include <dust/dust.hpp>
#include <dust/interface.hpp>

template <typename real_t, typename container>
real_t odin_sum1(const container x, size_t from, size_t to);
template <typename real_t, typename container>
real_t odin_sum2(const container x, int from_i, int to_i, int from_j, int to_j, int dim_x_1);
// [[dust::class(sirs)]]
// [[dust::param(I_ini, has_default = FALSE, default_value = NULL, rank = 0, min = -Inf, max = Inf, integer = FALSE)]]
// [[dust::param(m, has_default = FALSE, default_value = NULL, rank = 2, min = -Inf, max = Inf, integer = FALSE)]]
// [[dust::param(N_age, has_default = FALSE, default_value = NULL, rank = 0, min = -Inf, max = Inf, integer = FALSE)]]
// [[dust::param(alpha, has_default = TRUE, default_value = 0.1, rank = 0, min = -Inf, max = Inf, integer = FALSE)]]
// [[dust::param(beta, has_default = TRUE, default_value = 0.2, rank = 0, min = -Inf, max = Inf, integer = FALSE)]]
// [[dust::param(dt, has_default = TRUE, default_value = 1L, rank = 0, min = -Inf, max = Inf, integer = FALSE)]]
// [[dust::param(gamma, has_default = TRUE, default_value = 0.1, rank = 0, min = -Inf, max = Inf, integer = FALSE)]]
// [[dust::param(S_ini, has_default = TRUE, default_value = 1000L, rank = 0, min = -Inf, max = Inf, integer = FALSE)]]
class sirs {
public:
  typedef double real_t;
  typedef dust::no_data data_t;
  struct shared_t {
    real_t alpha;
    real_t beta;
    int dim_I;
    int dim_lambda;
    int dim_m;
    int dim_m_1;
    int dim_m_2;
    int dim_n_IR;
    int dim_n_RS;
    int dim_n_SI;
    int dim_p_SI;
    int dim_R;
    int dim_S;
    int dim_s_ij;
    int dim_s_ij_1;
    int dim_s_ij_2;
    real_t dt;
    real_t gamma;
    real_t I_ini;
    std::vector<real_t> initial_I;
    std::vector<real_t> initial_R;
    std::vector<real_t> initial_S;
    real_t initial_time;
    std::vector<real_t> m;
    int N_age;
    int offset_variable_I;
    int offset_variable_R;
    real_t p_IR;
    real_t p_RS;
    real_t S_ini;
  };
  struct internal_t {
    std::vector<real_t> lambda;
    std::vector<real_t> n_IR;
    std::vector<real_t> n_RS;
    std::vector<real_t> n_SI;
    std::vector<real_t> p_SI;
    std::vector<real_t> s_ij;
  };
  sirs(const dust::pars_t<sirs>& pars) :
    shared(pars.shared), internal(pars.internal) {
  }
  size_t size() {
    return 1 + shared->dim_S + shared->dim_I + shared->dim_R;
  }
  std::vector<real_t> initial(size_t step) {
    std::vector<real_t> state(1 + shared->dim_S + shared->dim_I + shared->dim_R);
    state[0] = shared->initial_time;
    std::copy(shared->initial_S.begin(), shared->initial_S.end(), state.begin() + 1);
    std::copy(shared->initial_I.begin(), shared->initial_I.end(), state.begin() + shared->offset_variable_I);
    std::copy(shared->initial_R.begin(), shared->initial_R.end(), state.begin() + shared->offset_variable_R);
    return state;
  }
  void update(size_t step, const real_t * state, dust::rng_state_t<real_t>& rng_state, real_t * state_next) {
    const real_t * S = state + 1;
    const real_t * I = state + shared->offset_variable_I;
    const real_t * R = state + shared->offset_variable_R;
    state_next[0] = (step + 1) * shared->dt;
    for (int i = 1; i <= shared->dim_n_IR; ++i) {
      internal.n_IR[i - 1] = dust::distr::rbinom(rng_state, std::round(I[i - 1]), shared->p_IR);
    }
    for (int i = 1; i <= shared->dim_n_RS; ++i) {
      internal.n_RS[i - 1] = dust::distr::rbinom(rng_state, std::round(R[i - 1]), shared->p_RS);
    }
    for (int i = 1; i <= shared->dim_s_ij_1; ++i) {
      for (int j = 1; j <= shared->dim_s_ij_2; ++j) {
        internal.s_ij[i - 1 + shared->dim_s_ij_1 * (j - 1)] = shared->m[shared->dim_m_1 * (j - 1) + i - 1] * I[i - 1];
      }
    }
    for (int i = 1; i <= shared->dim_R; ++i) {
      state_next[shared->offset_variable_R + i - 1] = R[i - 1] + internal.n_IR[i - 1] - internal.n_RS[i - 1];
    }
    for (int i = 1; i <= shared->dim_lambda; ++i) {
      internal.lambda[i - 1] = shared->beta * odin_sum2<real_t>(internal.s_ij.data(), 0, shared->dim_s_ij_1, i - 1, i, shared->dim_s_ij_1);
    }
    for (int i = 1; i <= shared->dim_p_SI; ++i) {
      internal.p_SI[i - 1] = 1 - std::exp(- internal.lambda[i - 1] * shared->dt);
    }
    for (int i = 1; i <= shared->dim_n_SI; ++i) {
      internal.n_SI[i - 1] = dust::distr::rbinom(rng_state, std::round(S[i - 1]), internal.p_SI[i - 1]);
    }
    for (int i = 1; i <= shared->dim_I; ++i) {
      state_next[shared->offset_variable_I + i - 1] = I[i - 1] + internal.n_SI[i - 1] - internal.n_IR[i - 1];
    }
    for (int i = 1; i <= shared->dim_S; ++i) {
      state_next[1 + i - 1] = S[i - 1] + internal.n_RS[i - 1] - internal.n_SI[i - 1];
    }
  }
private:
  std::shared_ptr<const shared_t> shared;
  internal_t internal;
};
namespace dust {
template <>
struct has_gpu_support<sirs> : std::true_type {};
}
template <>
size_t dust::device_shared_size_int<sirs>(dust::shared_ptr<sirs> shared) {
  return 16;
}
template <>
size_t dust::device_shared_size_real<sirs>(dust::shared_ptr<sirs> shared) {
  return 4 + shared->dim_m;
}
template <>
size_t dust::device_internal_size_int<sirs>(dust::shared_ptr<sirs> shared) {
  return 0;
}
template <>
size_t dust::device_internal_size_real<sirs>(dust::shared_ptr<sirs> shared) {
  return 0 + shared->dim_lambda + shared->dim_n_IR + shared->dim_n_RS + shared->dim_n_SI + shared->dim_p_SI + shared->dim_s_ij;
}
template <>
void dust::device_shared_copy<sirs>(dust::shared_ptr<sirs> shared, int * shared_int, sirs::real_t * shared_real) {
  shared_int = dust::shared_copy(shared_int, shared->dim_n_IR);
  shared_int = dust::shared_copy(shared_int, shared->dim_n_RS);
  shared_int = dust::shared_copy(shared_int, shared->dim_s_ij);
  shared_int = dust::shared_copy(shared_int, shared->dim_R);
  shared_int = dust::shared_copy(shared_int, shared->dim_lambda);
  shared_int = dust::shared_copy(shared_int, shared->dim_p_SI);
  shared_int = dust::shared_copy(shared_int, shared->dim_n_SI);
  shared_int = dust::shared_copy(shared_int, shared->dim_I);
  shared_int = dust::shared_copy(shared_int, shared->dim_S);
  shared_int = dust::shared_copy(shared_int, shared->dim_m);
  shared_int = dust::shared_copy(shared_int, shared->dim_m_1);
  shared_int = dust::shared_copy(shared_int, shared->dim_m_2);
  shared_int = dust::shared_copy(shared_int, shared->dim_s_ij_1);
  shared_int = dust::shared_copy(shared_int, shared->dim_s_ij_2);
  shared_int = dust::shared_copy(shared_int, shared->offset_variable_I);
  shared_int = dust::shared_copy(shared_int, shared->offset_variable_R);
  shared_real = dust::shared_copy(shared_real, shared->dt);
  shared_real = dust::shared_copy(shared_real, shared->p_IR);
  shared_real = dust::shared_copy(shared_real, shared->p_RS);
  shared_real = dust::shared_copy(shared_real, shared->beta);
  shared_real = dust::shared_copy(shared_real, shared->m);
}
template<>
void update_device<sirs>(size_t step, const dust::interleaved<sirs::real_t> state, dust::interleaved<int> internal_int, dust::interleaved<sirs::real_t> internal_real, const int * shared_int, const sirs::real_t * shared_real, dust::rng_state_t<sirs::real_t>& rng_state, dust::interleaved<sirs::real_t> state_next) {
  typedef sirs::real_t real_t;
  int dim_n_IR = shared_int[0];
  int dim_n_RS = shared_int[1];
  int dim_s_ij = shared_int[2];
  int dim_R = shared_int[3];
  int dim_lambda = shared_int[4];
  int dim_p_SI = shared_int[5];
  int dim_n_SI = shared_int[6];
  int dim_I = shared_int[7];
  int dim_S = shared_int[8];
  int dim_m = shared_int[9];
  int dim_m_1 = shared_int[10];
  int dim_m_2 = shared_int[11];
  int dim_s_ij_1 = shared_int[12];
  int dim_s_ij_2 = shared_int[13];
  int offset_variable_I = shared_int[14];
  int offset_variable_R = shared_int[15];
  real_t dt = shared_real[0];
  real_t p_IR = shared_real[1];
  real_t p_RS = shared_real[2];
  real_t beta = shared_real[3];
  const real_t * m = shared_real + 4;
  dust::interleaved<sirs::real_t> lambda = internal_real + 0;
  dust::interleaved<sirs::real_t> n_IR = lambda + dim_lambda;
  dust::interleaved<sirs::real_t> n_RS = n_IR + dim_n_IR;
  dust::interleaved<sirs::real_t> n_SI = n_RS + dim_n_RS;
  dust::interleaved<sirs::real_t> p_SI = n_SI + dim_n_SI;
  dust::interleaved<sirs::real_t> s_ij = p_SI + dim_p_SI;
  const dust::interleaved<real_t> S = state + 1;
  const dust::interleaved<real_t> I = state + offset_variable_I;
  const dust::interleaved<real_t> R = state + offset_variable_R;
  state_next[0] = (step + 1) * dt;
  for (int i = 1; i <= dim_n_IR; ++i) {
    n_IR[i - 1] = dust::distr::rbinom(rng_state, std::round(I[i - 1]), p_IR);
  }
  for (int i = 1; i <= dim_n_RS; ++i) {
    n_RS[i - 1] = dust::distr::rbinom(rng_state, std::round(R[i - 1]), p_RS);
  }
  for (int i = 1; i <= dim_s_ij_1; ++i) {
    for (int j = 1; j <= dim_s_ij_2; ++j) {
      s_ij[i - 1 + dim_s_ij_1 * (j - 1)] = m[dim_m_1 * (j - 1) + i - 1] * I[i - 1];
    }
  }
  for (int i = 1; i <= dim_R; ++i) {
    state_next[offset_variable_R + i - 1] = R[i - 1] + n_IR[i - 1] - n_RS[i - 1];
  }
  for (int i = 1; i <= dim_lambda; ++i) {
    lambda[i - 1] = beta * odin_sum2<real_t>(s_ij, 0, dim_s_ij_1, i - 1, i, dim_s_ij_1);
  }
  for (int i = 1; i <= dim_p_SI; ++i) {
    p_SI[i - 1] = 1 - std::exp(- lambda[i - 1] * dt);
  }
  for (int i = 1; i <= dim_n_SI; ++i) {
    n_SI[i - 1] = dust::distr::rbinom(rng_state, std::round(S[i - 1]), p_SI[i - 1]);
  }
  for (int i = 1; i <= dim_I; ++i) {
    state_next[offset_variable_I + i - 1] = I[i - 1] + n_SI[i - 1] - n_IR[i - 1];
  }
  for (int i = 1; i <= dim_S; ++i) {
    state_next[1 + i - 1] = S[i - 1] + n_RS[i - 1] - n_SI[i - 1];
  }
}
template <typename real_t, typename container>
real_t odin_sum2(const container x, int from_i, int to_i, int from_j, int to_j, int dim_x_1) {
  real_t tot = 0.0;
  for (int j = from_j; j < to_j; ++j) {
    int jj = j * dim_x_1;
    for (int i = from_i; i < to_i; ++i) {
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
#include <memory>
#include <vector>

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
  if (is_na(value)) {
    cpp11::stop("'%s' must not be NA", name);
  }
  if (!is_na(min) && value < min) {
    cpp11::stop("Expected '%s' to be at least %g", name, (double) min);
  }
  if (!is_na(max) && value > max) {
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

template <>
inline float user_get_scalar<float>(cpp11::list user, const char *name,
                                    const float previous, float min, float max) {
  double value = user_get_scalar<double>(user, name, previous, min, max);
  return static_cast<float>(value);
}

template <typename T>
std::vector<T> user_get_array_value(cpp11::sexp x, const char * name,
                                    T min, T max) {
  std::vector<T> ret = cpp11::as_cpp<std::vector<T>>(x);
  user_check_array_value<T>(ret, name, min, max);
  return ret;
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

  return user_get_array_value<T>(x, name, min, max);
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

  return user_get_array_value<T>(x, name, min, max);
}

template <>
inline std::vector<float> user_get_array_value(cpp11::sexp x, const char * name,
                                               float min, float max) {
  // NOTE: possible under/overflow here for min/max because we've
  // downcast this.
  std::vector<double> value = user_get_array_value<double>(x, name, min, max);
  std::vector<float> ret(value.size());
  std::copy(value.begin(), value.end(), ret.begin());
  return ret;
}

// This is sum with inclusive "from", exclusive "to", following the
// same function in odin
template <typename real_t, typename container>
real_t odin_sum1(const container x, size_t from, size_t to) {
  real_t tot = 0.0;
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
dust::pars_t<sirs> dust_pars<sirs>(cpp11::list user) {
  typedef typename sirs::real_t real_t;
  auto shared = std::make_shared<sirs::shared_t>();
  sirs::internal_t internal;
  shared->initial_time = 0;
  shared->I_ini = NA_REAL;
  shared->N_age = NA_INTEGER;
  shared->alpha = 0.10000000000000001;
  shared->beta = 0.20000000000000001;
  shared->dt = 1;
  shared->gamma = 0.10000000000000001;
  shared->S_ini = 1000;
  shared->alpha = user_get_scalar<real_t>(user, "alpha", shared->alpha, NA_REAL, NA_REAL);
  shared->beta = user_get_scalar<real_t>(user, "beta", shared->beta, NA_REAL, NA_REAL);
  shared->dt = user_get_scalar<real_t>(user, "dt", shared->dt, NA_REAL, NA_REAL);
  shared->gamma = user_get_scalar<real_t>(user, "gamma", shared->gamma, NA_REAL, NA_REAL);
  shared->I_ini = user_get_scalar<real_t>(user, "I_ini", shared->I_ini, NA_REAL, NA_REAL);
  shared->N_age = user_get_scalar<int>(user, "N_age", shared->N_age, NA_REAL, NA_REAL);
  shared->S_ini = user_get_scalar<real_t>(user, "S_ini", shared->S_ini, NA_REAL, NA_REAL);
  shared->dim_I = shared->N_age;
  shared->dim_lambda = shared->N_age;
  shared->dim_m_1 = shared->N_age;
  shared->dim_m_2 = shared->N_age;
  shared->dim_n_IR = shared->N_age;
  shared->dim_n_RS = shared->N_age;
  shared->dim_n_SI = shared->N_age;
  shared->dim_p_SI = shared->N_age;
  shared->dim_R = shared->N_age;
  shared->dim_S = shared->N_age;
  shared->dim_s_ij_1 = shared->N_age;
  shared->dim_s_ij_2 = shared->N_age;
  shared->p_IR = 1 - std::exp(- shared->gamma * shared->dt);
  shared->p_RS = 1 - std::exp(- shared->alpha * shared->dt);
  shared->initial_I = std::vector<real_t>(shared->dim_I);
  shared->initial_R = std::vector<real_t>(shared->dim_R);
  shared->initial_S = std::vector<real_t>(shared->dim_S);
  internal.lambda = std::vector<real_t>(shared->dim_lambda);
  internal.n_IR = std::vector<real_t>(shared->dim_n_IR);
  internal.n_RS = std::vector<real_t>(shared->dim_n_RS);
  internal.n_SI = std::vector<real_t>(shared->dim_n_SI);
  internal.p_SI = std::vector<real_t>(shared->dim_p_SI);
  shared->dim_m = shared->dim_m_1 * shared->dim_m_2;
  shared->dim_s_ij = shared->dim_s_ij_1 * shared->dim_s_ij_2;
  for (int i = 1; i <= shared->dim_I; ++i) {
    shared->initial_I[i - 1] = shared->I_ini;
  }
  for (int i = 1; i <= shared->dim_R; ++i) {
    shared->initial_R[i - 1] = 0;
  }
  for (int i = 1; i <= shared->dim_S; ++i) {
    shared->initial_S[i - 1] = shared->S_ini;
  }
  shared->offset_variable_I = 1 + shared->dim_S;
  shared->offset_variable_R = 1 + shared->dim_S + shared->dim_I;
  internal.s_ij = std::vector<real_t>(shared->dim_s_ij);
  shared->m = user_get_array_fixed<real_t, 2>(user, "m", shared->m, {shared->dim_m_1, shared->dim_m_2}, NA_REAL, NA_REAL);
  return dust::pars_t<sirs>(shared, internal);
}
template <>
cpp11::sexp dust_info<sirs>(const dust::pars_t<sirs>& pars) {
  const sirs::internal_t internal = pars.internal;
  const std::shared_ptr<const sirs::shared_t> shared = pars.shared;
  cpp11::writable::strings nms({"time", "S", "I", "R"});
  cpp11::writable::list dim(4);
  dim[0] = cpp11::writable::integers({1});
  dim[1] = cpp11::writable::integers({shared->dim_S});
  dim[2] = cpp11::writable::integers({shared->dim_I});
  dim[3] = cpp11::writable::integers({shared->dim_R});
  dim.names() = nms;
  cpp11::writable::list index(4);
  index[0] = cpp11::writable::integers({1});
  index[1] = integer_sequence(2, shared->dim_S);
  index[2] = integer_sequence(shared->offset_variable_I + 1, shared->dim_I);
  index[3] = integer_sequence(shared->offset_variable_R + 1, shared->dim_R);
  index.names() = nms;
  size_t len = shared->offset_variable_R + shared->dim_R;
  using namespace cpp11::literals;
  return cpp11::writable::list({
           "dim"_nm = dim,
           "len"_nm = len,
           "index"_nm = index});
}

// cpp11 register functions
#include "sirs.hpp"
