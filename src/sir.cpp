// Generated by dust (version 0.4.13) - do not edit
#include <iostream>
#include <dust/dust.hpp>
#include <dust/interface.hpp>

// Generated by odin.dust (version 0.0.10) - do not edit
template <typename T>
T odin_sum1(const T * x, size_t from, size_t to);
template <typename real_t, typename int_t>
real_t odin_sum2(const real_t * x, int_t from_i, int_t to_i, int_t from_j, int_t to_j, int_t dim_x_1);

// More sum templates
template <typename T>
T odin_sum1(const interleaved<T> x, size_t from, size_t to);
template <typename real_t, typename int_t>
real_t odin_sum2(const interleaved<real_t> x, int_t from_i, int_t to_i, int_t from_j, int_t to_j, int_t dim_x_1);

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
  size_t size() const {
    return 1 + internal.dim_S + internal.dim_I + internal.dim_R;
  }

  // We only return things needed to compute the update function;
  // odin.dust knows what these are, but I am guessing here.
  size_t size_internal_real() const {
    // 3: beta, dt, p_IR
    return 3 + internal.dim_lambda + internal.dim_m + internal.dim_n_IR +
      internal.dim_n_SI + internal.dim_p_SI + internal.dim_s_ij;
  }

  size_t size_internal_int() const {
    // 14: dim_I, dim_R, dim_S, dim_lambda, dim_m, dim_m_1, dim_n_IR,
    // dim_n_SI, dim_p_SI, dim_s_ij, dim_s_ij_1, dim_s_ij_2,
    // offset_variable_I, offset_variable_R
    return 14;
  }

  // Big mess of functions for converting from an object into flat storage
  template <typename T>
  void internal_real(T dest, size_t stride) const {
    size_t j = 0;
    j = stride_copy(dest, internal.beta, j, stride);
    j = stride_copy(dest, internal.dt, j, stride);
    j = stride_copy(dest, internal.p_IR, j, stride);
    j = stride_copy(dest, internal.lambda, j, stride);
    j = stride_copy(dest, internal.m, j, stride);
    j = stride_copy(dest, internal.n_IR, j, stride);
    j = stride_copy(dest, internal.n_SI, j, stride);
    j = stride_copy(dest, internal.p_SI, j, stride);
    j = stride_copy(dest, internal.s_ij, j, stride);
  }

  template <typename T>
  void internal_int(T dest, size_t stride) const {
    size_t j = 0;
    j = stride_copy(dest, internal.dim_I, j, stride);
    j = stride_copy(dest, internal.dim_R, j, stride);
    j = stride_copy(dest, internal.dim_S, j, stride);
    j = stride_copy(dest, internal.dim_lambda, j, stride);
    j = stride_copy(dest, internal.dim_m, j, stride);
    j = stride_copy(dest, internal.dim_m_1, j, stride);
    j = stride_copy(dest, internal.dim_n_IR, j, stride);
    j = stride_copy(dest, internal.dim_n_SI, j, stride);
    j = stride_copy(dest, internal.dim_p_SI, j, stride);
    j = stride_copy(dest, internal.dim_s_ij, j, stride);
    j = stride_copy(dest, internal.dim_s_ij_1, j, stride);
    j = stride_copy(dest, internal.dim_s_ij_2, j, stride);
    j = stride_copy(dest, internal.offset_variable_I, j, stride);
    j = stride_copy(dest, internal.offset_variable_R, j, stride);
  }

  std::vector<real_t> initial(size_t step) {
    std::vector<real_t> state(1 + internal.dim_S + internal.dim_I + internal.dim_R);
    state[0] = internal.initial_time;
    std::copy(internal.initial_S.begin(), internal.initial_S.end(), state.begin() + 1);
    std::copy(internal.initial_I.begin(), internal.initial_I.end(), state.begin() + internal.offset_variable_I);
    std::copy(internal.initial_R.begin(), internal.initial_R.end(), state.begin() + internal.offset_variable_R);
    return state;
  }

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

// More template specialisations, though looking at this we could just
// template against the input type as we seem to always return a real?
template <typename T>
T odin_sum1(const interleaved<T> x, size_t from, size_t to) {
  T tot = 0.0;
  for (size_t i = from; i < to; ++i) {
    tot += x[i];
  }
  return tot;
}
template <typename real_t, typename int_t>
real_t odin_sum2(const interleaved<real_t> x, int_t from_i, int_t to_i, int_t from_j, int_t to_j, int_t dim_x_1) {
  real_t tot = 0.0;
  for (int_t j = from_j; j < to_j; ++j) {
    int_t jj = j * dim_x_1;
    for (int_t i = from_i; i < to_i; ++i) {
      tot += x[i + jj];
    }
  }
  return tot;
}


// I can write out fairly easily here code that would do an update
// given a pointer into the data, interleaved as needed. Looks like
// we'll need to get the RNG state into the same basic setup though,
// so that we keep the RNG state interleaved normally.
//
// We're doing this as a free function as that's how we decided that
// it might be easiest to get things working with
template <>
void update2<sir>(size_t step, const interleaved<sir::real_t> state,
             interleaved<int> internal_int,
             interleaved<sir::real_t> internal_real,
             dust::rng_state_t<sir::real_t> rng_state,
             interleaved<sir::real_t> state_next) {
  typedef sir::real_t real_t;
  typedef int int_t;
  // Unpack the integer vector
  int dim_I = internal_int[0];
  int dim_R = internal_int[1];
  int dim_S = internal_int[2];
  int dim_lambda = internal_int[3];
  int dim_m = internal_int[4];
  int dim_m_1 = internal_int[5];
  int dim_n_IR = internal_int[6];
  int dim_n_SI = internal_int[7];
  int dim_p_SI = internal_int[8];
  int dim_s_ij = internal_int[9];
  int dim_s_ij_1 = internal_int[10];
  int dim_s_ij_2 = internal_int[11];
  int offset_variable_I = internal_int[12];
  int offset_variable_R = internal_int[13];

  // Unpack the real vector; the issue here is that we need another
  // whole swath of offsets to make this work well. I know these ahead
  // of time so am going to write them out here, but this is not
  // ideal, and will need adding into the object.
  int offset_internal_lambda = 3;
  int offset_internal_m = offset_internal_lambda + dim_lambda;
  int offset_internal_n_IR = offset_internal_m + dim_m;
  int offset_internal_n_SI = offset_internal_n_IR + dim_n_IR;
  int offset_internal_p_SI = offset_internal_n_SI + dim_n_SI;
  int offset_internal_s_ij = offset_internal_p_SI + dim_p_SI;

  double beta = internal_real[0];
  double dt = internal_real[1];
  double p_IR = internal_real[2];

  // TODO - alternative would be to make offsets relative, then could
  // use operator+ after first assignment
  interleaved<real_t> lambda = internal_real + offset_internal_lambda;
  interleaved<real_t> m = internal_real + offset_internal_m;
  interleaved<real_t> n_IR = internal_real + offset_internal_n_IR;
  interleaved<real_t> n_SI = internal_real + offset_internal_n_SI;
  interleaved<real_t> p_SI = internal_real + offset_internal_p_SI;
  interleaved<real_t> s_ij = internal_real + offset_internal_s_ij;

  interleaved<real_t> S = state + 1;
  interleaved<real_t> I = state + offset_variable_I;
  interleaved<real_t> R = state + offset_variable_R;

  real_t N = odin_sum1(S, 0, dim_S) + odin_sum1(I, 0, dim_I) + odin_sum1(R, 0, dim_R);
  state_next[0] = (step + 1) * dt;
  for (int_t i = 1; i <= dim_n_IR; ++i) {
    n_IR[i - 1] = dust::distr::rbinom(rng_state, std::round(I[i - 1]), p_IR);
  }
  for (int_t i = 1; i <= dim_s_ij_1; ++i) {
    for (int_t j = 1; j <= dim_s_ij_2; ++j) {
      s_ij[i - 1 + dim_s_ij_1 * (j - 1)] = m[dim_m_1 * (j - 1) + i - 1] * I[i - 1];
    }
  }
  for (int_t i = 1; i <= dim_R; ++i) {
    state_next[offset_variable_R + i - 1] = R[i - 1] + n_IR[i - 1];
  }
  for (int_t i = 1; i <= dim_lambda; ++i) {
    lambda[i - 1] = beta / (real_t) N * odin_sum2(s_ij, i - 1, i, 0, dim_s_ij_2, dim_s_ij_1);
  }
  for (int_t i = 1; i <= dim_p_SI; ++i) {
    p_SI[i - 1] = 1 - std::exp(- lambda[i - 1] * dt);
  }
  for (int_t i = 1; i <= dim_n_SI; ++i) {
    n_SI[i - 1] = dust::distr::rbinom(rng_state, std::round(S[i - 1]), p_SI[i - 1]);
  }
  for (int_t i = 1; i <= dim_I; ++i) {
    state_next[offset_variable_I + i - 1] = I[i - 1] + n_SI[i - 1] - n_IR[i - 1];
  }
  for (int_t i = 1; i <= dim_S; ++i) {
    state_next[1 + i - 1] = S[i - 1] - n_SI[i - 1];
  }
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

[[cpp11::register]]
SEXP dust_sir_alloc(cpp11::list r_data, size_t step, size_t n_particles,
                size_t n_threads, cpp11::sexp r_seed) {
  return dust_alloc<sir>(r_data, step, n_particles, n_threads, r_seed);
}

[[cpp11::register]]
SEXP dust_sir_run(SEXP ptr, size_t step_end) {
  return dust_run<sir>(ptr, step_end);
}

[[cpp11::register]]
SEXP dust_sir_run2(SEXP ptr, size_t step_end) {
  return dust_run2<sir>(ptr, step_end);
}

[[cpp11::register]]
SEXP dust_sir_set_index(SEXP ptr, cpp11::sexp r_index) {
  dust_set_index<sir>(ptr, r_index);
  return R_NilValue;
}

[[cpp11::register]]
SEXP dust_sir_set_state(SEXP ptr, SEXP r_state, SEXP r_step) {
  dust_set_state<sir>(ptr, r_state, r_step);
  return R_NilValue;
}

[[cpp11::register]]
SEXP dust_sir_reset(SEXP ptr, cpp11::list r_data, size_t step) {
  return dust_reset<sir>(ptr, r_data, step);
}

[[cpp11::register]]
SEXP dust_sir_state(SEXP ptr, SEXP r_index) {
  return dust_state<sir>(ptr, r_index);
}

[[cpp11::register]]
size_t dust_sir_step(SEXP ptr) {
  return dust_step<sir>(ptr);
}

[[cpp11::register]]
void dust_sir_reorder(SEXP ptr, cpp11::sexp r_index) {
  return dust_reorder<sir>(ptr, r_index);
}

[[cpp11::register]]
SEXP dust_sir_rng_state(SEXP ptr, bool first_only) {
  return dust_rng_state<sir>(ptr, first_only);
}

[[cpp11::register]]
SEXP dust_sir_set_rng_state(SEXP ptr, cpp11::raws rng_state) {
  dust_set_rng_state<sir>(ptr, rng_state);
  return R_NilValue;
}

[[cpp11::register]]
SEXP dust_sir_simulate(cpp11::sexp r_steps,
                            cpp11::list r_data,
                            cpp11::doubles_matrix r_state,
                            cpp11::sexp r_index,
                            const size_t n_threads,
                            cpp11::sexp r_seed) {
  return dust_simulate<sir>(r_steps, r_data, r_state, r_index,
                                 n_threads, r_seed);
}

[[cpp11::register]]
bool dust_sir_has_openmp() {
#ifdef _OPENMP
  return true;
#else
  return false;
#endif
}

// these ones we're adding
[[cpp11::register]]
size_t dust_sir_size_internal_real(SEXP ptr) {
  return dust_size_internal_real<sir>(ptr);
}

[[cpp11::register]]
size_t dust_sir_size_internal_int(SEXP ptr) {
  return dust_size_internal_int<sir>(ptr);
}

[[cpp11::register]]
std::vector<double> dust_sir_internal_real(SEXP ptr) {
  return dust_internal_real<sir>(ptr);
}

[[cpp11::register]]
std::vector<int> dust_sir_internal_int(SEXP ptr) {
  return dust_internal_int<sir>(ptr);
}
