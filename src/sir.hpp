// These only exist so that cpp11 finds them as it can't look within
// .cu files

#include <cpp11/list.hpp>

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
SEXP dust_sir_run_device(SEXP ptr, size_t step_end) {
  return dust_run_device<sir>(ptr, step_end);
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

[[cpp11::register]]
bool dust_sir_has_cuda() {
#ifdef __NVCC__
  return true;
#else
  return false;
#endif
}