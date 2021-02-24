// These only exist so that cpp11 finds them as it can't look within
// .cu files

#include <cpp11/list.hpp>

[[cpp11::register]]
SEXP dust_sirs_alloc(cpp11::list r_pars, bool pars_multi, size_t step,
                         size_t n_particles, size_t n_threads,
                         cpp11::sexp r_seed) {
  return dust_alloc<sirs>(r_pars, pars_multi, step, n_particles,
                               n_threads, r_seed);
}

[[cpp11::register]]
SEXP dust_sirs_run(SEXP ptr, size_t step_end, bool device) {
  return dust_run<sirs>(ptr, step_end, device);
}

[[cpp11::register]]
SEXP dust_sirs_simulate(SEXP ptr, cpp11::sexp step_end) {
  return dust_simulate<sirs>(ptr, step_end);
}

[[cpp11::register]]
SEXP dust_sirs_set_index(SEXP ptr, cpp11::sexp r_index) {
  dust_set_index<sirs>(ptr, r_index);
  return R_NilValue;
}

[[cpp11::register]]
SEXP dust_sirs_set_state(SEXP ptr, SEXP r_state, SEXP r_step) {
  dust_set_state<sirs>(ptr, r_state, r_step);
  return R_NilValue;
}

[[cpp11::register]]
SEXP dust_sirs_reset(SEXP ptr, cpp11::list r_pars, size_t step) {
  return dust_reset<sirs>(ptr, r_pars, step);
}

[[cpp11::register]]
SEXP dust_sirs_state(SEXP ptr, SEXP r_index) {
  return dust_state<sirs>(ptr, r_index);
}

[[cpp11::register]]
size_t dust_sirs_step(SEXP ptr) {
  return dust_step<sirs>(ptr);
}

[[cpp11::register]]
void dust_sirs_reorder(SEXP ptr, cpp11::sexp r_index) {
  return dust_reorder<sirs>(ptr, r_index);
}

[[cpp11::register]]
SEXP dust_sirs_resample(SEXP ptr, cpp11::doubles r_weights) {
  return dust_resample<sirs>(ptr, r_weights);
}

[[cpp11::register]]
SEXP dust_sirs_set_pars(SEXP ptr, cpp11::list r_pars) {
  return dust_set_pars<sirs>(ptr, r_pars);
}

[[cpp11::register]]
SEXP dust_sirs_rng_state(SEXP ptr, bool first_only) {
  return dust_rng_state<sirs>(ptr, first_only);
}

[[cpp11::register]]
SEXP dust_sirs_set_rng_state(SEXP ptr, cpp11::raws rng_state) {
  dust_set_rng_state<sirs>(ptr, rng_state);
  return R_NilValue;
}

[[cpp11::register]]
SEXP dust_sirs_set_data(SEXP ptr, cpp11::list data) {
  dust_set_data<sirs>(ptr, data);
  return R_NilValue;
}

[[cpp11::register]]
SEXP dust_sirs_compare_data(SEXP ptr) {
  return dust_compare_data<sirs>(ptr);
}

[[cpp11::register]]
SEXP dust_sirs_filter(SEXP ptr, bool save_history) {
  return dust_filter<sirs>(ptr, save_history);
}

[[cpp11::register]]
cpp11::sexp dust_sirs_capabilities() {
  return dust_capabilities<sirs>();
}

[[cpp11::register]]
void dust_sirs_set_n_threads(SEXP ptr, int n_threads) {
  return dust_set_n_threads<sirs>(ptr, n_threads);
}

[[cpp11::register]]
int dust_sirs_n_state(SEXP ptr) {
  return dust_n_state<sirs>(ptr);
}