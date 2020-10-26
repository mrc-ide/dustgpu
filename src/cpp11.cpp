// Generated by cpp11: do not edit by hand
// clang-format off


#include "cpp11/declarations.hpp"

// dustgpu.cpp
double cpp_add(double a, double b);
extern "C" SEXP _dustgpu_cpp_add(SEXP a, SEXP b) {
  BEGIN_CPP11
    return cpp11::as_sexp(cpp_add(cpp11::as_cpp<cpp11::decay_t<double>>(a), cpp11::as_cpp<cpp11::decay_t<double>>(b)));
  END_CPP11
}
// sir.cpp
SEXP dust_sir_alloc(cpp11::list r_data, size_t step, size_t n_particles, size_t n_threads, cpp11::sexp r_seed);
extern "C" SEXP _dustgpu_dust_sir_alloc(SEXP r_data, SEXP step, SEXP n_particles, SEXP n_threads, SEXP r_seed) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_sir_alloc(cpp11::as_cpp<cpp11::decay_t<cpp11::list>>(r_data), cpp11::as_cpp<cpp11::decay_t<size_t>>(step), cpp11::as_cpp<cpp11::decay_t<size_t>>(n_particles), cpp11::as_cpp<cpp11::decay_t<size_t>>(n_threads), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_seed)));
  END_CPP11
}
// sir.cpp
SEXP dust_sir_run(SEXP ptr, size_t step_end);
extern "C" SEXP _dustgpu_dust_sir_run(SEXP ptr, SEXP step_end) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_sir_run(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<size_t>>(step_end)));
  END_CPP11
}
// sir.cpp
SEXP dust_sir_set_index(SEXP ptr, cpp11::sexp r_index);
extern "C" SEXP _dustgpu_dust_sir_set_index(SEXP ptr, SEXP r_index) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_sir_set_index(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_index)));
  END_CPP11
}
// sir.cpp
SEXP dust_sir_set_state(SEXP ptr, SEXP r_state, SEXP r_step);
extern "C" SEXP _dustgpu_dust_sir_set_state(SEXP ptr, SEXP r_state, SEXP r_step) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_sir_set_state(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<SEXP>>(r_state), cpp11::as_cpp<cpp11::decay_t<SEXP>>(r_step)));
  END_CPP11
}
// sir.cpp
SEXP dust_sir_reset(SEXP ptr, cpp11::list r_data, size_t step);
extern "C" SEXP _dustgpu_dust_sir_reset(SEXP ptr, SEXP r_data, SEXP step) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_sir_reset(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<cpp11::list>>(r_data), cpp11::as_cpp<cpp11::decay_t<size_t>>(step)));
  END_CPP11
}
// sir.cpp
SEXP dust_sir_state(SEXP ptr, SEXP r_index);
extern "C" SEXP _dustgpu_dust_sir_state(SEXP ptr, SEXP r_index) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_sir_state(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<SEXP>>(r_index)));
  END_CPP11
}
// sir.cpp
size_t dust_sir_step(SEXP ptr);
extern "C" SEXP _dustgpu_dust_sir_step(SEXP ptr) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_sir_step(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr)));
  END_CPP11
}
// sir.cpp
void dust_sir_reorder(SEXP ptr, cpp11::sexp r_index);
extern "C" SEXP _dustgpu_dust_sir_reorder(SEXP ptr, SEXP r_index) {
  BEGIN_CPP11
    dust_sir_reorder(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_index));
    return R_NilValue;
  END_CPP11
}
// sir.cpp
SEXP dust_sir_rng_state(SEXP ptr, bool first_only);
extern "C" SEXP _dustgpu_dust_sir_rng_state(SEXP ptr, SEXP first_only) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_sir_rng_state(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<bool>>(first_only)));
  END_CPP11
}
// sir.cpp
SEXP dust_sir_set_rng_state(SEXP ptr, cpp11::raws rng_state);
extern "C" SEXP _dustgpu_dust_sir_set_rng_state(SEXP ptr, SEXP rng_state) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_sir_set_rng_state(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr), cpp11::as_cpp<cpp11::decay_t<cpp11::raws>>(rng_state)));
  END_CPP11
}
// sir.cpp
SEXP dust_sir_simulate(cpp11::sexp r_steps, cpp11::list r_data, cpp11::doubles_matrix r_state, cpp11::sexp r_index, const size_t n_threads, cpp11::sexp r_seed);
extern "C" SEXP _dustgpu_dust_sir_simulate(SEXP r_steps, SEXP r_data, SEXP r_state, SEXP r_index, SEXP n_threads, SEXP r_seed) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_sir_simulate(cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_steps), cpp11::as_cpp<cpp11::decay_t<cpp11::list>>(r_data), cpp11::as_cpp<cpp11::decay_t<cpp11::doubles_matrix>>(r_state), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_index), cpp11::as_cpp<cpp11::decay_t<const size_t>>(n_threads), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_seed)));
  END_CPP11
}
// sir.cpp
bool dust_sir_has_openmp();
extern "C" SEXP _dustgpu_dust_sir_has_openmp() {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_sir_has_openmp());
  END_CPP11
}
// sir.cpp
size_t dust_sir_size_internal_real(SEXP ptr);
extern "C" SEXP _dustgpu_dust_sir_size_internal_real(SEXP ptr) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_sir_size_internal_real(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr)));
  END_CPP11
}
// sir.cpp
size_t dust_sir_size_internal_int(SEXP ptr);
extern "C" SEXP _dustgpu_dust_sir_size_internal_int(SEXP ptr) {
  BEGIN_CPP11
    return cpp11::as_sexp(dust_sir_size_internal_int(cpp11::as_cpp<cpp11::decay_t<SEXP>>(ptr)));
  END_CPP11
}

extern "C" {
/* .Call calls */
extern SEXP _dustgpu_cpp_add(SEXP, SEXP);
extern SEXP _dustgpu_dust_sir_alloc(SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _dustgpu_dust_sir_has_openmp();
extern SEXP _dustgpu_dust_sir_reorder(SEXP, SEXP);
extern SEXP _dustgpu_dust_sir_reset(SEXP, SEXP, SEXP);
extern SEXP _dustgpu_dust_sir_rng_state(SEXP, SEXP);
extern SEXP _dustgpu_dust_sir_run(SEXP, SEXP);
extern SEXP _dustgpu_dust_sir_set_index(SEXP, SEXP);
extern SEXP _dustgpu_dust_sir_set_rng_state(SEXP, SEXP);
extern SEXP _dustgpu_dust_sir_set_state(SEXP, SEXP, SEXP);
extern SEXP _dustgpu_dust_sir_simulate(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _dustgpu_dust_sir_size_internal_int(SEXP);
extern SEXP _dustgpu_dust_sir_size_internal_real(SEXP);
extern SEXP _dustgpu_dust_sir_state(SEXP, SEXP);
extern SEXP _dustgpu_dust_sir_step(SEXP);

static const R_CallMethodDef CallEntries[] = {
    {"_dustgpu_cpp_add",                     (DL_FUNC) &_dustgpu_cpp_add,                     2},
    {"_dustgpu_dust_sir_alloc",              (DL_FUNC) &_dustgpu_dust_sir_alloc,              5},
    {"_dustgpu_dust_sir_has_openmp",         (DL_FUNC) &_dustgpu_dust_sir_has_openmp,         0},
    {"_dustgpu_dust_sir_reorder",            (DL_FUNC) &_dustgpu_dust_sir_reorder,            2},
    {"_dustgpu_dust_sir_reset",              (DL_FUNC) &_dustgpu_dust_sir_reset,              3},
    {"_dustgpu_dust_sir_rng_state",          (DL_FUNC) &_dustgpu_dust_sir_rng_state,          2},
    {"_dustgpu_dust_sir_run",                (DL_FUNC) &_dustgpu_dust_sir_run,                2},
    {"_dustgpu_dust_sir_set_index",          (DL_FUNC) &_dustgpu_dust_sir_set_index,          2},
    {"_dustgpu_dust_sir_set_rng_state",      (DL_FUNC) &_dustgpu_dust_sir_set_rng_state,      2},
    {"_dustgpu_dust_sir_set_state",          (DL_FUNC) &_dustgpu_dust_sir_set_state,          3},
    {"_dustgpu_dust_sir_simulate",           (DL_FUNC) &_dustgpu_dust_sir_simulate,           6},
    {"_dustgpu_dust_sir_size_internal_int",  (DL_FUNC) &_dustgpu_dust_sir_size_internal_int,  1},
    {"_dustgpu_dust_sir_size_internal_real", (DL_FUNC) &_dustgpu_dust_sir_size_internal_real, 1},
    {"_dustgpu_dust_sir_state",              (DL_FUNC) &_dustgpu_dust_sir_state,              2},
    {"_dustgpu_dust_sir_step",               (DL_FUNC) &_dustgpu_dust_sir_step,               1},
    {NULL, NULL, 0}
};
}

extern "C" void R_init_dustgpu(DllInfo* dll){
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
