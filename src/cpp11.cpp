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

extern "C" {
/* .Call calls */
extern SEXP _dustgpu_cpp_add(SEXP, SEXP);

static const R_CallMethodDef CallEntries[] = {
    {"_dustgpu_cpp_add", (DL_FUNC) &_dustgpu_cpp_add, 2},
    {NULL, NULL, 0}
};
}

extern "C" void R_init_dustgpu(DllInfo* dll){
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
