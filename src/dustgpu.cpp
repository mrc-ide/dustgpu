#include <cpp11/doubles.hpp>

[[cpp11::register]]
double cpp_add(double a, double b) {
  return a + b;
}
