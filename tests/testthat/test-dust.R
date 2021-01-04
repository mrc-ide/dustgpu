context("dust")

test_that("cpp11 setup is working", {
  expect_equal(cpp_add(1, 2), 3)
})

test_that("device run gives the same results", {
  skip_if(sir_model_d$has_cuda() == FALSE, "CUDA not supported")
  data(polymod, package = "socialmixr")

  age.limits = seq(0, 70, 10)

  contact <- socialmixr::contact_matrix(
    survey = polymod,
    countries = "United Kingdom",
    age.limits = age.limits,
    symmetric = TRUE)

  transmission <- contact$matrix

  N_age <- length(age.limits)
  S_ini = 1E3
  I_ini = 10
  beta = 0.3
  gamma = 0.1

  n_particles <- 10L
  dt <- 0.25
  n_steps <- 20L

  sir_model_d <- dustgpu::sir$new(data = list(dt = dt,
                                              S_ini = S_ini,
                                              I_ini = I_ini,
                                              beta = beta,
                                              gamma = gamma,
                                              m = transmission,
                                              N_age = N_age),
                                  step = 0,
                                  n_particles = n_particles,
                                  n_threads = 1L,
                                  seed = 1L)
  sir_model_h <- dustgpu::sir$new(data = list(dt = dt,
                                              S_ini = S_ini,
                                              I_ini = I_ini,
                                              beta = beta,
                                              gamma = gamma,
                                              m = transmission,
                                              N_age = N_age),
                                  step = 0,
                                  n_particles = n_particles,
                                  n_threads = 1L,
                                  seed = 1L)

  h_results <- sir_model_h$run(n_steps)
  d_results <- sir_model_d$run_device(n_steps)
  expect_identical(h_results, d_results)

  h_rng <- sir_model_h$rng_state()
  d_rng <- sir_model_d$rng_state()
  expect_identical(h_rng, d_rng)
})