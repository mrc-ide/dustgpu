data(polymod, package = "socialmixr")

age.limits = seq(0, 70, 10)

contact <- socialmixr::contact_matrix(
  survey = polymod,
  countries = "United Kingdom",
  age.limits = age.limits,
  symmetric = TRUE)

transmission <- contact$matrix

N_age <- length(age.limits)
n_particles <- 100000L
dt <- 0.25
n_steps <- 1L
steps_per_obs <- 20L

pkgload::load_all()

# CREATE MODEL

sir_model <- dustgpu::sir$new(data = list(dt = dt,
                                 S_ini = 1E3,
                                 I_ini = 10,
                                 beta = 0.3,
                                 gamma = 0.1,
                                 m = transmission,
                                 N_age = N_age),
                              step = 0,
                                  n_particles = n_particles,
                                  n_threads = 1L,
                                  seed = 1L)
sir_model$set_index(c(1L, 2L, 10L, 18L))
# RUN
for (obs_step in 1:n_steps) {
    results <- sir_model$run_device(obs_step * steps_per_obs)
    order <- sample.int(n_particles, replace = T)
    sir_model$reorder(order)
}
