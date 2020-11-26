data(polymod, package = "socialmixr")

age.limits = seq(0, 70, 10)

contact <- socialmixr::contact_matrix(
  survey = polymod,
  countries = "United Kingdom",
  age.limits = age.limits,
  symmetric = TRUE)

transmission <- contact$matrix /
  rep(contact$demography$population, each = ncol(contact$matrix))


# RECOMPILE

pkgload::load_all()

# CREATE MODEL

N_age <- length(age.limits)
n_particles <- 10L
dt <- 0.25
sir_model <- dustgpu::sir$new(data = list(dt = dt,
                                 S_ini = 6E7,
                                 I_ini = 10,
                                 beta = 0.2,
                                 gamma = 0.1,
                                 m = transmission,
                                 N_age = N_age),
                              step = 1,
                              n_particles = n_particles,
                              n_threads = 1L,
                              seed = 1L)

# RUN
sir_model$run_device(10)
