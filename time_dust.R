data(polymod, package = "socialmixr")

age.limits = seq(0, 70, 10)

contact <- socialmixr::contact_matrix(
  survey = polymod,
  countries = "United Kingdom",
  age.limits = age.limits,
  symmetric = TRUE)

transmission <- contact$matrix

N_age <- length(age.limits)

pkgload::load_all()

model_time <- function(grid_val, gpu = TRUE) {
    np = as.integer(grid_val[1])
    dt = grid_val[2]
    sir_model <- dustgpu::sir$new(data = list(dt = dt,
                                 S_ini = 1E3,
                                 I_ini = 10,
                                 beta = 0.3,
                                 gamma = 0.1,
                                 m = transmission,
                                 N_age = N_age),
                              step = 0,
                                  n_particles = np,
                                  n_threads = 32L,
                                  seed = 1L)
    sir_model$set_index(c(1L, 2L, 10L, 18L))

    steps_per_obs <- as.integer(1 / dt)
    n_steps <- 2000L
    # RUN
    if (gpu) {
        model_run <- function() {
            for (obs_step in 1:n_steps) {
                results <- sir_model$run_device(obs_step * steps_per_obs)
                order <- sample.int(np, replace = T)
                sir_model$reorder(order)
            }
        }
    } else {
        model_run <- function() {
            for (obs_step in 1:n_steps) {
                results <- sir_model$run(obs_step * steps_per_obs)
                order <- sample.int(np, replace = T)
                sir_model$reorder(order)
            }
        }
    }
    system.time(model_run())['elapsed']
}

np = 32 * 2^(seq_len(12))
dt = c(0.1, 0.3, 1)
grid = data.frame(np = rep(np, length(dt)), dt = rep(dt, each = length(np)))
gpu_time <- apply(grid, 1, model_time)
cpu_time <- apply(grid, 1, model_time, gpu = FALSE)

gpu_times <- data.frame(grid, cpu = cpu_time, gpu = gpu_time, speedup = cpu_time / gpu_time)

library(reshape2)
library(ggplot2)
melt(gpu_times, id.vars = c("np"), measure.vars = c("speedup"))

ggplot() +
  geom_line(data = gpu_times, mapping = aes(x = np, y = speedup, colour = factor(dt))) +
  scale_x_continuous(trans='log2') +
  geom_hline(yintercept = 1, linetype="dashed") +
  theme_bw()