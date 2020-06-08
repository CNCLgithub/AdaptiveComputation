num_observations = 6
num_particles = 5
T = 100
num_targets = 3

inertia = 0.8
sigma_x = 120.5
sigma_v = 2.2
sigma_w = 1.8
spring = 0.004
measurement_noise = 2.0
obs_noise = 1e-2

params = Params(inertia, spring, sigma_w, sigma_x, sigma_v,
				obs_noise, num_targets, num_observations)
(obs, avg_vel, dots) = collect_observations(T, params)
