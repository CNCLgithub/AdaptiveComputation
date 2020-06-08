using Base.Filesystem
using Base.Iterators
using Base.Threads

using MOT
using HDF5

num_targets = 4
num_observations = 8
num_steps = 64
num_reps = 2
num_trials = num_steps * num_reps * num_observations
T = 120

inertia = 0.8
sigma_v = 2.2

sigma_x = 120.5
l = Int(sqrt(num_steps))
springs = LinRange(0.0005, 0.002, l)
sigma_ws = LinRange(1.0, 2.5, l)
vars = product(springs, sigma_ws)

measurement_noise = 1E-11
num_rejuv = 0


h5open("/datasets/exp_0.h5", "w") do file
	dataset = g_create(file, "dataset")
	dataset["num_trials"] = num_trials
	scene_idx = 0
	for (spring, sigma_w) in vars, j in 1:num_reps
		params = Params(inertia,
						spring,
						sigma_w,
						sigma_x,
						sigma_v,
						measurement_noise,
						num_targets,
						num_observations,
						num_rejuv)
		scene = g_create(dataset, "$scene_idx")
		(obs, avg_vel, gt_dots, init_dots) = collect_observations(T, params)
		scene["obs"] = obs
		scene["avg_vel"] = avg_vel
		scene["gt_dots"] = gt_dots
		scene["init_dots"] = init_dots
		# saving parameters as well
		scene["inertia"] = inertia
		scene["spring"] = spring
		scene["sigma_w"] = sigma_w
		scene["sigma_x"] = sigma_x
		scene["sigma_v"] = sigma_v
		scene["num_targets"] = num_targets
		scene["num_observations"] = num_observations

		scene_idx += 1
	end
end
