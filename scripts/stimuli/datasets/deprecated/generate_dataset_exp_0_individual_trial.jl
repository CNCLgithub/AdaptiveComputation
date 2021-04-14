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
spring = 0.00114
sigma_w = 1.85714

measurement_noise = 1E-11
num_rejuv = 0


h5open("/datasets/exp_0.h5", "r+") do file
	scene_idx = 97
        params = Params(inertia,
			spring,
			sigma_w,
			sigma_x,
			sigma_v,
			measurement_noise,
			num_targets,
			num_observations,
			num_rejuv)
	(obs, avg_vel, gt_dots, init_dots) = collect_observations(T, params)
	# o_delete(file, "dataset/$scene_idx")
	scene = g_create(file, "/dataset/$scene_idx")
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

end
