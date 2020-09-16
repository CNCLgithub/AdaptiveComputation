using MOT
using HDF5

function generate_dataset_loc_error(num_trials)

	T = 120

	num_observations = 10
	num_targets = 3

	inertia = 0.8
	sigma_x = 120.5
	sigma_v = 2.2
	sigma_w = 2.0
	spring = 0.0015
	measurement_noise = 2.0
	num_rejuv = 5

	params = Params(inertia, spring, sigma_w,
					sigma_x, sigma_v, measurement_noise,
					num_targets, num_observations, num_rejuv)


	h5open("datasets/loc_error_dataset.h5", "w") do file
		dataset = g_create(file, "dataset")
		dataset["num_trials"] = num_trials
		for i=1:num_trials
			trial = g_create(dataset, string(i))
			(obs, avg_vel, gt_dots, init_dots) = collect_observations(T, params)
			trial["obs"] = obs
			trial["avg_vel"] = avg_vel
			trial["gt_dots"] = gt_dots
			trial["init_dots"] = init_dots
		end

		# saving parameters as well
		parameters = g_create(dataset, "parameters")

		parameters["inertia"] = inertia
		parameters["spring"] = spring
		parameters["sigma_w"] = sigma_w
		parameters["sigma_x"] = sigma_x
		parameters["sigma_v"] = sigma_v
		parameters["measurement_noise"] = measurement_noise
		parameters["num_targets"] = num_targets
		parameters["num_observations"] = num_observations
		parameters["num_rejuv"] = num_rejuv
	end
end

num_trials = 400
generate_dataset_loc_error(num_trials)
