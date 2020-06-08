using Gen
using Gen_Compose
using MOT
using Random
using HDF5
Random.seed!(3)

function load_from_file(filename, trial)
	file = h5open(filename, "r")
	dataset = read(file, "dataset")
	data = dataset["$(trial-1)"]

	obs = data["obs"]
	avg_vel = data["avg_vel"]
	dots = data["gt_dots"]
	init_dots = data["init_dots"]
		
	inertia = data["inertia"]
	spring = data["spring"]
	sigma_w = data["sigma_w"]	
	sigma_x = data["sigma_x"]	
	sigma_v = data["sigma_v"]	

	# adding measurement noise to simulate the perception module
	stds = fill(2.0, size(obs))
	obs = broadcasted_normal(obs, stds)

	return obs, avg_vel, dots, init_dots, inertia, spring, sigma_w, sigma_x, sigma_v
end

# using ProfileView

Gen.load_generated_functions()

T = 120
num_particles = 10

num_observations = 8
num_targets = 4

measurement_noise = 2.0
rejuv_count = 10

"""
inertia = 0.8
sigma_x = 120.5
sigma_v = 2.2
sigma_w = 1.8
spring = 0.004
measurement_noise = 2.0
obs_noise = 1e-2
rejuv_count = 1

params = Params(inertia, spring, sigma_w, sigma_x, sigma_v,
			obs_noise, num_targets, num_observations, rejuv_count)
(obs, avg_vel, dots, init_dots) = collect_observations(T, params)
"""

dataset = "datasets/exp_0.h5"
(obs, avg_vel, gt_dots, init_dots, inertia, spring, sigma_w, sigma_x, sigma_v) = load_from_file(dataset, 127)
params = Params(inertia, spring, sigma_w, sigma_x, sigma_v,
			measurement_noise, num_targets, num_observations, rejuv_count)


latents = Dict( :x => x -> :x )


init_obs = Gen.choicemap()
for i=1:params.num_targets
	init_obs[:initial_state => i => :x] = init_dots[i,1]
	init_obs[:initial_state => i => :y] = init_dots[i,2]
end

args = [(t, params) for t in 1:T]
observations = Vector{Gen.ChoiceMap}(undef, T)
for t = 1:T
	cm = Gen.choicemap()
	for i=1:num_observations
		cm[:chain => t => :points => i => :e] = obs[t, i,:]
	end
	observations[t] = cm
end

query = Gen_Compose.SequentialQuery(latents, #bogus for now
									generative_model,
									(0, params),
									init_obs,
									args,
									observations)

# confusability epsilon
epsilon = 1E-3

function rejuv!(state, stats)
		accepted = MOT.rejuvenate_assignment!(state, stats)
		stats = MOT.retrieve_confusability(state)
		accepted += rejuvenate_state!(state, stats)
		return 0.5 * accepted
end

procedure = PopParticleFilter(num_particles,
															num_particles/2, # ESS is in terms of effective particle count, not fraction
															npp_proposal,
															tuple(),
															rejuv!,
															MOT.retrieve_confusability,
															((n,o) -> sum(n - o) > -epsilon),
															10,
															3,
															true)

results = sequential_monte_carlo(procedure, query)
MOT.save_state(results, "test.jld2")
positions = retrieve_pf_positions("test.jld2")

#println(positions)
overlay(gt_dots[1:T,:,:], num_targets; pf_xy=positions, stimuli=false, highlighted=[1])
