using MOT
using Gen
Gen.load_generated_functions()

using Random
Random.seed!(3)

T = 120
num_targets = 3
inertia = 0.8
obs_noise = 1e-2
rejuv_count = 5
sigma_x = 120.5
sigma_v = 2.2
min_dots = 4
max_dots = 15

M = 100

for m=1:M
    println(m)
    spring = uniform(0.0005, 0.003)
    sigma_w = uniform(0.1, 2.3)
    num_observations = uniform_discrete(min_dots, max_dots)

    params = Params(inertia, spring, sigma_w, sigma_x, sigma_v,
		    obs_noise, num_targets, num_observations, rejuv_count)
    (obs, avg_vel, dots) = collect_observations(T, params)

    # we will use only the last or a random time step
    if uniform(0, 1) < 0.8
        obs = obs[T, :, :]
    else
        index = Int(floor(T * uniform(0, 1))) + 1
        obs = obs[index, :, :]
    end
    obs = reshape(obs, (1, num_observations, 2))
    overlay(obs, num_targets; pf_xy=nothing, stimuli=true, freeze_time=0, render_masks=m)
end
