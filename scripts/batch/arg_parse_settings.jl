using ArgParse

s = ArgParseSettings()
@add_arg_table s begin
    "combination"
        arg_type = Int
        help = "the index of the combination"
        required = true
    "job"
        arg_type = Int
        required = true
    "runs_per_job"
        arg_type = Int
        help = "how many runs to do in one job"
        required = true
    "timesteps"
        arg_type = Int
        required = true
    "num_observations"
        arg_type = Int
        required = true
    "num_targets"
        arg_type = Int
        required = true
    "num_particles"
        arg_type = Int
        required = true
    "inertia"
        arg_type = Float64
        required = true
    "spring"
        arg_type = Float64
        required = true
    "sigma_w"
        arg_type = Float64
        required = true
    "sigma_x"
        arg_type = Float64
        required = true
    "sigma_v"
        arg_type = Float64
        required = true
    "measurement_noise"
        arg_type = Float64
        required = true
    "num_rejuv"
        arg_type = Int
        required = true
end
