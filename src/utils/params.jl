export Params,
    default_params

@with_kw mutable struct Params
    inertia::Float64 = 0.8
    spring::Float64 = 0.002
    sigma_w::Float64 = 1.5
    sigma_v::Float64 = 5.0

    num_trackers::Int = 1
    num_distractors_rate::Float64 = 1.0
    
    rejuv_smoothness::Float64 = 1.03 # lower = smoother
    max_rejuv::Int = 15

    area_width::Int = 800
    area_height::Int = 800

    img_width::Int = 200
    img_height::Int = 200
    dot_radius::Float64 = 20.0

    attended_trackers::Vector{Vector{Int}} = [] # fake param to store attention info
end

const = default_params = Params()
