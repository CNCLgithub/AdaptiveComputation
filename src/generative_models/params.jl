

@with_kw struct HGMParams <: AbstractGMParams
    n_trackers::Int = 4
    distractor_rate::Real = 4.0
    init_pos_spread::Real = 300.0
    polygon_radius::Real = 130.0
    
    # graphics parameters
    dot_radius::Real = 20.0
    img_height::Int = 200
    img_width::Int = 200
    area_height::Int = 800
    area_width::Int = 800

    # parameters for the drawing the mask random variable arguments
    dot_p::Float64 = 0.5 # prob of pixel on in the dot region
    gauss_amp::Float64 = 0.5 # gaussian amplitude for the gaussian component of the mask
    gauss_std::Float64 = 2.5 # standard deviation --||--

    # flow masks
    fmasks::Bool = false
    fmasks_decay_function::Function = MOT.default_decay_function
    fmasks_n = 5

    # probes
    probe_flip::Float64 = 0.0

    targets::Vector{Bool} = zeros(8)
end

function load(::Type{HGMParams}, path; kwargs...)
    HGMParams(;read_json(path)..., kwargs...)
end

function load(::Type{HGMParams}, path::String)
    HGMParams(;read_json(path)...)
end

#const default_hgm = HGMParams()

export HGMParams
