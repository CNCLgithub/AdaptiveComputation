

@with_kw struct HGMParams <: AbstractGMParams
    n_trackers::Int64 = 4
    distractor_rate::Float64 = 4.0
    init_pos_spread::Float64 = 320.0
    polygon_radius::Float64 = 130.0
    max_vertices::Int64 = 7
    
    # graphics parameters
    dot_radius::Real = 20.0
    img_height::Int = 200
    img_width::Int = 200
    area_height::Int = 2.5 * init_pos_spread
    area_width::Int = 2.5 * init_pos_spread

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
