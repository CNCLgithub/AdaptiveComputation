abstract type AbstractGMParams end

# The most basic generative model parameters
@with_kw struct GMParams <: AbstractGMParams
    n_trackers::Int = 4
    distractor_rate::Real = 4.0
    init_pos_spread::Real = 300.0
    area_height::Float64 = 800.0
    area_width::Float64 = 800.0
    dot_radius::Float64 = 20.0
    targets::Vector{Bool} = zeros(8)
end

function load(::Type{GMParams}, path; kwargs...)
    GMParams(;read_json(path)..., kwargs...)
end


# Polygon/hierarchical version of GMParams
@with_kw struct HGMParams <: AbstractGMParams
    n_trackers::Int64 = 4
    distractor_rate::Float64 = 4.0
    init_pos_spread::Float64 = 320.0
    dist_pol_verts::Float64 = 100.0
    max_vertices::Int64 = 7
    dot_radius::Real = 20.0
    area_height::Int64 = 800
    area_width::Int64 = 800
    targets::Vector{Bool} = zeros(8)
end

function load(::Type{HGMParams}, path; kwargs...)
    HGMParams(;read_json(path)..., kwargs...)
end

export GMParams, HGMParams
