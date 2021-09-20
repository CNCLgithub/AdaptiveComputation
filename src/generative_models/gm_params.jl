export GMParams

# The most basic generative model parameters
@with_kw struct GMParams <: AbstractGMParams
    max_things::Int64 = 8
    death_rate::Float64 = 0.1
    n_targets::Int64 = 4
    init_pos_spread::Real = 300.0
    area_height::Float64 = 800.0
    area_width::Float64 = 800.0
    dot_radius::Float64 = 20.0
    n_trackers::Int = 4 # TODO: Depricate
    distractor_rate::Real = 4.0 # TODO: Depricate
    targets::Vector{Bool} = zeros(8) # TODO: Depricate
end

function load(::Type{GMParams}, path; kwargs...)
    GMParams(;read_json(path)..., kwargs...)
end

function tracker_bounds(gm::GMParams)
    @unpack area_width, area_height, dot_radius = gm
    xs = (-0.5*area_width + dot_radius, 0.5*area_width - dot_radius)
    ys = (-0.5*area_height + dot_radius, 0.5*area_height - dot_radius)
    (xs, ys, dot_radius)
end
