@with_kw struct GMParams
    n_trackers::Int = 4
    distractor_rate::Real = 4.0
    init_pos_spread::Real = 300.0

    # graphics parameters
    dot_radius::Real = 20.0
    area_height::Int = 800
    area_width::Int = 800
end

function load(::Type{GMParams}, path; kwargs...)
    GMParams(;read_json(path)..., kwargs...)
end

function load(::Type{GMParams}, path::String)
    GMParams(;read_json(path)...)
end

const default_gm_params = GMParams()

