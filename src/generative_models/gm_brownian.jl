using LinearAlgebra

struct State
    graph::CausalGraph{Dot, SimpleGraph}
end

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


# ##### INIT STATE ######
@gen function sample_init_tracker(init_pos_spread::Real)::Dot
    x = @trace(uniform(-init_pos_spread, init_pos_spread), :x)
    y = @trace(uniform(-init_pos_spread, init_pos_spread), :y)
    vx = 0.0
    vy = 0.0

    # z (depth) drawn at beginning
    z = @trace(uniform(0, 1), :z)

    # initial velocity is zero
    return Dot([x,y,z], [vx, vy])
end

init_trackers_map = Gen.Map(sample_init_tracker)

@gen function sample_init_state(gm::GMParams)
    init_pos_spread_vec = fill(gm.init_pos_spread, gm.n_trackers)
    trackers = @trace(init_trackers_map(init_pos_spread_vec), :trackers)
    trackers = collect(Dot, trackers)

    # add each tracker to the graph as independent vertices
    graph = CausalGraph(trackers, SimpleGraph)

    State(graph)
end


@gen static function brownian_kernel(t::Int,
                            prev_state::State,
                            dynamics_model::BrownianDynamics,
                            params::GMParams)
    new_graph = @trace(brownian_update(dynamics_model, prev_state.graph), :dynamics)
    new_state = State(new_graph)
    return new_state
end

brownian_chain = Gen.Unfold(brownian_kernel)

# generative model for the brownian motion
@gen static function gm_brownian(T::Int, motion::BrownianDynamics,
                                 params::GMParams)
    init_state = @trace(sample_init_state(params), :init_state)
    states = @trace(brownian_chain(T, init_state, motion, params), :kernel)
    result = (init_state, states)
    return result
end

export GMParams, gm_brownian, default_gm_params
