export BrownianDynamicsModel

@with_kw struct BrownianDynamicsModel <: AbstractDynamicsModel
    inertia::Float64 = 0.8
    spring::Float64 = 0.002
    sigma_w::Float64 = 1.5
end

function load(Type{BrownianDynamicsModel}, path::String)
    BrownianDynamicsModel(;read_json(path)...)
end

@gen function step(model::BrownianDynamicsModel, dot::Dot)
    _x,_y,z = dot.pos
    _vx,_vy = dot.vel

    vx = @trace(normal(model.inertia * _vx - model.spring * _x,
                               model.sigma_w), :vx)
    vy = @trace(normal(model.inertia * _vy - model.spring * _y,
                               model.sigma_w), :vy)
    x = _x + _vx
    y = _y + _vy
    d = Dot([x,y,z], [vx,vy])
    return d
end

_step = Map(step)

@gen function update(model::BrownianDynamicsModel, cg::CausalGraph)
    dots = cg.elements
    graph = cg.graph
    new_dots = @trace(_step(fill(model, length(dots)), dots), :brownian)
    cg = CausalGraph(new_dots, graph)
    return cg
end
