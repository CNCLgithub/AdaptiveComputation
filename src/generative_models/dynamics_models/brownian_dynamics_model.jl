export BrownianDynamicsModel

@with_kw struct BrownianDynamicsModel <: AbstractDynamicsModel
    inertia::Float64 = 0.8
    spring::Float64 = 0.002
    sigma_x::Float64 = 1.5
    sigma_y::Float64 = 1.5
end

function load(::Type{BrownianDynamicsModel}, path::String)
    BrownianDynamicsModel(;read_json(path)...)
end

@gen function brownian_step(model::BrownianDynamicsModel, dot::Dot)
    _x,_y,_ = dot.pos
    _vx,_vy = dot.vel

    vx = @trace(normal(model.inertia * _vx - model.spring * _x,
                               model.sigma_x), :vx)
    vy = @trace(normal(model.inertia * _vy - model.spring * _y,
                               model.sigma_y), :vy)

    z = @trace(uniform(0, 1), :z)
    x = _x + vx
    y = _y + vy
    d = Dot([x,y,z], [vx,vy])
    return d
end

_brownian_step = Map(brownian_step)

@gen function brownian_update(model::BrownianDynamicsModel, cg::CausalGraph)
    dots = cg.elements
    new_dots = @trace(_brownian_step(fill(model, length(dots)), dots), :brownian)
    new_dots = collect(Dot, new_dots)
    cg = update(cg, new_dots)
    return cg
end
