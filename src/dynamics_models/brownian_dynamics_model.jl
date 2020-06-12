export BrownianDynamicsModel,
       update
        
struct BrownianDynamicsModel <: DynamicsModel
    inertia::Float64
    spring::Float64
    sigma_w::Float64
end

@gen function update_individual(dot::Dot, model::BrownianDynamicsModel)
    _x,_y,z = dot.pos
    _vx,_vy = dot.vel

    vx = @trace(normal(model.inertia * _vx - model.spring * _x,
                               model.sigma_w), :vx)
    vy = @trace(normal(model.inertia * _vy - model.spring * _y,
                               model.sigma_w), :vy)
    x = _x + _vx
    y = _y + _vy
    Dot([x,y,z], [vx,vy])
end

_update_map = Map(update_individual)

@gen function update(dots::Vector{Dot}, model::BrownianDynamicsModel)
    _update_map(dots, fill(model, length(dots)))
end

