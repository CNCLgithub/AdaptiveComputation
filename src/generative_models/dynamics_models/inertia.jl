
@with_kw struct InertiaModel <: AbstractDynamicsModel
    sigma_inertia::Float64 = 3.5
    acc_w::Float64 = 3.5
    v_w::Float64 = 0.05
end

function load(::Type{InertiaModel}, path::String)
    InertiaModel(;read_json(path)...)
end

@gen function inertial_step(model::InertiaModel, dot::Dot)
    _x, _y, z = dot.pos
    _vx, _vy = dot.vel
    _ax, _ay = dot.acc

    a = @trace(beta(0.9, 0.9), :acc)
    # ax = @trace(uniform(0.1, 0.9), :ax)
    # ay = @trace(uniform(0.1, 0.9), :ay)

    vel_sd = min(model.v_w / (a+1E-3), model.sigma_inertia)
    vx = @trace(normal(a * _vx, vel_sd), :vx)
    vy = @trace(normal(a * _vy, vel_sd), :vy)

    x = _x + vx
    y = _y + vy
    z = @trace(uniform(0, 1), :z)

    d = Dot(pos = [x,y,z], vel = [vx,vy], acc = [a, a])
    return d
end

_inertial_step = Map(inertial_step)

@gen function inertial_update(model::InertiaModel, cg::CausalGraph)
    dots = cg.elements
    new_dots = @trace(_inertial_step(fill(model, length(dots)), dots), :brownian)
    new_dots = collect(Dot, new_dots)
    cg = update(cg, new_dots)
    return cg
end

export InertiaModel
