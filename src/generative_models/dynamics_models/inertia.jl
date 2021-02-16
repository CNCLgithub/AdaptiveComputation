
@with_kw struct InertiaModel <: AbstractDynamicsModel
    vel::Float64 = 10 # base vel
    low_w::Float64 = 0.05
    high_w::Float64 = 3.5
    a::Float64 = 0.1
    b::Float64 = 0.3
end

function load(::Type{InertiaModel}, path::String)
    InertiaModel(;read_json(path)...)
end

@gen function inertial_step(model::InertiaModel, dot::Dot)
    _x, _y, z = dot.pos
    _vx, _vy = dot.vel
    _ax, _ay = dot.acc

    acc = @trace(beta(model.a, model.b), :acc)

    vel_sd = min(model.low_w/acc, model.high_w)
    vx = @trace(normal(acc * _vx, vel_sd), :vx)
    vy = @trace(normal(acc * _vy, vel_sd), :vy)

    vel = [vx, vy] .+ 1e-10
    vel *= model.vel/norm(vel)
    vel = @trace(broadcasted_normal(vel, model.low_w), :v)

    # vel_sd = min(model.low_w/acc, model.high_w)
    # vx = @trace(normal(acc * _vx, vel_sd), :vx)
    # vy = @trace(normal(acc * _vy, vel_sd), :vy)
    # vel = [vx, vy]

    x = _x + vel[1]
    y = _y + vel[2]
    z = @trace(uniform(0, 1), :z)

    d = Dot(pos = [x,y,z], vel = vel, acc = [acc, acc])
    return d
end

_inertial_step = Map(inertial_step)

@gen function inertial_update(model::InertiaModel, cg::CausalGraph)
    dots = cg.elements
    new_dots = @trace(_inertial_step(fill(model, length(dots)), dots), :brownian)
    new_dots = collect(Object, new_dots)
    cg = update(cg, new_dots)
    return cg
end

export InertiaModel
