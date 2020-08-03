export ConstrainedBDM

@with_kw struct ConstrainedBDM <: AbstractDynamicsModel
    inertia::Float64 = 0.8
    spring::Float64 = 0.002
    sigma_x::Float64 = 1.5
    sigma_y::Float64 = 1.5
end

function load(::Type{ConstrainedBDM}, path::String)
    ConstrainedBDM(;read_json(path)...)
end

@gen function brownian_step(model::ConstrainedBDM, dot::Dot)
    _x, _y, _z = dot.pos
    _vx, _vy = dot.vel

    vx = @trace(normal(model.inertia * _vx - model.spring * _x,
                               model.sigma_x), :vx)
    vy = @trace(normal(model.inertia * _vy - model.spring * _y,
                               model.sigma_y), :vy)

    x = _x + vx
    y = _y + vy
    z = @trace(uniform(0, 1), :z)

    d = Dot([x,y,z], [vx,vy])
    return d
end

_cbm_step = Map(brownian_step)


function resolve!(dots, tries::Int64)
    tries == 0 && return nothing
    n = length(dots)
    distances = Matrix{Float64}(undef, dots, dots)
    pairs = product(dots, dots)
    for i = 1:n, j = 1:n
        dr = 0.5 * (dots[i].radius + dots[j].radius[j])
        distances[i,j] = i==j ? false : norm(dots[i].pos - dots[j].pos) <= dr
    end
    sum(distances) == 0 && return nothing
    for i = 1:n
        collided = findall(pairs[:, i])
        isempty(collided) && continue
        vels = mean(vcat(map(x -> dots[x].vel, collided)...), dims = 1)
    end

end


@gen function cbm_update(model::ConstrainedBDM, cg::CausalGraph)
    dots = cg.elements
    new_dots = @trace(_cbm_step(fill(model, length(dots)), dots), :brownian)
    new_dots = collect(Dot, new_dots)
    cg = update(cg, new_dots)
    return cg
end
