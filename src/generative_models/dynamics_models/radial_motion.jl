export RadialMotion

@with_kw struct RadialMotion <: AbstractDynamicsModel
    inertia::Float64 = 0.8
    sigma_mag::Float64 = 0.5
    rim::Float64 = 100
    eps::Float64 = 1E-5
end

function load(::Type{RadialMotion}, path::String)
    RadialMotion(;read_json(path)...)
end

@gen function radial_step(model::RadialMotion, dot::Dot)
    x,y,z = dot.pos
    _vx,_vy = dot.vel

    mag = norm(dot.vel)
    new_mag = @trace(gamma(mag, model.sigma_mag), :mag)

    θ = atan(y, x)
    bearing = norm([x,y]) > model.rim ? θ : θ + π
    κ = exp(1 - log(max(norm(mag - model.rim), model.eps)))
    new_θ = @trace(von_mises(bearing, κ), :bearing)
    vx = cos(new_θ) * new_mag
    vy = sin(new_θ) * new_mag
    d = Dot([x+vx,y+vy,z], [vx,vy])
    return d
end

_radial_step = Map(radial_step)

@gen function radial_update(model::RadialMotion, cg::CausalGraph)
    dots = cg.elements
    new_dots = @trace(_radial_step(fill(model, length(dots)), dots), :radial)
    new_dots = collect(Dot, new_dots)
    cg = update(cg, new_dots)
    return cg
end
