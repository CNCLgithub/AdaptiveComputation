export HGMDynamicsModel

@with_kw struct HGMDynamicsModel <: AbstractDynamicsModel
    inertia::Float64 = 0.8
    spring::Float64 = 0.001
    sigma_x::Float64 = 1.5
    sigma_y::Float64 = 1.5
    
    pol_inertia::Float64 = 0.8
    pol_spring::Float64 = 0.001
    pol_sigma_x::Float64 = 0.5
    pol_sigma_y::Float64 = 0.5
end

function load(::Type{HGMDynamicsModel}, path::String)
    HGMDynamicsModel(;read_json(path)...)
end

# TODO implement polygon collapse (and formation?)
@gen function hierarchical_brownian_step(model::HGMDynamicsModel, object::Object)

    _x, _y, _z = object.pos
    _vx, _vy = object.vel

    vx = @trace(normal(model.inertia * _vx - model.spring * _x,
                               model.sigma_x), :vx)
    vy = @trace(normal(model.inertia * _vy - model.spring * _y,
                               model.sigma_y), :vy)
    x = _x + vx
    y = _y + vy

    z = @trace(uniform(0, 1), :z)

    if isa(object, Dot)
        return Dot([x,y,z], [vx,vy])

    elseif isa(object, Polygon)
        dots = Vector{Dot}(undef, length(object.dots))

        for i=1:length(dots)
            _dot_x, _dot_y = object.dots[i].pos
            _dot_vx, _dot_vy = object.dots[i].vel
            
            # finding the center position defined by the polygon
            r = object.radius
            c_dot_x = x + r * cos(2*pi*i/length(dots))
            c_dot_y = y + r * sin(2*pi*i/length(dots))

            dot_vx = @trace(normal(model.pol_inertia * _dot_vx - model.pol_spring * (_dot_x - c_dot_x),
                                       model.pol_sigma_x), i => :vx)
            dot_vy = @trace(normal(model.pol_inertia * _dot_vy - model.pol_spring * (_dot_y - c_dot_y),
                                       model.pol_sigma_y), i => :vy)
            dot_x = _dot_x + dot_vx + vx
            dot_y = _dot_y + dot_vy + vy

            dots[i] = Dot([dot_x, dot_y, z], [dot_vx, dot_vy])
        end

        return Polygon([x,y,z], [vx,vy], object.radius, dots)
    end
end

_hierarchical_brownian_step = Map(hierarchical_brownian_step)

@gen function hgm_update(model::HGMDynamicsModel, cg::CausalGraph)
    objects = cg.elements
    new_objects = @trace(_hierarchical_brownian_step(fill(model, length(objects)), objects), :brownian)
    new_objects = collect(Object, new_objects)
    cg = update(cg, new_objects)
    return cg
end
