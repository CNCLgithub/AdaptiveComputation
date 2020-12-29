export HGMDynamicsModel

@with_kw struct HGMDynamicsModel <: AbstractDynamicsModel
    inertia::Float64 = 0.8
    spring::Float64 = 0.001
    sigma_x::Float64 = 0.5
    sigma_y::Float64 = 0.5
    
    # brownian motion on a spring with regards to polygon positions
    pol_inertia::Float64 = 0.8
    pol_spring::Float64 = 0.03
    pol_sigma_x::Float64 = 0.01
    pol_sigma_y::Float64 = 0.01

    # repulsion
    dot_repulsion::Float64 = 80.0
    wall_repulsion::Float64 = 50.0
    distance::Float64 = 60.0
    vel::Float64 = 10.0 # base velocity
    rep_inertia::Float64 = 0.9
end

function load(::Type{HGMDynamicsModel}, path::String)
    HGMDynamicsModel(;read_json(path)...)
end

@gen function hgm_brownian_step(model::HGMDynamicsModel, object::Object)

    _x, _y, _z = object.pos
    _vx, _vy = object.vel

    vx = @trace(normal(model.inertia * _vx - model.spring * _x,
                               model.sigma_x), :vx)
    vy = @trace(normal(model.inertia * _vy - model.spring * _y,
                               model.sigma_y), :vy)
    x = _x + vx
    y = _y + vy

    #z = @trace(uniform(0, 1), :z)
    z = _z

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

_hgm_brownian_step = Map(hgm_brownian_step)


function get_new_dot(index, model, rep_forces, dot)
    vel = dot.vel

    if sum(vel) != 0
        vel *= model.vel/norm(vel)
    end
    vel *= model.rep_inertia
    vel += (1.0-model.rep_inertia)*(rep_forces[index])
    index += 1
    return Dot(dot.pos, vel), index
end


function hgm_repulsion_step(model, objects, hgm)
    dots = []
    for object in objects
        if isa(object, Polygon)
            dots = [dots; object.dots]
        elseif isa(object, Dot)
            push!(dots, object)
        end
    end

    rep_forces = get_repulsion_force(model, dots, hgm)
    println(rep_forces)
    index = 1
    
    for i=1:length(objects)
        if objects[i] isa Polygon
            for j=1:length(objects[i].dots)
                objects[i].dots[j], index = get_new_dot(index, model, rep_forces, objects[i].dots[j])
            end
        elseif objects[i] isa Dot
            objects[i], index = get_new_dot(index, model, rep_forces, objects[i])
        end
    end

    return objects
end

@gen function hgm_update(model::HGMDynamicsModel, cg::CausalGraph, hgm)
    objects = cg.elements
    objects = hgm_repulsion_step(model, objects, hgm)
    objects = @trace(_hgm_brownian_step(fill(model, length(objects)), objects), :brownian)
    objects = collect(Object, objects)
    cg = update(cg, objects)
    return cg
end
