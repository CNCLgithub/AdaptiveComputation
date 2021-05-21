export ConstrainedBDM

@with_kw struct ConstrainedBDM <: AbstractDynamicsModel
    #inertia::Float64 = 1.0
    #spring::Float64 = 0.000001
    #sigma_x::Float64 = 1.2
    #sigma_y::Float64 = 1.2
    bearing_k::Float64 = 8

    # collision detection radius
    distance::Float64 = 50.0
end

function load(::Type{ConstrainedBDM}, path::String)
    ConstrainedBDM(;read_json(path)...)
end

@gen function cbm_step(model::ConstrainedBDM, dot::BDot)
    _x, _y, _z = dot.pos

    # add some angle spice
    bearing = @trace(von_mises(dot.bearing, model.bearing_k), :b)
    bearing = (bearing - pi)%(2*pi) + pi
    
    x = _x + dot.vel*cos(bearing)
    y = _y + dot.vel*sin(bearing)

    d = BDot([x,y,_z], bearing, dot.vel)
    return d
end

_cbm_step = Map(cbm_step)

function resolve!(model, gm_params, dots, tries::Int64)
    tries == 0 && return nothing

    n = length(dots)
    collisions = Matrix{Bool}(undef, n, n)
    walls = Matrix{Bool}(undef, n, 4)

    for i = 1:n, j = 1:n
        d = norm(dots[i].pos - dots[j].pos)
        collisions[i,j] = i==j ? false : d  <= model.distance

        # NESW order
        walls[i,1] = gm_params.area_height/2 - gm_params.dot_radius - dots[i].pos[2] <= model.distance
        walls[i,2] = gm_params.area_width/2 - gm_params.dot_radius - dots[i].pos[1] <= model.distance
        walls[i,3] = gm_params.area_height/2 + gm_params.dot_radius + dots[i].pos[2] <= model.distance
        walls[i,4] = gm_params.area_width/2 + gm_params.dot_radius + dots[i].pos[1] <= model.distance
        @show dots[i].pos
        @show walls[i,:]
    end
    #sum(distances) == 0 && return nothing # what does this do?

    for i = 1:n
        collided_i = findall(collisions[i,:])
        collided_walls_i = findall(walls[i,:])
        println("$i collided with $collided_i")
        println("$i collided with walls $collided_walls_i")

        # we want to find the mean vector of where everything is with respect to
        # the dot in question
        angles = Vector{Float64}(undef, length(collided_i))
        for (j, x) in enumerate(collided_i)
            vec = dots[x].pos - dots[i].pos
            angles[j] = acos(vec[1]/norm(vec))
        end
    
        wall_angles = []
        walls[1] && push!(wall_angles, 3/2*pi)
        walls[2] && push!(wall_angles, pi)
        walls[3] && push!(wall_angles, 1/2*pi)
        walls[4] && push!(wall_angles, 0)

        all_angles = [angles; wall_angles]
        println("old bearing $(dots[i].bearing)")
        if length(all_angles) != 0
            new_bearing = -mean(all_angles)
            println("new bearing $new_bearing")
            #dots[i].bearing = 0.5 * dots[i].bearing + 0.5 * new_bearing
            dots[i].bearing = new_bearing #0.5 * dots[i].bearing + 0.5 * new_bearing
        end
    end
end

@gen function cbm_update(model::ConstrainedBDM, cg::CausalGraph, gm_params) #gm_params::GMMaskParams)
    dots = cg.elements
    new_dots = @trace(_cbm_step(fill(model, length(dots)), dots), :brownian)
    resolve!(model, gm_params, new_dots, 1)
    new_dots = collect(BDot, new_dots)
    cg = update(cg, new_dots)
    return cg
end
