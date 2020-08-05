export ISRDynamics

@with_kw struct ISRDynamics <: AbstractDynamicsModel
    dot_repulsion::Float64 = 100.4
    wall_repulsion::Float64 = 110.9
    distance::Float64 = 40.0
    inertia::Float64 = 0.95
    sigma_x::Float64 = 1.0
    sigma_y::Float64 = 1.0
end

function load(::Type{ISRDynamics}, path::String)
    ISRDynamics(;read_json(path)...)
end

@gen function isr_noise_step(model::ISRDynamics, dot::Dot)
    _x, _y, _z = dot.pos
    _vx, _vy = dot.vel

    #vx = @trace(normal(model.inertia * _vx, model.sigma_x), :vx)
    #vy = @trace(normal(model.inertia * _vy, model.sigma_y), :vy)

    #x = _x + vx
    #y = _y + vy
    
    x = _x
    y = _y

    d = Dot([x,y,_z], [vx,vy])
    return d
end

_isr_noise_step = Map(isr_noise_step)

function isr_step(model, dots, gm_params)

    @show dots
    n = length(dots)

    for i = 1:n
        # repulsion from other dots
        force = zeros(3)
        for j = 1:n
            i == j && continue
            v = dots[i].pos - dots[j].pos
            #force .+= exp(-norm(v))*v/norm(v)
            force .+= model.dot_repulsion*exp(-(v[1]^2 + v[2]^2)/(2*model.dot_repulsion^2)) * v / norm(v)
        end
        #println("dot force $force")
        #dot_applied_force = model.dot_repulsion*force/norm(force)
        dot_applied_force = force
        @show dot_applied_force
        #@show dots[i].pos
        #dots[i].pos .+= applied_force
        #@show dots[i].pos
                
        # repulsion from walls
        walls = Matrix{Float64}(undef, 4, 3)
        walls[1,:] = [gm_params.area_width/2, dots[i].pos[2], dots[i].pos[3]]
        walls[2,:] = [dots[i].pos[1], gm_params.area_height/2, dots[i].pos[3]]
        walls[3,:] = [-gm_params.area_width/2, dots[i].pos[2], dots[i].pos[3]]
        walls[4,:] = [dots[i].pos[1], -gm_params.area_height/2, dots[i].pos[3]]

        force = zeros(3)
        for j = 1:4
            v = dots[i].pos - walls[j,:]
            force .+= model.dot_repulsion*exp(-(v[1]^2 + v[2]^2)/(2*model.wall_repulsion^2)) * v / norm(v)
        end
        #println("wall force $force")
        #wall_applied_force = model.wall_repulsion*force/norm(force)
        wall_applied_force = force
        @show i wall_applied_force
        println()
        #dots[i].pos .+= dot_applied_force + wall_applied_force
        dots[i].vel = 10*dots[i].vel/norm(dots[i].vel)
        dots[i].vel = model.inertia*dots[i].vel + (1.0-model.inertia)*(dot_applied_force[1:2]+wall_applied_force[1:2])
    end
    
    return dots
end


@gen function isr_update(model::ISRDynamics, cg::CausalGraph, gm_params) #gm_params::GMMaskParams)
    dots = cg.elements
    dots = @trace(_isr_noise_step(fill(model, length(dots)), dots), :dynamics)
    dots = isr_step(model, dots, gm_params)
    dots = collect(Dot, dots)
    cg = update(cg, dots)
    return cg
end
