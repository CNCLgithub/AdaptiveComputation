export ISRPylonsDynamics

@with_kw struct ISRPylonsDynamics <: AbstractDynamicsModel
    repulsion::Bool = true
    dot_repulsion::Float64 = 80.0
    wall_repulsion::Float64 = 50.0
    distance::Float64 = 80.0
    vel::Float64 = 10.0 # base velocity
    rep_inertia::Float64 = 0.9

    brownian::Bool = true
    inertia::Float64 = 0.8
    spring::Float64 = 0.002
    sigma_x::Float64 = 1.0
    sigma_y::Float64 = 1.0
    
    # there are 4 pylons, all symmetrically positioned from origin (0,0)
    pylon_strength = 18.0
    pylon_radius = 100.0
    pylon_x = 150.0 
    pylon_y = 150.0
end


function get_pylons_force(model, dots, pylons, gm_params)

    n = length(dots)
    pylons_forces = Vector{Vector{Float64}}(undef, n)

    for i = 1:n
        dot = dots[i]

        force = zeros(3)
        for pylon in pylons
            v = dot.pos - pylon.pos
            (norm(v) > pylon.radius) && continue
            force .+= -dot.pylon_interaction * pylon.strength * exp(-(v[1]^2 + v[2]^2)/(pylon.radius^2)) * v / norm(v)
        end

        println(force)

        pylons_forces[i] = force[1:2]
    end
    
    #println(pylons_forces)

    pylons_forces
end


function isr_pylons_step(model, dots, pylons, gm_params)
    rep_forces = get_repulsion_force(model, dots, gm_params)
    pylons_forces = get_pylons_force(model, dots, pylons, gm_params)
    
    n = length(dots)

    for i = 1:n
        vel = dots[i].vel
        if sum(vel) != 0
            vel *= model.vel/norm(vel)
        end
        vel *= model.rep_inertia
        vel += (1.0-model.rep_inertia)*(rep_forces[i]+pylons_forces[i])
        dots[i] = Dot(pos=dots[i].pos, vel=vel,
                      pylon_interaction=dots[i].pylon_interaction)
    end
    
    dots
end



@gen function isr_pylons_update(model::ISRDynamics, cg::CausalGraph, gm_params) #gm_params::GMMaskParams)
    dots = filter(x->isa(x, Dot), cg.elements)
    pylons = filter(x->isa(x, Pylon), cg.elements)

    if model.repulsion
        dots = isr_pylons_step(model, dots, pylons, gm_params)
    end
    
    dots = @trace(_isr_brownian_step(fill(model, length(dots)), dots), :brownian)
    println("pylon interaction: ", map(d->d.pylon_interaction, dots))

    dots = collect(Dot, dots)
    cg = update(cg, [dots; pylons])
    return cg
end
