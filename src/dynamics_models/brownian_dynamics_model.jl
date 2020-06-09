export BrownianDynamicsModel,
        update
        
struct BrownianDynamicsModel <: DynamicsModel
    inertia::Float64
    spring::Float64
    sigma_w::Float64
end

function update!(model::BrownianDynamicsModel, dots::Vector{Dot})
    for dot in dots
        x = dot.pos[1]
        y = dot.pos[2]
        vx = dot.vel[1]
        vy = dot.vel[2]

        dot.vel[1] = normal(model.inertia * vx - model.spring * x, model.sigma_w)
        dot.vel[2] = normal(model.inertia * vy - model.spring * y, model.sigma_w)

        dot.pos[1] += dot.vel[1]
        dot.pos[2] += dot.vel[2]
    end
end
