export BrownianDynamicsModel,
        update
        
struct BrownianDynamicsModel <: DynamicsModel
    inertia::Float64
    spring::Float64
    sigma_w::Float64
end

@gen function update_individual!(model::BrownianDynamicsModel, dot::Dot)
    x = dot.pos[1]
    y = dot.pos[2]
    vx = dot.vel[1]
    vy = dot.vel[2]

    dot.vel[1] = @trace(normal(model.inertia * vx - model.spring * x, model.sigma_w), :vx)
    dot.vel[2] = @trace(normal(model.inertia * vy - model.spring * y, model.sigma_w), :vy)

    dot.pos[1] += dot.vel[1]
    dot.pos[2] += dot.vel[2]
end

function update!(model::BrownianDynamicsModel, dots::Vector{Dot})
    map(update_individual!, fill(model, length(dots)), dots)
end

