abstract type GraphicalObject end

evolve(::GraphicalObject) = error("ni")

mutable struct GraphicalDot <: GraphicalObject
    flow::Flow
    bern_existence_prob::Float64
end

get_flow(d::GraphicalDot) = d.flow
get_bern_existence_prob(d::GraphicalDot) = d.bern_existence_prob


function evolve(d::GraphicalDot, space::Space)
    flow = evolve(d.flow, space)
    GraphicalDot(flow, d.bern_existence_prob)
end
