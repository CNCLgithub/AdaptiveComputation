abstract type AbstractGraphicalState end

evolve(::AbstractGraphicalState, spaces::Vector{Space})

struct GraphicalState <: AbstractGraphicalState
    reps::Vector{GraphicalRep}
    #graphical_ensemble::GraphicalEnsemble
end

function evolve(gs::GraphicalState, spaces::Vector{Space})
    graphical_objects = @>> gs get_graphical_objects
    #graphical_ensemble = @>> gs get_graphical_ensemble
    
    graphical_objects = map(evolve, graphical_objects, spaces)
    graphical_ensemble = evolve(graphical_ensemble)

    GraphicalState(graphical_objects, graphical_ensemble)
end

function GraphicalState(graphics::Graphics,
                        space::Space,
                        n_dots::Int64,
                        flow_decay_rate::Float64,
                        bern_existence_prob::Float64,
                        ppp_rate::Float64,
                        ppp_pixel_prob::Float64)

    flows = @>> 1:n_dots begin
        map(i -> ExponentialFlow{T}(flow_decay_rate,
                                    Matrix{Float64},
                                    graphics.img_dims))
    end
    
    graphical_objects = @>> flows map(flow -> GraphicalDot(flow, bern_existence_prob))
    graphical_ensemble = PPP(ppp_rate, ppp_pixel_prob)

    GraphicalState(graphical_objects, graphical_ensemble)
end

