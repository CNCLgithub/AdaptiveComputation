# loader.jl
#
# Given a dataset, render a subset of trials with a subset of objects
using Gen
using MOT
using MOT: @set, choicemap
using JLD2
using Lazy: @>, @>>
using Random

using Images
using FileIO

# Set seed for reproducability
Random.seed!(1234)

@gen function sample_without_replacement(ids::Vector{Int64}, k::Int64)
end

"""
Returns the CausalGraphs of everyframe in the scene for the given dataset
"""
function load_scene(dataset_path::String, scene::Int64)::Vector{CausalGraph}
    # see JLD2 docs
end

"""
Returns the states of the subset, `vs`, as causal graphs
"""
function select_subset(cgs::Vector{CausalGraph}, vs::Vector{Int64})::Vector{CausalGraph}
    # see LightGraphs package
end

"""
Renders all objects as an ensemble (one resulting image)
"""
function render_subset(cgs::Vector{CausalGraph})::Matrix{Float64}
    # see MOT.render; under generative_models/graphics/graphics_module.jl
    # see MOT.aggregate_masks; under visuals/render_masks.jl
end

function render_trial(dpath::String, sid::Int64, k::Int64)

    # use functions above to load scene, then randomly sample a subset, and render that subset
    scene = load_scene()
    obj_vs = @> scene begin
        first # first causal graph
        get_object_verts(MOT.Dot) # Vector{Int64}
    end
    subset = sample_without_replacement(obj_vs, k)
    # use `select_subset` and `render_subset`
    img = render_subset()

    # save the image
    save("${render_path}/${sid}_${subset_name}.png", img)
end
