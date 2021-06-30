export generate_dataset

include("generate_dataset_helpers.jl")

"""
    generate_dataset(dataset_path::String, n_scenes::Int64,
                                               k::Int64, gms::Vector{AbstractGMParams},
                                               dms::Vector{AbstractDynamicsModel};
                                               min_distance = 50.0,
                                               cms::Union{Nothing, Vector{ChoiceMap}} = nothing,
                                               aux_data::Union{Nothing, Vector{Any}} = nothing,
                                               ff_ks::Union{Nothing, Vector{Int64}} = nothing)

    Generates a JLD2 MOT dataset.

...
# Arguments:
- dataset_path: String
- n_scenes: number of scenes to be generated
- k: number of timesteps per scene
- gms: vector of generative model parameters for each scene
- dm: dynamics model parameters
...
# Optional arguments:
- min_distance: minimum distance between dots at t0
- cms: vector of choicemaps for each scene constraining the generative model
- aux_data: vector of auxiliary data for each scene needed to be saved
- ff_ks: vector of integers describing for each scene how many frames to generate
         for the initial constraint satisfaction (dots being inside + min distance)

"""
function generate_dataset(dataset_path::String, n_scenes::Int64,
                          k::Int64, gms, dms;
                          min_distance = 50.0,
                          cms::Union{Nothing, Vector} = nothing,
                          aux_data::Union{Nothing, Vector{Any}} = nothing,
                          ff_ks::Union{Nothing, Vector{Int64}} = nothing)
    
    jldopen(dataset_path, "w") do file 
        file["n_scenes"] = n_scenes
        for i=1:n_scenes
            println("generating scene $i/$n_scenes")
            init_gt_cgs = nothing

            # if no choicemaps, then create an empty one
            cm = isnothing(cms) ? choicemap() : cms[i]
            
            # this loop tries to satisfy min distance between dots and
            # dots being generated inside of the area
            tries = 0
            while true
                tries += 1
                println("$tries \r")

                # if ff_ks is not empty, then only generating those frames
                init_k = !isnothing(ff_ks) ? ff_ks[i]+1 : k
                init_gt_cgs = dgp(init_k, dms[i], gms[i];
                                  cm=cm)

                # shifting scene data to the end if ff_ks are present
                if !isnothing(ff_ks)
                    init_gt_cgs = init_gt_cgs[ff_ks[i]:end]
                end
                
                # checking whether dots are inside the area
                di=are_dots_inside(init_gt_cgs, gms[i])
                println("dots inside: $di")
                di || continue

                # checking whether the minimum distance between dots is satisfied
                md=is_min_distance_satisfied(first(init_gt_cgs), min_distance)
                println("minimum distance ($min_distance): $md")
                md || continue
                
                # if both satisfied, then breaking from the loop
                di && md && break
            end
            
            # generating the whole scene using the initial constraints from the loop above
            gt_cgs = dgp(k, dms[i], gms[i];
                         cm=init_constraint_from_cg(first(init_gt_cgs)))
            
            # checking whether any dots escaped
            !are_dots_inside(gt_cgs, gms[i]) && error("dots escaped the area")

            # saving the scene to a JLD2 structure
            scene = JLD2.Group(file, "$i")
            scene["gm"] = gms[i]
            scene["dm"] = dms[i]
            scene["aux_data"] = isnothing(aux_data) ? nothing : aux_data[i]
            scene["gt_causal_graphs"] = gt_cgs
        end
    end
end
