export generate_dataset,
        is_min_distance_satisfied,
        are_dots_inside,
        forward_scene_data!,
        init_constraint_from_cg

include("generate_dataset_helpers.jl")


"""
    generate_dataset(dataset_path::String, n_scenes::Int64,
                                               k::Int64, gms::Vector{AbstractGMParams},
                                               dms::Vector{AbstractDynamicsModel};
                                               min_distance = 50.0,
                                               cms::Union{Nothing, Vector{ChoiceMap}} = nothing,
                                               aux_data::Union{Nothing, Vector{Any}} = nothing,
                                               ff_ks::Union{Nothing, Vector{Int64}} = nothing)

    generates a JLD2 dataset

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
            scene_data = nothing

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
                scene_data = dgp(init_k, gms[i], dms[i];
                                 generate_masks=false,
                                 cm=cm,
                                 generate_cm=true)

                # shifting scene data to the end if ff_ks are present
                !isnothing(ff_ks) && forward_scene_data!(scene_data, ff_ks[i])
                
                break # TESTING

                # checking whether dots are inside the area
                di=are_dots_inside(scene_data, gms[i])
                println("dots inside: $di")
                di || continue

                # checking whether the minimum distance between dots is satisfied
                md=is_min_distance_satisfied(scene_data, min_distance)
                println("minimum distance ($min_distance): $md")
                md || continue
                
                # if both satisfied, then breaking from the loop
                di && md && break
            end
            
            # generating the whole scene using the initial constraints from the loop above
            scene_data = dgp(k, gms[i], dms[i];
                             generate_masks=false,
                             cm=init_constraint_from_cg(first(scene_data[:gt_causal_graphs]), scene_data[:cm]))
            
            # saving the scene to a JLD2 structure
            scene = JLD2.Group(file, "$i")
            scene["gm"] = gms[i]
            scene["dm"] = dms[i]
            scene["aux_data"] = isnothing(aux_data) ? nothing : aux_data[i]
            scene["gt_causal_graphs"] = scene_data[:gt_causal_graphs]
        end
    end
end
