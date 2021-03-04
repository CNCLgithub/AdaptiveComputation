export generate_dataset,
        is_min_distance_satisfied,
        are_dots_inside,
        forward_scene_data!,
        init_constraint_from_cg

include("generate_dataset_helpers.jl")

function generate_dataset(dataset_path, n_scenes, k, gms, dm;
                          min_distance = 50.0,
                          cms::Union{Nothing, Vector{ChoiceMap}} = nothing,
                          aux_data::Union{Nothing, Vector{Any}} = nothing,
                          ff_ks::Union{Nothing, Vector{Int64}} = nothing)
    
    jldopen(dataset_path, "w") do file 
        file["n_scenes"] = n_scenes
        for i=1:n_scenes
            println("generating scene $i/$n_scenes")
            scene_data = nothing

            # if no choicemaps, then create an empty one
            cm = isnothing(cms) ? choicemap() : cms[i]
            
            tries = 0
            while true
                tries += 1
                println("$tries \r")
                init_k = !isnothing(ff_ks) ? ff_ks[i]+1 : k
                scene_data = dgp(init_k, gms[i], dm;
                                 generate_masks=false,
                                 cm=cm,
                                 generate_cm=true)
                !isnothing(ff_ks) && forward_scene_data!(scene_data, ff_ks[i])
                di=are_dots_inside(scene_data, gms[i])
                println("dots inside: $di")
                di || continue
                md=is_min_distance_satisfied(scene_data, min_distance)
                println("minimum distance ($min_distance): $md")
                md || continue
                di && md && break
            end

            scene_data = dgp(k, gms[i], dm;
                             generate_masks=false,
                             cm=init_constraint_from_cg(first(scene_data[:gt_causal_graphs]), scene_data[:cm]))

            scene = JLD2.Group(file, "$i")
            scene["gm"] = gms[i]
            scene["dm"] = dm
            scene["aux_data"] = isnothing(aux_data) ? nothing : aux_data[i]
            scene["gt_causal_graphs"] = scene_data[:gt_causal_graphs]
        end
    end
end
