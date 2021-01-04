"""
    functions to calculate the structure values for each scene
"""

using MOT
using DataFrames
using JLD2
using CSV

"""
    computes the motion correlation between dots
"""
function get_correlation(dataset_path::String, scene::Int)
    scene_data = MOT.load_scene(scene, dataset_path, default_hgm;
                                generate_masks=false)
    gt_cgs = scene_data[:gt_causal_graphs]
    targets = scene_data[:aux_data]
    dots = map(_ -> true, targets)
    pos = map(cg -> MOT.get_hgm_positions(cg, dots), gt_cgs)
    vels = map(t -> pos[t] - pos[t+1], 1:length(pos)-1)
    println(vels)
    return 0.5
end

"""
    compute target concentration, mean(n_targets/n_dots for each polygon that has targets)
"""
function get_target_concentration(dataset_path::String, scene::Int)
    return 0.5
end



function quantify_structure(dataset_path::String, output_path::String)
    file = jldopen(dataset_path, "r")
    n_scenes = file["n_scenes"]
    close(file)
    
    correlations = map(i -> get_correlation(dataset_path, i), 1:n_scenes)
    target_concentrations = map(i -> get_target_concentration(dataset_path, i), 1:n_scenes)
    
    df = DataFrame(scene = 1:n_scenes,
                   correlations = correlations,
                   target_concentrations = target_concentrations)
    df |> CSV.write(output_path)
end
