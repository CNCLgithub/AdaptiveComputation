"""
    functions to calculate the structure values for each scene
"""

using MOT
using DataFrames
using JLD2
using CSV
using Statistics

"""
    computes the motion correlation between dots
"""
function get_correlations(dataset_path::String, scene::Int)
    scene_data = MOT.load_scene(scene, dataset_path, default_hgm;
                                generate_masks=false)
    gt_cgs = scene_data[:gt_causal_graphs]
    targets = scene_data[:aux_data]
    dots = map(_ -> true, targets)
    pos = map(cg -> MOT.get_hgm_positions(cg, dots), gt_cgs)
    vels = map(t -> pos[t+1] - pos[t], 1:length(pos)-1)
    vels_x = Matrix{Float64}(undef, length(vels), length(dots))
    vels_y = Matrix{Float64}(undef, length(vels), length(dots))
    for i=1:length(vels), j=1:length(dots)
        vels_x[i,j] = vels[i][j][1]
        vels_y[i,j] = vels[i][j][2]
    end
    cor_vels_x = cor(vels_x)
    cor_vels_y = cor(vels_y)
    cor_vels = (cor_vels_x + cor_vels_y) / 2
    return mean(cor_vels)
end

"""
    structure = 1/n_objects
"""
function get_structure(dataset_path::String, scene::Int)
    scene_data = MOT.load_scene(scene, dataset_path, default_hgm;
                                generate_masks=false)
    gt_cgs = scene_data[:gt_causal_graphs]
    n_objects = length(first(gt_cgs).elements)
    return 1/n_objects   
end

# helpers for get_target_concentration
_n_dots(x::MOT.Dot) = 1
_n_dots(x::MOT.Polygon) = length(x.dots)

"""
    compute target concentration, mean(n_targets/n_dots for each polygon that has targets)
"""
function get_target_concentration(dataset_path::String, scene::Int)
    scene_data = MOT.load_scene(scene, dataset_path, default_hgm;
                                generate_masks=false)
    gt_cgs = scene_data[:gt_causal_graphs]
    targets = scene_data[:aux_data]
    objects = first(gt_cgs).elements
    polygon_dots = map(x->_n_dots(x), objects)
    polygon_target_concentrations = []
    index = 1
    for pol in polygon_dots
        pol_targets = targets[index:index+pol-1]
        ptc = sum(pol_targets)/length(pol_targets)
        ptc != 0 && push!(polygon_target_concentrations, ptc)
        index += pol
    end
    
    return mean(polygon_target_concentrations)
end



function quantify_structure(dataset_path::String, output_path::String)
    file = jldopen(dataset_path, "r")
    n_scenes = file["n_scenes"]
    close(file)
    
    #correlations = map(i -> get_correlations(dataset_path, i), 1:n_scenes)
    structures = map(i -> get_structure(dataset_path, i), 1:n_scenes)
    target_concentrations = map(i -> get_target_concentration(dataset_path, i), 1:n_scenes)
    
    df = DataFrame(scene = 1:n_scenes,
                   structure = structures,
                   #correlation = correlations,
                   target_concentration = target_concentrations)
    df |> CSV.write(output_path)
end
