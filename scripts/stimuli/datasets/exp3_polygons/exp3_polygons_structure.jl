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
    targets = scene_data[:aux_data][:targets]
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
get_structure_value(scene_data) = 1/length(scene_data[:aux_data][:polygon_structure])

# helpers for get_target_concentration
_n_dots(x::MOT.Dot) = 1
_n_dots(x::MOT.Polygon) = length(x.dots)

"""
    compute target concentration, mean(n_targets/n_dots for each polygon that has targets)
"""
function get_target_concentration(pol_structure, targets)
    #println("pol_structure $pol_structure, targets $targets")
    pol_target_concentrations = []
    index = 1
    for pol in pol_structure
        # if a dot, then count as 0
        if pol == 1 && targets[index] == 1
            push!(pol_target_concentrations, 0.0)
        else
            pol_targets = targets[index:index+pol-1]
            ptc = sum(pol_targets)/length(pol_targets)
            ptc != 0 && push!(pol_target_concentrations, ptc)
        end
        index += pol
    end
    
    #println(pol_target_concentrations)
    if sum(pol_target_concentrations) == 0.0
        return 0.0
    else
        return mean(pol_target_concentrations)
    end
end

get_polygon_structure(scene_data) = scene_data[:aux_data][:polygon_structure]
get_targets(scene_data) = scene_data[:aux_data][:targets]

function quantify_structure(dataset_path::String,
                            output_path::String;
                            alpha::Float64 = 1.0,
                            beta::Float64 = 1.0)
    file = jldopen(dataset_path, "r")
    n_scenes = file["n_scenes"]
    close(file)
   
    scenes = map(i -> MOT.load_scene(i, dataset_path, default_hgm; generate_masks=false), 1:n_scenes)

    polygons = map(i -> get_polygon_structure(scenes[i]), 1:n_scenes)
    targets = map(i -> get_targets(scenes[i]), 1:n_scenes)
    structure_values = map(i -> get_structure_value(scenes[i]), 1:n_scenes)
    target_concentrations = map(i -> get_target_concentration(polygons[i], targets[i]), 1:n_scenes)

    rel_structures = (alpha .* structure_values + beta .* target_concentrations)/(alpha + beta)
    
    df = DataFrame(scene = 1:n_scenes,
                   #correlation = correlations,
                   polygons = polygons,
                   targets = targets,
                   n_targets = map(sum, targets),
                   structure_value = structure_values,
                   target_concentration = target_concentrations,
                   rel_structure = rel_structures)
    df |> CSV.write(output_path)
end
