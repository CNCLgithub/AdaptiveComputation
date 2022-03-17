using MOT
using CSV
using Statistics
using LinearAlgebra:norm
using Base.Iterators:take
using DataFrames
using FileIO

function add_nearest_distractor(att_tps::String, att_tps_out::String, dataset_path::String;
                                min_distance::Float64=0.0)

    df = DataFrame(CSV.File(att_tps))

    # adding new cols
    df[!,:nd] .= 0
    df[!,:dist_to_nd] .= 0.0
    df[!,:tracker_to_origin] .= 0.0 # perhaps to control for eccentricity?
    df[!,:tracker_to_tracker_mean] .= 0.0 # perhaps to control for eccentricity?
    df[!,:tracker_to_dot_mean] .= 0.0 # another control for eccentricity
    df[!,:cumulative_dist] .= 0.0 # distance to all other objects

    for (i, trial_row) in enumerate(eachrow(df))
        scene = trial_row.scene
        scene_data = MOT.load_scene(scene, dataset_path, default_gm;
                                    generate_masks=false)
        # getting the corresponding causal graph elements
        #
        # (+1 because the first causal graph is for the init state)
        # dots = scene_data[:gt_causal_graphs][trial_row.frame+1].elements
        # actually, not sure if needed
        dots = scene_data[:gt_causal_graphs][trial_row.frame].elements
        pos = collect(map(x->x.pos[1:2], dots))

        tracker_pos = pos[trial_row.tracker]
        
        tracker_mean = Statistics.mean(pos[1:4])
        dot_mean = Statistics.mean(pos)
        df[i, :tracker_to_origin] = norm(tracker_pos - zeros(2))
        df[i, :tracker_to_tracker_mean] = norm(tracker_pos - tracker_mean)
        df[i, :tracker_to_dot_mean] = norm(tracker_pos - dot_mean)
        
        tracker_distances = map(x->norm(tracker_pos - x), pos[setdiff(1:4, trial_row.tracker)])
        distractor_distances = map(distr_pos->norm(tracker_pos - distr_pos), pos[5:8])
        
        df[i, :cumulative_dist] = sum(tracker_distances) + sum(distractor_distances)

        distractor_distances = map(x-> x < min_distance ? Inf : x, distractor_distances)

        df[i, :nd] = argmin(distractor_distances)+4
        df[i, :dist_to_nd] = minimum(distractor_distances)
    end

    display(df)
    CSV.write(att_tps_out, df)
end

