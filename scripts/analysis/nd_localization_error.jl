using CSV
using JSON
using DataFrames
using Statistics
using UnicodePlots

# 1. Compute centroid per frame and per chain
# 2. Aggregate across chains
# 3. Explore time window
#
# Raw data format:
# frame	tracker	importance	cycles	pred_x	pred_y	scene	chain


function load_gt_positions(path::String)
    scenes = JSON.parsefile(path)
    ns = length(scenes)
    df = DataFrame(scene = Int64[],
                   frame = Int64[],
                   object = Int64[],
                   x = Float64[],
                   y = Float64[])
    for s = 1:ns
        steps = scenes[s]["positions"]
        nt = length(steps)
        ni = length(scenes[s]["aux_data"]["targets"])
        for t = 1:nt, i = 1:ni
            x,y = steps[t][i]
            push!(df, (s, t, i, x, y))
        end
    end
    return df
end

function nearest_distractor_distance(scene, frame, tx, ty, gt_gd)
    g = gt_gd[(scene = scene, frame = frame)]
    l2d = (x,y) -> l2_distance(tx,ty,x,y)
    transform!(g,
               [:x, :y] => l2d => :d)
    minimum(g.d)
end


function localization_error(scene, frame, tracker, tx, ty, gt_gd)
    g = gt_gd[(scene = scene, frame = frame, object = tracker)]
    l2d = (x,y) -> l2_distance(tx,ty,x,y)
    transform!(g,
               [:x, :y] => l2d => :d)
    first(g.d)
end

function weighted_centroid(tracker, weight, pred_x, pred_y)
    mu_x = sum(pred_x .* weight)
    mu_y = sum(pred_y .* weight)
    (ic_x = mu_x, ic_y = mu_y)
end

function l2_distance(cx, cy, x, y)
    @. sqrt((cx - x)^2 + (cy - y)^2)
end


# see `scripts/aggregate_chains.jl`
model = "td"

model_inferences = "/spaths/experiments/exp3_localization_error_adaptive_computation_$(model)_att.csv"
df = DataFrame(CSV.File(model_inferences))
min_frame = 24 # skip the first second
filter!(row -> row.frame >  min_frame, df)
select!(df, Cols(:chain, :scene, :frame, :tracker, :pred_x, :pred_y))
# filter!(row -> row.scene == 1, df) # TODO: remove after debugging

# distance to the nearest distractor for each frame x tracker
gt_positions = load_gt_positions("/spaths/datasets/exp3_localization_error.json")
filter!(row -> row.frame >  min_frame, gt_positions)

grouped_gt_positions = groupby(gt_positions, Cols(:scene, :frame, :object))
loc_error_f = ByRow((s, f, t, x, y) -> localization_error(s,f,t,x,y,grouped_gt_positions))
transform!(df,
           [:scene, :frame, :tracker, :pred_x, :pred_y] => loc_error_f => :loc_error)

distractor_positions = filter(row -> row.object > 3, gt_positions)
grouped_dis_positions = groupby(distractor_positions, Cols(:scene, :frame))
nd_dist_f = ByRow((s, f, x, y) -> nearest_distractor_distance(s,f,x,y,grouped_dis_positions))
transform!(df,
           [:scene, :frame, :pred_x, :pred_y] => nd_dist_f => :nn_dist)


CSV.write("/spaths/experiments/exp3_localization_error_adaptive_computation_$(model)_dnd.csv",
          df)

