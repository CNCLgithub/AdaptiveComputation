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
            x,y,_ = steps[t][i]
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

function label_probe(scene, frame, pdf)
    spdf = pdf[pdf.scene .== scene, :]
    probe = 0
    for r in eachrow(spdf)
        probe = r.tracker
        r.frame >= frame && break
    end
    probe
end

function l2_distance(cx, cy, x, y)
    @. sqrt((cx - x)^2 + (cy - y)^2)
end


# probe timing for experiment
probe_timings = "/spaths/datasets/exp2_probe_map_random.csv"
pdf = DataFrame(CSV.File(probe_timings))
filter!(row -> row.scene <= 40, pdf)
# probe frames
probed_frames = select(pdf, Cols(:scene, :frame))

# see `scripts/aggregate_chains.jl`
model = "td"
# chain performance
model_perf_csv = "/spaths/experiments/exp2_probes_ac_$(model)_perf.csv"
# model_perf_csv = "/spaths/experiments/exp2_probes_adaptive_computation_$(model)_perf.csv"
# model_perf_csv = "/spaths/experiments/exp2_probes_adaptive_computation_perf.csv"
model_perf = DataFrame(CSV.File(model_perf_csv))
model_perf = groupby(model_perf, Cols(:chain, :scene))
model_perf = combine(model_perf, Cols(:td_acc) => mean => :avg_acc)
transform!(model_perf, [:avg_acc] => ByRow(acc -> acc > 0.75) => :passed)
filter!(row -> row.passed, model_perf)
passed_chains = select(model_perf, Cols(:chain, :scene))


model_inferences = "/spaths/experiments/exp2_probes_ac_$(model)_att.csv"
# model_inferences = "/spaths/experiments/exp2_probes_adaptive_computation_$(model)_att.csv"
# model_inferences = "/spaths/experiments/exp2_probes_adaptive_computation_att.csv"
df = DataFrame(CSV.File(model_inferences))
df = leftjoin(passed_chains, df, on = [:chain, :scene])
df = select(df, Cols(:chain, :scene, :frame, :tracker, :pred_x, :pred_y))
df = leftjoin(probed_frames, df, on = [:scene, :frame])
# filter!(row -> row.scene == 1, df) # TODO: remove after debugging


# distance to the nearest distractor for each frame x tracker
gt_positions = load_gt_positions("/spaths/datasets/exp2_probes.json")

grouped_gt_positions = groupby(gt_positions, Cols(:scene, :frame, :object))
loc_error_f = ByRow((s, f, t, x, y) -> localization_error(s,f,t,x,y,grouped_gt_positions))
transform!(df,
           [:scene, :frame, :tracker, :pred_x, :pred_y] => loc_error_f => :loc_error)

distractor_positions = filter(row -> row.object > 4, gt_positions)
grouped_dis_positions = groupby(distractor_positions, Cols(:scene, :frame))
nd_dist_f = ByRow((s, f, x, y) -> nearest_distractor_distance(s,f,x,y,grouped_dis_positions))
transform!(df,
           [:scene, :frame, :pred_x, :pred_y] => nd_dist_f => :nn_dist)


gdf = groupby(df, Cols(:scene, :frame, :tracker))
gdf = combine(gdf, Cols(:pred_x, :pred_y, :loc_error, :nn_dist) .=> [mean std])

CSV.write("/spaths/experiments/exp2_probes_adaptive_computation_$(model)_dnd.csv",
          gdf)

df = select(gdf, Not(r"std"))
rename!(df,
        :pred_x_mean => :pred_x,
        :pred_y_mean => :pred_y,
        :nn_dist_mean => :nn_dist
        )

# normalize each tracker's nearest distactor distance
# by the the aggregate across trackers
sf = groupby(df, Cols(:scene, :frame))
total_distance = combine(sf,
                          :nn_dist => sum)
leftjoin!(df, total_distance,
          on = [:scene, :frame])
transform!(df,
           [:nn_dist, :nn_dist_sum] => ByRow((x, s) ->  x / s) => :weight)

# compute the weighted centroid according to proportion of nearest
# neighbor "mass"
sf = groupby(df, Cols(:scene, :frame))
# aggregate x and y across trackers
# scene    frame   icx  icy
ic = combine(sf,
     [:tracker, :weight, :pred_x, :pred_y] => weighted_centroid =>
         AsTable)

s1 = filter(x -> x.scene == 1, ic)
display(s1)

# extract frames with probe present
# format: scene chain frame icx icy probe
label_probe_f = ByRow((x,y) -> label_probe(x,y,pdf))
transform!(ic, [:scene, :frame] => label_probe_f => :probe)
filter!(:probe => x -> x > 0, ic)

# compute the distances of the centroid to each probe
distances = select(df, Not(:weight))
# format: scene chain frame tracker pred_x pred_y icx icy probe
distances = innerjoin(ic, distances, on = [:scene, :frame])
distances = distances[distances.tracker .== distances.probe, All()]
# format: scene chain frame pred_x pred_y icx  icy
select!(distances, Not(Cols(:tracker, :probe)))
# scene   chain   frame   pred_x   pred_y   icx  icy   d_ic
transform!(distances,
           [:ic_x, :ic_y, :pred_x, :pred_y] => l2_distance => :d_ic)
# scene   chain   frame   ic_x  ic_y   d_ic
select!(distances, Not(Cols(:pred_x, :pred_y)))
display(distances)

CSV.write("/spaths/experiments/exp2_probes_ac_$(model)_dnd_centroid.csv", distances)
