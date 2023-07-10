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
        for t = 1:nt, i = 5:ni
            x,y,_ = steps[t][i]
            push!(df, (s, t, i, x, y))
        end
    end
    gdf = groupby(df, Cols(:scene, :frame))
    return gdf
end

function nearest_distractor_distance(scene, frame, tx, ty, gt_gd)
    println("scene $scene, frame $frame")
    g = gt_gd[(scene = scene, frame = frame)]
    l2d = (x,y) -> l2_distance(tx,ty,x,y)
    transform!(g,
               [:x, :y] => l2d => :d)
    minimum(g.d)
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

# see `scripts/aggregate_chains.jl`
model = "td"
model_inferences = "/spaths/experiments/exp2_probes_adaptive_computation_$(model)_att.csv"
df = DataFrame(CSV.File(model_inferences))
df = select(df, Cols(:chain, :scene, :frame, :tracker, :pred_x, :pred_y))
# filter!(row -> row.scene == 1, df) # TODO: remove after debugging

# probe timing for experiment
probe_timings = "/spaths/datasets/exp2_probe_map_random.csv"
pdf = DataFrame(CSV.File(probe_timings))

# distance to the nearest distractor for each frame x tracker
gt_positions = load_gt_positions("/spaths/datasets/exp2_probes.json")
nd_dist_f = ByRow((s, f, x, y) -> nearest_distractor_distance(s,f,x,y,gt_positions))
transform!(df,
           [:scene, :frame, :pred_x, :pred_y] => nd_dist_f => :nn_dist)

gdf = groupby(df, Cols(:scene, :frame, :tracker))
gdf = combine(gdf, Cols(:pred_x, :pred_y, :nn_dist) .=> [mean std])

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

# a sanity check, looking at all chains on one scene
s1 = filter(x -> x.scene == 1, distances)
plt = scatterplot(s1.frame, s1.d_ic,
                  height = 50, width = 50)
display(plt)

CSV.write("/spaths/experiments/exp2_probes_adaptive_computation_$(model)_dnd_centroid.csv", distances)
