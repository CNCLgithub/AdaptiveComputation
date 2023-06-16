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

df = DataFrame(CSV.File("/spaths/experiments/exp2_probes_target_designation_att.csv"))
gdf = groupby(df, Cols(:scene, :frame, :tracker))
gdf = combine(gdf, Cols(:pred_x, :pred_y) .=> mean, renamecols = false)
df = gdf
display(df)

pdf = DataFrame(CSV.File("/spaths/datasets/exp2_probe_map_random.csv"))
gt_positions = load_gt_positions("/spaths/datasets/exp2_probes.json")
nd_dist_f = ByRow((s, f, x, y) -> nearest_distractor_distance(s,f,x,y,gt_positions))
transform!(df,
           [:scene, :frame, :pred_x, :pred_y] => nd_dist_f => :nn_dist)

sf = groupby(df, Cols(:scene, :frame))
total_distance = combine(sf,
                          :nn_dist => sum)
leftjoin!(df, total_distance,
          on = [:scene, :frame])
transform!(df,
           [:nn_dist, :nn_dist_sum] => ByRow((x, s) ->  x / s) => :weight)

sf = groupby(df, Cols(:scene, :frame))
# aggregate x and y across trackers
# scene    frame   icx  icy
ic = combine(sf,
     [:tracker, :weight, :pred_x, :pred_y] => weighted_centroid =>
         AsTable)

s1 = filter(x -> x.scene == 1, ic)
display(s1)

# scene   chain   frame   icx  icy    probe
label_probe_f = ByRow((x,y) -> label_probe(x,y,pdf))
# transform(c1, [:scene, :frame] => label_probe_f => :probe)
transform!(ic, [:scene, :frame] => label_probe_f => :probe)
filter!(:probe => x -> x > 0, ic)

distances = select(df, Not(:weight))
display(distances)
# distances = distances[distances.tracker .== distances.probe]

# scene   chain   frame   tracker   pred_x   pred_y   icx  icy   probe
distances = innerjoin(ic, distances, on = [:scene, :frame])
display(distances)
distances = distances[distances.tracker .== distances.probe, All()]

# scene   chain   frame   pred_x   pred_y   icx  icy
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

CSV.write("/spaths/experiments/exp2_probes_dnd.csv", distances)
