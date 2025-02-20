"""
Computes the nearest distractor distances.

It is required to use the ground-truth distractor positions
since the `Inertia` generative model does not directly represent
distractor positions.

Will produce two files:
1. `exp_probes_dnd.csv` : raw distance to nearest distractor for each target
2. `exp_probes_dnd_centroid.csv` : computes a centroid distance metric
"""

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

function nearest_distractor_pos(row)
    g = gt_gd[(scene = scene, frame = frame)]
    l2d = (x,y) -> l2_distance(tx,ty,x,y)
    transform!(g,
               [:x, :y] => l2d => :d)
    idx = argmin(g.d)
    g[idx, [:x, :y]]
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
probed_frames = select(pdf, Cols(:scene, :frame, :tracker))

# distance to the nearest distractor for each frame x tracker
gt_positions = load_gt_positions("/spaths/datasets/exp_probes.json")
gt_positions = leftjoin(probed_frames, gt_positions, on = [:scene, :frame])

gt_probe_pos = filter(row -> row.tracker == row.object, gt_positions)
rename!(gt_probe_pos,
        :x => :probe_x,
        :y => :probe_y)
select!(gt_probe_pos, [:scene, :frame, :probe_x, :probe_y])
display(gt_probe_pos)

leftjoin!(gt_positions, gt_probe_pos, on = [:scene, :frame])

distractor_positions = filter(row -> row.object > 4, gt_positions)
grouped_dis_positions = groupby(distractor_positions, Cols(:scene, :frame))
distractor_center = combine(grouped_dis_positions, [:x, :y] .=> mean .=> [:dcx, :dcy])

transform!(distractor_positions,
           [:x, :y, :probe_x, :probe_y] => l2_distance => :l2d)
grouped_dis_positions = groupby(distractor_positions, Cols(:scene, :frame))
subset!(grouped_dis_positions,
        :l2d => x -> x .== minimum(x)
        )
rename!(distractor_positions,
        :x => :dx,
        :y => :dy)
select!(distractor_positions, Cols(:scene, :frame, :dx, :dy))
leftjoin!(distractor_positions, distractor_center, on = [:scene, :frame])

CSV.write("/spaths/experiments/exp_probes_dis_cov.csv",
          distractor_positions)
