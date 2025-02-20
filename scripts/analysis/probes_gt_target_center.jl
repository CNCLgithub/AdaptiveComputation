"""
Computes the target center using the ground-truth positions

Will produce one file:
1. `exp_probes_gt_tc.csv` : The ground truth target centers
"""

using CSV
using JSON
using DataFrames

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
                   tx = Float64[],
                   ty = Float64[])
    for s = 1:ns
        steps = scenes[s]["positions"]
        nt = length(steps)
        ni = count(scenes[s]["aux_data"]["targets"])
        for t = 1:nt
            x = 0.0; y = 0.0;
            for i = 1:ni
                _x,_y,_ = steps[t][i]
                x += _x
                y += _y
            end
            x *= 1.0 / ni
            y *= 1.0 / ni
            push!(df, (s, t, x, y))
        end
    end
    return df
end

gt_positions = load_gt_positions("/spaths/datasets/exp_probes.json")
CSV.write("/spaths/experiments/exp_probes_gt_tc.csv", gt_positions)
