using MOT
using Random
using FileIO
using VideoIO
Random.seed!(3)


q = Exp1(k=120, trial=1,
         gm="scripts/inference/exp1/gm.json",
         proc="scripts/inference/exp1/proc.json",
         dataset_path="output/datasets/exp1_isr.jld2")
gm = MOT.load(GMMaskParams, q.gm)

positions = last(MOT.load_trial(1, q.dataset_path, gm; generate_masks=false))

probes = zeros(Bool, 120, 8)
probes[45:48,2] .= true
render(gm, dot_positions=positions, freeze_time=30, stimuli=true, highlighted=[1], probes=probes)

path="render"
imgnames = filter(x->occursin(".png",x), readdir(path))
intstrings =  map(x->split(x,".")[1],imgnames)
p = sortperm(parse.(Int,intstrings))
imgstack = []
for imgname in imgnames[p]
    push!(imgstack, load(joinpath(path, imgname)))
end
encodevideo(joinpath("render", "video.mp4"), imgstack)
