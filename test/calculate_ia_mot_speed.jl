using MOT
using Lazy
using LinearAlgebra
using Statistics

scene = 1
dataset = "/datasets/ia_mot.jld2"
gm_params = MOT.load(GMMaskParams, "scripts/inference/ia_rf/gm.json")

scene_data = load_scene(scene, dataset, gm_params;
                        generate_masks=true)
gt_causal_graphs = scene_data[:gt_causal_graphs]

raw_pos = @>> gt_causal_graphs begin
    map(cg -> map(x -> x.pos, cg.elements))
end

k = size(pos,1)
n_dots = size(first(pos), 1)

pos = Array{Float64}(undef, k, n_dots, 2)
for t=1:k, i=1:n_dots, j=1:2
    pos[t,i,j] = raw_pos[t][i][j]
end

vel = @>> begin 1:size(pos,1)-1
    map(t -> norm(pos[t+1,:,:] .- pos[t,:,:]))
    mean
end



