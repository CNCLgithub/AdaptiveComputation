using MOT
using MOT: CausalGraph
using Statistics: norm
using Gen
using Setfield
using Lazy: @>, @>>
using BayesianOptimization, GaussianProcesses

function get_angs_mags(cgs::Vector{MOT.CausalGraph})
    n_frames = length(cgs)
    n_objects = length(MOT.get_objects(first(cgs), Dot))

    pos = Array{Float64}(undef, n_frames, n_objects, 2)
    for i=1:n_frames
        dots = MOT.get_objects(cgs[i], Dot)
        for j=1:n_objects
            pos[i,j,:] = dots[j].pos[1:2]
        end
    end

    pos_t0 = pos[1:end-1,:,:]
    pos_t1 = pos[2:end,:,:]
    delta_pos = pos_t1 - pos_t0
    
    # getting velocity vectors
    vels = @>> Iterators.product(1:n_frames-1, 1:n_objects) begin
        map(ij -> delta_pos[ij[1], ij[2], :])
    end

    angs = @>> vels map(vel -> atan(vel...))
    mags = norm.(vels)

    return angs, mags
end

function get_constraints!(cm::ChoiceMap, gt_cgs::Vector{CausalGraph})
    angs, mags = get_angs_mags(gt_cgs)

    for t=1:length(gt_cgs)-1
        for i=1:length(MOT.get_objects(gt_cgs[t], Dot))
            addr = :kernel => t => :dynamics => :brownian => i => :mag
            cm[addr] = mags[t, i]
            addr = :kernel => t => :dynamics => :brownian => i => :ang
            cm[addr] = angs[t, i]
        end
    end
end

function get_f(scene)
    return x -> get_score(x, scene)
end

# function to optimize
function get_score(params, scene)
    bern = params[1]
    k_min = params[2]
    k_max = params[3]
    w_min = params[4]
    w_max = params[5]
    
    gm_path = "$(@__DIR__)/gm.json"
    dataset_path = "/datasets/exp1_difficulty.jld2"
    
    # loading scene data
    scene_data = MOT.load_scene(scene, dataset_path) 
    gt_cgs = scene_data[:gt_causal_graphs][1:end]
    aux_data = scene_data[:aux_data]
    
    gm = MOT.load(GMParams, gm_path)
    @set gm.n_trackers = sum(aux_data.targets) + sum(aux_data.n_distractors)
    @set gm.distractor_rate = 0.0

    dm = InertiaModel(vel=aux_data[:vel], bern=bern, k_min=k_min, k_max=k_max,
                      w_min=w_min, w_max=w_max)

    cm = MOT.get_init_constraints(first(gt_cgs))
    get_constraints!(cm, gt_cgs)
    display(cm)
    
    scores = []
    for i=1:10
        trace, score = Gen.generate(MOT.gm_inertia_pos, (length(gt_cgs), gm, dm), cm)
        push!(scores, score)
    end

    return scores
end

scene = 1
g = get_f(scene)
scores = g([0.8, 1, 50, 0.01, 0.2])
@show scores

