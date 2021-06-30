using MOT
using MOT: CausalGraph
using Statistics: norm, mean
using Gen
using Setfield
using Lazy: @>, @>>
using BayesianOptimization, GaussianProcesses, Distributions

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
    global ITERATIONS += 1
    @show ITERATIONS

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
    
    scores = []

    for t=1:length(gt_cgs)-1
        cgs = gt_cgs[t:t+1]
        cm = MOT.get_init_constraints(first(cgs))
        get_constraints!(cm, cgs)
        #display(cm)
        
        trace, score = Gen.generate(MOT.gm_inertia_pos, (length(cgs), gm, dm), cm)
        score = isinf(score) ? -100000.0 : score
        push!(scores, score)
    end
    
    score = -mean(scores)
    @show score
    return score
end

scene = 1
func = get_f(scene)
#score = g([0.8, 1, 50, 0.01, 0.2])
#@show score

# Choose as a model an elastic GP with input dimensions 2.
# The GP is called elastic, because data can be appended efficiently.
model = ElasticGPE(2,                            # 2 input dimensions
                   mean = MeanConst(0.),         
                   kernel = SEArd([0., 0.], 5.),
                   logNoise = 0.,
                   capacity = 3000)              # the initial capacity of the GP is 3000 samples.
set_priors!(model.mean, [Normal(1, 2)])

# Optimize the hyperparameters of the GP using maximum a posteriori (MAP) estimates every 50 steps
modeloptimizer = MAPGPOptimizer(every = 50, noisebounds = [-4, 3],       # bounds of the logNoise
                                kernbounds = [[-1, -1, 0], [4, 4, 10]],  # bounds of the 3 parameters GaussianProcesses.get_param_names(model.kernel)
                                maxeval = 40)
opt = BOpt(func,
           model,
           UpperConfidenceBound(),                   # type of acquisition
           modeloptimizer,                        
           [0.01, 0.1, 10.0, 0.01, 0.01], [0.99, 20, 200, 0.5, 2.0], # lowerbounds, upperbounds       
           repetitions = 5,                          # evaluate the function for each input 5 times
           maxiterations = 100,                      # evaluate at 100 input positions
           sense = Min,                              # minimize the function
           acquisitionoptions = (method = :LD_LBFGS, # run optimization of acquisition function with NLopts :LD_LBFGS method
                                 restarts = 5,       # run the NLopt method from 5 random initial conditions each time.
                                 maxtime = 0.1,      # run the NLopt method for at most 0.1 second each time
                                 maxeval = 1000),    # run the NLopt methods for at most 1000 iterations (for other options see https://github.com/JuliaOpt/NLopt.jl)
            verbosity = Progress)

ITERATIONS = 0
result = boptimize!(opt)

