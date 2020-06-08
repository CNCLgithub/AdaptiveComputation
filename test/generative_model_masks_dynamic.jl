using MOT
using Gen
using Gen_Compose
using FileIO
using Random

using Images, ImageDraw

using StatProfilerHTML


function generate_data(T, num_particles)
    args = (T, default_params, true)
    trace, weight = Gen.generate(generative_model_masks_dynamic, args)
    choices = Gen.get_choices(trace)
    full_imgs, _ = Gen.get_retval(trace)

    println("weight: $weight")
    println("score: $(Gen.get_score(trace))")

    return choices, full_imgs
end

# function for the latent map
function extract_tracker_positions(trace::Gen.Trace)
    _, tracker_positions = Gen.get_retval(trace)
    tracker_positions = tracker_positions[end,:,:]
    tracker_positions = reshape(tracker_positions, (1,1,size(tracker_positions)...))
    return tracker_positions
end

function extract_chain(r::Gen_Compose.SequentialChain)
    weighted = []
    unweighted = []
    log_scores = []
    ml_est = []
    states = []
    for t = 1:length(r.buffer)
        state = r.buffer[t]
        push!(weighted, state["weighted"])
        push!(unweighted, state["unweighted"])
        push!(log_scores, state["log_scores"])
        push!(ml_est, state["ml_est"])
    end
    weighted = merge(vcat, weighted...)
    unweighted = merge(vcat, unweighted...)
    log_scores = vcat(log_scores...)
    extracts = Dict("weighted" => weighted,
                    "unweighted" => unweighted,
                    "log_scores" => log_scores,
                    "ml_est" => ml_est)
    return extracts
end

function run_inference(choices, T, num_particles)
    println()
    println("INFERENCE TIME WOWOWOOWOWOWOWO!!!!")
    println()

    # inference with observations from the generative model
    latent_map = LatentMap(Dict(
                                :tracker_positions => extract_tracker_positions
                               ))

    println(latent_map)
    
    init_obs = Gen.choicemap()
    for i=1:default_params.num_trackers
        init_obs[0 => :x => i] = choices[0 => :x => i]
        init_obs[0 => :y => i] = choices[0 => :y => i]
    end
    
    args = [(t, default_params, false) for t in 1:T]
    println(args)
    observations = Vector{Gen.ChoiceMap}(undef, T)
    for t = 1:T
        cm = Gen.choicemap()
        cm[t => :masks] = choices[t => :masks]
        observations[t] = cm
    end

    query = Gen_Compose.SequentialQuery(latent_map, #bogus for now
                                        generative_model_masks_dynamic,
                                        (0, default_params, false),
                                        init_obs,
                                        args,
                                        observations)
    do_nothing() = nothing

    procedure = PopParticleFilter(num_particles,
                                            num_particles/2, # ESS is in terms of effective particle count, not fraction
                                            nothing,
                                            tuple(),
                                            nothing,
                                            nothing,
                                            nothing,
                                            3,
                                            3,
                                            true)

    results = sequential_monte_carlo(procedure, query,
                                     buffer_size = T)
    
    return results
end

function visualize(xy, full_imgs, T, num_particles)
    # just some non file magic kind of thing
    #xy = Array{Float64}(undef, num_particles, T, default_params.num_trackers, 2)
    
    for t=1:length(full_imgs)
        img = AbstractArray{Float64}
        #img = zeros(Gray, default_params.img_height, default_params.img_width)
        
        img = Gray.(full_imgs[t])
        #for i=1:default_params.img_height, j=1:default_params.img_width
        #    img[i,j] = full_imgs[t,i,j]
        #end

        for p=1:num_particles
            for i=1:default_params.num_trackers
                x = xy[t,p,i,1]
                y = xy[t,p,i,2]
                x = round(Int, x+default_params.img_width/2) 
                y = round(Int, y+default_params.img_height/2)
                circle = CirclePointRadius(x, y, 7.0)
                draw!(img, circle, Gray(0.0))
                circle = CirclePointRadius(x, y, 5.0)
                draw!(img, circle, Gray(1.0))
                circle = CirclePointRadius(x, y, 3.0)
                draw!(img, circle, Gray(0.0))
                circle = CirclePointRadius(x, y, 1.0)
                draw!(img, circle, Gray(1.0))
            end
        end
        save("image_$(lpad(t, 3, "0")).png", img)
    end

    # WHAT TO DO WITH SAVE STATE
    #MOT.save_state(results, "test.jld2")
    #positions = retrieve_pf_positions("test.jld2")

    #println(positions)
    #overlay(gt_dots[1:T,:,:], num_targets; pf_xy=positions, stimuli=false, highlighted=[1])
end

function test()
    Random.seed!(3)

    T = 20
    num_particles = 5
    
    # starting the profiler
    #statprofilehtml()

    choices, full_imgs = generate_data(T, num_particles)
    results = run_inference(choices, T, num_particles)
    extracted = extract_chain(results)
    xy = extracted["unweighted"][:tracker_positions]
    visualize(xy, full_imgs, T, num_particles)

    return extracted
end


