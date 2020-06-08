using MOT
using Gen
using FileIO
using Random

Random.seed!(3)

function test()
    T = 2

    # getting observations
    args = (nothing, 1, default_params)
    println("timestep 1")
    trace, _ = Gen.generate(generative_model_segments_dynamic, args)
    for t=2:T
        println("timestep $t")
        args = (Gen.get_retval(trace), t, default_params)
        trace, _ = Gen.update(trace, args, (), Gen.choicemap())
    end
    choices = Gen.get_choices(trace)
    #println(choices[1 => :x => 1])
    #segments = choices[(3, :segments)]
    #for (i, img) in enumerate(segments)
    #    save("image_$i.png", img)
    #end

    """
    # inference with observations from the generative model
    latents = Dict( :x => x -> :x )

    init_obs = Gen.choicemap()
    for i=1:default_params.num_targets
        init_obs[1 => :x => i] = choices[1 => :x => i]
        init_obs[1 => :y => i] = choices[1 => :y => i]
    end
    
    trackers = nothing
    args = [(trackers, t, default_params) for t in 1:T]
    observations = Vector{Gen.ChoiceMap}(undef, T)
    for t = 1:T
        cm = Gen.choicemap()
        for i=1:default_params.num_targets
            cm[t => :segment] = choices[t => :segment]
        end
        observations[t] = cm
    end

    query = Gen_Compose.SequentialQuery(latents, #bogus for now
                                        generative_model_segments_dynamic,
                                        (0, default_params),
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
                                            false,
                                            3,
                                            3,
                                            true)

    results = sequential_monte_carlo(procedure, query)
    MOT.save_state(results, "test.jld2")
    positions = retrieve_pf_positions("test.jld2")

    #println(positions)
    overlay(gt_dots[1:T,:,:], num_targets; pf_xy=positions, stimuli=false, highlighted=[1])
    """
end

test()
