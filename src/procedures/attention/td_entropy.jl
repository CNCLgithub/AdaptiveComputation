export TDEntropyAttentionModel

@with_kw struct TDEntropyAttentionModel <: AbstractAttentionModel
    rejuv_smoothness::Float64 = 1.005
    max_sweeps::Int = 0
    max_fails::Int = 3 
    perturb_function::Union{Function, Nothing} = nothing
end

function load(::Type{TDEntropyAttentionModel}, path; kwargs...)
    TDEntropyAttentionModel(;read_json(path)..., kwargs...)
end

function get_stats(attention::TDEntropyAttentionModel, state::Gen.ParticleFilterState)
    num_particles = length(state.traces)

    samples = sample_unweighted_traces(state, num_particles)

    t, motion, gm = Gen.get_args(first(samples))

    # let's do td_entropy in terms of trackers!!!
    td_entropy = fill(-Inf, gm.n_trackers)

    for i=1:num_particles
        masks = samples[i][:states => t => :masks]
        
        # getting tracker designation, assignment and the weights for TD
        # from saved state in pmbrfs_params (Gen hack)
        pmbrfs_stats = Gen.get_retval(samples[i])[2][t].pmbrfs_params.pmbrfs_stats
        tds, td_weights = pmbrfs_stats.partitions, pmbrfs_stats.ll_partitions
         
        # saving main TD and assignment hypothesis
        main_td = tds[1][1]
        main_A = tds[1][2]

        # comparing them to the other hypotheses
        for j=2:length(tds)

            # these are in the main hypothesis, but not in the alternative
            differing_obs = setdiff(main_td, tds[j][1])
           
            # finding the trackers that are involved
            # i.e. trackers having differing observations different hypotheses
            # (this part was/is tricky)
            differing_obs_indices = []
            for ob in differing_obs
                push!(differing_obs_indices, findall(x->x==ob, main_td)[1])
            end

            trackers = []
            for index in differing_obs_indices
                push!(trackers, main_A[index])
            end
            
            for tracker in trackers
                td_entropy[tracker] = logsumexp(td_entropy[tracker], td_weights[j])
            end
        end
    end
    
    # dividing td_entropy by num_particles to normalize
    td_entropy .-= log(num_particles)

    # making it smoother
    td_entropy = attention.max_sweeps * attention.rejuv_smoothness.^td_entropy

    return td_entropy
end

function get_sweeps(attention::TDEntropyAttentionModel, stats)
    #return round(Int, attention.max_sweeps * attention.rejuv_smoothness^logsumexp(stats))
    sweeps = min(attention.max_sweeps, sum(stats))
    round(Int, sweeps)
end

"""
for now returns false, no early stopping
"""
function early_stopping(::TDEntropyAttentionModel, new_stats, prev_stats)
    return false
end
