export visualize_inference


function _states(t::Gen.Trace)
    @>> t begin
        get_retval
        last
    end
end

"""
Returns a matrix whose columns contain the indeces of the living descendants
for each time step
"""
function living_descendants(geneology::Vector{Vector{Int64}})
    t = length(geneology)
    p = length(geneology[1])
    family_tree = Matrix{Int64}(undef, p, t)
    # last generation is coppied
    family_tree[:, end] = collect(1:p)
    t == 1 && return family_tree
    for i = (t-1):-1:1
        pg = geneology[i + 1]
        pft = family_tree[:, i + 1]
        family_tree[:, i] = pg[pft]
    end
    return family_tree
end

"""
    creates visuals of many things for a given inference chain:
        1) plots attention statistics
        2) renders masks (observed and predicted)
        3) renders the scene
"""
function visualize_inference(chain, chain_path, gt_causal_graphs, gm,
                             graphics, attention, path;
                             render_model=false,
                             padding = 3,
                             n_back_cgs = 3)

    np = length(chain.state.traces)
    dg = extract_digest(chain_path)
    aux_state = dg[:, :auxillary]
    geneology = dg[:, :parents]

    k = size(aux_state, 1)
    # attention_weights = [aux_state[t].sensitivities for t = 1:k]
    # attention_weights = collect(hcat(attention_weights...)')
    # plot_compute_weights(attention_weights, path)

    # attended = Dict{Pair{Int64, Int64}, Vector{Int64}}
    ld = living_descendants(geneology)
    attended = Dict()
    for t = 1:k, p = 1:np
        ancestral_id = ld[p, t]
        ne = length(aux_state[t].sensitivities[ancestral_id])
        alloc = get(aux_state[t].allocated, ancestral_id, zeros(Int64, ne))
        attended[t => p] = alloc
    end
    # attended = Matrix{Float64}(undef, size(attention_weights, 2), k)
    # for t=1:k
    #     attended[:, t] = aux_state[t].allocated
    # end
    max_c = attention.sweeps
    arrousal = Float64[logsumexp(aux_state[t].arrousal) - log(np) for t = 1:k]
    plot_compute_weights(arrousal, path)
    cycles = Int64[sum(aux_state[t].cycles) for t = 1:k]
    plot_rejuvenation(cycles, max_c, path)

    # # rendering observed flow masks from receptive fields
    choices = @>> chain.state.traces first get_choices
    for t=1:min(k, 10)
        render_masks(choices, t, gm, graphics,
                     joinpath(path, "obs_rf_masks"))
    end


    # # visualizing inference on stimuli
    # unweighted = Gen.sample_unweighted_traces(chain.state, np)
    traces = chain.state.traces
    pf_st = Matrix{InertiaKernelState}(undef, np, k)
    for i = 1:np
        pf_st[i, :] = _states(traces[i])
    end

    render_scene(gm, gt_causal_graphs[2:end], pf_st, attended;
                 base = joinpath(path, "render"))



    # # rendering predicted distribution flow masks from receptive fields
    states = map(world, pf_st[1, :])
    for t=1:min(k, 10)
        render_masks(states, t, gm, graphics,
                     joinpath(path, "pred_dist_rf_masks"))
    end
end
