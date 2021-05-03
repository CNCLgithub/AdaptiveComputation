export visualize_inference


"""
    creates visuals of many things for a given inference chain:
        1) plots attention statistics
        2) renders masks (observed and predicted)
        3) renders the scene
"""
function visualize_inference(results, gt_causal_graphs, gm,
                             receptive_fields, rf_dims, attention, path;
                             render_model=false,
                             padding = 3,
                             n_back_cgs = 3)

    extracted = extract_chain(results)
    causal_graphs = extracted["unweighted"][:causal_graph]
    traces = extracted["unweighted"][:trace]

    k = size(causal_graphs, 1)
    
    aux_state = extracted["aux_state"]
    attention_weights = [aux_state[t].stats for t = 1:k]
    attention_weights = collect(hcat(attention_weights...)')

    plot_compute_weights(attention_weights, path)

    attempts = Vector{Int}(undef, k)
    attended = Vector{Vector{Float64}}(undef, k)
    for t=1:k
        attempts[t] = aux_state[t].attempts
        attended[t] = aux_state[t].attended_trackers
    end
    MOT.plot_attention(attended, attention.sweeps, path)
    plot_rejuvenation(attempts, path)
        
    # rendering observed flow masks from receptive fields
    choices = get_choices(traces[end,1])
    for t=1:k
        render_rf_masks(choices, t, gm, receptive_fields,
                        joinpath(path, "obs_rf_masks"))
    end
    
    # rendering predicted distribution flow masks from receptive fields
    states = get_retval(traces[end,1])[2]
    states = collect(RFState, states)
    for t=1:k
        render_rf_masks(states, t, gm, receptive_fields,
                        joinpath(path, "pred_dist_rf_masks"))
    end

    # visualizing inference on stimuli
    #pf_cgs = @>> 1:size(causal_graphs, 1) map(i -> causal_graphs[i,1])
    # T x n_steps_back of cgs
    pf_cgs = @>> traces[:,1] map(trace -> get_n_back_cgs(trace, n_back_cgs))
    render_scene(gm, gt_causal_graphs, pf_cgs,
                 rf_dims, attended;
                 base = joinpath(path, "render"))
end

function get_n_back_cgs(trace::Trace, n_back_cgs::Int64)::Vector{CausalGraph}
    t, dm, gm = Gen.get_args(trace)
    ret = Gen.get_retval(trace)
    @>> 0:n_back_cgs-1 reverse map(i -> ret[2][max(1, t-i)].cg)
end
