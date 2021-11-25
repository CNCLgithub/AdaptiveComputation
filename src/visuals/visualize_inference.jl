export visualize_inference


function _states(t::Gen.Trace)
    @>> t begin
        get_retval
        last
    end
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



    k = size(aux_state, 1)
    attention_weights = [aux_state[t].sensitivities for t = 1:k]
    attention_weights = collect(hcat(attention_weights...)')
    plot_compute_weights(attention_weights, path)

    attended = Matrix{Float64}(undef, size(attention_weights, 2), k)
    for t=1:k
        attended[:, t] = aux_state[t].allocated
    end
    max_c = attention.sweeps * np
    plot_attention(attended, max_c, path)
    plot_rejuvenation(sum(attended, dims = 1), max_c, path)

    # # rendering observed flow masks from receptive fields
    choices = @>> chain.state.traces first get_choices
    # for t=1:k
    #     render_masks(choices, t, gm, graphics,
    #                  joinpath(path, "obs_rf_masks"))
    # end


    # # visualizing inference on stimuli
    unweighted = Gen.sample_unweighted_traces(chain.state, np)
    pf_st = Matrix{InertiaKernelState}(undef, np, k)
    for i = 1:np
        pf_st[i, :] = _states(unweighted[i])
    end

    render_scene(gm, gt_causal_graphs, pf_st, attended;
                 base = joinpath(path, "render"))



    # # rendering predicted distribution flow masks from receptive fields
    states = map(world, pf_st[1, :])
    for t=1:k
        render_masks(states, t, gm, graphics,
                     joinpath(path, "pred_dist_rf_masks"))
    end
end
