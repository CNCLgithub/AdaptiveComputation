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
function visualize_inference(chain, dg, gt_states, gm::T, path) where {T}

    @unpack state, proc = chain
    np = length(state.traces)
    aux_state = dg[:, :auxillary]
    k = length(aux_state)

    # aggregate cycles per latent
    max_arrousal = proc.attention.max_arrousal
    attended = Vector{Vector{Float64}}(undef, k)
    for t=1:k
        importance = aux_state[t].importance
        arrousal  = aux_state[t].arrousal
        attended[t] = @. importance * arrousal / max_arrousal
    end

    # # visualizing inference on stimuli
    # unweighted = Gen.sample_unweighted_traces(chain.state, np)
    traces = state.traces
    pf_st = Matrix{GMState{T}}(undef, np, k)
    for i = 1:np
        pf_st[i, :] = _states(traces[i])
    end

    render_scene(gm, gt_states, pf_st, attended;
                 base = path)
end
