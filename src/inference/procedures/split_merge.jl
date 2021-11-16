@gen function split_merge_proposal(trace::Gen.Trace, elem::Int64)
    # what moves are possible?
    split_weight, merge_weights = move_weights(trace)

    if ({:split} ~ bernoulli(split_weight))
        cg = graph_from_trace(trace)
        {:tracker} ~ inertia_tracker(cg)

    end
end


@transform split_merge_involution (t, u) to (t_prime, u_prime) begin

    cg = graph_from_trace(t)
    _, elem = get_args(u)
    split = @read(u[:split], :discrete)

    if split

        # copy tracker from u to t_prime
        @copy(u[:tracker] => t_prime[:kernel => t => ne + 1])

    else
    end


end
